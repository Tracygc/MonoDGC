from typing import Optional, List
import math
import copy
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from utils.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn


from .graph_cross_former2 import GraphCrossFormerBlock


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class DepthAwareDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, group_num=1):
        super().__init__()

        # Standard Attention Modules
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn_depth = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_depth = nn.Dropout(dropout)
        self.norm_depth = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.group_num = group_num

        self.use_graph = True
        if self.use_graph:
            self.graph_block = GraphCrossFormerBlock(d_model, n_heads, dropout=dropout)
            self.norm_graph = nn.LayerNorm(d_model)


    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self,
                tgt,
                query_pos,
                reference_points,
                src,
                src_spatial_shapes,
                level_start_index,
                src_padding_mask,
                depth_pos_embed,
                mask_depth,
                bs,
                predicted_3d_centers=None):

        # 1. Depth Cross Attention
        tgt2 = self.cross_attn_depth(tgt.transpose(0, 1),
                                     depth_pos_embed,
                                     depth_pos_embed,
                                     key_padding_mask=mask_depth)[0].transpose(0, 1)
        tgt = tgt + self.dropout_depth(tgt2)
        tgt = self.norm_depth(tgt)

        # 2. Self Attention
        q = k = self.with_pos_embed(tgt, query_pos)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = tgt.transpose(0, 1)


        num_queries = q.shape[0]
        if self.training:
            num_noise = num_queries - self.group_num * 50
            q_noise = q[:num_noise].repeat(1, self.group_num, 1)
            k_noise = k[:num_noise].repeat(1, self.group_num, 1)
            v_noise = v[:num_noise].repeat(1, self.group_num, 1)
            q = torch.cat([q_noise, q[num_noise:]], dim=0)
            k = torch.cat([k_noise, k[num_noise:]], dim=0)
            v = torch.cat([v_noise, v[num_noise:]], dim=0)


        tgt2 = self.self_attn(q, k, v)[0]
        tgt2 = tgt2.transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 3. Visual Cross Attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 4. FFN
        tgt = self.forward_ffn(tgt)

        # --- 5. Graph Cross-Former Refinement ---
        if self.use_graph and predicted_3d_centers is not None:
            tgt_graph = self.graph_block(tgt, predicted_3d_centers)
            tgt = self.norm_graph(tgt + tgt_graph)

        return tgt


class DepthAwareDecoder_MonoDGC(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, depth_pos_embed=None, mask_depth=None, bs=None):
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        bs = src.shape[0]

        for lid, layer in enumerate(self.layers):
            # Calculate Reference Points Input
            if reference_points.shape[-1] == 6:
                reference_points_input = reference_points[:, :, None] * torch.cat(
                    [src_valid_ratios, src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]


            current_3d_centers = None
            if self.bbox_embed is not None:

                tmp = self.bbox_embed[lid](output)  # [B, N, 6]
                if reference_points.shape[-1] == 6:
                    new_ref = tmp + inverse_sigmoid(reference_points)
                    new_ref = new_ref.sigmoid()
                else:
                    new_ref = tmp
                    new_ref[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_ref = new_ref.sigmoid()

                current_3d_centers = torch.cat([reference_points, torch.zeros_like(reference_points[..., :1])], dim=-1)


            output = layer(output,
                           query_pos,
                           reference_points_input,
                           src,
                           src_spatial_shapes,
                           src_level_start_index,
                           src_padding_mask,
                           depth_pos_embed,
                           mask_depth,
                           bs,
                           predicted_3d_centers=current_3d_centers)

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 6:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class GraphDet3DTransformer_MonoDGC(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_decoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, group_num=1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.group_num = group_num

        decoder_layer = DepthAwareDecoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points, group_num=group_num)
        self.decoder = DepthAwareDecoder_MonoDGC(decoder_layer, num_decoder_layers, return_intermediate_dec)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(self, intermediate_output, query_embeds, depth_pos_embed):
        memory = intermediate_output['memory']
        reference_points = intermediate_output['reference_points']
        spatial_shapes = intermediate_output['spatial_shapes']
        level_start_index = intermediate_output['level_start_index']
        valid_ratios = intermediate_output['valid_ratios']
        mask_flatten = intermediate_output['mask_flatten']

        bs, _, c = memory.shape
        tgt = query_embeds
        init_reference_out = reference_points

        depth_pos_embed = depth_pos_embed.flatten(2).permute(2, 0, 1)
        mask_depth = None
        query_embeds = None

        hs, inter_references = self.decoder(
            tgt,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_embeds,
            mask_flatten,
            depth_pos_embed,
            mask_depth,
            bs=bs)

        return hs, init_reference_out, inter_references


def build_graph_det3d_transformer(cfg):
    """ Builder function for the new Graph-based Transformer """
    return GraphDet3DTransformer_MonoDGC(
        d_model=cfg['hidden_dim'],
        dropout=cfg['dropout'],
        activation="relu",
        nhead=cfg['nheads'],
        dim_feedforward=cfg['dim_feedforward'],
        num_decoder_layers=cfg['dec_layers'],
        return_intermediate_dec=cfg['return_intermediate_dec'],
        num_feature_levels=cfg['num_feature_levels'],
        dec_n_points=cfg['dec_n_points'])