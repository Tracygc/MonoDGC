import torch
import torch.nn as nn
# import torch.nn.functional as F


class GraphCrossFormerBlock(nn.Module):

    def __init__(self, d_model=256, nhead=8, k_neighbors=9, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.k_neighbors = k_neighbors
        self.nhead = nhead

        self.node_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_node = nn.LayerNorm(d_model)

        self.struct_embed = nn.Sequential(
            nn.Linear(k_neighbors, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.struct_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_struct = nn.LayerNorm(d_model)


        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)

        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def build_dynamic_topology(self, pred_3d_centers):
        """
        Constructs the graph topology based on current 3D predictions.
        Args:
            pred_3d_centers: [Batch, Num_Queries, 3] (x, y, z)
        Returns:
            topology_feat: [Batch, Num_Queries, K_Neighbors] -> Distance features
        """
        # B, N, C = pred_3d_centers.shape
        dist_matrix = torch.cdist(pred_3d_centers, pred_3d_centers, p=2)

        topk_values, topk_indices = torch.topk(dist_matrix, k=self.k_neighbors, dim=-1, largest=False)

        topology_feat = torch.exp(-topk_values)

        return topology_feat

    def forward(self, query_content, pred_3d_centers):

        return "The code will be released soon."