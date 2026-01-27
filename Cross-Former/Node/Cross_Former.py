import torch_geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
import scipy.sparse as sp
import utils
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian
from torch_geometric.nn import MessagePassing
from scipy.special import comb
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix
import scipy.sparse as sp
import scipy
from torch_geometric.nn import global_mean_pool

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



class Cross_Former(nn.Module):
    def __init__(self, dataset, args, num_heads=1, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(Cross_Former, self).__init__()

        self.dropout = args.dropout
        self.hidden = args.hidden
        self.l = args.l
        self.device = args.device
        # self.hidden_dim = args.hidden_dim
        # self.ffn_dim = 2 * args.hidden_dim
        self.ffn_dim = 2 * args.hidden_dim
        self.num_heads = num_heads
        self.device=args.device
        # # self.n_class = dataset.num_classes
       # self.dropout_rate = args.dropout_rate
        self.attention_dropout_rate = args.attention_dropout

        self.lin1 = nn.Linear(dataset.num_features, args.hidden)
        self.dataset=args.dataset
        # Transformer layers
        N=dataset[0].x.shape[0]
        encoders = \
            [EncoderLayer(N,args.hidden, self.ffn_dim, self.dropout, self.attention_dropout_rate, self.num_heads,self.device)
             for _ in range(self.l)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(args.hidden)


        self.attn_layer = nn.Linear(2*args.hidden, 1)
        self.linear = nn.Linear(2*args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.prop.reset_parameters()


    def forward(self, data):
        data=data.to(self.device)
        x, edge_index = data.x, data.edge_index
        #(183,1703)  (2,574)
        edge_index = torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index)  #(83,183)
        # edge_index = normalize_adj(edge_index)
        edge_index = sparse_mx_to_torch_sparse_tensor(edge_index)#(83,183)
        adj = edge_index.to(self.device)


        adj = adj.to_dense()

        x = F.dropout(x, self.dropout, training=self.training)
        x0 = self.lin1(x)  # 自身
        x1 = self.lin1(x)  # 自身
        #x1 = self.prop1(x, edge_index)



        for enc_layer in self.layers:
            x1 = enc_layer(x1,adj)

       # print(x1.shape)
            #(2708,64)
        #x = self.transformer_encoder(x)
        x1 = self.final_ln(x1)



        # Aggregation (e.g., weighted sum or pooling)
        x1 = F.relu(x1)  # Apply ReLU activation

        h_n = torch.cat((x0,x1), dim=1)  # [V, 2*H]
        h_n = F.dropout(h_n, self.dropout, training=self.training)
        x = self.linear(h_n)

        # h_n = x0 + x1  # [V, 2*H]
        # # print(h_n.shape)
        # h_n = torch.relu(h_n)
        # h_n = F.dropout(h_n, self.dropout, training=self.training)
        # x = self.linear(h_n)

        #print(x.shape)

        # X = x.cpu().detach().numpy()
        # # torch.save(tensor, 'tensor_data.pth')
        # # np.save('./z',  z)
        # np.save('./z/X_{}'.format(self.dataset), X)
        #
        # A = adj.cpu().detach().numpy()
        # # torch.save(tensor, 'tensor_data.pth')
        # # np.save('./z',  z)
        # # np.save('./z/A_{}'.format(self.dataset), A)

        return torch.log_softmax(x, dim=1)



class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, N,hidden_size, attention_dropout_rate, num_heads,device):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.hidden_size= hidden_size
        self.att_size = att_size = hidden_size // num_heads

        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)

        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)
        self.layer = nn.Linear(N, hidden_size)
        self.attn_layer = nn.Linear(2 * hidden_size, 1)

        self.linear = nn.Linear(hidden_size, hidden_size)
        self.attw = torch.nn.Linear(hidden_size * 2, 2)

        self.att_0, self.att_1 = 0, 0
        self.att_vec_0, self.att_vec_1 = (
            Parameter(torch.FloatTensor(1 * hidden_size, 1).to(device)),
            Parameter(torch.FloatTensor(1 * hidden_size, 1).to(device))
        )
        self.att_vec = Parameter(torch.FloatTensor(2, 2).to(device))
        self.reset_parameters()

    def reset_parameters(self):
        std_att = 1.0 / math.sqrt(self.att_vec_1.size(1))
        std_att_vec = 1.0 / math.sqrt(self.att_vec.size(1))

        self.att_vec_1.data.uniform_(-std_att, std_att)
        self.att_vec_0.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

    def attention2(self, output_0, output_1):
        logits = (
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm((output_0), self.att_vec_0),
                            torch.mm((output_1), self.att_vec_1),
                        ],
                        1,
                    )
                ),
                self.att_vec
            )
        )
        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None]


    def forward(self, q, adj,  attn_bias=None):
       # q = self.linear_q(q)  # (183,64)
        # k = self.linear_k(k)
       # v = self.linear_v(v)
        k = q.transpose(0, 1)  # (64,183)

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
       # q = q * self.scale
        x = torch.matmul(q, k)
        # print(x.shape) #(183,183)

        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=1)  # (183,183)
        # print("1",x)

        x = self.att_dropout(x)
        x = adj.matmul(x)
        # print(x.shape) #torch.Size([183, 183])

        x1 = self.layer(x)

        AT = adj.t()  # 计算A的转置矩阵
       # adj = adj * self.scale
        AAT = torch.matmul(adj, AT)  # 计算A与AT的乘积

        AAT = torch.softmax(AAT, dim=1)  # (183,183)
        AAT = self.att_dropout(AAT)

        x2 = AAT.matmul(q)  # 9183,64)
        # print(x2.shape)
        x2 = self.output_layer(x2)

        self.att_0, self.att_1 = self.attention2((x1), (x2))
        # print(self.att_0.size())
        x = self.att_0 * x1 + self.att_1 * x2


        return x



class EncoderLayer(nn.Module):
    def __init__(self,N, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads,device):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            N,hidden_size, attention_dropout_rate, num_heads,device)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adj, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, adj, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x



