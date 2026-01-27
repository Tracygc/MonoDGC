import torch
import math
import numpy as np
import random
import scipy.sparse as sp
import torch_geometric
import dgl
import torch_geometric.utils as utils
from torch_scatter import scatter_add
import networkx as nx
from scipy.linalg import fractional_matrix_power, inv

def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, seed=12134):
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]
    #print(test_idx)

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    
    return data


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def edge_index_to_adj(edge_index):
    #adj = to_sparse_tensor(edge_index)
    edge_index = torch_geometric.utils.to_scipy_sparse_matrix(edge_index)
    adj = sparse_mx_to_torch_sparse_tensor(edge_index)
    adj = adj.to_dense()
    one = torch.ones_like(adj)
    adj = adj + adj.t()  # 对称化
    adj = torch.where(adj < 1, adj, one)
    diag = torch.diag(adj)
    a_diag = torch.diag_embed(diag)  # 去除自环
    adj = adj - a_diag
    # adjaddI = adj + torch.eye(adj.shape[0]) #加自环
    # d1 = torch.sum(adjaddI, dim=1)
    return adj  # 稠密矩阵

def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian

    #adjacency_matrix(transpose, scipy_fmt="csr")
    #A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    A = g.adj(scipy_fmt='csr')
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N #通过将邻接矩阵归一化处理，然后使用单位矩阵减去归一化的邻接矩阵，得到拉普拉斯矩阵

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs #返回指定数量（k=pos_enc_dim+1）的最小实部特征值和对应的特征向量
    EigVec = EigVec[:, EigVal.argsort()] # increasing order 排序
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()

    return lap_pos_enc

def yLapEncoding(graph, pos_enc_dim):
        edge_attr = graph.edge_attr
        edge_index, edge_attr = utils.get_laplacian(
            graph.edge_index, edge_attr, normalization=None,
            num_nodes=graph.num_nodes)
        L = utils.to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort()  # increasing order
        EigVal, EigVec = np.real(EigVal[idx]), np.real(EigVec[:, idx])

        return torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

def LapEncoding(graph, pos_enc_dim):
        edge_attr = graph.edge_attr
        edge_index, edge_attr = utils.get_laplacian(
            graph.edge_index, edge_attr, normalization=None,
            num_nodes=graph.num_nodes)
        L = utils.to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = np.real(EigVal[idx]), np.real(EigVec[:,idx])
        idx2 = (-EigVal).argsort()  # decreasing order
        EigVal2, EigVec2 = np.real(EigVal[idx2]), np.real(EigVec[:, idx2])
        #从大到小排列了
        return torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float(),torch.from_numpy(EigVec2[:, 1:pos_enc_dim+1]).float()



#第二种获取最大到小
def LapEncoding2(graph, pos_enc_dim):
    edge_attr = graph.edge_attr
    edge_index, edge_attr = utils.get_laplacian(
        graph.edge_index, edge_attr, normalization=None,
        num_nodes=graph.num_nodes)
    L = utils.to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = np.real(EigVal[idx]), np.real(EigVec[:, idx])
    #直接提取特征向量中的后面几个列（即最后 pos_enc_dim 列）
    return torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float(), torch.from_numpy(EigVec[:, -pos_enc_dim:]).float()


def RWEncoding(graph, pos_enc_dim):
        W0 = normalize_adj(graph.edge_index, num_nodes=graph.num_nodes).tocsc()
        W = W0
        vector = torch.zeros((graph.num_nodes, pos_enc_dim))
        vector[:, 0] = torch.from_numpy(W0.diagonal())
        for i in range(pos_enc_dim - 1):
            W = W.dot(W0)
            vector[:, i + 1] = torch.from_numpy(W.diagonal())
        return vector.float()


def normalize_adj(edge_index, edge_weight=None, num_nodes=None):
    edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1),
                                 device=edge_index.device)
    num_nodes = utils.num_nodes.maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    return utils.to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)