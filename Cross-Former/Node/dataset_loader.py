import torch
import math
import pickle
import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import Actor
# from torch_geometric.datasets import PCQM4Mv2
from torch_geometric.nn import APPNP
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_geometric.io import read_npz
#from cSBM_dataset import dataset_ContextualSBM
import torch_geometric
import dgl
import utils
from utils import random_planetoid_splits
from torch_geometric.datasets import TUDataset





class WebKB(InMemoryDataset):

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           'master/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def DataLoader(name):
    # name = name.lower()

    if name in ['cora', 'citeseer', 'pubmed']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures())
    elif name in ['computers', 'photo']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Amazon(path, name, T.NormalizeFeatures())


    elif name in ['chameleon', 'squirrel']:
        # use everything from "geom_gcn_preprocess=False" and
        # only the node label y from "geom_gcn_preprocess=True"
        preProcDs = WikipediaNetwork(
            root='./data/', name=name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(
            root='./data/', name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index
        return dataset

    elif name in ['film']:
        dataset = Actor(
            root='/home/luwei/die/data/film', transform=T.NormalizeFeatures())
    elif name in ['texas', 'cornell','wisconsin']:
        dataset = WebKB(root='/home/luwei/die/data/',
                        name=name, transform=T.NormalizeFeatures())
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')


    return dataset

