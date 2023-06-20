import gzip
import json
import torch
import os.path as osp
import pickle, os, numpy as np
import scipy.io as sio
# from math import comb
from scipy.special import comb
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.data.data import Data
from torch_geometric.utils import to_undirected
import networkx as nx

# two more simulation dataset from PNA and SMP paper
from core.data_utils.data_pna import GraphPropertyDataset
from core.data_utils.data_cycles import CyclesDataset
from core.data_utils.sbm_cliques import CliqueSBM
from core.data_utils.tudataset_gin_split import TUDatasetGINSplit


class QM9SplitsDataset(InMemoryDataset):
    """A PyTorch Geometric dataset with the splits from GNN-FiLM,
    following the code from SPNNs.

    See: https://arxiv.org/pdf/2206.01003.pdf
    """
    def __init__(self, root, split="train", transform=None):
        self.split = split
        super().__init__(root, transform)
        graphs = self.read_qm9(root, split, transform, root)
        if transform:
            graphs = [transform(g) for g in graphs]
        self.data, self.slices = self.collate(graphs)

        # Store the preprocessed dataset
        prepro_path = self.processed_paths[0]
        os.makedirs(osp.dirname(prepro_path), exist_ok=True)
        torch.save((self.data, self.slices), prepro_path)

    @property
    def processed_file_names(self):
        return f'{self.split}-qm9.pt'

    def read_qm9(self, direc, file, transform_f, processed_root):
        path = osp.join(direc, file + ".jsonl.gz")
        with gzip.open(path, "r") as f:
            data = f.read().decode("utf-8")
            graphs = [json.loads(jline) for jline in data.splitlines()]
            pyg_graphs = [
                self.map_qm9_to_pyg(graph, make_undirected=True, remove_dup=False)
                for graph in graphs
            ]
            return pyg_graphs

    def map_qm9_to_pyg(self, json_file, make_undirected=True, remove_dup=False):
        edge_index = np.array([[g[0], g[2]] for g in json_file["graph"]]).T
        edge_attributes = np.array(
            [g[1] - 1 for g in json_file["graph"]]
        )

        # This will invariably cost us edge types because we reduce duplicates
        if make_undirected:
            edge_index_reverse = edge_index[[1, 0], :]
            # Concat and remove duplicates
            if remove_dup:
                edge_index = torch.LongTensor(
                    np.unique(
                        np.concatenate([edge_index, edge_index_reverse], axis=1), axis=1
                    )
                )
            else:
                edge_index = torch.LongTensor(
                    np.concatenate([edge_index, edge_index_reverse], axis=1)
                )
                edge_attributes = torch.LongTensor(
                    np.concatenate([edge_attributes, np.copy(edge_attributes)], axis=0)
                )
        x = torch.FloatTensor(np.array(json_file["node_features"]))
        y = torch.FloatTensor(np.array(json_file["targets"]).T)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attributes, y=y)


class RookShrikhandeDataset(InMemoryDataset):
    """Implements an in-memory 4x4 Rook and Shrikhande (3-WL, Strongly Regular) dataset
    with the two graphs.

    This is helpful to study 3-WL expressivity.
    """
    ROOK_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 8), (0, 12), (1, 2), (1, 3), (1, 5), (1, 9), (1, 13), (2, 3), (2, 6), (2, 10), (2, 14), (3, 7), (3, 11), (3, 15), (4, 5), (4, 6), (4, 7), (4, 8), (4, 12), (5, 6), (5, 7), (5, 9), (5, 13), (6, 7), (6, 10), (6, 14), (7, 11), (7, 15), (8, 9), (8, 10,), (8, 11), (8, 12), (9, 10), (9, 11), (9, 13), (10, 11), (10, 14), (11, 15), (12, 13), (12, 14), (12, 15), (13, 14), (13, 15), (14, 15)]
    SHRIKHANDE_EDGES = [(0, 1), (0, 3), (0, 4), (0, 5), (0, 12), (0, 15), (1, 2), (1, 5), (1, 6), (1, 12), (1, 13), (2, 3), (2, 6), (2, 7), (2, 13), (2, 14), (3, 4), (3, 7), (3, 14), (3, 15), (4, 5), (4, 7), (4, 8), (4, 9), (5, 6), (5, 9), (5, 10), (6, 7), (6, 10), (6, 11), (7, 8), (7, 11), (8, 9), (8, 11), (8, 12), (8, 13), (9, 10), (9, 13), (9, 14), (10, 11), (10, 14), (10, 15), (11, 12), (11, 15), (12, 13), (12, 15), (13, 14), (14, 15)]

    def __init__(self, root, transform=None):
        super().__init__('.', transform)
        x = torch.ones((16, 1)).long()
        rook_tensor = to_undirected(torch.Tensor([list(x) for x in zip(*self.ROOK_EDGES)]).long())
        shrik_tensor = to_undirected(torch.Tensor([list(x) for x in zip(*self.SHRIKHANDE_EDGES)]).long())

        rook_data = Data(x=x, edge_index=rook_tensor, y=0)
        shrik_data = Data(x=x, edge_index=shrik_tensor, y=1)
        self.data, self.slices = self.collate([rook_data, shrik_data])


class ColoredEdgesDataset(InMemoryDataset):
    """Implements an in-memory distance-to-colored edges dataset

    This is helpful to study expressivity from the point of view of edge information.
    """
    @property
    def raw_file_names(self):
        return f"{self.root}/data.json.gz"

    @property
    def processed_file_names(self):
        return f"{self.task}-Coloring.pt"

    def undirected_repeat(self, edge_index):
        return torch.cat([edge_index, edge_index[[1, 0], :]], dim=-1)

    def process(self):
        with gzip.open(self.raw_file_names, "rt") as f:
            flat_dataset = [sample for sample in json.load(f) if sample["task"] == self.task]
        data_as_tensors = [
            Data(
                x=torch.Tensor(sample["x"]).long().reshape(-1, 1),
                y=sample["y"],
                edge_index=self.undirected_repeat(torch.Tensor(sample["edge_index"]).long()),
                edge_attr=torch.Tensor(sample["edge_attr"]).long().repeat(2).reshape(-1, 1),
                edge_rel_mask=torch.Tensor(sample["edge_rel_mask"]).long().repeat(2).reshape(-1, 1)
            ) for sample in flat_dataset]
        os.makedirs(f"{self.root}/processed/", exist_ok=True)
        torch.save(self.collate(data_as_tensors), self.processed_paths[0])

    def __init__(self, root, task, transform=None):
        self.task = task
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


class ProximityDataset(InMemoryDataset):
    """Implements an in-memory dataset for the K-Proximity datasets
    proposed by Abboud et al. in Shortest Path Networks for Graph Property Prediction
    (LoG 2022).
    """
    url = "https://zenodo.org/record/6557736/files/Proximity.zip?download=1"
    valid_k = (1, 3, 5, 8, 10)
    def __init__(self, root, k=1, pre_transform=None, transform=None):
        self.k = k
        if self.k not in self.valid_k:
            raise ValueError(f"The value of `k` must be one of {self.valid_k}.")
        super().__init__(root, pre_transform, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"{self.k}-Prox/raw/data_list.pickle"]

    @property
    def processed_file_names(self):
        return [f"{self.k}-data.pt"]

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)

    def process(self):
        raw_data = osp.join(self.root, f"{self.k}-Prox", "raw", "data_list.pickle")
        with open(raw_data, "rb") as f:
            data_list = pickle.load(f)
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        torch.save(self.collate(data_list), self.processed_paths[0])


class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PlanarSATPairsDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["GRAPHSAT.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass  

    def process(self):
        # Read data into huge `Data` list.
        # The data pickle used an older PyG version, so failed loading -- the raw version should work.
        # data_list = pickle.load(open(os.path.join(self.root, "raw/GRAPHSAT.pkl"), "rb"))
        data_list = pickle.load(open(os.path.join(self.root, "raw/GRAPHSAT_Raw.pkl"), "rb"))
        data_list = [Data(**tensor_dict) for tensor_dict in data_list]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GraphCountDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphCountDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        a=sio.loadmat(self.raw_paths[0])
        self.train_idx = torch.from_numpy(a['train_idx'][0])
        self.val_idx = torch.from_numpy(a['val_idx'][0]) 
        self.test_idx = torch.from_numpy(a['test_idx'][0]) 

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]
        # list of output
        Y=a['F']

        data_list = []
        for i in range(len(A)):
            a=A[i]
            A2=a.dot(a)
            A3=A2.dot(a)
            tri=np.trace(A3)/6
            tailed=((np.diag(A3)/2)*(a.sum(0)-2)).sum()
            cyc4=1/8*(np.trace(A3.dot(a))+np.trace(A2)-2*A2.sum())
            cus= a.dot(np.diag(np.exp(-a.dot(a).sum(1)))).dot(a).sum()

            deg=a.sum(0)
            star=0
            for j in range(a.shape[0]):
                star+=comb(int(deg[j]),3)

            expy=torch.tensor([[tri,tailed,star,cyc4,cus]])

            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.ones(A[i].shape[0],1).long() # change to category
            #y=torch.tensor(Y[i:i+1,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=expy))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class SRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]  #sr251256  sr351668

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i,datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(),1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def calculate_stats(dataset):
    num_graphs = len(dataset)
    ave_num_nodes = np.array([g.num_nodes for g in dataset]).mean()
    ave_num_edges = np.array([g.num_edges for g in dataset]).mean()
    print(f'# Graphs: {num_graphs}, average # nodes per graph: {ave_num_nodes}, average # edges per graph: {ave_num_edges}.')
    

if __name__ == "__main__":
    # dataset = PlanarSATPairsDataset('data/EXP')
    dataset = GraphCountDataset('data/subgraphcount')
    print(dataset.data.x.max(), dataset.data.x.min())
