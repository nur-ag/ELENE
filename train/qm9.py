import torch
from core.config import cfg, update_cfg
from core.train_helper import run 
from core.model import GNNAsKernel
from core.transform import SubgraphsTransform

from core.data import QM9SplitsDataset, calculate_stats


NUM_NODE_FEATURES = 15
NUM_TASKS = 13
# This file is essentially tailor-made for QM9
TASKS = [
    "mu",
    "alpha",
    "HOMO",
    "LUMO",
    "gap",
    "R2",
    "ZPVE",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
    "Omega",
]
# QM9 y values were normalized, so we need to de-normalize.
CHEMICAL_ACC_NORMALISING_FACTORS = [
    0.066513725,
    0.012235489,
    0.071939046,
    0.033730778,
    0.033486113,
    0.004278493,
    0.001330901,
    0.004165489,
    0.004128926,
    0.00409976,
    0.004527465,
    0.012292586,
    0.037467458,
]

def create_dataset(cfg): 
    # No need to do offline transformation
    transform = SubgraphsTransform(cfg.subgraph.hops, 
                                   walk_length=cfg.subgraph.walk_length, 
                                   p=cfg.subgraph.walk_p, 
                                   q=cfg.subgraph.walk_q, 
                                   repeat=cfg.subgraph.walk_repeat,
                                   sampling_mode=cfg.sampling.mode, 
                                   minimum_redundancy=cfg.sampling.redundancy, 
                                   shortest_path_mode_stride=cfg.sampling.stride, 
                                   random_mode_sampling_rate=cfg.sampling.random_rate,
                                   random_init=True)

    transform_eval = SubgraphsTransform(cfg.subgraph.hops, 
                                        walk_length=cfg.subgraph.walk_length, 
                                        p=cfg.subgraph.walk_p, 
                                        q=cfg.subgraph.walk_q, 
                                        repeat=cfg.subgraph.walk_repeat,
                                        sampling_mode=None, 
                                        random_init=False)

    root = 'data/QM9'
    train_dataset = QM9SplitsDataset(root, split='train', transform=transform)
    val_dataset = QM9SplitsDataset(root, split='valid', transform=transform_eval) 
    test_dataset = QM9SplitsDataset(root, split='test', transform=transform_eval)   

    # When without randomness, transform the data to save a bit time
    # torch.set_num_threads(cfg.num_workers)
    if (cfg.sampling.mode is None and cfg.subgraph.walk_length == 0) or (cfg.subgraph.online is False):
        train_dataset = [x for x in train_dataset]
    val_dataset = [x for x in val_dataset]
    test_dataset = [x for x in test_dataset]

    # print('------------Train--------------')
    # calculate_stats(train_dataset)
    # print('------------Validation--------------')
    # calculate_stats(val_dataset)
    # print('------------Test--------------')
    # calculate_stats(test_dataset)
    # print('------------------------------')

    return train_dataset, val_dataset, test_dataset

def create_model(cfg):
    model = GNNAsKernel(NUM_NODE_FEATURES, None, 
                        nhid=cfg.model.hidden_size, 
                        nout=NUM_TASKS if cfg.task < 0 else 1, 
                        nlayer_outer=cfg.model.num_layers,
                        nlayer_inner=cfg.model.mini_layers,
                        gnn_types=[cfg.model.gnn_type], 
                        hop_dim=cfg.model.hops_dim,
                        use_normal_gnn=cfg.model.use_normal_gnn, 
                        vn=cfg.model.virtual_node, 
                        pooling=cfg.model.pool,
                        embs=cfg.model.embs,
                        embs_combine_mode=cfg.model.embs_combine_mode,
                        mlp_layers=cfg.model.mlp_layers,
                        dropout=cfg.train.dropout, 
                        subsampling=True if cfg.sampling.mode is not None else False,
                        online=cfg.subgraph.online,
                        igel_length=cfg.igel.embedded_vector_length,
                        igel_edge_encodings=cfg.igel.use_edge_encodings,
                        eigel_max_degree=cfg.eigel.max_degree,
                        eigel_max_distance=cfg.eigel.max_distance,
                        eigel_relative_degrees=cfg.eigel.relative_degrees,
                        eigel_model_type=cfg.eigel.model_type,
                        eigel_embedding_dim=cfg.eigel.embedding_dim,
                        eigel_reuse_embeddings=cfg.eigel.reuse_embeddings,
                        eigel_layer_indices=cfg.eigel.layer_indices,
                        use_gnn=cfg.use_gnn)
    model.task = cfg.task
    return model

def train(train_loader, model, optimizer, device):
    total_loss = 0
    N = 0
    ntask = model.task
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if ntask >= 0:
            loss = (model(data).squeeze() - data.y[:,ntask:ntask+1].squeeze()).abs().mean()
        else:
            loss = (model(data) - data.y).abs().mean() 

        loss.backward()
        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs
        optimizer.step()

    return total_loss / N

@torch.no_grad()
def test(loader, model, evaluator, device):
    total_error = 0
    N = 0 
    ntask = model.task
    for data in loader:
        data = data.to(device)
        if ntask >= 0:
            total_error += (model(data).squeeze() - data.y[:,ntask:ntask+1].squeeze()).abs().sum().item()
        else:
            total_error += (model(data) - data.y).abs().sum().item()
        N += data.num_graphs

    if ntask >= 0:
        return - total_error / N
    else:
        return - total_error / (N * CHEMICAL_ACC_NORMALISING_FACTORS[ntask])

if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('train/configs/qm9.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test)