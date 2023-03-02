import torch
from core.config import cfg, update_cfg
from core.train_helper import run_k_fold 
from core.model import GNNAsKernel
from core.transform import SubgraphsTransform
from core.data import ProximityDataset, calculate_stats
import shutil


def create_dataset(cfg):
    if cfg.num_workers:
        torch.set_num_threads(cfg.num_workers)
    dataset = ProximityDataset('data/Proximity', cfg.task)

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
    
    print('------------Stats--------------')
    calculate_stats(dataset)
    return dataset, transform, transform_eval

def create_model(cfg):
    model = GNNAsKernel(None, None, 
                        nhid=cfg.model.hidden_size, 
                        nout=1, 
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
    return model

def train(train_loader, model, optimizer, device):
    total_loss = 0
    N = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.reshape(-1, 1).float())
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs
        optimizer.step()
    return total_loss / N

@torch.no_grad()
def test(loader, model, evaluator, device):
    y_preds, y_trues = [], []
    for data in loader:
        data = data.to(device)
        y_preds.append(model(data).sigmoid().round().long().reshape(-1))
        y_trues.append(data.y.long())
    y_preds = torch.cat(y_preds, -1)
    y_trues = torch.cat(y_trues, -1)
    return (y_preds == y_trues).float().mean()


if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('train/configs/proximity.yaml')
    cfg = update_cfg(cfg)
    run_k_fold(cfg, create_dataset, create_model, train, test)