import copy
import torch
from core.config import cfg, update_cfg
from core.train_helper import run 
from core.model import GNNAsKernel
from core.transform import SubgraphsTransform
from core.data import ColoredEdgesDataset, calculate_stats

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

    root = 'data/EdgeColoring'
    dataset = ColoredEdgesDataset(root, cfg.task, transform=transform)
    full_dataset = [x for x in dataset]

    # Validate that we know our dataset
    SAMPLES_PER_TASK = 4800
    assert len(full_dataset) == SAMPLES_PER_TASK

    train_dataset = full_dataset[:int(4800 * 0.8)]
    val_dataset = full_dataset[int(4800 * 0.8):int(4800 * 0.9)]
    test_dataset = full_dataset[int(4800 * 0.9):SAMPLES_PER_TASK]
    print('------------All--------------')
    calculate_stats(dataset)
    # exit(0)
    return train_dataset, val_dataset, test_dataset

def create_model(cfg):
    model = GNNAsKernel(None, None, 
                        nhid=cfg.model.hidden_size, 
                        nout=2, 
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
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).squeeze()
        loss = torch.nn.CrossEntropyLoss()(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(loader, model, evaluator, device):
    y_preds, y_trues = [], []
    for data in loader:
        data = data.to(device)
        y_preds.append(torch.argmax(model(data), dim=-1))
        y_trues.append(data.y)
    y_preds = torch.cat(y_preds, -1)
    y_trues = torch.cat(y_trues, -1)
    return (y_preds == y_trues).float().mean()

if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('train/configs/edge_colors.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test)
