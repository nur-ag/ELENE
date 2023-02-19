import torch
from torch_geometric.data import DataLoader, Batch
import numpy as np
import igraph as ig

import sys
sys.path = ['../IGEL/src', '../IGEL/gnnml-comparisons'] + sys.path


class AddIGELNodeFeatures:
    '''A preprocessor to include IGEL (neighbourhood structure) node features.

    Args:
        seed (int): random seed to pass to the unsupervised IGEL training job.
        distance (int): IGEL encoding distance. If set to less than 1, this preprocessor is a no-op.
        vector_length (int): length of the learned unsupervised IGEL embeddings. If set to a negative number, use IGEL structural encoding features (no training).
        use_relative_degrees (boolean): whether or not to use 'relative' degrees from nodes to their neighbours closer to the ego-root.
    '''
    def __init__(
            self, 
            seed=0, 
            distance=1, 
            vector_length=-1,
            use_relative_degrees=False,
            use_edge_encodings=False):
        self.distance = distance
        self.seed = seed
        self.vector_length = vector_length
        self.use_encoding = vector_length < 0
        self.use_relative_degrees = use_relative_degrees
        self.use_edge_encodings = use_edge_encodings
        self.model = None

    def __call__(self, data):
        if self.distance < 1:
            return data

        if type(data) != list:
            data = [datum for datum in data]

        G = self.global_graph(data)
        if self.model is None:
            self.model = self.train_igel_model(G)

        igel_emb = torch.Tensor(self.model(G.vs, G).cpu().detach().numpy())
        datum_shift = 0
        for datum in data:
            num_nodes = datum.x.shape[0]
            end_datum_shift = datum_shift + num_nodes
            datum_igel_emb = igel_emb[datum_shift:end_datum_shift].to(datum.x.device)
            datum.x = torch.cat([datum.x, datum_igel_emb], dim=-1)
            datum_edge_attr = getattr(datum, "edge_attr", None)
            if datum_edge_attr is not None and self.use_edge_encodings:
                if len(datum_edge_attr.shape) == 1:
                    datum_edge_attr = datum_edge_attr.reshape(-1, 1)
                datum_igel_edge_attr = datum_igel_emb[datum.edge_index[0, :]] * datum_igel_emb[datum.edge_index[1, :]]
                datum.edge_attr = torch.cat([datum_edge_attr, datum_igel_edge_attr], dim=-1)
            datum_shift = end_datum_shift
        return data

    def global_graph(self, data):
        num_nodes = [datum.x.shape[0] for datum in data]
        edges = [datum.clone().edge_index.numpy() for datum in data]
        shift_index = 0
        for i, graph_edges in enumerate(edges):
            edges[i] += shift_index
            shift_index += num_nodes[i]
        global_edges = np.concatenate(edges, axis=-1)
        edge_tuples = list(zip(global_edges[0], global_edges[1]))

        # The 'global' graph is the graph with all the nodes and edges as disconnected components
        G = ig.Graph()
        G.add_vertices(shift_index)
        G.add_edges(edge_tuples)
        G.vs['name'] = [str(n.index) for n in G.vs]
        return G

    def train_igel_model(self, G):
        # Note: Imported from IGEL.gnnml-comparisons
        from igel_embedder import get_unsupervised_embedder, TRAINING_OPTIONS
        import torch.nn as nn

        # Do not train if we are using the encoding
        embedder_length = self.vector_length
        if self.use_encoding:
            TRAINING_OPTIONS.epochs = 0
            embedder_length = 0

        # Get the model using the modified training options (with/without epochs)
        trained_model = get_unsupervised_embedder(G, 
                                         self.distance, 
                                         self.seed,
                                         embedder_length,
                                         self.use_relative_degrees,
                                         train_opts=TRAINING_OPTIONS)
        # We force the model in the CPU
        trained_model.device = torch.device('cpu')
        trained_model.to(trained_model.device)

        # If we are using encodings, override the parameter matrix
        if self.use_encoding:
            encoder = trained_model.structural_mapper
            embedder_length = encoder.num_elements()

            # Using encodings: simply replace the learned embedding matrix with an identity matrix
            # This will return the structural features as-is.
            input_identity = torch.eye(embedder_length).to(trained_model.device)
            trained_model.matrix = nn.Parameter(input_identity, requires_grad=False).to(trained_model.device)
            trained_model.output_size = embedder_length
        print(f'Prepared IGEL model (encoding-only: {self.use_encoding}, relative: {self.use_relative_degrees}) with {embedder_length} features.')
        return trained_model
