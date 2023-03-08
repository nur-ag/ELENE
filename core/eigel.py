import math
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from core.model_utils.elements import MLP


class EIGELEmbedder(nn.Module):
    def __init__(self,
            max_degree=50,
            max_distance=3,
            use_relative_degrees=True,
            joint_embeddings=False,
            embedding_dim=None,
            given_embedders=None,
            node_only=False,
        ):
        super().__init__()
        self.max_degree = max_degree
        self.max_distance = max_distance
        self.use_relative_degrees = use_relative_degrees
        self.joint_embeddings = joint_embeddings
        self.node_only = node_only

        self._num_degree_types = 3 if use_relative_degrees else 1
        if not self.joint_embeddings:
            if given_embedders is not None:
                degree, delta, distance = given_embedders
                self.degree_embedders = degree
                self.delta_embedder = delta
                self.distance_embedder = distance
            else:
                self.degree_embedders = nn.ModuleList([
                    nn.Embedding(
                        self.max_degree + 1, 
                        self.max_degree + 1 if not embedding_dim else embedding_dim
                    ) for i in range(self._num_degree_types)
                ])
                self.delta_embedder = nn.Embedding(
                    self._num_degree_types * (self.max_distance + 1), 
                    self._num_degree_types * (self.max_distance + 1) if not embedding_dim else embedding_dim
                ) if self.use_relative_degrees else None
                self.distance_embedder = nn.Embedding(
                    self.max_distance + 1, 
                    self.max_distance + 1 if not embedding_dim else embedding_dim
                )

            distance_size = self.max_distance + 1
            rel_degree_size = self._num_degree_types * (self.max_degree + 1)
            self.node_embedder_size = (self.degree_embedders[0].embedding_dim * self._num_degree_types + 
                                       self.distance_embedder.embedding_dim)
            self.edge_embedder_size = (self.degree_embedders[0].embedding_dim * self._num_degree_types + 
                                       self.delta_embedder.embedding_dim)
            assert self.delta_embedder.weight.shape[0] >= distance_size * self._num_degree_types, "Delta embedder must be able to represent all deltas."
            assert self.distance_embedder.weight.shape[0] >= distance_size, "Distance embedder must be able to represent all distances."
            for i in range(self._num_degree_types):
                assert self.degree_embedders[i].weight.shape[0] >= self.max_degree + 1, "Degree embedders must be able to represent all degrees."
        else:
            joint_embedding_size = self._num_degree_types * (self.max_degree + 1) * (self.max_distance + 1)
            if given_embedders is not None:
                edge, node = given_embedders
                self.edge_degree_delta_embedders = edge
                self.node_degree_distance_embedders = node
            else:
                self.edge_degree_delta_embedders = nn.ModuleList([
                    nn.Embedding(
                        self._num_degree_types * joint_embedding_size, 
                        self._num_degree_types * joint_embedding_size if not embedding_dim else embedding_dim
                    ) for i in range(self._num_degree_types)
                ])
                self.node_degree_distance_embedders = nn.ModuleList([
                    nn.Embedding(
                        joint_embedding_size, joint_embedding_size if not embedding_dim else embedding_dim
                    ) for i in range(self._num_degree_types)
                ])

            self.node_embedder_size = self.node_degree_distance_embedders[0].embedding_dim * self._num_degree_types
            self.edge_embedder_size = self.edge_degree_delta_embedders[0].embedding_dim * self._num_degree_types

            # Validate the embeders
            edge_embs = self.edge_degree_delta_embedders
            node_embs = self.node_degree_distance_embedders
            for i in range(self._num_degree_types):
                assert edge_embs[i].weight.shape[0] >= joint_embedding_size * self._num_degree_types, "Joint edge embedders must have equal or larger capacity than max delta and degree composite."
                assert node_embs[i].weight.shape[0] >= joint_embedding_size, "Joint edge embedders must have equal or larger capacity than max distance and degree composite."

    def reset_parameters(self):
        SQRT_GAIN = math.sqrt(2)
        if self.joint_embeddings:
            embedders = list(self.edge_degree_delta_embedders) + list(self.node_degree_distance_embedders)
        else:
            embedders = list(self.degree_embedders) + [self.delta_embedder, self.distance_embedder]
        for emb in embedders:
            emb.reset_parameters()
            nn.init.xavier_normal_(emb.weight, gain=1/emb.embedding_dim)

    def compute_node_embedding(self, data):
        node_dist = torch.clamp(data.subgraph_node_hops, min=0, max=self.max_distance)
        node_degrees = torch.clamp(data.subgraph_node_degrees, min=0, max=self.max_degree)
        node_degrees[:, 1] = torch.clamp(node_degrees[:, 0] + node_degrees[:, 1] + node_degrees[:, 2], min=0, max=self.max_degree)

        # If not using relative degrees, only use the 'full' degree
        if not self.use_relative_degrees:
            node_degrees = node_degrees[:, 1].reshape(-1, 1)

        # Compute the node-level embedding
        if self.joint_embeddings:
            node_joint_idx = (node_degrees * (self.max_distance + 1) + 
                              node_dist.repeat(self._num_degree_types).reshape(node_degrees.shape))
            node_eigel_emb = torch.cat([
                self.node_degree_distance_embedders[i](node_joint_idx[:, i]) 
                for i in range(self._num_degree_types)
            ], dim=-1)
        else:
            node_dist_emb = self.distance_embedder(node_dist)
            node_degrees_emb = torch.cat([
                self.degree_embedders[i](node_degrees[:, i]) 
                for i in range(self._num_degree_types)
            ], dim=-1)
            node_eigel_emb = torch.cat([node_degrees_emb, node_dist_emb], dim=-1)
        return node_eigel_emb

    def compute_edge_embedding(self, data):
        combined_subgraphs_edge_idx = data.edge_index[:, data.subgraphs_edges_mapper]
        edge_indices_a = combined_subgraphs_edge_idx[0]
        edge_indices_b = combined_subgraphs_edge_idx[1]

        # Extract all relative node degrees alongside the edge
        edge_degree_idx_a = torch.clamp(data.subgraph_edge_degrees[:, 0], min=0, max=self.max_degree)
        edge_degree_idx_b = torch.clamp(data.subgraph_edge_degrees[:, 1], min=0, max=self.max_degree)
        edge_degree_idx_a[:, 1] = torch.clamp(edge_degree_idx_a[:, 0] + edge_degree_idx_a[:, 1] + edge_degree_idx_a[:, 2], min=0, max=self.max_degree)
        edge_degree_idx_b[:, 1] = torch.clamp(edge_degree_idx_b[:, 0] + edge_degree_idx_b[:, 1] + edge_degree_idx_b[:, 2], min=0, max=self.max_degree)

        # If not using relative degrees, only use the actual node degree
        if not self.use_relative_degrees:
            edge_degree_idx_a = edge_degree_idx_a[:, 1].reshape(-1, 1)
            edge_degree_idx_b = edge_degree_idx_b[:, 1].reshape(-1, 1)

        # Extract all node distances
        edge_distance_a = torch.clamp(data.subgraph_edge_hops[:, 0], min=0, max=self.max_distance)
        edge_distance_b = torch.clamp(data.subgraph_edge_hops[:, 1], min=0, max=self.max_distance)
        if self.use_relative_degrees:
            edge_delta_a = edge_distance_a - edge_distance_b + 1
            edge_delta_b = edge_distance_b - edge_distance_a + 1
        else:
            edge_delta_a = torch.zeros(edge_distance_a.shape)
            edge_delta_b = torch.zeros(edge_distance_b.shape)

        # Compute the embeddings
        if self.joint_embeddings:
            edge_joint_idx_a = (edge_degree_idx_a * (self.max_distance + 1) * self._num_degree_types +
                                edge_distance_a.repeat(self._num_degree_types).reshape(edge_degree_idx_a.shape) * self._num_degree_types + 
                                edge_delta_a.repeat(self._num_degree_types).reshape(edge_degree_idx_a.shape))
            edge_joint_idx_b = (edge_degree_idx_b * (self.max_distance + 1) * self._num_degree_types +
                                edge_distance_b.repeat(self._num_degree_types).reshape(edge_degree_idx_b.shape) * self._num_degree_types + 
                                edge_delta_b.repeat(self._num_degree_types).reshape(edge_degree_idx_b.shape))

            edge_emb_degree_a = torch.cat([
                self.edge_degree_delta_embedders[i](edge_joint_idx_a[:, i]) for i in range(self._num_degree_types)
            ], dim=-1)
            edge_emb_degree_b = torch.cat([
                self.edge_degree_delta_embedders[i](edge_joint_idx_b[:, i]) for i in range(self._num_degree_types)
            ], dim=-1)
            edge_eigel_emb = edge_emb_degree_a + edge_emb_degree_b
        else:
            edge_emb_degree_a = torch.cat([
                self.degree_embedders[i](edge_degree_idx_a[:, i]) for i in range(self._num_degree_types)
            ], dim=-1)
            edge_emb_degree_b = torch.cat([
                self.degree_embedders[i](edge_degree_idx_b[:, i]) for i in range(self._num_degree_types)
            ], dim=-1)

            edge_delta_emb_a = self.delta_embedder(edge_distance_a * self._num_degree_types + edge_delta_a)
            edge_delta_emb_b = self.delta_embedder(edge_distance_b * self._num_degree_types + edge_delta_b)
            edge_eigel_emb = torch.cat([edge_emb_degree_a * edge_emb_degree_b, edge_delta_emb_a * edge_delta_emb_b], dim=-1)
        return edge_eigel_emb

    def forward(self, data):
        node_eigel_emb = self.compute_node_embedding(data)
        if self.node_only:
            return node_eigel_emb
        edge_eigel_emb = self.compute_edge_embedding(data)
        return node_eigel_emb, edge_eigel_emb


class EIGELMessagePasser(nn.Module):
    def __init__(self,
            nhid,
            edge_emd_dim,
            mlp_layers=1,
            max_degree=50,
            max_distance=3,
            pooling="mean",
            use_relative_degrees=True,
            joint_embeddings=False,
            eigel_embedder=None,
        ):
        super().__init__()
        self.max_degree = max_degree
        self.max_distance = max_distance
        self.pooling = pooling
        if eigel_embedder is not None:
            self.eigel_embedder = eigel_embedder
        else:
            self.eigel_embedder = EIGELEmbedder(max_degree, max_distance, use_relative_degrees, joint_embeddings)
        self.subgraph_transform = MLP(2 * nhid + self.eigel_embedder.node_embedder_size, nhid, nlayer=mlp_layers, with_final_activation=True)
        if not self.eigel_embedder.node_only:
            self.relative_transform = MLP(2 * nhid + edge_emd_dim + self.eigel_embedder.edge_embedder_size, edge_emd_dim, nlayer=mlp_layers, with_final_activation=True)
            self.output_transform = MLP(2 * nhid + edge_emd_dim, nhid, nlayer=mlp_layers, with_final_activation=True)
        else:
            self.output_transform = MLP(2 * nhid, nhid, nlayer=mlp_layers, with_final_activation=True)
        self.sum_weights = nn.Parameter(torch.ones(4))

    def reset_parameters(self):
        self.eigel_embedder.reset_parameters()
        self.subgraph_transform.reset_parameters()
        self.output_transform.reset_parameters()
        if not self.eigel_embedder.node_only:
            self.relative_transform.reset_parameters()

    def forward(self, data):
        combined_subgraphs_x = data.x[data.subgraphs_nodes_mapper]
        source_indices = data.subgraphs_edge_source
        combined_subgraphs_edge_idx = data.edge_index[:, data.subgraphs_edges_mapper]
        combined_subgraphs_edge_attr = data.edge_attr[data.subgraphs_edges_mapper]
        edge_indices_a = combined_subgraphs_edge_idx[0]
        edge_indices_b = combined_subgraphs_edge_idx[1]

        # Compute EIGEL features for the subgraph nodes and edges
        node_distance_mask = (data.subgraph_node_hops <= self.max_distance).reshape(-1, 1)
        if self.eigel_embedder.node_only:
            node_eigel_emb = self.eigel_embedder(data)
            edge_eigel_emb = None
        else:
            edge_pair_distance_mask = data.subgraph_edge_hops <= self.max_distance
            edge_distance_mask = (edge_pair_distance_mask[:, 0] * edge_pair_distance_mask[:, 1]).reshape(-1, 1)
            node_eigel_emb, edge_eigel_emb = self.eigel_embedder(data)

        # Compute subgraph message, collecting all node information with the root
        root_features = data.x[data.subgraphs_batch]
        node_features = data.x[data.subgraphs_nodes_mapper]
        subgraph_message = torch.cat([root_features, node_features, node_eigel_emb], dim=-1)
        subgraph_message = self.subgraph_transform(subgraph_message)

        # Compute the aggregate subgraph and relative messages if necessary
        subgraph_mask_weight = scatter(node_distance_mask, data.subgraphs_batch, dim_size=data.num_nodes, dim=0, reduce="sum")
        subgraph_mask_weight = torch.maximum(subgraph_mask_weight, subgraph_mask_weight.new_ones(subgraph_mask_weight.shape))
        subgraph_raw_msg = scatter(subgraph_message * node_distance_mask, data.subgraphs_batch, dim_size=data.num_nodes, dim=0, reduce="sum")
        subgraph_msg = subgraph_raw_msg / subgraph_mask_weight
        assert data.x.shape[0] == subgraph_msg.shape[0] and data.x.shape[1] == subgraph_msg.shape[1], "Subgraph message shapes must match the input node feature shape!"

        if edge_eigel_emb is not None:
            # Get features alongside the edge and ego root
            ego_source_x = combined_subgraphs_x[source_indices]
            edge_messages_a = combined_subgraphs_x[edge_indices_a]
            edge_messages_b = combined_subgraphs_x[edge_indices_b]

            # Represent edge-wise combination as an element-wise product
            edge_messages = edge_messages_a * edge_messages_b

            # The message is the edge messages (root features, edge features, aggregated edges, EIGEL degrees and deltas) merged
            edge_message = torch.cat([ego_source_x, combined_subgraphs_edge_attr, edge_messages, edge_eigel_emb], dim=-1)
            rel_message = self.relative_transform(edge_message)

            relative_mask_weight = scatter(edge_distance_mask, source_indices, dim_size=data.num_nodes, dim=0, reduce="sum")
            relative_mask_weight = torch.maximum(relative_mask_weight, relative_mask_weight.new_ones(relative_mask_weight.shape))
            relative_raw_msg = scatter(rel_message * edge_distance_mask, source_indices, dim_size=data.num_nodes, dim=0, reduce="sum")
            relative_msg = relative_raw_msg / relative_mask_weight
            assert data.x.shape[0] == relative_msg.shape[0] and data.edge_attr.shape[1] == relative_msg.shape[1], f"Relative message shapes must match the input edge feature shape ({data.edge_attr.shape} vs {relative_msg.shape})!"
            combined_msg = torch.cat([data.x, subgraph_msg, relative_msg], dim=-1)

            # Produce a replacement for node and edge features _with_ EIGEL encodings
            x_out = self.output_transform(combined_msg)
            composite_x = data.x * self.sum_weights[0] + x_out * self.sum_weights[1]
            edge_attr_out = scatter(rel_message, data.subgraphs_edges_mapper, dim_size=data.edge_index.shape[1], dim=0, reduce=self.pooling)
            composite_edge_attr = data.edge_attr * self.sum_weights[2] + edge_attr_out * self.sum_weights[3]
            return composite_x, composite_edge_attr
        # Produce a replacement for node features _with_ EIGEL encodings
        combined_msg = torch.cat([data.x, subgraph_msg], dim=-1)
        x_out = self.output_transform(combined_msg)
        composite_x = data.x * self.sum_weights[0] + x_out * self.sum_weights[1]
        return composite_x, None

