import os
import gzip
import json
import random
import argparse
import numpy as np
import igraph as ig
from functools import reduce
from collections import Counter, defaultdict
import matplotlib.colors as mcolors


index_to_color = {i: x for i, x in enumerate(mcolors.TABLEAU_COLORS)}
def get_color(index):
    return mcolors.TABLEAU_COLORS[index_to_color[index]]


def select_largest_component(G):
    components = G.components()
    largest_comp = np.argmax(components.sizes())
    return G.subgraph(components[largest_comp])


def generate_seeded_graph(
        seed, 
        node_range=(32, 64),
        num_colors=5,
        graph_types=["lattice", "erdos_renyi", "k_regular", "barabasi"]
    ):
    random.seed(seed)
    graph_choice = random.choice(graph_types)
    if graph_choice == "erdos_renyi":
        n = random.randint(*node_range)
        max_p = np.log(6 / n)
        min_p = np.log(1 / (n * 1.2))
        p = np.exp(random.random() * (max_p - min_p) + min_p)
        G = ig.Graph.Erdos_Renyi(n, p=p)
        G = select_largest_component(G)
    elif graph_choice == "lattice":
        num_layers = random.randint(2, 7)
        node_range = tuple(round(np.power(x, 1 / num_layers)) for x in node_range)
        lattice_shape = [random.randint(*node_range) for l in range(num_layers)]
        G = ig.Graph.Lattice(lattice_shape, 1)
    elif graph_choice == "k_regular":
        G = None
        while G is None:
            try:
                n = random.randint(*node_range)
                branching = random.randint(2, 7)
                G = ig.Graph.K_Regular(n, branching)
            except ig._igraph.InternalError:
                G = None
    elif graph_choice == "barabasi":
        n = random.randint(*node_range)
        edges = random.randint(1, 3)
        G = ig.Graph.Barabasi(n, edges, power=0.8 + 0.6 * random.random())
        G = select_largest_component(G)
    else:
        raise ValueError(f"Unknown graph type '{graph_choice}'!")
    # Track the graph type
    G["graph_type"] = graph_choice
    # Color the nodes and edges at random
    G.vs["c"] = [random.randint(0, num_colors-1) for _ in G.vs]
    G.es["c"] = [random.randint(0, num_colors-1) for _ in G.es]
    return G


def find_edges_in_distance_of_edge(G, edge_index, edge_distance):
    edge = G.es[edge_index]
    if edge_distance == 0:
        return [edge]
    src_d, dst_d = G.distances(edge.tuple)
    edges_in_dist = [
        edge
        for edge in G.es
        if min(
            src_d[edge.source], 
            src_d[edge.target],
            dst_d[edge.source], 
            dst_d[edge.target]
        ) + 1 == edge_distance
    ]
    return edges_in_dist


def check_graph_label(
        G, 
        edge_index, 
        edge_color=8, 
        ring_colors=[1, 3], 
        edge_distance=3,
        min_edges=2,
        num_edges_must_match=True
    ):
    edge = G.es[edge_index]
    if edge["c"] != edge_color:
        return False
    edges_in_dist = find_edges_in_distance_of_edge(G, edge_index, edge_distance)
    edges_per_color = [
        sum(1 for edge in edges_in_dist if edge["c"] == color) 
        for color in ring_colors
    ]
    all_colors_appear = len([edges for edges in edges_per_color if edges > 0]) == len(ring_colors)
    all_colors_above_min = all(num_edges >= min_edges for num_edges in edges_per_color)
    equal_edges_of_each = not num_edges_must_match or len(set(edges_per_color)) == 1
    label = all_colors_appear and all_colors_above_min and equal_edges_of_each
    return label


def find_valid_edge(G, *args):
    for edge in G.es:
        is_valid = check_graph_label(G, edge.index, *args)
        if is_valid:
            return edge
    return None


def generate(
        node_range=(32, 64),
        edge_color=0,
        ring_colors=[1, 3],
        min_edges=3,
        num_edges_must_match=True,
        num_colors=5,
        graph_types=["lattice", "erdos_renyi", "k_regular", "barabasi"],
        required_per_pair=600,
        max_distance=5,
        plotting=False
    ):
    total_per_pair = defaultdict(int)
    labelled_pairs = []
    graph_index = 0
    for edge_distance in range(1, max_distance + 1):
        graph_count = 0
        graph_pairs = []
        while graph_count < required_per_pair * len(graph_types):
            graph_index += 1
            G = generate_seeded_graph(
                graph_index,
                node_range=node_range,
                num_colors=num_colors,
                graph_types=graph_types
            )
            G["edge_distance"] = edge_distance
            pair_key = (edge_distance, G["graph_type"])
            if total_per_pair[pair_key] >= required_per_pair:
                continue
            valid_edge = find_valid_edge(G, edge_color, ring_colors, edge_distance, min_edges, num_edges_must_match)
            if valid_edge is not None:
                G_neg = G.copy()
                G.es["width"] = "0.1"
                found_neg = False
                while not found_neg:
                    G_neg.es["c"] = [random.randint(0, num_colors-1) for _ in G.es]
                    found_neg = find_valid_edge(G_neg) is None
                if plotting:
                    G.es["width"] = 0.1
                    G.vs["color"] = [get_color(c) for c in G.vs["c"]]
                    G.es["color"] = [get_color(c) for c in G.es["c"]]
                    for dist in range(edge_distance + 1):
                        edges = find_edges_in_distance_of_edge(G, valid_edge.index, dist)
                        for e in edges:
                            e["label"] = str(dist)
                            e["width"] = 8.0 - dist * 1.5
                    valid_edge["label"] = "0"
                    valid_edge["color"] = "red"
                    valid_edge["width"] = 8.0
                    ig.plot(G)
                graph_count += 1
                total_per_pair[pair_key] += 1
                graph_pairs.append((G, G_neg))
        # Shuffle samples after generation so we ensure graph types are found at random
        random.shuffle(graph_pairs)
        labelled_pairs.extend(graph_pairs)
    return labelled_pairs


def get_parser():
    parser = argparse.ArgumentParser(prog="Colored Edge Dataset Generator")
    parser.add_argument("-n", "--node-range", default=(32, 64), type=lambda x: tuple(int(v.strip()) for v in x.split(",")))
    parser.add_argument("-e", "--edge-color", default=0, type=int)
    parser.add_argument("-r", "--ring-colors", default=[1, 3], type=lambda x: [int(v.strip()) for v in x.split(",")])
    parser.add_argument("-m", "--min-edges", default=3, type=int)
    parser.add_argument("-M", "--num-edges-must-match", default=True, action="store_true")
    parser.add_argument("-N", "--num-colors", default=5, type=int)
    parser.add_argument("-t", "--graph-types", default=["lattice", "erdos_renyi", "k_regular", "barabasi"], type=lambda x: [v.strip().lower() for v in x.split(",")])
    parser.add_argument("-R", "--required-per-pair", default=600, type=int)
    parser.add_argument("-d", "--max-distance", default=5, type=int)
    parser.add_argument("-P", "--plotting", default=False, action="store_true")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    labelled_pairs = generate(
        node_range=args.node_range,
        edge_color=args.edge_color,
        ring_colors=args.ring_colors,
        min_edges=args.min_edges,
        num_edges_must_match=args.num_edges_must_match,
        num_colors=args.num_colors,
        graph_types=args.graph_types,
        required_per_pair=args.required_per_pair,
        max_distance=args.max_distance,
        plotting=args.plotting,
    )

    type_to_index = {graph_type: i for i, graph_type in enumerate(args.graph_types)}
    flat_dataset = [
        {
            "index": 2 * index + label,
            "task_index": (2 * index + label) % (2 * args.required_per_pair * len(args.graph_types)),
            "x": G.vs["c"],
            "edge_attr": [edge["c"] for edge in G.es],
            "edge_index": [list(tensor) for tensor in zip(*[edge.tuple for edge in G.es])],
            "type_index": type_to_index[G["graph_type"]],
            "task": G["edge_distance"],
            "y": label
        } 
        for index, graph_pair in enumerate(labelled_pairs)
        for label, G in enumerate(reversed(graph_pair))
    ]
    print(len(labelled_pairs), Counter([(G["graph_type"], G["edge_distance"]) for (G, G_neg) in labelled_pairs]).most_common())
    print(np.mean([len(G.es) for G, G_neg in labelled_pairs]), np.mean([len(G.vs) for G, G_neg in labelled_pairs]))
    with gzip.open("data/EdgeColoring/data.json.gz", "wt") as f:
        json.dump(flat_dataset, f)
