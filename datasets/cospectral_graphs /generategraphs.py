import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pathlib import Path

import torch
import networkx as nx
from helpers.graph6 import encode_to_graph6
from helpers.general import random_walk_probability_matrices, all_nodes_shortest_paths, geng_graphs


def save_graph(graph6_string : str, cospectral_nodes: list):
    out_dir = Path(__file__).resolve().parent / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "graphs.txt"
    with out_file.open("a", encoding="utf-8") as file:
        file.write(graph6_string + f"{"".join([f"{node[0]}{node[1]}" for node in cospectral_nodes])}\n")

def tunneling_behaviour(probability_matrices: torch.Tensor, distances: torch.Tensor):
    node1_idx = node1.item()
    node2_idx = node2.item()
    n_nodes,_,_ = distances.shape()
    returning = torch.tensor(3, n_nodes, n_nodes)
    for node1, node2 in torch.combinations(torch.Tensor([index for index in range(n_nodes)])):
        limit = distances[node1_idx][node2_idx]
        class_ = [0, 0, 1] # no tunneling at all
        node1_return_probabilities = probability_matrices[:limit, node1_idx, node1_idx]
        node2_return_probabilities = probability_matrices[:limit, node2_idx, node2_idx]
        complete_tunneling = torch.equal(node1_return_probabilities, node2_return_probabilities)
        partial_tunneling = torch.equal(node2_return_probabilities[:-1], node2_return_probabilities[:-1])
        if complete_tunneling:
            class_ = [1, 0, 0]
        elif partial_tunneling:
            class_ = [0, 1 ,0]
        returning[node1_idx][node2_idx] = class_
        returning[node2_idx][node1_idx] = class_
    return returning

for n_nodes in range(5, 15):
    ranges = [(n_nodes - 1, 3 * n_nodes * (n_nodes - 1) / 20), (7/10 *n_nodes, n_nodes * (n_nodes - 1) / 2)]
    for min_range, max_range in ranges:
        for graph in geng_graphs('geng', ['-c', f"{n_nodes}", "{}:{}".format(min_range, max_range)]):
            graph = nx.to_numpy_array(nx.from_graph6_bytes(graph.encode()), dtype=int)
            distances = all_nodes_shortest_paths(graph)
            probability_matrices = random_walk_probability_matrices(torch.tensor(graph), torch.max(distances))
            cospectral_nodes = tunneling_behaviour(probability_matrices, distances)
            if bool(cospectral_nodes):
                save_graph(encode_to_graph6(graph), cospectral_nodes)
                