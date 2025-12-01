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

def check_cospectrality(probability_matrices: torch.Tensor, distances: torch.Tensor):
    returning = []
    for node1, node2 in torch.combinations(torch.Tensor([index for index in range(len(distances))])):
        if torch.equal(probability_matrices[:, int(node1.item()), int(node1.item())], probability_matrices[:, int(node2.item()), int(node2.item())]):
            returning.append((int(node1.item()), int(node2.item())))
    
    return returning
    

for n_nodes in range(5, 15):
    for graph in geng_graphs('geng', ['-c', f"{n_nodes}"]):
        graph = nx.to_numpy_array(nx.from_graph6_bytes(graph.encode()), dtype=int)
        probability_matrices = random_walk_probability_matrices(torch.tensor(graph))
        cospectral_nodes = check_cospectrality(probability_matrices, all_nodes_shortest_paths(graph))
        if bool(cospectral_nodes):
            save_graph(encode_to_graph6(graph), cospectral_nodes)