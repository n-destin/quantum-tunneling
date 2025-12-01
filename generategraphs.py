import torch
from torch import Tensor
import networkx as nx
from torch.nn import functional as F
from collections import defaultdict
import subprocess

root = "."

def connected_graph(graph: Tensor):
    dict_graph, n_nodes = adj_to_dict(graph)
    return all(bfs(dict_graph, 0, n_nodes)[1:])

def geng_graphs(command, command_args):
    p = subprocess.run([command] + command_args, capture_output=True, text=True, check=True)
    lines = [ln.strip() for ln in p.stdout.splitlines() if ln and not ln.startswith(('>', 'geng'))]
    return lines

def bfs(graph: dict, node: int, n_nodes : int):
    distances = [0 for _ in range(n_nodes)]
    visited = [0 for _ in range(n_nodes)]
    distance_count = 0
    current_nodes = [node]
    visited[node] = 1
    while bool(current_nodes):
        next_level_nodes = []
        for node_ in current_nodes:
            visited[node_] = 1
            distances[node_] = distance_count
            for neighbor in graph[node_]:
                if not bool(visited[neighbor]) and neighbor not in current_nodes:
                    next_level_nodes.append(neighbor)
        distance_count += 1
        next_level_nodes = list(set(next_level_nodes))
        current_nodes = next_level_nodes

    return Tensor(distances)


def adj_to_dict(graph: Tensor):
    n_nodes, _ = graph.shape
    dict_graph = defaultdict(list)
    for node1 in range(n_nodes):
        for node2 in range(node1, n_nodes):
            if bool(graph[node1][node2]):
                dict_graph[node1] += [node2]
                dict_graph[node2] += [node1]
    
    return dict_graph, n_nodes

def all_nodes_shortest_paths(graph: Tensor):
    dict_graph, n_nodes = adj_to_dict(graph)
    distances = torch.zeros(n_nodes, n_nodes)
    for node in range(len(graph)):
        distances[node] = bfs(dict_graph, node, n_nodes)
    
    return distances


def random_walk_probability_matrices(graph: Tensor, z_dim : int):
    n_nodes, _ = graph.shape
    probability_matrices = torch.zeros(n_nodes, n_nodes, z_dim)
    row_sums = torch.sum(graph.clone(), dim = 1, keepdim=True)
    transition_probability_matrix = graph.clone() / row_sums
    probability_matrices[0, :, :] = transition_probability_matrix
    for index in range(1, len(graph)):
        probability_matrices[index, :, :] = torch.matmul(probability_matrices[index - 1, :, :], transition_probability_matrix)

    return probability_matrices


def save_graph(graph6:str):
    out_file = root / "graphs.g6"
    with out_file.open("a", encoding="utf-8") as file:
        file.write(graph6)

def tunneling_behaviour(probability_matrices: Tensor, distances: Tensor):
    node1_idx = node1.item()
    node2_idx = node2.item()
    n_nodes,_,_ = distances.shape()
    returning = Tensor(3, n_nodes, n_nodes)
    for node1, node2 in torch.combinations(Tensor([index for index in range(n_nodes)])):
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
            nx_graph = nx.to_numpy_array(nx.from_graph6_bytes(graph.encode()), dtype=int)
            distances = all_nodes_shortest_paths(nx_graph)
            probability_matrices = random_walk_probability_matrices(Tensor(nx_graph), torch.max(distances))
            labels = tunneling_behaviour(probability_matrices, distances)
            save_graph(graph.encode())