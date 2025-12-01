import torch
from torch.nn import functional as F
from collections import defaultdict
import subprocess, igraph as ig

def connected_graph(graph: torch.Tensor):
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

    return torch.Tensor(distances)


def adj_to_dict(graph: torch.Tensor):
    n_nodes, _ = graph.shape
    dict_graph = defaultdict(list)
    for node1 in range(n_nodes):
        for node2 in range(node1, n_nodes):
            if bool(graph[node1][node2]):
                dict_graph[node1] += [node2]
                dict_graph[node2] += [node1]
    
    return dict_graph, n_nodes

def all_nodes_shortest_paths(graph: torch.Tensor):
    dict_graph, n_nodes = adj_to_dict(graph)
    distances = torch.zeros(n_nodes, n_nodes)
    for node in range(len(graph)):
        distances[node] = bfs(dict_graph, node, n_nodes)
    
    return distances


def random_walk_probability_matrices(graph: torch.Tensor, z_dim : int):
    n_nodes, _ = graph.shape
    probability_matrices = torch.zeros(n_nodes, n_nodes, z_dim)
    row_sums = torch.sum(graph.clone(), dim = 1, keepdim=True)
    transition_probability_matrix = graph.clone() / row_sums
    probability_matrices[0, :, :] = transition_probability_matrix
    for index in range(1, len(graph)):
        probability_matrices[index, :, :] = torch.matmul(probability_matrices[index - 1, :, :], transition_probability_matrix)

    return probability_matrices