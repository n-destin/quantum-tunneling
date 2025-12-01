import torch

def encode_to_graph6(adj):    
    binary_string = ""
    graph6 = ""
    for row in range(len(adj)):
        if row != 0:
            binary_string += "".join(str(adj[row][col]) for col in range(row))
    binary_string = binary_string + "0"*(6 - len(binary_string) % 6) if len(binary_string) % 6 != 0 else binary_string
    chunks = [binary_string[index : index + 6] for index in range(0, len(binary_string), 6)]
    graph6 = chr(len(adj) + 63) + "".join(chr(int(chunk, 2) + 63) for chunk in chunks) 
    
    return graph6

def sum_up_to_n(n):
    return int(n * (n + 1) / 2)

def decode_from_graph6(string):

    n_nodes = ord(string[0]) - 63
    n_entries = sum_up_to_n(n_nodes - 1)
    adj_matrix = [[0 for _ in range(n_nodes)] for _ in range(n_nodes)]

    values = [ord(char) - 63 for char in string[1:len(string)]]
    bits = [f"{value:b}" for value in values]

    processed_bits = "".join([bit[:((n_entries % 6) - 6)] if index == len(bits) - 1 else bit for index, bit in enumerate(bits)])

    left = 0
    right = 1

    while left < right and right <= n_entries:
        row = right - left
        for index in range(row):
            adj_matrix[row][index] = int(processed_bits[left : right][index])
            adj_matrix[index][row] = int(processed_bits[left : right][index])
        temp = left
        left = right 
        right += (right - temp) + 1

    return adj_matrix