import networkx as nx
import numpy as np
import scipy.sparse as sp

EDGE_WEIGHT_MIN = 1.0
EDGE_WEIGHT_MAX = 5.0
SOURCE_WEIGHT = 10e3


def random_tree_reverse_dfs_edges(n, seed=None):
    """
    Generate a random tree and return an (n-1,2) int32 numpy array of edges,
    with nodes labeled by reverse DFS order starting from original node 0.
    0 = first node removed (last in DFS), n-1 = last node removed (first in DFS).
    """
    # Generate random tree
    tree = nx.random_labeled_tree(n, seed=seed)
    G = tree.copy()
    # Get DFS order starting from vertex 0 in original labeling
    dfs_order = list(nx.dfs_preorder_nodes(G, source=0))
    # Map: node in original tree -> label (reverse of DFS order)
    label_map = {node: n - 1 - idx for idx, node in enumerate(dfs_order)}
    # Remap edges
    edges = np.array([[label_map[u], label_map[v]] for u, v in tree.edges()], dtype=np.int32)
    return edges


def cycle_edges(n):
    edges = np.zeros((n, 2), dtype=np.int32)
    edges[:, 0] = np.arange(n)
    edges[:, 1] = np.arange(1, n + 1) % n
    fill_ins = np.zeros((n - 3, 2), dtype=np.int32)
    fill_ins[:, 0] = n - 1
    fill_ins[:, 1] = np.arange(1, n - 2)
    return edges, fill_ins


def build_matrix_from_edges(edges, n, fill_ins=None):
    """
    Build an (n,n) adjacency matrix from an (n-1,2) array of edges.
    """
    A = np.zeros((n, n), dtype=np.float64)

    # Vectorized updates: count degree of each node and accumulate edge weights
    i_nodes, j_nodes = edges[:, 0], edges[:, 1]
    edge_weight = np.random.uniform(EDGE_WEIGHT_MIN, EDGE_WEIGHT_MAX, size=len(edges))
    np.add.at(A, (i_nodes, i_nodes), edge_weight)
    np.add.at(A, (j_nodes, j_nodes), edge_weight)
    np.add.at(A, (i_nodes, j_nodes), -edge_weight)
    np.add.at(A, (j_nodes, i_nodes), -edge_weight)

    A[-1, -1] += SOURCE_WEIGHT  # Add large value to last node to make it well-conditioned

    # build sparse matrix with one as non-zeros
    if fill_ins is not None:
        full_edges = np.concatenate([edges, fill_ins], axis=0)
    else:
        full_edges = edges
    i_nodes, j_nodes = full_edges[:, 0], full_edges[:, 1]
    row_indices = np.r_[i_nodes, j_nodes, np.arange(n)]
    col_indices = np.r_[j_nodes, i_nodes, np.arange(n)]
    data = np.ones(len(row_indices), dtype=np.int32)
    A_sparse = sp.csr_array((data, (row_indices, col_indices)), shape=(n, n))
    A_sparse.sum_duplicates()
    A_sparse.sort_indices()
    A_sparse.data[:] = 1

    return A, A_sparse


def produce_random_tree_matrix(n, seed=None):
    """
    Generate a random tree and return its adjacency matrix.
    """
    edges = random_tree_reverse_dfs_edges(n, seed=seed)
    A, A_sparse = build_matrix_from_edges(edges, n)
    return A, A_sparse


def produce_cycle_matrix(n):
    edges, fill_ins = cycle_edges(n)
    A, A_sparse = build_matrix_from_edges(edges, n, fill_ins=fill_ins)
    return A, A_sparse
