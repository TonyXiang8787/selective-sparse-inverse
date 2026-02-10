import networkx as nx
import numpy as np
import scipy.sparse as sp

EDGE_WEIGHT = 5.0
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


def build_matrix_from_edges(edges, n):
    """
    Build an (n,n) adjacency matrix from an (n-1,2) array of edges.
    """
    A = np.zeros((n, n), dtype=np.float64)

    # Vectorized updates: count degree of each node and accumulate edge weights
    i_nodes, j_nodes = edges[:, 0], edges[:, 1]
    np.add.at(A, (i_nodes, i_nodes), EDGE_WEIGHT)
    np.add.at(A, (j_nodes, j_nodes), EDGE_WEIGHT)
    np.add.at(A, (i_nodes, j_nodes), -EDGE_WEIGHT)
    np.add.at(A, (j_nodes, i_nodes), -EDGE_WEIGHT)

    A[-1, -1] += SOURCE_WEIGHT  # Add large value to last node to make it well-conditioned

    # build sparse matrix with one as non-zeros
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
