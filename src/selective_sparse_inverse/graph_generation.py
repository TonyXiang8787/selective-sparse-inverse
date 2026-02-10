import networkx as nx
import numpy as np


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
