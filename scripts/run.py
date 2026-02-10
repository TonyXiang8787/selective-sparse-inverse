from selective_sparse_inverse.graph_generation import random_tree_reverse_dfs_edges


def main():
    n = 10
    edges = random_tree_reverse_dfs_edges(n, seed=42)
    print(edges)
    print(edges.shape)  # Should be (n-1, 2)


if __name__ == "__main__":
    main()
