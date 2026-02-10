from selective_sparse_inverse.graph_generation import produce_random_tree_matrix


def main():
    n = 5
    A = produce_random_tree_matrix(n, seed=42)
    print(A)
    print(A.shape)  # Should be (n, n)


if __name__ == "__main__":
    main()
