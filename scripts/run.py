from selective_sparse_inverse.graph_generation import produce_random_tree_matrix
from selective_sparse_inverse.lu_decomposition import lu_no_pivot


def main():
    n = 5
    A = produce_random_tree_matrix(n, seed=42)
    print(A)
    lu = lu_no_pivot(A)
    print(lu)


if __name__ == "__main__":
    main()
