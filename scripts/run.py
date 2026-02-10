import numpy as np
import scipy.linalg as sclin

from selective_sparse_inverse.full_inverse import lu_inv_full
from selective_sparse_inverse.graph_generation import produce_random_tree_matrix
from selective_sparse_inverse.lu_decomposition import lu_no_pivot


def main():
    n = 5
    A = produce_random_tree_matrix(n, seed=42)
    print(A)
    lu = lu_no_pivot(A)
    print(lu)
    lu_inv = lu_inv_full(lu)
    lu_inv_scipy = sclin.inv(A)
    max_diff = np.max(np.abs(lu_inv - lu_inv_scipy))
    print(f"Max absolute difference between lu_inv and scipy inv: {max_diff:.6e}")


if __name__ == "__main__":
    main()
