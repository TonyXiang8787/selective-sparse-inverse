import numpy as np
import scipy.linalg as sclin

from selective_sparse_inverse.full_inverse import lu_inv_full
from selective_sparse_inverse.graph_generation import produce_random_tree_matrix
from selective_sparse_inverse.lu_decomposition import lu_no_pivot


def main():
    n = 5
    A, A_sparse = produce_random_tree_matrix(n)
    lu = lu_no_pivot(A)
    lu_inv = lu_inv_full(lu)
    lu_inv_scipy = sclin.inv(A)
    max_diff = np.max(np.abs(lu_inv - lu_inv_scipy))
    print(f"Matrix size: {n}x{n}")
    print(f"A nnz: {A_sparse.nnz}")
    print(f"A sparsity: {A_sparse.nnz / (n * n):.4e}")
    print(f"Max absolute difference between lu_inv and scipy inv: {max_diff:.6e}")


if __name__ == "__main__":
    main()
