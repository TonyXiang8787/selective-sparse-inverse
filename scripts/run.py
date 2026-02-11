import numpy as np
import scipy.linalg as sclin

from selective_sparse_inverse.full_inverse import lu_inv_full
from selective_sparse_inverse.sparse_inverse import lu_inv_sparse
from selective_sparse_inverse.graph_generation import produce_random_tree_matrix, produce_cycle_matrix
from selective_sparse_inverse.lu_decomposition import lu_no_pivot


def main():
    n = 1000
    A, A_sparse = produce_random_tree_matrix(n)
    lu = lu_no_pivot(A)
    inv = lu_inv_full(lu)
    inv_scipy = sclin.inv(A)
    max_diff = np.max(np.abs(inv - inv_scipy))
    print("Random tree matrix:")
    print(f"Matrix size: {n}x{n}")
    print(f"A nnz: {A_sparse.nnz}")
    print(f"A sparsity: {A_sparse.nnz / (n * n):.4e}")
    print(f"Max absolute difference between full lu_inv and scipy inv: {max_diff:.6e}")
    inv_sparse = lu_inv_sparse(lu, A_sparse)
    max_diff_sparse = np.nanmax(np.abs(inv_sparse - inv_scipy))
    print(f"Max absolute difference between sparse lu_inv and scipy inv: {max_diff_sparse:.6e}")

    print("\nCycle matrix:")
    A_cycle, A_cycle_sparse = produce_cycle_matrix(n)
    lu_cycle = lu_no_pivot(A_cycle)
    inv_cycle = lu_inv_full(lu_cycle)
    inv_cycle_scipy = sclin.inv(A_cycle)
    max_diff_cycle = np.max(np.abs(inv_cycle - inv_cycle_scipy))
    print(f"Matrix size: {n}x{n}")
    print(f"A nnz: {A_cycle_sparse.nnz}")
    print(f"A sparsity: {A_cycle_sparse.nnz / (n * n):.4e}")
    print(f"Max absolute difference between full lu_inv and scipy inv: {max_diff_cycle:.6e}")
    inv_cycle_sparse = lu_inv_sparse(lu_cycle, A_cycle_sparse)
    max_diff_cycle_sparse = np.nanmax(np.abs(inv_cycle_sparse - inv_cycle_scipy))
    print(f"Max absolute difference between sparse lu_inv and scipy inv: {max_diff_cycle_sparse:.6e}")


if __name__ == "__main__":
    main()
