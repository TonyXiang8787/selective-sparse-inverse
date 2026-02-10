import numpy as np
import scipy.sparse as sp


def lu_inv_sparse(lu: np.ndarray, sparsity_pattern: sp.csr_array):
    inv = np.full_like(lu, np.nan)
    _lu_inv_sparse(lu, inv, sparsity_pattern)
    return inv


def _lu_inv_sparse(lu: np.ndarray, inv: np.ndarray, sparsity_pattern: sp.csr_array):
    n = inv.shape[0]
    if n == 1:
        inv[0, 0] = 1.0 / lu[0, 0]
        return
    non_zero_indices = sparsity_pattern.indices[sparsity_pattern.indptr[0] : sparsity_pattern.indptr[1]]
    assert non_zero_indices[0] == 0
    non_zero_indices = non_zero_indices[1:]

    #  recursive call
    sparsity_pattern_next = sparsity_pattern[1:, 1:]
    _lu_inv_sparse(lu[1:, 1:], inv[1:, 1:], sparsity_pattern_next)

    # calculate current pivot
    a00 = lu[0, 0]
    l0_sparse = lu[non_zero_indices, 0]
    u0_sparse = lu[0, non_zero_indices]
    u0_sparse /= a00
    z_sparse = inv[non_zero_indices.reshape(-1, 1), non_zero_indices.reshape(1, -1)]
    inv[0, 0] = 1.0 / a00 + (u0_sparse @ z_sparse @ l0_sparse)
    inv[non_zero_indices, 0] = -(z_sparse @ l0_sparse)
    inv[0, non_zero_indices] = -(u0_sparse @ z_sparse)
