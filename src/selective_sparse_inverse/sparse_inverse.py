import numpy as np
import scipy.sparse as sp


def lu_inv_sparse(lu: np.ndarray, sparsity_pattern: sp.csr_array):
    inv = np.full_like(lu, np.nan)
    _lu_inv_sparse(lu, inv, sparsity_pattern)
    return inv


def _lu_inv_sparse(lu: np.ndarray, inv: np.ndarray, sparsity_pattern: sp.csr_array):
    n = inv.shape[0]

    # Process blocks from size 1 to n, starting from bottom-right
    for k in range(1, n + 1):
        offset = n - k  # Offset for current block from top

        if k == 1:
            # Base case: invert single element
            inv[offset, offset] = 1.0 / lu[offset, offset]
        else:
            # Create views for dense matrices (cheap operation)
            lu_reduced = lu[offset:, offset:]
            inv_reduced = inv[offset:, offset:]

            # Extract non-zero indices from row `offset` of original pattern, filtered to >= offset
            row_start = sparsity_pattern.indptr[offset]
            row_end = sparsity_pattern.indptr[offset + 1]
            all_col_indices = sparsity_pattern.indices[row_start:row_end]

            # Filter to columns > offset and adjust by offset
            non_zero_indices = all_col_indices[all_col_indices > offset] - offset

            for row_idx in non_zero_indices:
                # Check that these row indices have all the required columns
                row_orig = offset + row_idx
                row_start_col = sparsity_pattern.indptr[row_orig]
                row_end_col = sparsity_pattern.indptr[row_orig + 1]
                col_indices = sparsity_pattern.indices[row_start_col:row_end_col]
                col_indices = col_indices[col_indices > offset] - offset
                assert np.all(np.isin(non_zero_indices, col_indices))

            # Calculate current pivot using already-inverted submatrix
            a00 = lu_reduced[0, 0]
            l0_sparse = lu_reduced[non_zero_indices, 0]
            u0_sparse = lu_reduced[0, non_zero_indices]
            u0_sparse /= a00
            z_sparse = inv_reduced[non_zero_indices.reshape(-1, 1), non_zero_indices.reshape(1, -1)]
            inv_reduced[0, 0] = 1.0 / a00 + (u0_sparse @ z_sparse @ l0_sparse)
            inv_reduced[non_zero_indices, 0] = -(z_sparse @ l0_sparse)
            inv_reduced[0, non_zero_indices] = -(u0_sparse @ z_sparse)
