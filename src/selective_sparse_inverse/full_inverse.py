import numpy as np


def lu_inv_full(lu: np.ndarray):
    lu_inv = lu.copy()
    _lu_inv_inplace(lu_inv)
    return lu_inv


def _lu_inv_inplace(lu_inv: np.ndarray):
    n = lu_inv.shape[0]

    # Process blocks from size 1 to n, starting from bottom-right
    for k in range(1, n + 1):
        i = n - k  # Starting row/column for current block

        if k == 1:
            # Base case: invert single element
            lu_inv[i, i] = 1.0 / lu_inv[i, i]
        else:
            # Get submatrix z (already inverted from previous iteration)
            z = lu_inv[i + 1 :, i + 1 :]

            # Calculate current pivot
            a00 = lu_inv[i, i]
            l0 = lu_inv[i + 1 :, i]
            u0 = lu_inv[i, i + 1 :]
            u0 /= a00
            lu_inv[i, i] = 1.0 / a00 + (u0 @ z @ l0)
            lu_inv[i + 1 :, i] = -(z @ l0)
            lu_inv[i, i + 1 :] = -(u0 @ z)
