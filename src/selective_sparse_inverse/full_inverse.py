import numpy as np


def lu_inv_full(lu: np.ndarray):
    lu_inv = lu.copy()
    _lu_inv_inplace(lu_inv)
    return lu_inv


def _lu_inv_inplace(lu_inv: np.ndarray):
    n = lu_inv.shape[0]
    if n == 1:
        lu_inv[0, 0] = 1.0 / lu_inv[0, 0]
        return

    #  recursive call
    z = lu_inv[1:, 1:]
    _lu_inv_inplace(z)

    # calculate current pivot
    a00 = lu_inv[0, 0]
    l0 = lu_inv[1:, 0]
    u0 = lu_inv[0, 1:]
    lu_inv[0, 0] = 1.0 / a00 + 1.0 / a00 / a00 * (u0 @ z @ l0)
    lu_inv[1:, 0] = -1.0 / a00 * (z @ l0)
    lu_inv[0, 1:] = -1.0 / a00 * (u0 @ z)
