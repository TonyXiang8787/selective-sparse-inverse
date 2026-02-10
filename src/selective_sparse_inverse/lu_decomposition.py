import numpy as np


def lu_no_pivot(A: np.ndarray):
    n = A.shape[0]
    lu = A.copy()
    for k in range(n):
        lu[k + 1 :, k] /= lu[k, k]
        lu[k + 1 :, k + 1 :] -= np.outer(lu[k + 1 :, k], lu[k, k + 1 :])

    l = np.tril(lu, k=-1) + np.eye(n)
    u = np.triu(lu)
    assert np.allclose(l @ u, A)
    return lu
