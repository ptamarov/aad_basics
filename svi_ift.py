import numpy as np
from numpy.typing import NDArray

array = NDArray[np.float64]


def jacobian_svi(z: array, P: array) -> array:
    """
    Input:
        - z of length N
        - P of length 5

    Output:
        - matrix J of N rows and 5 columns.

    """
    _, b, r, m, s = P
    u = z - m
    root = np.sqrt(u**2 + s**2)
    dfda = np.ones_like(z)
    dfdb = r * u + root
    dfdr = b * u
    dfds = b * s / root
    dfdm = -b * (r + u / root)
    return np.column_stack([dfda, dfdb, dfdr, dfdm, dfds])


def residual_h(r: array, z: array, P: array) -> array:
    """
    Input:
     - z of length N
     - P of length 5

    Output:
     - residual symmetric matrix of size 5 x 5
    """
    _, b, rho, m, s = P

    x = z - m
    q = np.sqrt(x**2 + s**2)
    q3 = q**3

    H = np.zeros((5, 5))

    H[1, 2] = H[2, 1] = np.sum(r * x)
    H[1, 3] = H[3, 1] = np.sum(r * (-rho - x / q))
    H[1, 4] = H[4, 1] = np.sum(r * s / q)

    H[2, 3] = H[3, 2] = -b * np.sum(r)
    H[3, 3] = b * np.sum(r * s**2 / q3)
    H[3, 4] = H[4, 3] = b * np.sum(r * x * s / q3)
    H[4, 4] = b * np.sum(r * x**2 / q3)

    return H


def amat_solve(r: array, z: array, P: array, dt: array) -> array:
    J = jacobian_svi(z, P)
    Res = residual_h(r, z, P)
    A = J.T @ J + Res
    x0 = np.linalg.solve(A, dt.T)
    return x0
