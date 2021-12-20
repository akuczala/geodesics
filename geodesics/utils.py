import numpy as np

from geodesics.constants import EPSILON


def sympy_matrix_to_numpy(sp_mat):
    return np.array(sp_mat, dtype=np.float)


def solve_real_quad(a, b, c):
    """
    Solve a x^2 + b x + c == 0 for real roots
    """
    det = b ** 2 - 4 * a * c
    if det < EPSILON:
        return []
    if det > EPSILON:
        return list((-b + np.array([-1, 1]) * np.sqrt(det)) / (2 * a))
    return [-b / (2 * a)]
