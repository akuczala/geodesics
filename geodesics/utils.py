from functools import reduce
from typing import List, Iterable

import numpy as np

from geodesics.constants import EPSILON


def sympy_matrix_to_numpy(sp_mat):
    return np.array(sp_mat, dtype=float)


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


def gram_schmidt(inner, basis: Iterable[np.ndarray]) -> List[np.ndarray]:
    return reduce(
        lambda vecs, b: vecs + [b - sum(project(inner, v, b) for v in vecs)],
        basis[1:], [basis[0]]
    )


def calc_orthogonal(inner, v: np.ndarray, d: np.ndarray) -> np.ndarray:
    return d - project(inner, v, d)


def project(inner, onto_u, v):
    return inner(onto_u, v) / inner(onto_u, onto_u) * onto_u
