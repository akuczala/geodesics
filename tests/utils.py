from typing import List

import numpy as np

from geodesics.metric_space import MetricSpace
from geodesics.tangent_vector import TangentVector, TangentVectorType


def calc_inner_products(inner, vecs):
    return np.array([[inner(v1, v2) for v1 in vecs] for v2 in vecs])


def is_diagonal(mat) -> bool:
    return np.all(np.isclose(mat, 0) | np.eye(len(mat)).astype(bool))


def are_orthogonal(inner, vecs):
    return is_diagonal(calc_inner_products(inner, vecs))


# brute force a timelike vector
# todo produce timelike vector systematically
def gen_timelike_vector(metric_space: MetricSpace, pos: np.ndarray):
    N_TRIES = 100
    for _ in range(N_TRIES):
        r = np.random.randn(metric_space.dim)
        if metric_space.classify_tangent_vector(TangentVector(x=pos, u=r)) == TangentVectorType.TIMELIKE:
            return r
    raise Exception(f"Could not produce timelike vector at {pos} after {N_TRIES} tries")


def gen_S1_vecs(n: int, min_angle=0, max_angle=2 * np.pi) -> List[np.ndarray]:
    return [np.array([np.cos(t), np.sin(t)]) for t in np.linspace(min_angle, max_angle, n + 1)[:-1]]


def normalize_euclid(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)
