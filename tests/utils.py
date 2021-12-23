import numpy as np


def calc_inner_products(inner, vecs):
    return np.array([[inner(v1, v2) for v1 in vecs] for v2 in vecs])


def is_diagonal(mat) -> bool:
    return np.all(np.isclose(mat, 0) | np.eye(len(mat)).astype(bool))


def are_orthogonal(inner, vecs):
    return is_diagonal(calc_inner_products(inner, vecs))
