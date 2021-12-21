import sympy as sp

SympyArray = sp.tensor.array.dense_ndim_array.ImmutableDenseNDimArray
SympyMatrix = sp.matrices.dense.MutableDenseMatrix
SympySymbol = sp.core.symbol.Symbol
EPSILON = 1e-8
EPSILON_SQ = EPSILON * EPSILON
