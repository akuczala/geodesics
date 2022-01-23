import numpy as np
import sympy as sp

from geodesics.coordinate_map import CoordinateMap

r, th, ph = sp.symbols('r th ph')
x, y, z = sp.symbols('x y z')
POLAR_MAPPING = CoordinateMap(
    domain_coordinates=sp.symbols('r ph'),
    image_coordinates=sp.symbols('x y'),
    mapping=sp.Array([r * sp.cos(ph), r * sp.sin(ph)])
)
SPHERICAL_MAPPING = CoordinateMap(
    domain_coordinates=sp.symbols('r th ph'),
    image_coordinates=sp.symbols('x y z'),
    mapping=sp.Array([r * sp.cos(ph) * sp.sin(th), r * sp.sin(ph) * sp.sin(th), r * sp.cos(th)])
)
INVERSE_SPHERICAL_MAPPING = CoordinateMap(
    domain_coordinates=sp.symbols('x y z'),
    image_coordinates=sp.symbols('r th ph'),
    mapping=sp.Array([
        sp.sqrt(x ** 2 + y ** 2 + z ** 2), sp.acos(z / sp.sqrt(x ** 2 + y ** 2 + z ** 2)), sp.atan2(y, x)
    ])
)


def get_bispherical_mapping(a, beta, **kwargs):
    s, ps, ph = sp.symbols('s ps ph')
    tau = -beta * sp.cos(ps)
    b = a / (sp.cosh(tau) - sp.cos(s))
    return CoordinateMap(
        domain_coordinates=sp.symbols('ps s ph'),
        image_coordinates=sp.symbols('x y z'),
        mapping=sp.Array([
            b * sp.sinh(tau),
            b * sp.sin(s) * sp.sin(ph),
            b * sp.sin(s) * sp.cos(ph)
        ]), **kwargs
    )
