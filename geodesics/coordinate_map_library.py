import sympy as sp

from geodesics.coordinate_map import CoordinateMap

r, th, ph = sp.symbols('r th ph')
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