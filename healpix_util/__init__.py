"""
todo
    test quad_check with real weight map
"""

import healpy
from healpy import *

from .healpix import \
        HealPix, \
        RING, NESTED, NEST, \
        get_scheme_name, get_scheme_num

from .maps import Map, DensityMap

from .fileio import \
        load_map, load_maps, \
        load_density_map, load_density_maps, \
        writeMap, writeMaps

from .coords  import \
        eq2ang, ang2eq, eq2vec, \
        randsphere, randcap, get_posangle_eq, get_quadrant_eq

from .version import __version__
