"""
todo
    test quad_check with real weight map
"""

try:
    import healpy
    from healpy import *
except ImportError:
    pass

from .healpix import \
        HealPix, \
        RING, NESTED, NEST, \
        get_scheme_name, get_scheme_num

from .maps import Map, DensityMap

from .fileio import \
        readMap, readMaps, \
        readDensityMap, readDensityMaps, \
        writeMap, writeMaps

from .coords  import \
        eq2ang, ang2eq, eq2vec, \
        randsphere, randcap, get_posangle_eq, get_quadrant_eq

from .version import __version__
