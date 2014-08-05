"""
todo

    Map class to hold a map
        allow conversions .convert()

    test quad_check with real weight map

    in coords.py use coordinate system converts from C code

"""

from .version import __version__

from .healpix import \
        HealPix, Map, \
        nside_is_ok, npix_is_ok, nside2npix, npix2nside, \
        RING, NEST

from .fileio import read_map, read_maps, read_density_map, read_density_maps

from .coords  import \
        eq2ang, ang2eq, eq2xyz, ang2xyz, \
        randsphere, randcap, get_posangle_eq, get_quadrant_eq
