"""
todo

    Map class to hold a map
        allow conversions .convert()

    add quad_check to Map

    in coords.py use coordinate system converts from C code

"""

from .version import __version__
from .healpix import HealPix, Map, read_fits
from .healpix import nside_is_ok, npix_is_ok, nside2npix, npix2nside
from .healpix import RING, NEST
from .coords  import eq2ang, ang2eq, eq2xyz, ang2xyz
