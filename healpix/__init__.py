"""
todo

    read_map to read a map from fits file

    Map class to hold a map
        allow conversions .as_ring() .as_nest() return
        new maps, only copying if necessary

    in coords.py use coordinate system converts from C code

"""

from .version import __version__
from .healpix import HealPix, Map, read_fits
from .healpix import nside_is_ok, npix_is_ok, nside2npix, npix2nside
from .healpix import RING, NEST
from .coords  import eq2ang, ang2eq, eq2xyz, ang2xyz
