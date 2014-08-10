"""
classes
-------
HealPix:
    class to work with healpixels

functions
---------
get_scheme_name():
    Get string form of input scheme specification
get_scheme_num():
    Get integer form of input scheme specification

constants
----------
RING=1
    integer referring to ring scheme
NESTED=2
    integer referring to nested scheme

POINT_OK=1<<0
    bit mask to indicate a point is "good" in a density map
QUAD12_OK=1<<1
    bit mask to indicate pair of quadrants 1+2 is "good" around a point
    in a density map
QUAD23_OK=1<<2
    bit mask to indicate pair of quadrants 2+3 is "good" around a point
    in a density map
QUAD34_OK=1<<3
    bit mask to indicate pair of quadrants 3+4 is "good" around a point
    in a density map
QUAD41_OK=1<<4
    bit mask to indicate pair of quadrants 4+1 is "good" around a point
    in a density map
"""
from __future__ import print_function
import numpy

import healpy
from . import _healpix
from . import coords

RING=1
NESTED=2
NEST=2

class HealPix(_healpix.HealPix):
    """
    class representing a healpix resolution

    parameters
    ----------
    scheme: string or int
        if a string is input, the value should be
            'ring' or 'nested' (case insensitive)
        if an int is input, the value should be
            healpix.RING (1) or healpix.NESTED (2)

    nside: int
        healpix resolution

    read-only attributes
    --------------------
    scheme
    scheme_num
    nside
    npix
    ncap
    area

    methods
    -------
    # see docs for each method for more information

    scheme="ring"
    nside=4096
    hp=HealPix(scheme,nside)

    eq2pix:
        Get pixnums for the input equatorial ra,dec in degrees
    ang2pix:
        Get pixnums for the input angular theta,phi in radians
    pix2eq:
        Get equatorial ra,dec in degrees for the input pixels
    pix2ang:
        Get angular theta,phi in radians for the input pixels
    query_disc:
        Get ids of pixels whose centers are contained with the disc
        or that intersect the disc (inclusive=True)

    # the following methods are the same as the read only attributes above
    get_scheme()     # scheme name
    get_scheme_num() # scheme number
    get_nside()
    get_npix()
    get_ncap()
    get_area()
    """

    def __init__(self, scheme, nside):
        scheme_num = get_scheme_num(scheme)
        super(HealPix,self).__init__(scheme_num, nside)

    def eq2pix(self, ra, dec):
        """
        get the pixel number(s) for the input ra,dec

        parameters
        ----------
        ra: scalar or array
            right ascension
        dec: scalar or array
            declination

        returns
        -------
        pixnum: scalar array
            The pixel number(s)
        """

        is_scalar=numpy.isscalar(ra)

        ra  = numpy.array(ra, dtype='f8', ndmin=1, copy=False)
        dec = numpy.array(dec, dtype='f8', ndmin=1, copy=False)

        if ra.size != dec.size:
            raise ValueError("ra,dec must have same size, "
                             "got %s,%s" % (ra.size,dec.size))
        pixnum = numpy.zeros(ra.size, dtype='i8')

        super(HealPix,self)._fill_eq2pix(ra, dec, pixnum)

        if is_scalar:
            pixnum=pixnum[0]
        return pixnum

    def ang2pix(self, theta, phi):
        """
        get the pixel number(s) for the input angular theta,phi

        parameters
        ----------
        theta: scalar or array
            theta in radians
        phi: scalar or array
            phi in radians

        returns
        -------
        pixnum: scalar array
            The pixel number(s)
        """

        is_scalar=numpy.isscalar(theta)

        theta = numpy.array(theta, dtype='f8', ndmin=1, copy=False)
        phi = numpy.array(phi, dtype='f8', ndmin=1, copy=False)

        if theta.size != phi.size:
            raise ValueError("theta,phi must have same size, "
                             "got %s,%s" % (theta.size,phi.size))
        pixnum = numpy.zeros(theta.size, dtype='i8')

        super(HealPix,self)._fill_ang2pix(theta, phi, pixnum)

        if is_scalar:
            pixnum=pixnum[0]
        return pixnum


    def pix2eq(self, pixnum):
        """
        get the nominal pixel center in equatorial ra,dec for the input pixel
        numbers

        parameters
        ----------
        pixnum: scalar array
            The pixel number(s)

        returns
        -------
        theta, phi: scalars or arrays
            theta in radians, phi in radians
        """

        is_scalar=numpy.isscalar(pixnum)

        pixnum = numpy.array(pixnum, dtype='i8', ndmin=1, copy=False)

        ra = numpy.zeros(pixnum.size, dtype='f8')
        dec = numpy.zeros(pixnum.size, dtype='f8')

        super(HealPix,self)._fill_pix2eq(pixnum, ra, dec)

        if is_scalar:
            ra=ra[0]
            dec=dec[0]
        return ra, dec

    def pix2ang(self, pixnum):
        """
        get the nominal pixel center in angular theta,phi for the input pixel
        numbers

        parameters
        ----------
        pixnum: scalar array
            The pixel number(s)

        returns
        -------
        theta, phi: scalars or arrays
            theta in radians, phi in radians
        """

        is_scalar=numpy.isscalar(pixnum)

        pixnum = numpy.array(pixnum, dtype='i8', ndmin=1, copy=False)

        theta = numpy.zeros(pixnum.size, dtype='f8')
        phi = numpy.zeros(pixnum.size, dtype='f8')

        super(HealPix,self)._fill_pix2ang(pixnum, theta, phi)

        if is_scalar:
            theta=theta[0]
            phi=phi[0]
        return theta, phi

    def query_disc(self, coord1, coord2, radius, system='eq', **kw):
        """
        get pixels that are contained within or intersect the disc

        hpix.query_disc(ra,dec,radius_degrees)
        hpix.query_disc(theta,phi,radius_radians,system='ang')

        parameters
        ----------
        coord1: scalar
            If system=='eq' this is ra degrees
            If system=='ang' this is theta radians
        coord2: scalar
            If system=='eq' this is dec degrees
            If system=='ang' this is phi radians
        radius: scalar
            radius of disc
            If system=='eq' this is in degrees
            If system=='ang' this is in radians
        system: string
            'eq' for equatorial ra,dec in degrees
            'ang' for angular theta,phi in radians

            default 'eq'

        keywords for healpy.query_disc

        inclusive: bool, optional
            If False, return the exact set of pixels whose pixel centers lie
            within the disk; if True, return all pixels that overlap with the
            disk, and maybe a few more. Default: False

        fact : int, optional
            Only used when inclusive=True. The overlapping test will be done at
            the resolution fact*nside. For NESTED ordering, fact must be a
            power of 2, else it can be any positive integer. Default: 4.
        nest: bool, optional
            if True, assume NESTED pixel ordering, otherwise, RING pixel
            ordering
        buff: int array, optional
            if provided, this numpy array is used to contain the return values
            and must be at least long enough to do so

        returns
        -------
        pixnums: array
            Array of pixel numbers that are contained or intersect the disc

        examples
        --------
        from hpix_util import HealPix
        hpix = HealPix("ring", 4096)

        ra=200.0
        dec=0.0
        radius=1.0
        pixnums=hpix.query_disc(ra, dec, radius)
        """


        if system=='eq':
            vec=coords.eq2vec(coord1, coord2)
            rad_send=numpy.deg2rad(radius)
        elif system=='ang':
            vec=healpy.ang2vec(coord1, coord2)
            rad_send=radius
        else:
            raise ValueError("system should be 'eq' or 'ang'")

        kw['nest'] = self.nested
        pixnums=healpy.query_disc(self.nside, vec, radius, **kw)

        return pixnums

    # read-only attributes
    scheme = property(_healpix.HealPix.get_scheme,doc="get the healpix scheme name")
    scheme_num = property(_healpix.HealPix.get_scheme_num,doc="get the healpix scheme number")
    nested = property(_healpix.HealPix.is_nested,doc="1 if nested else 0")
    nside = property(_healpix.HealPix.get_nside,doc="get the resolution")
    npix = property(_healpix.HealPix.get_npix,doc="number of pixels in the sky")
    ncap = property(_healpix.HealPix.get_ncap,doc="number of pixels in the northern cap")
    area = property(_healpix.HealPix.get_area,doc="area of a pixel")



def get_scheme_name(scheme):
    """
    get the string version of a scheme.

    parameters
    ----------
    scheme: int or string
        'ring' or 'nested' or in lower or upper case,
        or 1 for ring, 2 for nested

        The numerical versions can be gotten with healpix.RING and
        healpix.NESTED

    returns
    -------
    'ring' or 'ntest'
    """
    if scheme not in _scheme_name_map:
        raise ValueError("bad scheme specification: '%s'" % scheme)
    return _scheme_name_map[scheme]

def get_scheme_num(scheme):
    """
    get the integer version of a scheme.

    parameters
    ----------
    scheme: int or string
        'ring' or 'nested' or 'nested' in lower or upper case,
        or 1 for ring, 2 for nested

        The numerical versions can be gotten with healpix.RING and
        healpix.NESTED

    returns
    -------
    1 for ring, 2 for nested
    """
    if scheme not in _scheme_num_map:
        raise ValueError("bad scheme specification: '%s'" % scheme)
    return _scheme_num_map[scheme]


_scheme_num_map={'ring':RING,
                 'RING':RING,
                 RING:RING,
                 'nest':NESTED,
                 'nested':NESTED,
                 'NESTED':NESTED,
                 'NESTED':NESTED,
                 NESTED:NESTED}
_scheme_name_map={'ring':'RING',
                  'RING':'RING',
                  RING:'RING',
                  'nest':'NESTED',
                  'nested':'NESTED',
                  'NESTED':'NESTED',
                  'NESTED':'NESTED',
                  NESTED:'NESTED'}


'''
def nest2ring(nside, ipnest):
    """
    convert the input pixel number(s) in nested scheme to ring scheme

    parameters
    ----------
    nside: int
        healpix resolution
    ipnest: scalar or array
        The pixel number(s) in nested scheme

    returns
    -------
    ipring: scalar array
        The pixel number(s) in ring scheme
    """

    # just to hold some metadata
    is_scalar=numpy.isscalar(ipnest)

    ipnest = numpy.array(ipnest, dtype='i8', ndmin=1, copy=False)
    ipring = numpy.zeros(ipnest.size, dtype='i8')

    _healpix._fill_nest2ring(nside, ipnest, ipring)

    if is_scalar:
        ipring=ipring[0]
    return ipring

def ring2nest(nside, ipring):
    """
    convert the input pixel number(s) in ring scheme to nested scheme

    parameters
    ----------
    nside: int
        healpix resolution
    ipring: scalar or array
        The pixel number(s) in ring scheme

    returns
    -------
    ipnest: scalar or array
        The pixel number(s) in nested scheme
    """

    # just to hold some metadata
    is_scalar=numpy.isscalar(ipring)

    ipring = numpy.array(ipring, dtype='i8', ndmin=1, copy=False)
    ipnest = numpy.zeros(ipring.size, dtype='i8')

    _healpix._fill_ring2nest(nside, ipring, ipnest)

    if is_scalar:
        ipnest=ipnest[0]
    return ipnest
'''


