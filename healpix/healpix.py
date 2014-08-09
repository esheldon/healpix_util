"""
classes
-------
HealPix:
    class to work with healpixels
Map:
    class to contain a healpix map
DensityMap:
    class to contain a density healpix map

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
from . import _healpix
from ._healpix import nside_is_ok, npix_is_ok, nside2npix, npix2nside
from . import coords

RING=1
NESTED=2
NEST=2
POINT_OK=1<<0
QUAD12_OK=1<<1
QUAD23_OK=1<<2
QUAD34_OK=1<<3
QUAD41_OK=1<<4

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

    def query_disc(self, coord1, coord2, radius, system='eq', inclusive=False):
        """
        get pixels that are contained within or intersect the disc

        hpix.query_disc(ra,dec,radius_degrees)
        hpix.query_disc(theta,phi,radius_degrees,system='ang')

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
        inclusive: bool
            If False, include only pixels whose centers are within the disc.
            If True, include any pixels that intersect the disc

            Default is False

        returns
        -------
        pixnums: array
            Array of pixel numbers that are contained or intersect the disc

        examples
        --------
        import healpix
        hpix = healpix.HealPix("ring", 4096)

        ra=200.0
        dec=0.0
        radius=1.0
        pixnums=hpix.query_disc(ra, dec, radius)
        """

        if not inclusive:
            inclusive_int=0
        else:
            inclusive_int=1

        if radius is None:
            raise ValueError("send radius=")

        if system=='eq':
            pixnums=super(HealPix,self)._query_disc(coord1,
                                                    coord2,
                                                    radius,
                                                    coords.SYSTEM_EQ,
                                                    inclusive_int)
        elif system=='ang':
            pixnums=super(HealPix,self)._query_disc(coord1,
                                                    coord2,
                                                    radius,
                                                    coords.SYSTEM_ANG,
                                                    inclusive_int)
        else:
            raise ValueError("system should be 'eq' or 'ang'")

        return pixnums

    # read-only attributes
    scheme = property(_healpix.HealPix.get_scheme,doc="get the healpix scheme name")
    scheme_num = property(_healpix.HealPix.get_scheme_num,doc="get the healpix scheme number")
    nside = property(_healpix.HealPix.get_nside,doc="get the resolution")
    npix = property(_healpix.HealPix.get_npix,doc="number of pixels in the sky")
    ncap = property(_healpix.HealPix.get_ncap,doc="number of pixels in the northern cap")
    area = property(_healpix.HealPix.get_area,doc="area of a pixel")


class Map(object):
    """
    class to represent a healpix map.

    parameters
    ----------
    scheme: string or int
        if a string is input, the value should be
            'ring' or 'nested' (case insensitive)
        if an int is input, the value should be
            healpix.RING or healpix.NESTED.
    array: sequence or array
        array representing the healpix map data

    attributes
    ----------
    .hpix    # A HealPix object, the pixel specification
    .data    # the healpix map as a numpy array.  .data.size is 
             # the number of pixels
    .scheme  # the healpix scheme used (gotten from .hpix)
    .nside   # resolution of healpix map (gotten from .hpix)

    examples
    --------

    m=Map(scheme, array)
    print(m)

    metadata:
    scheme:       1
    scheme_name:  RING
    nside:        4096
    npix:         201326592
    ncap:         33546240
    area:         6.24178e-08 square degrees
    area:         0.000224704 square arcmin
    area:         0.808935 square arcsec

    map data:
    array([ -1.63750000e+30,  -1.63750000e+30,  -1.63750000e+30, ...,
            -1.63750000e+30,  -1.63750000e+30,  -1.63750000e+30], dtype=float32)

    print("scheme:",m.hpix.scheme)
    print("nside:",m.hpix.nside)
    print("number of pixels:",m.data.size)
    print("should match:",m.hpix.npix)

    print("pixel 35:",m.data[35])
    print("pixels 200-209:",m.data[200:210])
    print("pixels",indices,":,m.data[indices])

    # get a new map converted to the requested scheme
    hmap = m.convert("nested")
    hmap = m.convert("ring")

    methods
    -------
    get_mapval:
        Get the value of the map at the given coordinates
    convert:
        convert the map to the specified scheme.  If the current map is already
        in the specified scheme, no copy of the underlying data is made
    """
    def __init__(self, scheme, array):

        array = array.ravel()
        nside = npix2nside(array.size)

        self.hpix = HealPix(scheme, nside)
        self.data = numpy.array(array, ndmin=1, copy=False)

    def get_mapval(self, coord1, coord2, system='eq'):
        """
        get the value of the map for the input coordinates

        parameters
        ----------
        coord1: scalar or array
            If system=='eq' this is ra in degrees
            If system=='ang' this is theta in radians
        coord2: scalar or array
            If system=='eq' this is dec in degrees
            If system=='ang' this is phi in radians

        system: string
            'eq' for equatorial ra,dec in degrees
            'ang' for angular theta,phi in radians

            default 'eq'

        returns
        -------
        the value of the map at the given coordinates
        """
        if system=='eq':
            pixnums=self.hpix.eq2pix(coord1, coord2)
        elif system=='ang':
            pixnums=self.hpix.ang2pix(coord1, coord2)
        else:
            raise ValueError("bad system: '%s'" % system)

        return self.data[pixnums]

    def convert(self, scheme):
        """
        get a new Map with the specified scheme

        if the scheme would be unchanged, a reference to the internal map data
        is used

        parameters
        ----------
        scheme: string or int
            if a string is input, the value should be
                'ring' or 'nested'
            if an int is input, the value should be
                healpix.RING or healpix.NESTED.

        returns
        -------
        newmap:
            A new map with the requested ordering
        """

        newdata = self._get_converted_data(scheme)
        return Map(scheme, newdata)

    def _get_converted_data(self, scheme):
        """
        internal routine to get the data converted to the requested
        scheme

        If the scheme would be unchanged, a reference to the data is returned
        """
        scheme_num = get_scheme_num(scheme)
        if scheme_num == self.hpix.scheme_num:
            return self.data
        
        if scheme_num==NESTED:
            ipring=numpy.arange(self.hpix.npix,dtype='i8')
            ipnest=ring2nest(self.hpix.nside, ipring)

            newdata=self.data.copy()
            newdata[ipnest]=self.data
        else:
            ipnest=numpy.arange(self.hpix.npix,dtype='i8')
            ipring=nest2ring(self.hpix.nside, ipnest)

            newdata=self.data.copy()
            newdata[ipring]=self.data

        return newdata

    def get_scheme(self):
        """
        get the scheme used in the map
        """
        return self.hpix.scheme

    def get_scheme_num(self):
        """
        get the scheme number used in the map
        """
        return self.hpix.scheme_num

    def get_nside(self):
        """
        get the nside used in the map
        """
        return self.hpix.nside

    scheme = property(get_scheme,doc="get the healpix scheme name")
    scheme_num = property(get_scheme_num,doc="get the healpix scheme number")
    nside = property(get_nside,doc="get the resolution")

    def __repr__(self):
        tname=str(type(self))
        hrep = self.hpix.__repr__()
        array_repr=self.data.__repr__()
        rep="""
%s

%s
map data:
%s""" % (tname, hrep, array_repr)
        return rep


class DensityMap(Map):
    """
    A healpix Map to represent a spatial density.  Provides additional methods
    beyond Map.

    The minimum value in the map should not go below zero.

    The overall scaling of the map does not matter, e.g. it could be on [0,1].

    parameters
    ----------
    scheme: string or int
        if a string is input, the value should be
            'ring' or 'nested' (case insensitive)
        if an int is input, the value should be
            healpix.RING or healpix.NESTED.
    array: sequence or array
        array representing the healpix map data

        the minimum value in the array should not be below zero.

    extra methods beyond Map
    ------------------------
    get_weight:
        get the weight for the input coordinates, equivalent to
        map.get_mapval()/map.data.max()
    check_quad:
        Check quadrants around the specified point.  Only makes sens if the map
        is a weight map (or Neffective, etc)
    genrand:
        generate random ra,dec points

    examples
    --------
    See docs for Map for basic examples.

    m=DensityMap(scheme, array)

    # generate random points according to the map
    ra,dec=m.genrand(10000)
    theta,phi=m.genrand(10000,system='ang')

    # check quadrants around the specified center
    maskflags=m.check_quad(ra=200., dec=0., radius=1.0)
    """

    def __init__(self, scheme, array):
        super(DensityMap,self).__init__(scheme, array)

        # do not allow values less than zero
        if numpy.any(self.data < 0.0):
            raise ValueError("found %d values < 0 in density map "
                             "density maps must be positive")

        self._maxval=self.data.max()
        self._maxval_inv=1.0/self._maxval

    def convert(self, scheme):
        """
        get a new DensityMap with the specified scheme

        if the scheme would be unchanged, a reference to the internal map data
        is used

        parameters
        ----------
        scheme: string or int
            if a string is input, the value should be
                'ring' or 'nested'
            if an int is input, the value should be
                healpix.RING or healpix.NESTED.

        returns
        -------
        newmap:
            A new density map with the requested ordering
        """

        newdata = self._get_converted_data(scheme)
        return DensityMap(scheme, newdata)

    def get_weight(self, coord1, coord2, system='eq'):
        """
        get the weight for the input coordinates

        this is equivalent to map.get_mapval()/map.data.max()

        parameters
        ----------
        coord1: scalar or array
            If system=='eq' this is ra degrees
            If system=='ang' this is theta radians
        coord2: scalar or array
            If system=='eq' this is dec degrees
            If system=='ang' this is phi radians

        system: string
            'eq' for equatorial ra,dec in degrees
            'ang' for angular theta,phi in radians

            default 'eq'

        returns
        -------
        the weight for the given coordinates.  This is equivalent
        to map.get_mapval()/map.data.max()
        """
        res = self.get_mapval(coord1, coord2, system=system)
        res *= self._maxval_inv
        return res

    def check_quad(self, ra, dec, radius,
                   system='eq',
                   inclusive=False,
                   pmin=0.95,
                   verbose=False):
        """
        Check quadrants around the specified point.  Only makes sens if the map
        is a weight map (or Neffective, etc)

        currently works in ra,dec only

        parameters
        ----------
        ra: scalar
            right ascension in degrees
        dec: scalar
            declination in degrees
        radius: scalar
            radius in degrees
        pmin: scalar
            Minimum relative "masked fraction" for quadrants to
            qualify as "good".
            
            This is essentially 
                weights.sum()/weights.max()/npoints
            but max weight is determined from the two adjacent
            quadrants that are being checked.

        inclusive: bool
            If False, include only pixels whose centers are within the disc.
            If True, include any pixels that intersect the disc

            Default is False


        verbose: bool
            If True, print some info about each quadrant

        returns
        -------
        maskflags: scalar 
            bitmask describing the point

            2**0: set if central point has weight > 0
            2**1: set if quadrants 1+2 area good pair
            2**2: set if quadrants 2+3 area good pair
            2**3: set if quadrants 3+4 area good pair
            2**4: set if quadrants 4+1 area good pair
        """

        hpix=self.hpix

        maskflags=0

        # check central point
        pixnum = hpix.eq2pix(ra,dec)
        weight = self.data[pixnum]
        if weight <= 0.0:
            return maskflags

        # mark central point as OK
        maskflags |= POINT_OK

        pixnums = hpix.query_disc(ra, dec, radius,
                                  system='eq',
                                  inclusive=inclusive)


        # the "weights" from our map (actually the raw values).
        weights = self.data[pixnums]

        # location of center of each pixel
        rapix,decpix=hpix.pix2eq(pixnums)

        # quadrants around central point for each pixel
        quadrants = coords.get_quadrant_eq(ra,dec,rapix,decpix)

        count=numpy.zeros(4,dtype='i8')
        wtmax=numpy.zeros(4)
        wsum=numpy.zeros(4)

        for i in xrange(4):
            quad=i+1
            w,=numpy.where(quadrants == quad)
            if w.size > 0:
                wts = weights[w]

                count[i] = w.size
                wtmax[i] = wts.max()
                wsum[i] = wts.sum()

        for ipair in xrange(4):
            ifirst = ipair % 4
            isec   = (ipair+1) % 4

            wtmax1=wtmax[ifirst]
            wtmax2=wtmax[isec]
            if wtmax1 > 0. and wtmax2 > 0.:
                wtmax12 = max(wtmax1,wtmax2)
                fm1 = wsum[ifirst]/wtmax12/count[ifirst]
                fm2 = wsum[isec]/wtmax12/count[isec]
                if verbose:
                    print(ifirst+1,isec+1,fm1,fm2)
                if fm1 > pmin and fm2 > pmin:
                    maskflags |= 1<<(ipair+1)

        return maskflags

    def genrand(self, nrand, system='eq', **keys):
        """
        generate random points from the map

        the points will follow the density in the map

        parameters
        ----------
        nrand: int
            number of randoms
        system: string
            'eq' for equatorial ra,dec in degrees
            'ang' for angular theta,phi in radians
        **kw:
            ra_range=, dec_range= to limit range of randoms
                in ra,dec system
            theta_range=, phi_range= to limit range of randoms
                in theta,phi system
        """
        from .coords import randsphere

        coord1=numpy.zeros(nrand,dtype='f8')
        coord2=numpy.zeros(nrand,dtype='f8')

        ngood=0
        nleft=nrand
        while nleft > 0:
            t1,t2=randsphere(nleft, system=system, **keys)

            weights=self.get_weight(t1,t2,system=system)

            # keep with probability equal to weight
            ru = numpy.random.random(nleft)

            w,=numpy.where( ru < weights )
            if w.size > 0:
                beg=ngood
                end=ngood+w.size
                coord1[beg:end] = t1[w]
                coord2[beg:end] = t2[w]

                ngood += w.size
                nleft -= w.size
        return coord1, coord2

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



