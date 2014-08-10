"""
classes
-------
Map:
    class to contain a healpix map
DensityMap:
    class to contain a density healpix map

constants
----------
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

POINT_OK=1<<0
QUAD12_OK=1<<1
QUAD23_OK=1<<2
QUAD34_OK=1<<3
QUAD41_OK=1<<4

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

    from healpix_util import Map
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
        nside = healpy.npix2nside(array.size)

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
            ipnest=healpy.ring2nest(self.hpix.nside, ipring)

            newdata=self.data.copy()
            newdata[ipnest]=self.data
        else:
            ipnest=numpy.arange(self.hpix.npix,dtype='i8')
            ipring=healpy.nest2ring(self.hpix.nside, ipnest)

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

    from hpix_util import DensityMap
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

    def check_quad(self, ra, dec, radius,
                   inclusive=False,
                   pmin=0.95,
                   verbose=False):
        """
        Check quadrants around the specified point.  Only makes sens if the map
        is a weight map (or Neffective, etc).  ra,dec only for now.

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


