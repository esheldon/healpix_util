"""
classes
-------
HealPix:
    class to work with healpixels

Map:
    class to contain a healpix map

functions
---------
read_fits:
    Read a healpix map from a fits file
get_scheme_string():
    Get string form of input scheme specification
get_scheme_int():
    Get integer form of input scheme specification

constants
----------
RING=1
    integer referring to ring scheme
NEST=2
    integer referring to nest scheme

"""
from __future__ import print_function
import numpy
from . import _healpix
from ._healpix import nside_is_ok, npix_is_ok, nside2npix, npix2nside
from . import coords

RING=1
NEST=2

class HealPix(_healpix.HealPix):
    """
    class representing a healpix resolution

    parameters
    ----------
    scheme: string or int
        if a string is input, the value should be
            'ring' or 'nest' or 'nested'
        if an int is input, the value should be
            healpix.RING (1) or healpix.NEST (2)
        
        Currently, only the ring scheme is fully supported

    nside: int
        healpix resolution

    read-only attributes
    --------------------
    scheme
    scheme_name
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

    pixnums=hp.eq2pix(ra, dec)
        Get pixnums for the input equatorial ra,dec in degrees
    pixnums=hp.ang2pix(ra, dec) 
        Get pixnums for the input angular theta,phi in radians
    ra,dec=hp.pix2eq(pixnums)
        Get equatorial ra,dec in degrees for the input pixels
    theta,phi=hp.pix2ang(pixnums)
        Get angular theta,phi in radians for the input pixels
    pixnums = hp.query_disc(ra=,dec=,theta=,phi=,radius=,inclusive=)
        Get ids of pixels whose centers are contained with the disc
        or that intersect the disc (inclusive=True)

    # the following methods are the same as the read only attributes above
    get_scheme()
    get_scheme_name()
    get_nside()
    get_npix()
    get_ncap()
    get_area()
    """

    def __init__(self, scheme, nside):
        scheme_int = get_scheme_int(scheme)
        if scheme_int != RING:
            raise ValueError("only ring scheme is currently supported")

        super(HealPix,self).__init__(scheme_int, nside)

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

    def query_disc(self,
                   ra=None,
                   dec=None,
                   theta=None,
                   phi=None,
                   radius=None,
                   inclusive=False):
        """
        get pixels that are contained within or intersect the disc

        Send either
            ra=,dec=,radius= in degrees
        or
            theta=,phi=,radius in radians

        parameters
        ----------
        Either of the following set of keywords

        ra=,dec=,radius=: scalars
            equatorial coordinates and radius in degrees
        theta=,phi=,radius=: scalars
            angular coordinates and radius in radians

        inclusive: bool
            If False, include only pixels whose centers are within the disc.
            If True, include any pixels that intersect the disc

            Default is False

        returns
        -------
        pixnums: array
            Array of pixel numbers that are contained or intersect the disc
        """

        if not inclusive:
            inclusive_int=0
        else:
            inclusive_int=1

        if radius is None:
            raise ValueError("send radius=")

        if ra is not None and dec is not None:
            pixnums=super(HealPix,self)._query_disc(ra,
                                                    dec,
                                                    radius,
                                                    coords.SYSTEM_EQ,
                                                    inclusive_int)
        elif theta is not None and phi is not None:
            pixnums=super(HealPix,self)._query_disc(theta,
                                                    phi,
                                                    radius,
                                                    coords.SYSTEM_ANG,
                                                    inclusive_int)
        else:
            raise ValueError("send ra=,dec= or theta=,phi=")

        return pixnums

    # read-only attributes
    scheme = property(_healpix.HealPix.get_scheme,doc="get the healpix scheme name")
    scheme_num = property(_healpix.HealPix.get_scheme_num,doc="get the healpix scheme number")
    nside = property(_healpix.HealPix.get_nside,doc="get the resolution")
    npix = property(_healpix.HealPix.get_npix,doc="number of pixels in the sky")
    ncap = property(_healpix.HealPix.get_ncap,doc="number of pixels in the northern cap")
    area = property(_healpix.HealPix.get_area,doc="area of a pixel")

def read_fits(filename, **keys):
    """
    read healpix map(s) from the specified file

    parameters
    ----------
    filename: string
        The fits filename
    scheme: string or int
        Optional scheme specification.  If the scheme is not specified
        in the header as 'ORDERING', then you can specify it with
        this keyword.

        Also if the specified scheme does not match the ORDERING in the
        header, the maps will be converted to the requested scheme.

    **keys:
        other keywords for the fits reading, such as 
            ext= (default 1)
            columns= (default is to read all columns)
            header=True to return the header
        See the fitsio documentation for more details

    returns
    -------
    a Map object holding the requested maps.
    
    """
    import fitsio

    scheme = keys.get("scheme",None)
    if scheme is not None:
        scheme=get_scheme_string(scheme)

    # ensure None or a sequence, which makes
    # fitsio return a recarray
    columns=keys.get('columns',None)
    if columns is not None:
        if numpy.isscalar(columns):
            keys['columns']=[columns]

    with fitsio.FITS(filename) as fits:

        ext=keys.get('ext',1)
        hdu = fits[ext]
        if not isinstance(hdu,fitsio.fitslib.TableHDU):
            raise ValueError("extension %s is not a table" % ext)

        header = hdu.read_header()

        # user may specify columns= here
        data = hdu.read(**keys)


        if 'ordering' in header:
            scheme_in_file = header['ordering'].strip()
            scheme_in_file = get_scheme_string(scheme_in_file)
        else:
            # we need input from the user
            if scheme is None:
                raise ValueError("ORDERING not in header, send scheme= "
                                 "to specify")
            scheme_in_file = scheme

        hmap = Map(scheme_in_file, data)

        if scheme is not None and scheme != scheme_in_file:
            print("converting from scheme '%s' "
                  "to '%s'" % (scheme_in_file,scheme))
            hmap = hmap.convert(scheme)

    gethead=keys.get("header",False)
    if gethead:
        return hmap, header
    else:
        return hmap

def read_fits_old(filename, **keys):
    """
    read healpix map(s) from the specified file

    parameters
    ----------
    filename: string
        The fits filename
    scheme: string or int
        Optional scheme specification.  If the scheme is not specified
        in the header as 'ORDERING', then you can specify it with
        this keyword.

        Also if the specified scheme does not match the ORDERING in the
        header, the maps will be converted to the requested scheme.

    **keys:
        other keywords for the fits reading, such as 
            ext= (default 1)
            columns= (default is to read all columns)
            header=True to return the header
        See the fitsio documentation for more details

    returns
    -------
    By default, a dict of healpix.Map is returned, keyed by the map column
    name.
    
    If a single scalar columns= is specified, a single map is returned
    instead of a list.

    if header=True is specified, a tuple (maps, header) is returned
    """
    import fitsio

    scheme = keys.get("scheme",None)
    if scheme is not None:
        scheme=get_scheme_string(scheme)

    with fitsio.FITS(filename) as fits:

        ext=keys.get('ext',1)
        hdu = fits[ext]
        if not isinstance(hdu,fitsio.fitslib.TableHDU):
            raise ValueError("extension %s is not a table" % ext)

        header = hdu.read_header()

        # user may specify columns= here
        data = hdu.read(**keys)

        if 'ordering' in header:
            scheme_in_file = header['ordering'].strip()
            scheme_in_file = get_scheme_string(scheme_in_file)
        else:
            # we need input from the user
            if scheme is None:
                raise ValueError("ORDERING not in header, send scheme= "
                                 "to specify")
            scheme_in_file = scheme

        if scheme is not None and scheme != scheme_in_file:
            print("converting from scheme '%s' "
                  "to '%s'" % (scheme_in_file,scheme))
            do_convert=True
        else:
            do_convert=False

        names=data.dtype.names
        if names is not None:
            # there were multiple columns read
            map_dict={}
            for name in names:
                m = Map(scheme_in_file,data[name])
                if do_convert:
                    m=m.convert(scheme)
                map_dict[name] = m

            result = map_dict
        else:
            # columns=scalar was used to read
            m = Map(scheme_in_file,data)
            if do_convert:
                m=m.convert(scheme)
            result = m

    gethead=keys.get("header",False)
    if gethead:
        return result, header
    else:
        return result


class Map(object):
    """
    class to represent a healpix map.

    parameters
    ----------
    scheme: string or int
        if a string is input, the value should be
            'ring' or 'nest'
        if an int is input, the value should be
            healpix.RING or healpix.NEST.
    array: sequence or array
        array representing the healpix map data

    attributes
    ----------
    .hpix    # A HealPix object, the pixel specification
    .data    # the healpix map as a numpy array.  .data.size is 
             # the number of pixels

    data access
    -----------
    # access the .data attribute, which is a numpy array
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

    # convert map to nested
    nmap = m.convert("nest")
    # convert map to ring
    rmap = m.convert("ring")

    methods
    -------
    convert(scheme):
        convert the map to the specified scheme.  If the current map is already
        in the specified scheme, no copy of the underlying data is made
    """
    def __init__(self, scheme, array):

        array = array.ravel()
        nside = npix2nside(array.size)

        self.hpix = HealPix(scheme, nside)
        self.data = numpy.array(array, ndmin=1, copy=False)
    
    def quad_check(self,
                   ra=None, dec=None,
                   theta=None, phi=None,
                   radius=None,
                   inclusive=False,
                   pmin=0.95):
        """
        Check quadrants around the specified point.
        """

        pixnums = self.hpix.query_disc(ra=ra,
                                       dec=dec,
                                       theta=theta,
                                       phi=phi,
                                       radius=radius,
                                       inclusive=inclusive)
        # the "weights" from our map.
        weights = self.data[pixnums]
        wmax = weights.max()

        wsum_norm=weights.sum()/wmax
        print("wsum norm:",wsum_norm)

    def __getitem__(self, args):
        return self.data[args]

    def __repr__(self):
        hrep = self.hpix.__repr__()
        array_repr=self.data.__repr__()
        rep="""
healpix Map

%s
map data:
%s""" % (hrep, array_repr)
        return rep

def get_scheme_string(scheme):
    """
    get the string version of a scheme.

    parameters
    ----------
    scheme: int or string
        'ring' or 'nest' or 'nested' in lower or upper case,
        or 1 for ring, 2 for nest

        The numerical versions can be gotten with healpix.RING and
        healpix.NEST

    returns
    -------
    'ring' or 'ntest'
    """
    if scheme not in _scheme_string_map:
        raise ValueError("bad scheme specification: '%s'" % scheme)
    return _scheme_string_map[scheme]

def get_scheme_int(scheme):
    """
    get the string version of a scheme.

    parameters
    ----------
    scheme: int or string
        'ring' or 'nest' or 'nested' in lower or upper case,
        or 1 for ring, 2 for nest

        The numerical versions can be gotten with healpix.RING and
        healpix.NEST

    returns
    -------
    1 for ring, 2 for nest
    """
    if scheme not in _scheme_int_map:
        raise ValueError("bad scheme specification: '%s'" % scheme)
    return _scheme_int_map[scheme]


_scheme_int_map={'ring':RING,
                 'RING':RING,
                 RING:RING,
                 'nest':NEST,
                 'nested':NEST,
                 'NEST':NEST,
                 'NESTED':NEST,
                 NEST:NEST}
_scheme_string_map={'ring':'ring',
                    'RING':'ring',
                    RING:'ring',
                    'nest':'nest',
                    'nested':'nest',
                    'NEST':'nest',
                    'NESTED':'nest',
                    NEST:'nest'}



