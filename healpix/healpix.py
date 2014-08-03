import numpy
from . import _healpix

RING=1
NEST=2

_scheme_int_map={'ring':RING,
                 RING:RING,
                 'nest':NEST,
                 NEST:NEST}
_scheme_string_map={'ring':'ring',
                    RING:'ring',
                    'nest':'nest',
                    NEST:'nest'}


class HealPix(_healpix.HealPix):
    """
    class representing a healpix resolution

    parameters
    ----------
    nside: int
        healpix resolution
    scheme: string or int
        if a string is input, the value should be
            'ring' or 'nest'
        if an int is input, the value should be
            healpix.RING or healpix.NEST.
        
        Only the ring scheme is fully supported currently

    attributes
    ----------
    scheme
    scheme_name
    nside
    npix
    ncap
    area

    methods
    -------
    hp=HealPix(nside)
    pixnum=hp.eq2pix(ra, dec) 

    # the following methods are the same as the read only attributes above
    get_scheme()
    get_scheme_name()
    get_nside()
    get_npix()
    get_ncap()
    get_area()
    """

    def __init__(self, nside, scheme='ring'):
        scheme_int = _scheme_int_map.get(scheme,None)
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



    # read-only attributes
    scheme = property(_healpix.HealPix.get_scheme,doc="get the healpix scheme")
    scheme_name = property(_healpix.HealPix.get_scheme_name,doc="get the healpix scheme name")
    nside = property(_healpix.HealPix.get_nside,doc="get the resolution")
    npix = property(_healpix.HealPix.get_npix,doc="number of pixels in the sky")
    ncap = property(_healpix.HealPix.get_ncap,doc="number of pixels in the northern cap")
    area = property(_healpix.HealPix.get_area,doc="area of a pixel")
