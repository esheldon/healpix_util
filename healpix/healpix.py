import numpy
from . import _healpix

class HealPix(_healpix.HealPix):
    """
    class representing a healpix resolution

    parameters
    ----------
    nside: int
        healpix resolution

    attributes
    ----------
    nside
    npix
    ncap
    area

    methods
    -------
    hp=HealPix(nside)
    pixnum=hp.eq2pix(ra, dec) 

    # these are the same as the read only attributes above
    get_nside()
    get_npix()
    get_ncap()
    get_area()
    """

    def eq2pix(self, ra, dec, scheme='ring'):
        """
        get the pixel number(s) for the input ra,dec

        parameters
        ----------
        ra: scalar or array
            right ascension
        dec: scalar or array
            declination
        scheme: string
            'ring' or 'nest'.  Only ring supported currently

        returns
        -------
        pixnum: scalar array
            The pixel number(s)
        """

        if scheme != 'ring':
            raise ValueError("only ring scheme is currently supported")

        is_scalar=numpy.isscalar(ra)

        ra  = numpy.array(ra, dtype='f8', ndmin=1, copy=False)
        dec = numpy.array(dec, dtype='f8', ndmin=1, copy=False)

        if ra.size != dec.size:
            raise ValueError("ra,dec must have same size, "
                             "got %s,%s" % (ra.size,dec.size))
        pixnum = numpy.zeros(ra.size, dtype='i8')

        super(HealPix,self)._fill_eq2pix_ring(ra, dec, pixnum)

        if is_scalar:
            pixnum=pixnum[0]
        return pixnum

    # read-only attributes
    nside = property(_healpix.HealPix.get_nside,doc="get the resolution")
    npix = property(_healpix.HealPix.get_npix,doc="number of pixels in the sky")
    ncap = property(_healpix.HealPix.get_ncap,doc="number of pixels in the northern cap")
    area = property(_healpix.HealPix.get_area,doc="area of a pixel")
