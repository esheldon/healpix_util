"""
functions
---------
eq2ang:
    convert equatorial ra,dec in degrees to angular theta, phi in radians
ang2eq:
    convert angular theta, phi in radians to equatorial ra,dec in degrees
eq2vec:
    Convert equatorial ra,dec in degrees to x,y,z on the unit sphere
randsphere:
    generate random points on the unit sphere
randcap:
    generate random points in a spherical cap

constants
---------
SYSTEM_ANG=1
    angular theta,phi system in radians
SYSTEM_EQ=2
    equatorial ra,dec system in degrees
"""

from __future__ import print_function
import numpy as np
from . import _healpix

SYSTEM_ANG = 1
SYSTEM_EQ = 2


def eq2ang(ra, dec):
    """
    convert equatorial ra,dec in degrees to angular theta, phi in radians

    parameters
    ----------
    ra: scalar or array
        Right ascension in degrees
    dec: scalar or array
        Declination in degrees

    returns
    -------
    theta,phi: tuple
        theta = pi/2-dec*D2R # in [0,pi]
        phi   = ra*D2R       # in [0,2*pi]
    """
    is_scalar = np.isscalar(ra)

    ra = np.array(ra, dtype='f8', ndmin=1, copy=False)
    dec = np.array(dec, dtype='f8', ndmin=1, copy=False)
    if ra.size != dec.size:
        raise ValueError("ra,dec not same size: %s,%s" % (ra.size, dec.size))
    theta = np.zeros(ra.size, dtype='f8')
    phi = np.zeros(ra.size, dtype='f8')

    _healpix._fill_eq2ang(ra, dec, theta, phi)

    if is_scalar:
        theta = theta[0]
        phi = phi[0]

    return theta, phi


def ang2eq(theta, phi):
    """
    convert angular theta, phi in radians to equatorial ra,dec in degrees

    ra = phi*R2D            # [0,360]
    dec = (pi/2-theta)*R2D  # [-90,90]

    parameters
    ----------
    theta: scalar or array
        angular theta in radians
    phi: scalar or array
        angular phi in radians

    returns
    -------
    ra,dec: tuple
        ra  = phi*R2D          # in [0,360]
        dec = (pi/2-theta)*R2D # in [-90,90]
    """

    is_scalar = np.isscalar(theta)

    theta = np.array(theta, dtype='f8', ndmin=1, copy=False)
    phi = np.array(phi, dtype='f8', ndmin=1, copy=False)
    if theta.size != phi.size:
        raise ValueError(
            "ra,dec not same size: %s,%s" % (theta.size, phi.size)
        )
    ra = np.zeros(theta.size, dtype='f8')
    dec = np.zeros(theta.size, dtype='f8')

    _healpix._fill_ang2eq(theta, phi, ra, dec)

    if is_scalar:
        ra = ra[0]
        dec = dec[0]

    return ra, dec


def eq2vec(ra, dec):
    """
    Convert equatorial ra,dec in degrees to x,y,z on the unit sphere

    parameters
    ----------
    ra: scalar or array
        ra in degrees
    dec: scalar or array
        decin radians

    returns
    -------
    vec with x,y,z on the unit sphere
        A 1-d vector[3] or a 2-d array[npoints, 3]

    """
    is_scalar = np.isscalar(ra)

    ra = np.array(ra, dtype='f8', ndmin=1, copy=False)
    dec = np.array(dec, dtype='f8', ndmin=1, copy=False)
    if ra.size != dec.size:
        raise ValueError("ra,dec not same size: %s,%s" % (ra.size, dec.size))

    vec = np.zeros((ra.size, 3))
    x = vec[:, 0]
    y = vec[:, 1]
    z = vec[:, 2]

    _healpix._fill_eq2xyz(ra, dec, x, y, z)

    if is_scalar:
        vec = vec[0, :]

    return vec


def sphdist(ra1, dec1, ra2, dec2):
    """
    Get the arc length between two points on the unit sphere

    parameters
    ----------
    ra1, dec1: scalar or array
        Coordinates of two points or sets of points.
        Must be the same length.
    ra2, dec2: scalar or array
        Coordinates of two points or sets of points.
        Must be the same length.
    """

    is_scalar1 = np.isscalar(ra1)
    is_scalar2 = np.isscalar(ra2)

    xyz1 = eq2vec(ra1, dec1)
    xyz2 = eq2vec(ra2, dec2)

    if is_scalar1:
        xyz1 = xyz1[np.newaxis, :]
    if is_scalar2:
        xyz2 = xyz2[np.newaxis, :]

    x1, y1, z1 = xyz1[:, 0], xyz1[:, 1], xyz1[:, 2]
    x2, y2, z2 = xyz2[:, 0], xyz2[:, 1], xyz2[:, 2]

    costheta = x1*x2 + y1*y2 + z1*z2
    costheta.clip(-1.0, 1.0, out=costheta)

    theta = np.arccos(costheta)

    np.rad2deg(theta, theta)

    if is_scalar1 and is_scalar2:
        theta = theta[0]

    return theta


def get_posangle_eq(ra_cen, dec_cen, ra, dec):
    """
    get the position angle at which each point lies as defined by the center
    point

    parameters
    ----------
    ra_cen: scalar
        right ascensioni of center in degrees
    dec_cen: scalar
        declination of center in degrees
    ra: scalar or array
        right ascension point(s) around the center
    dec: scalar or array
        declination point(s) around the center

    returns
    -------
    posangle: scalar or array
        angle in degrees
    """

    is_scalar = np.isscalar(ra)

    ra = np.array(ra, dtype='f8', ndmin=1, copy=False)
    dec = np.array(dec, dtype='f8', ndmin=1, copy=False)

    if ra.size != dec.size:
        raise ValueError("ra,dec should be same size, "
                         "got %s,%s" % (ra.size, dec.size))

    posangle = np.zeros(ra.size, dtype='f8')

    _healpix._fill_posangle_eq(ra_cen, dec_cen, ra, dec, posangle)

    if is_scalar:
        posangle = posangle[0]

    return posangle


def get_quadrant_eq(ra_cen, dec_cen, ra, dec, more=False):
    """
    get the quadrant at which each point lies as defined by the center
    point.  Optionally get radius and position angle

    parameters
    ----------
    ra_cen: scalar
        right ascensioni of center in degrees
    dec_cen: scalar
        declination of center in degrees
    ra: scalar or array
        right ascension point(s) around the center
    dec: scalar or array
        declination point(s) around the center

    returns
    -------
    if more==False:
        quadrant: scalar or array
            quadrant 1,2,3,4

    if more==True
        quadrant, r, posangle:
            with r,posangle in degrees
    """

    is_scalar = np.isscalar(ra)

    ra = np.array(ra, dtype='f8', ndmin=1, copy=False)
    dec = np.array(dec, dtype='f8', ndmin=1, copy=False)

    if ra.size != dec.size:
        raise ValueError("ra,dec should be same size, "
                         "got %s,%s" % (ra.size, dec.size))

    quadrant = np.zeros(ra.size, dtype='i4')

    if not more:
        _healpix._fill_quadrant_eq(ra_cen, dec_cen, ra, dec, quadrant)
        if is_scalar:
            quadrant = quadrant[0]
        return quadrant
    else:
        r = np.zeros(ra.size, dtype='f8')
        posangle = np.zeros(ra.size, dtype='f8')
        _healpix._fill_quad_info_eq(
            ra_cen, dec_cen, ra, dec, quadrant, r, posangle
        )
        if is_scalar:
            quadrant = quadrant[0]
            r = r[0]
            posangle = posangle[0]

        return quadrant, r, posangle


def get_quad_ellip_eq(ra_cen, dec_cen, ra, dec, n_min=10, weight=None):
    """
    get the weighted ellipticity of points around the centreal position in
    2*theta space for pairs of quadrants.

    if the number of points in a set of quadrants is less than n_min, the
    ellipticity is set to 1.0

    parameters
    ----------
    ra_cen: scalar
        right ascensioni of center in degrees
    dec_cen: scalar
        declination of center in degrees
    ra: array
        right ascension point(s) around the center
    dec: array
        declination point(s) around the center
    n_min: scalar
        minimum required number in each pair of quadrants to perform a
        measurement.  default is 10
    weights: array, optional
        weight for each point, default equal weights

    returns
    -------
    ellip: array
        ellipticity for each of the 4 quadrant pairs
            1-2, 2-3, 3-4, 4-1
    """
    quadrant, r, pa = get_quadrant_eq(ra_cen, dec_cen,
                                      ra, dec, more=True)

    if weight is None:
        weight = np.ones(ra.size)

    # work in 2*theta space
    np.deg2rad(pa, pa)
    pa *= 2.

    x2theta = r*np.cos(pa)
    y2theta = r*np.sin(pa)

    ellip = np.ones(4)

    for i in range(4):
        quad1 = i+1
        quad2 = quad1 + 1
        if quad2 > 4:
            quad2 = 1

        w, = np.where((quadrant == quad1) | (quadrant == quad2))
        n = w.size
        if n > 4:
            ww = weight[w]
            wsum = ww.sum()
            if wsum > 0.:
                wsum_inv = 1.0/wsum
                xx = x2theta[w]
                yy = y2theta[w]

                xmean = (xx*ww).sum() * wsum_inv
                ymean = (yy*ww).sum() * wsum_inv

                xmod = xx-xmean
                ymod = yy-ymean
                xxvar = (xmod*xmod*ww).sum() * wsum_inv
                xyvar = (xmod*ymod*ww).sum() * wsum_inv
                yyvar = (ymod*ymod*ww).sum() * wsum_inv

                T = xxvar + yyvar
                if T > 0.0:
                    Tinv = 1.0/T
                    e1 = (xxvar-yyvar)*Tinv
                    e2 = 2*xyvar*Tinv
                    e = np.sqrt(e1**2 + e2**2)
                    ellip[i] = e

    return ellip


def randsphere(num, system='eq', rng=None, **kw):
    """
    Generate random points on the sphere

    parameters
    ----------
    num: integer
        The number of randoms to generate
    system: string
        'eq' for equatorial ra,dec in degrees
            ra  in [0,360]
            dec in [-90,90]
        'ang' for angular theta,phi in radians
            theta in [0,pi]
            phi   in [0,2*pi]
        'vec' for x,y,z on the unit sphere
    rng: numpy.random.RandomState, optional
        The random number generator to use.  If not sent, one
        is created
    **kw:
        ra_range=, dec_range= to limit range of randoms
            in ra,dec system
        theta_range=, phi_range= to limit range of randoms
            in theta,phi system

    output
    ------
    ra,dec: tuple of arrays
        the random points
    """

    if system == 'eq':
        return randsphere_eq(num, rng=rng, **kw)
    elif system == 'ang':
        return randsphere_ang(num, rng=rng, **kw)
    elif system == 'vec':
        import healpy
        theta, phi = randsphere_ang(num, rng=rng)
        return healpy.ang2vec(theta, phi)
    else:
        raise ValueError("bad system: '%s'" % system)


def randsphere_eq(num, ra_range=None, dec_range=None, rng=None):
    """
    Generate random equatorial ra,dec points on the sphere

    parameters
    ----------
    num: integer
        The number of randoms to generate
    ra_range: 2-element sequence
        [min,max] range in which to generate ra
    dec_range: 2-element sequence
        [min,max] range in which to generate dec

    output
    ------
    ra,dec: tuple of arrays
        the random points
    """

    if rng is None:
        rng = np.random.RandomState()

    ra_range = _check_range(ra_range, [0.0, 360.0])
    dec_range = _check_range(dec_range, [-90.0, 90.0])

    ra = rng.uniform(low=ra_range[0], high=ra_range[1], size=num)

    cosdec_min = np.cos(np.deg2rad(90.0+dec_range[0]))
    cosdec_max = np.cos(np.deg2rad(90.0+dec_range[1]))

    v = rng.uniform(low=cosdec_min, high=cosdec_max, size=num)

    v.clip(min=-1.0, max=1.0, out=v)

    # Now this generates on [0,pi)
    dec = np.arccos(v)

    # convert to degrees
    np.rad2deg(dec, dec)

    # now in range [-90,90.0)
    dec -= 90.0

    return ra, dec


def randsphere_ang(num, theta_range=None, phi_range=None, rng=None):
    """
    Generate random angular theta,phi points on the sphere

    parameters
    ----------
    num: integer
        The number of randoms to generate
    theta_range: 2-element sequence
        [min,max] range in which to generate theta
    phi_range: 2-element sequence
        [min,max] range in which to generate phi
    rng: numpy.random.RandomState, optional
        The random number generator to use.  If not sent, one
        is created

    output
    ------
    theta,phi: tuple of arrays
        the random points

        theta in [0,pi]
        phi   in [0,2*pi]
    """

    if rng is None:
        rng = np.random.RandomState()

    theta_range = _check_range(theta_range, [0.0, np.pi])
    phi_range = _check_range(phi_range, [0, 2.*np.pi])

    phi = rng.uniform(low=phi_range[0], high=phi_range[1], size=num)

    cos_theta_min = np.cos(theta_range[0])
    cos_theta_max = np.cos(theta_range[1])

    v = rng.uniform(low=cos_theta_min, high=cos_theta_max, size=num)

    v.clip(min=-1.0, max=1.0, out=v)

    # Now this generates on [0,pi)
    theta = np.arccos(v)

    return theta, phi


def _check_range(rng, allowed):
    if rng is None:
        rng = allowed
    else:
        if not hasattr(rng, '__len__'):
            raise ValueError("range object does not have len() method")

        if rng[0] <= allowed[0] or rng[1] >= allowed[1]:
            raise ValueError("lon_range should be within [%s,%s]" % allowed)
    return rng


def randcap(nrand, coord1, coord2, radius, system='eq',
            rng=None, get_radius=False):
    """
    Generate random equatorial ra,ec points in a sherical cap

    parameters
    ----------
    nrand:
        The number of random points
    coord1: scalar
        If system=='eq' this is ra in degrees
        If system=='ang' this is theta in radians
    coord2: scalar
        If system=='eq' this is dec in degrees
        If system=='ang' this is dec in radians
    radius:
        radius of disc
        If system=='eq' this is in degrees
        If system=='ang' this is in radians

    system: string
        'eq' for equatorial ra,dec in degrees
        'ang' for angular theta,phi in radians

    rng: numpy.random.RandomState, optional
        The random number generator to use.  If not sent, one
        is created

    get_radius: bool, optional
        if true, return radius of each point in radians
    """

    if system == 'eq':
        res = randcap_eq(
            nrand, coord1, coord2, radius, rng=rng, get_radius=get_radius,
        )
    elif system == 'ang':
        ra, dec = ang2eq(coord1, coord2)
        radius_degrees = np.rad2deg(radius)
        res = randcap_eq(
            nrand, ra, dec, radius_degrees, rng=rng, get_radius=get_radius,
        )

        if get_radius:
            rarand, decrand, radout_degrees = res
            radout_radians = np.deg2rad(radout_degrees)
        else:
            rarand, decrand = res

        theta, phi = eq2ang(rarand, decrand)

        if get_radius:
            res = theta, phi, radout_radians
        else:
            res = theta, phi
    else:
        raise ValueError("bad system: '%s'" % system)

    return res


def randcap_eq(nrand, ra, dec, rad, rng=None, get_radius=False):
    """
    Generate random equatorial ra,ec points in a sherical cap

    parameters
    ----------
    nrand:
        The number of random points
    ra,dec:
        The center of the cap in degrees.  The ra should be within [0,360) and
        dec from [-90,90]
    rad:
        radius of the cap, degrees

    get_radius: bool, optional
        if true, return radius of each point in radians
    """

    if rng is None:
        rng = np.random.RandomState()

    # generate uniformly in r**2
    rand_r = rng.uniform(size=nrand)
    rand_r = np.sqrt(rand_r)*rad

    # put in degrees
    np.deg2rad(rand_r, rand_r)

    # generate position angle uniformly 0,2*pi
    rand_posangle = rng.uniform(low=0.0, high=2*np.pi, size=nrand)

    theta = np.array(dec, dtype='f8', ndmin=1, copy=True)
    phi = np.array(ra, dtype='f8', ndmin=1, copy=True)
    theta += 90

    np.deg2rad(theta, theta)
    np.deg2rad(phi, phi)

    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    # sinphi = sin(phi)
    # cosphi = cos(phi)

    sinr = np.sin(rand_r)
    cosr = np.cos(rand_r)

    cospsi = np.cos(rand_posangle)
    costheta2 = costheta*cosr + sintheta*sinr*cospsi

    np.clip(costheta2, -1, 1, costheta2)

    # gives [0,pi)
    theta2 = np.arccos(costheta2)
    sintheta2 = np.sin(theta2)

    cosDphi = (cosr - costheta*costheta2)/(sintheta*sintheta2)

    np.clip(cosDphi, -1, 1, cosDphi)
    Dphi = np.arccos(cosDphi)

    # note fancy usage of where
    phi2 = np.where(rand_posangle > np.pi, phi+Dphi, phi-Dphi)

    np.rad2deg(phi2, phi2)
    np.rad2deg(theta2, theta2)
    rand_ra = phi2
    rand_dec = theta2-90.0

    if get_radius:
        return rand_ra, rand_dec, rand_r
    else:
        return rand_ra, rand_dec


class Points(object):
    """
    A wrapper class for coordinates in various systems

    parameters
    ----------
    There are three conventions

    Points(ra=ra,dec=dec)
        ra,dec in degrees
    Points(theta=theta,phi=phi0
        theta,phi in radians

    example
    -------
    pts=Points(ra=ra, dec=dec)

    # access as attributes or by name
    pts.ra[35]
    pts.dec[25:88]
    # theta and phi would be generated the first time you try to access them
    pts.theta[55]
    pts.phi[ [3,4,5 ]]

    pts['ra'][35]
    pts['dec'][25:88]
    pts['theta'][55]
    pts['phi'][ [3,4,5 ]]
    """
    def __init__(self,
                 ra=None,
                 dec=None,
                 theta=None,
                 phi=None):

        self.clear()

        if ra is not None and dec is not None:
            self.set_eq(ra, dec)
        elif theta is not None and phi is not None:
            self.set_ang(theta, phi)

    def clear(self):
        """
        set all coords to None
        """
        self._ra = None
        self._dec = None
        self._theta = None
        self._phi = None

    def set_eq(self, ra, dec):
        """
        set equatorial ra,dec in degrees

        parameters
        ----------
        ra: array or scalar
            right ascension in degrees
        dec: array or scalar
            declination in degrees
        """

        self.clear()
        self._ra = np.array(ra,  dtype='f8', ndmin=1, copy=False)
        self._dec = np.array(dec, dtype='f8', ndmin=1, copy=False)
        self._defsystem = "eq"

    def set_ang(self, theta, phi):
        """
        set angular theta,dec in radians

        parameters
        ----------
        theta: array or scalar
            angular theta in radians
        phi: array or scalar
            angular phi in radians
        """

        self.clear()
        self._theta = np.array(theta, dtype='f8', ndmin=1, copy=False)
        self._phi = np.array(phi, dtype='f8', ndmin=1, copy=False)
        self._defsystem = "ang"

    def get_ra(self):
        """
        get a reference to the ra data
        """
        if self._ra is None:
            # the ang system must have been the initial
            self._ra, self._dec = ang2eq(self._theta, self._phi)
        return self._ra

    def get_dec(self):
        """
        get a reference to the dec data
        """
        if self._dec is None:
            # the ang system must have been the initial
            self._ra, self._dec = ang2eq(self._theta, self._phi)
        return self._dec

    def get_theta(self):
        """
        get a reference to the theta data
        """
        if self._theta is None:
            # the ang system must have been the initial
            self._theta, self._phi = eq2ang(self._ra, self._dec)
        return self._theta

    def get_phi(self):
        """
        get a reference to the phi data
        """
        if self._phi is None:
            # the ang system must have been the initial
            self._theta, self._phi = eq2ang(self._ra, self._dec)
        return self._phi

    ra = property(get_ra)
    dec = property(get_dec)
    theta = property(get_theta)
    phi = property(get_phi)

    def __getitem__(self, arg):
        if arg == 'ra':
            return self.get_ra()
        elif arg == 'dec':
            return self.get_dec()
        elif arg == 'theta':
            return self.get_theta()
        elif arg == 'phi':
            return self.get_phi()
        else:
            raise IndexError("no such item '%s'" % arg)
