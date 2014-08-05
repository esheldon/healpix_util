"""
functions
---------
eq2ang:
    convert equatorial ra,dec in degrees to angular theta, phi in radians
ang2eq:
    convert angular theta, phi in radians to equatorial ra,dec in degrees
eq2xyz:
    Convert equatorial ra,dec in degrees to x,y,z on the unit sphere
ang2xyz:
    Convert angular theta,phi in radians to x,y,z on the unit sphere
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

import numpy
from . import _healpix

SYSTEM_ANG=1
SYSTEM_EQ=2

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
    theta: scalar or array
        pi/2-dec*D2R # in [0,pi]
    phi: scalar or array
        phi = ra*D2R # in [0,2*pi]
    """
    theta = 0.5*numpy.pi - numpy.deg2rad(dec)
    phi = numpy.deg2rad(ra)

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
    ra: scalar or array
        phi*R2D          # in [0,360]
    dec: scalar or array
        (pi/2-theta)*R2D # in [-90,90]
    """

    ra = numpy.rad2deg(phi)
    dec = numpy.rad2deg(numpy.pi*0.5 - theta)

    return ra,dec


def eq2xyz(ra, dec):
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
    x,y,z on the unit sphere
    """
    theta, phi = eq2ang(ra,dec)
    return ang2xyz(theta, phi)

def ang2xyz(theta, phi):
    """
    Convert angular theta,phi in radians to x,y,z on the unit sphere

    parameters
    ----------
    theta: scalar or array
        theta in radians
    phi: scalar or array
        phi in radians
    
    returns
    -------
    x,y,z on the unit sphere
    """
    sintheta = numpy.sin(theta)
    x = sintheta * numpy.cos(phi)
    y = sintheta * numpy.sin(phi)

    z = numpy.cos(theta)

    return x,y,z


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

    is_scalar=numpy.isscalar(ra)

    ra=numpy.array(ra, dtype='f8', ndmin=1, copy=False)
    dec=numpy.array(dec, dtype='f8', ndmin=1, copy=False)

    if ra.size != dec.size:
        raise ValueError("ra,dec should be same size, "
                         "got %s,%s" % (ra.size,dec.size))

    posangle=numpy.zeros(ra.size, dtype='f8')

    _healpix._fill_posangle_eq(ra_cen, dec_cen, ra, dec, posangle)

    if is_scalar:
        posangle=posangle[0]

    return posangle

def get_quadrant_eq(ra_cen, dec_cen, ra, dec):
    """
    get the quadrant at which each point lies as defined by the center
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
    quadrant: scalar or array
        quadrant 1,2,3,4
    """

    is_scalar=numpy.isscalar(ra)

    ra=numpy.array(ra, dtype='f8', ndmin=1, copy=False)
    dec=numpy.array(dec, dtype='f8', ndmin=1, copy=False)

    if ra.size != dec.size:
        raise ValueError("ra,dec should be same size, "
                         "got %s,%s" % (ra.size,dec.size))

    quadrant=numpy.zeros(ra.size, dtype='i4')

    _healpix._fill_quadrant_eq(ra_cen, dec_cen, ra, dec, quadrant)

    if is_scalar:
        quadrant=quadrant[0]

    return quadrant


def randsphere(num, system='eq'):
    """
    Generate random points on the sphere

    parameters
    ----------
    num: integer 
        The number of randoms to generate
    system: string
        'eq' for equatorial ra,dec in degrees
        'ang' for angular theta,phi in radians
        'xyz' for x,y,z on the unit sphere
    output
    ------
    ra,dec: tuple of arrays
        the random points
    """

    if system=='eq':
        return randsphere_eq(num)
    elif system=='ang':
        return randsphere_ang(num)
    elif system=='xyz':
        theta,phi=randsphere_ang(num)
        return ang2xyz(theta,phi)
    else:
        raise ValueError("bad system: '%s'" % sytem)

def randsphere_eq(num):
    """
    Generate random equatorial ra,dec points on the sphere

    parameters
    ----------
    num: integer 
        The number of randoms to generate

    output
    ------
    ra,dec: tuple of arrays
        the random points
    """

    ra = numpy.random.random(num)
    ra *= 360.0

    # number [0,1)
    v = numpy.random.random(num)
    # [0,2)
    v *= 2.0
    # [-1,1)
    v -= 1.0

    # Now this generates on [0,pi)
    dec = numpy.arccos(v)

    # convert to degrees
    numpy.rad2deg(dec,dec)
    # now in range [-90,90.0)
    dec -= 90.0
    
    return ra, dec

def randsphere_ang(num, system='eq'):
    """
    Generate random angular theta,phi points on the sphere


    parameters
    ----------
    num: integer 
        The number of randoms to generate

    output
    ------
    theta,phi: tuple of arrays
        the random points

        theta in [0,pi]
        phi   in [0,2*pi]
    """

    phi = numpy.random.random(num)
    phi *= 2.0*numpy.pi

    # number [0,1)
    v = numpy.random.random(num)
    # [0,2)
    v *= 2.0
    # [-1,1)
    v -= 1.0

    # Now this generates on [0,pi)
    theta = numpy.arccos(v)

    return theta, phi

def randcap(nrand, coord1, coord2, radius, system='eq', get_radius=False):
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

    get_radius: bool, optional
        if true, return radius of each point in radians
    """
    if system=='eq':
        res=randcap_eq(nrand, coord1, coord2, radius, get_radius=get_radius)
    elif system=='ang':
        ra,dec=ang2eq(coord1,coord2)
        radius_degrees=numpy.rad2deg(radius)
        res=randcap_eq(nrand, ra, dec, radius_degrees, get_radius=get_radius)

        if get_radius:
            rarand,decrand,radout_degrees=res
            radout_radians=numpy.deg2rad(radout_degrees)
        else:
            rarand,decrand=res

        theta,phi=eq2ang(rarand,decrand)

        if get_radius:
            res=theta,phi,radout_radians
        else:
            res=theta,phi
    else:
        raise ValueError("bad system: '%s'" % system)

    return res

def randcap_eq(nrand, ra, dec, rad, get_radius=False):
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
    from numpy import sqrt, sin, cos, arccos, pi

    # generate uniformly in r**2
    rand_r = numpy.random.random(nrand)
    rand_r = sqrt(rand_r)*rad

    # put in degrees
    numpy.deg2rad(rand_r,rand_r)

    # generate position angle uniformly 0,2*pi
    rand_posangle = numpy.random.random(nrand)*2*pi

    theta = numpy.array(dec, dtype='f8',ndmin=1,copy=True)
    phi = numpy.array(ra,dtype='f8',ndmin=1,copy=True)
    theta += 90

    numpy.deg2rad(theta,theta)
    numpy.deg2rad(phi,phi)

    sintheta = sin(theta)
    costheta = cos(theta)
    sinphi = sin(phi)
    cosphi = cos(phi)

    sinr = sin(rand_r)
    cosr = cos(rand_r)

    cospsi = cos(rand_posangle)
    costheta2 = costheta*cosr + sintheta*sinr*cospsi

    numpy.clip(costheta2, -1, 1, costheta2)                    

    # gives [0,pi)
    theta2 = arccos(costheta2)
    sintheta2 = sin(theta2)

    cosDphi = (cosr - costheta*costheta2)/(sintheta*sintheta2)

    numpy.clip(cosDphi, -1, 1, cosDphi)                    
    Dphi = arccos(cosDphi)

    # note fancy usage of where
    phi2=numpy.where(rand_posangle > pi, phi+Dphi, phi-Dphi)

    numpy.rad2deg(phi2,phi2)
    numpy.rad2deg(theta2,theta2)
    rand_ra  = phi2
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
            self.set_eq(ra,dec)
        elif theta is not None and phi is not None:
            self.set_ang(theta,phi)

    def clear(self):
        """
        set all coords to None
        """
        self._ra=None
        self._dec=None
        self._theta=None
        self._phi=None

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
        self._ra  = numpy.array(ra,  dtype='f8', ndmin=1, copy=False)
        self._dec = numpy.array(dec, dtype='f8', ndmin=1, copy=False)
        self._defsystem="eq"

    def set_ang(self, theta, dec):
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
        self._theta = numpy.array(theta, dtype='f8',ndmin=1, copy=False)
        self._phi   = numpy.array(phi,   dtype='f8',ndmin=1, copy=False)
        self._defsystem="ang"

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
            self._theta, self._phi= eq2ang(self._ra, self._dec)
        return self._theta

    def get_phi(self):
        """
        get a reference to the phi data
        """
        if self._phi is None:
            # the ang system must have been the initial
            self._theta, self._phi= eq2ang(self._ra, self._dec)
        return self._phi

    ra = property(get_ra)
    dec = property(get_dec)
    theta = property(get_theta)
    phi = property(get_phi)

    def __getitem__(self, arg):
        if arg=='ra':
            return self.get_ra()
        elif arg=='dec':
            return self.get_dec()
        elif arg=='theta':
            return self.get_theta()
        elif arg=='phi':
            return self.get_phi()
        else:
            raise IndexError("no such item '%s'" % arg)
