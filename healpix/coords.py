"""
functions
---------
eq2ang(ra, dec):
    convert equatorial ra,dec in degrees to angular theta, phi in radians
ang2eq(theta, phi):
    convert angular theta, phi in radians to equatorial ra,dec in degrees
eq2xyz(ra, dec):
    Convert equatorial ra,dec in degrees to x,y,z on the unit sphere
ang2xyz(theta, phi):
    Convert angular theta,phi in radians to x,y,z on the unit sphere

constants
---------
SYSTEM_ANG=1
    angular theta,phi system in radians
SYSTEM_EQ=2
    equatorial ra,dec system in degrees
"""
import numpy

SYSTEM_ANG=1
SYSTEM_EQ=2

def eq2ang(ra, dec):
    """
    convert equatorial ra,dec in degrees to angular theta, phi in radians

    theta = pi/2-dec*D2R
    phi = ra*D2R

    parameters
    ----------
    ra: scalar or array
        Right ascension in degrees
    dec: scalar or array
        Declination in degrees

    returns
    -------
    theta: scalar or array
        pi/2-dec*D2R
    phi: scalar or array
        phi = ra*D2R
    """
    theta = 0.5*numpy.pi - numpy.deg2rad(dec)
    phi = numpy.deg2rad(ra)

    return theta, phi

def ang2eq(theta, phi):
    """
    convert angular theta, phi in radians to equatorial ra,dec in degrees

    ra = phi*R2D
    dec = (pi/2-theta)*R2D

    parameters
    ----------
    theta: scalar or array
        angular theta in radians
    phi: scalar or array
        angular phi in radians

    returns
    -------
    ra: scalar or array
        phi*R2D
    dec: scalar or array
        (pi/2-theta)*R2D
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
