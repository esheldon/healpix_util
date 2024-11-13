healpix
=======

Some tools to work with healpix.

Note reader utilities use camelCase to avoid name collision
with the healpy readers.

a few examples
==============

```python
import healpix_util as hu

# create a HealPix object.  This will carry around the nside, scheme, npix,
# etc.

hpix=hu.HealPix("ring", 4096)

print(hpix)
scheme:       1
scheme_name:  RING
nside:        4096
npix:         201326592
ncap:         33546240
area:         6.24178e-08 square degrees
area:         0.000224704 square arcmin
area:         0.808935 square arcsec

# has many attributes, e.g. .scheme, .nside, .npix, .area, etc.

# find pixel number for input ra,dec arrays
pixnums = hpix.eq2pix(ra, dec)
ra, dec = hpix.pix2eq(pixnums)

# or for theta,phi
pixnums = hpix.ang2pix(theta, phi)
theta, phi = hpix.pix2ang(pixnums)

# query disc 
pixnums = hpix.query_disc(ra, dec, radius_degrees)
pixnums = hpix.query_disc(ra, dec, radius_degrees, inclusive=True)
pixnums = hpix.query_disc(theta, phi, radius_radians, system='ang')


# load a healpix.Map object
m = hu.readMap(filename)
m = hu.readMap(filename, column='I')

# load multiple Maps into an ordered dict, keyed by column name
maps = hu.readMaps(filename)
maps = hu.readMaps(filename,columns=["I","Q"])

# convert between schemes
mnest = m.convert("nest")
mring = m.convert("ring")

# find the mapval for the input coordinates
mapvals = m.get_mapval(ra, dec)
mapvals = m.get_mapval(theta, phi, system='ang')

# load a healpix.DensityMap, a special map that represents
# a spatial density
dmap = hu.readDensityMap(filename)
dmaps = hu.readDensityMaps(filename)

# generate random points from the map
ra, dec = dmap.genrand(100000)

# attach a random number generator for repeatability
# when generating random points
rng = np.random.RandomState(8312)
dmap = hu.readDensityMap(filename, rng=rng)

# limit the range to a region known to have non-zero density, or
# weight.  This will speed it up greatly
ra, dec = dmap.genrand(100000, ra_range=[60.,95.], dec_range=[-62.,-42.])

# generate randoms in theta,phi
theta, phi = dmap.genrand(100000, system='ang', theta_range=tr, phi_range=pr)

# healpy routines are pulled into the healpix_util namespace
npix = hu.nside2npix(nside)


# obscure features
# check quadrants around the input points
# make sure weighted position ellipticity in adjacent quadrants
# less than 0.05
ellip_max = 0.05
for i in xrange(ra.size):
    maskflags[i] = dmap.check_quad(ra[i], dec[i], radius_degrees[i], ellip_max)

# see docs on quad_check
# checking maskflags > 1 indicates the cluster is "good", but note during
# a lensing measurement you should be careful to always use sources from
# the same pair of quadrants (unless all are good)
```


dependencies
============
- numpy
- healpy
- fitsio # for file reading
