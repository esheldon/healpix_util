import distutils
from distutils.core import setup, Extension, Command
import os
import numpy

ext=Extension("healpix._healpix", ["healpix/_healpix.c"])

exec(open('healpix/version.py').read())

setup(name="healpix", 
      packages=['healpix'],
      version=__version__,
      ext_modules=[ext],
      include_dirs=numpy.get_include())


