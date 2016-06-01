import distutils
from distutils.core import setup, Extension, Command
import os
import numpy

ext=Extension(
    "healpix_util._healpix",
    ["healpix_util/_healpix.c"],
    include_dirs=[numpy.get_include()],
)

exec(open('healpix_util/version.py').read())

setup(
    name="healpix_util", 
    packages=['healpix_util'],
    version=__version__,
    ext_modules=[ext],
)


