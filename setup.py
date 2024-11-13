from setuptools import setup, Extension
import numpy as np

ext = Extension(
    "healpix_util._healpix",
    ["healpix_util/_healpix.c"],
    include_dirs=[np.get_include()],
)

__version__ = None
exec(open('healpix_util/version.py').read())

setup(
    name="healpix_util",
    packages=['healpix_util'],
    version=__version__,
    ext_modules=[ext],
)
