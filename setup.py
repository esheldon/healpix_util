from setuptools import setup, find_packages, Extension
import numpy

ext = Extension(
    "healpix_util._healpix",
    ["healpix_util/_healpix.c"],
    include_dirs=[numpy.get_include()],
)

exec(open('healpix_util/version.py').read())

setup(
    name="healpix_util",
    license="GNU GPLv3",
    author="Erin Scott Sheldon",
    author_email="erin.sheldon@gmail.com",
    packages=find_packages(),
    version=__version__,
    ext_modules=[ext],
)
