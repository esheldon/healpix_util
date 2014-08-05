"""
functions
---------
read_map:
    Read a healpix map into a Map object
read_maps:
    Read multiple healpix maps
read_density_map:
    Read a healpix density map into a DensityMap object
read_density_maps:
    Read multiple healpix density maps
"""
from __future__ import print_function
import numpy
from .healpix import HealPix, Map, DensityMap, get_scheme_string

def read_map(filename, column=0, **keys):
    """
    read a healpix map from the specified file

    to read multiple maps, use read_maps
    to read a density map(s), use read_density_map/read_density_maps

    parameters
    ----------
    filename: string
        The fits filename
    column: string or int, optional
        The column to read, default is the first.
    scheme: string or int, optional
        Optional scheme specification.  If the scheme is not specified
        in the header as 'ORDERING', then you can specify it with
        this keyword.

        Also if the specified scheme does not match the ORDERING in the
        header, the maps will be converted to the requested scheme.

    **keys:
        other keywords for the fits reading, such as 
            ext= (default 1)
            columns= (default is to read all columns)
            header=True to return the header
        See the fitsio documentation for more details

    returns
    -------
    A Map object representing the healpix map.

    if header=True is specified, a tuple (maps, header) is returned
    """
    import fitsio

    scheme = keys.get("scheme",None)
    if scheme is not None:
        scheme=get_scheme_string(scheme)

    if not numpy.isscalar(column):
        column=column[0]

    with fitsio.FITS(filename) as fits:

        ext=keys.get('ext',1)
        hdu = fits[ext]
        if not isinstance(hdu,fitsio.fitslib.TableHDU):
            raise ValueError("extension %s is not a table" % ext)

        header = hdu.read_header()

        # user may specify columns= here
        data = hdu.read_column(column,**keys)

        if 'ordering' in header:
            scheme_in_file = header['ordering'].strip()
            scheme_in_file = get_scheme_string(scheme_in_file)
        else:
            # we need input from the user
            if scheme is None:
                raise ValueError("ORDERING not in header, send scheme= "
                                 "to specify")
            scheme_in_file = scheme

        if scheme is not None and scheme != scheme_in_file:
            print("converting from scheme '%s' "
                  "to '%s'" % (scheme_in_file,scheme))
            do_convert=True
        else:
            do_convert=False

        hmap = Map(scheme_in_file,data)
        if do_convert:
            hmap=hmap.convert(scheme)

    gethead=keys.get("header",False)
    if gethead:
        return hmap, header
    else:
        return hmap

def read_maps(filename, **keys):
    """
    read healpix map(s) from the specified file

    parameters
    ----------
    filename: string
        The fits filename
    scheme: string or int
        Optional scheme specification.  If the scheme is not specified
        in the header as 'ORDERING', then you can specify it with
        this keyword.

        Also if the specified scheme does not match the ORDERING in the
        header, the maps will be converted to the requested scheme.

    **keys:
        other keywords for the fits reading, such as 
            ext= (default 1)
            columns= (default is to read all columns)
            header=True to return the header
        See the fitsio documentation for more details

    returns
    -------
    a dict of healpix.Map is returned, keyed by the column names.
    if header=True is specified, a tuple (maps, header) is returned
    """
    import fitsio

    scheme = keys.get("scheme",None)
    if scheme is not None:
        scheme=get_scheme_string(scheme)

    # make sure columns is a sequence to ensure we get
    # an array with fields back
    columns=keys.get('columns',None)
    if columns is not None:
        if numpy.isscalar(columns):
            keys['columns'] = [columns]

    with fitsio.FITS(filename) as fits:

        ext=keys.get('ext',1)
        hdu = fits[ext]
        if not isinstance(hdu,fitsio.fitslib.TableHDU):
            raise ValueError("extension %s is not a table" % ext)

        header = hdu.read_header()

        data = hdu.read(**keys)

        if 'ordering' in header:
            scheme_in_file = header['ordering'].strip()
            scheme_in_file = get_scheme_string(scheme_in_file)
        else:
            # we need input from the user
            if scheme is None:
                raise ValueError("ORDERING not in header, send scheme= "
                                 "to specify")
            scheme_in_file = scheme

        if scheme is not None and scheme != scheme_in_file:
            print("converting from scheme '%s' "
                  "to '%s'" % (scheme_in_file,scheme))
            do_convert=True
        else:
            do_convert=False

        names=data.dtype.names
        # there were multiple columns read
        map_dict={}
        for name in names:
            m = Map(scheme_in_file,data[name])
            if do_convert:
                m=m.convert(scheme)
            map_dict[name] = m

    gethead=keys.get("header",False)
    if gethead:
        return map_dict, header
    else:
        return map_dict

def read_density_map(filename, column=0, **keys):
    """
    read a density healpix map from the specified file

    The difference between a DensityMap and Map is a density map represents a
    density, possibly arbitrarily scaled.  A normal map can represent any field
    on the sky.

    parameters
    ----------
    filename: string
        The fits filename
    column: string or int, optional
        The column to read, default is the first.
    scheme: string or int, optional
        Optional scheme specification.  If the scheme is not specified
        in the header as 'ORDERING', then you can specify it with
        this keyword.

        Also if the specified scheme does not match the ORDERING in the
        header, the maps will be converted to the requested scheme.

    **keys:
        other keywords for the fits reading, such as 
            ext= (default 1)
            columns= (default is to read all columns)
            header=True to return the header
        See the fitsio documentation for more details

    returns
    -------
    A DensityMap object representing the healpix map.

    if header=True is specified, a tuple (map, header) is returned
    """
    res=read_map(filename, column=column, **keys)
    if isinstance(res,tuple):
        hmap,header=res
    else:
        hmap=res

    density_hmap = DensityMap(hmap.scheme, hmap.data)
    if isinstance(res,tuple):
        return density_hmap, header
    else:
        return density_hmap

def read_density_maps(filename, **keys):
    """
    read multiple density healpix maps from the specified file

    The difference between a DensityMap and Map is a density map represents a
    density, possibly arbitrarily scaled.  A normal map can represent any field
    on the sky.

    parameters
    ----------
    filename: string
        The fits filename
    scheme: string or int, optional
        Optional scheme specification.  If the scheme is not specified
        in the header as 'ORDERING', then you can specify it with
        this keyword.

        Also if the specified scheme does not match the ORDERING in the
        header, the maps will be converted to the requested scheme.

    **keys:
        other keywords for the fits reading, such as 
            ext= (default 1)
            columns= (default is to read all columns)
            header=True to return the header
        See the fitsio documentation for more details

    returns
    -------
    A dict of DensityMap keyed by the column names.

    if header=True is specified, a tuple (maps, header) is returned
    """
    res=read_maps(filename, column=column, **keys)
    if isinstance(res,tuple):
        map_dict,header=res
    else:
        map_dict=res

    density_map_dict={}
    for name,hmap in map_dict.iteritems():
        density_map_dict[name] = DensityMap(hmap.scheme, hmap.data)

    if isinstance(res,tuple):
        return density_map_dict, header
    else:
        return density_map_dict
