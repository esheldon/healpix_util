"""
functions
---------
these read and write functions are named load_map etc. to avoid
conflice with the healpy read_map etc. functions

load_map:
    Read a healpix map into a Map object
load_maps:
    Read multiple healpix maps
load_density_map:
    Read a healpix density map into a DensityMap object
load_density_maps:
    Read multiple healpix density maps
"""
from __future__ import print_function
import numpy
from .healpix import HealPix, get_scheme_name
from .maps import Map, DensityMap

def load_map(filename, column=0, **kw):
    """
    read a healpix map from the specified file

    to read multiple maps, use load_maps
    to read a density map(s), use load_density_map/load_density_map

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

    **kw:
        other keywords for the fits reading, such as 
            ext= (default 1)
            header=True to return the header
        See the fitsio documentation for more details

    returns
    -------
    A Map object representing the healpix map.

    if header=True is specified, a tuple (maps, header) is returned
    """
    import fitsio

    scheme = kw.get("scheme",None)
    if scheme is not None:
        scheme=get_scheme_name(scheme)

    if not numpy.isscalar(column):
        column=column[0]

    with fitsio.FITS(filename) as fits:

        ext=kw.get('ext',1)
        hdu = fits[ext]
        if not isinstance(hdu,fitsio.fitslib.TableHDU):
            raise ValueError("extension %s is not a table" % ext)

        hdr = hdu.read_header()

        # user may specify columns= here
        data = hdu.read_column(column,**kw)

    if 'ordering' in hdr:
        scheme_in_file = get_scheme_name(hdr['ordering'].strip())
    else:
        # we need input from the user
        if scheme is None:
            raise ValueError("ORDERING not in header, send scheme= "
                             "to specify")
        scheme_in_file = scheme

    hmap = Map(scheme_in_file,data)
    if scheme is not None and scheme != scheme_in_file:
        print("converting from scheme '%s' "
              "to '%s'" % (scheme_in_file,scheme))
        hmap=hmap.convert(scheme)

    gethead=kw.get("header",False)
    if gethead:
        return hmap, hdr
    else:
        return hmap

def load_maps(filename, **kw):
    """
    read healpix map(s) from the specified file

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

    **kw:
        other keywords for the fits reading, such as 
            ext= (default 1)
            columns= (default is to read all columns)
            header=True to return the header
        See the fitsio documentation for more details

    returns
    -------
    An ordered dict of healpix.Map is returned, keyed by the column names.  if
    header=True is specified, a tuple (maps, header) is returned
    """
    from collections import OrderedDict
    import fitsio

    scheme = kw.get("scheme",None)
    if scheme is not None:
        scheme=get_scheme_name(scheme)

    # make sure columns is a sequence to ensure we get
    # an array with fields back
    columns=kw.get('columns',None)
    if columns is not None:
        if numpy.isscalar(columns):
            kw['columns'] = [columns]

    with fitsio.FITS(filename) as fits:

        ext=kw.get('ext',1)
        hdu = fits[ext]
        if not isinstance(hdu,fitsio.fitslib.TableHDU):
            raise ValueError("extension %s is not a table" % ext)

        hdr = hdu.read_header()
        data = hdu.read(**kw)

    if 'ordering' in hdr:
        scheme_in_file = get_scheme_name(hdr['ordering'].strip())
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
    map_dict=OrderedDict()
    for name in names:
        m = Map(scheme_in_file,data[name])
        if do_convert:
            m=m.convert(scheme)
        map_dict[name] = m

    gethead=kw.get("header",False)
    if gethead:
        return map_dict, hdr
    else:
        return map_dict

def load_density_map(filename, **kw):
    """
    read a density healpix map from the specified file

    The difference between a DensityMap and Map is a density map represents a
    density, possibly arbitrarily scaled.  The DensityMap provides additional
    methods.

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

    **kw:
        other keywords for the fits reading, such as 
            ext= (default 1)
            header=True to return the header
        See the fitsio documentation for more details

    returns
    -------
    A DensityMap object representing the healpix map.

    if header=True is specified, a tuple (map, header) is returned
    """
    res=load_map(filename, **kw)
    if isinstance(res,tuple):
        hmap,hdr=res
    else:
        hmap=res

    density_hmap = DensityMap(hmap.scheme, hmap.data)
    if isinstance(res,tuple):
        return density_hmap, hdr
    else:
        return density_hmap

def load_density_maps(filename, **kw):
    """
    read multiple density healpix maps from the specified file

    The difference between a DensityMap and Map is a density map represents a
    density, possibly arbitrarily scaled.  The DensityMap provides additional
    methods.

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

    **kw:
        other keywords for the fits reading, such as 
            ext= (default 1)
            columns= (default is to read all columns)
            header=True to return the header
        See the fitsio documentation for more details

    returns
    -------
    An ordered dict of DensityMap keyed by the column names.

    if header=True is specified, a tuple (maps, header) is returned
    """
    from collections import OrderedDict
    res=load_maps(filename, **kw)
    if isinstance(res,tuple):
        map_dict,hdr=res
    else:
        map_dict=res

    density_map_dict=OrderedDict()
    for name,hmap in map_dict.iteritems():
        density_map_dict[name] = DensityMap(hmap.scheme, hmap.data)

    if isinstance(res,tuple):
        return density_map_dict, hdr
    else:
        return density_map_dict

def writeMap(filename, hmap, colname='I', **kw):
    """
    write the map to the specified file name

    data are written in what has become the "standard" for
    healpix, as a table with each row holding 1024 pixels.

    parameters
    ----------
    filename: string
        where to write the data
    hmap: healpix Map
        A healpix.Map or child inheriting from Map
    colname: string, optional
        defaults to 'I'
    scheme: string or int, optional
        Force the requested scheme

    other keywords for fitsio
    -------------------------
    for example

    header: optional
        An optional header to write.  Should be a header
        type supported by fitsio
    clobber: bool
        If true, overwrite existing files, otherwise append
        a new HDU.  Default False.
    etc.
    """

    import fitsio

    scheme_to_write=kw.get('scheme',None)
    if scheme_to_write is not None:
        hmap=hmap.convert(scheme_to_write)

    npix=hmap.data.size
    if npix > 1024:
        dt=hmap.data.dtype.descr[0][1]
        view_dtype=[(colname,dt,1024)]
    else:
        view_dtype=[(colname,dt,npix)]

    dataview=hmap.data.view(view_dtype)
    with fitsio.FITS(filename,'rw',**kw) as fits:

        fits.write(dataview, **kw)

        fits[-1].write_key("ORDERING",hmap.scheme)
        fits[-1].write_key("NSIDE",hmap.nside)

def writeMaps(filename, hmap_dict,  **kw):
    """
    write multiple maps to the same file and hdu

    column names are derived from the dict keys

    for maps of different size or scheme, it is better to
    use writeMap and different hdus

    data are written in what has become the "standard" for
    healpix, as a table with each row holding 1024 pixels.

    parameters
    ----------
    filename: string
        where to write the data
    hmap_dict: ordered dict of Map
        An ordered dict holding healpix.Map or children inheriting from Map
    scheme: string or int, optional
        Specify to force all maps to this scheme.  By default all maps
        are forced to that of the first map in the dict.

    other keywords for fitsio
    -------------------------
    for example

    header: optional
        An optional header to write.  Should be a header
        type supported by fitsio
    clobber: bool
        If true, overwrite existing files, otherwise append
        a new HDU.  Default False.
    etc.
    """
    from collections import OrderedDict
    import fitsio

    keys=list(hmap_dict.keys())
    scheme=kw.get('scheme',None)
    if scheme is None:
        scheme=hmap_dict[keys[0]].scheme
    else:
        scheme=get_scheme_name(scheme)

    npix=hmap_dict[keys[0]].data.size

    if npix > 1024:
        nrows=npix/1024
    else:
        nrows=1

    use_dict=OrderedDict()
    for key,hmap in hmap_dict.iteritems():

        if hmap.data.size != npix:
            raise ValueError("not all maps are the same size")

        if hmap.scheme != scheme:
            use_dict[key]=hmap.convert(scheme)
        else:
            use_dict[key]=hmap

    dtype=[]
    for key,hmap in use_dict.iteritems():
        datatype=hmap.data.dtype.descr[0][1]
        if npix > 1024:
            dt=(key, datatype, 1024)
        else:
            dt=(key, datatype, npix)

        dtype.append( dt )

    output=numpy.zeros(nrows, dtype=dtype)

    for key,hmap in use_dict.iteritems():
        output[key] = hmap.data.reshape(output[key].shape)

    with fitsio.FITS(filename,'rw',**kw) as fits:

        fits.write(output, **kw)

        fits[-1].write_key("ORDERING",scheme)
        fits[-1].write_key("NSIDE",hmap.nside)

