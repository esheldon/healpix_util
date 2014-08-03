#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <Python.h>
#include "numpy/arrayobject.h" 

#ifndef M_PI
# define M_PI		3.14159265358979323846	/* pi */
#endif

#define HPX_TWO_PI   6.28318530717958647693 /* 2*pi */
#define HPX_TWOTHIRD 0.66666666666666666666

#define HPX_D2R  0.017453292519943295
#define HPX_R2D  57.295779513082323


struct PyHealPix {
    PyObject_HEAD

    int64_t nside;
    int64_t npix;
    int64_t ncap;
    double area;
};

static int64_t hpix_npix(int64_t nside) {
    return 12*nside*nside;
}

static double hpix_area(int64_t nside) {
    int64_t npix = hpix_npix(nside);
    return 4.0*M_PI/npix;
}

static inline int64_t i64max(int64_t v1, int64_t v2) {
    return v1 > v2 ? v1 : v2;
}
static inline int64_t i64min(int64_t v1, int64_t v2) {
    return v1 < v2 ? v1 : v2;
}

/*
static PyObject*
make_double_array(npy_intp size, const char* name, long double** ptr)
{
    PyObject* array=NULL;
    npy_intp dims[1];
    int ndims=1;
    if (size <= 0) {
        PyErr_Format(PyExc_ValueError, "size of %s array must be > 0",name);
        return NULL;
    }

    dims[0] = size;
    array = PyArray_ZEROS(ndims, dims, NPY_DOUBLE, 0);
    if (array==NULL) {
        PyErr_Format(PyExc_MemoryError, "could not create %s array",name);
        return NULL;
    }

    *ptr = PyArray_DATA((PyArrayObject*)array);
    return array;
}
*/

/*
   convert ra,dec degrees to theta,phi radians
*/

static inline void hpix_eq2tp(double ra, double dec, double* theta, double* phi) {

    if (ra < 0.) {
        ra=0.;
    }
    if (ra > 360.) {
        ra=360.;
    }
    if (dec < -90.) {
        dec=-90.;
    }
    if (dec > 90.0) {
        dec=90.;
    }

    *phi = ra*HPX_D2R;
    *theta = M_PI_2 -dec*HPX_D2R;
}

/*
   ra,dec to standard x,y,z
*/
static inline void hpix_eq2xyz(double ra, double dec, double* x, double* y, double* z) {

    double theta=0, phi=0;

    hpix_eq2tp(ra, dec, &theta, &phi);

    double sintheta = sin(theta);
    *x = sintheta * cos(phi);
    *y = sintheta * sin(phi);
    *z = cos(theta);
}


/*
   convert equatorial coordinates to pixel number in the ring scheme
*/
static int64_t hpix_eq2pix_ring(const struct PyHealPix* hpix,
                                double ra,
                                double dec) {
    int64_t nside=hpix->nside;
    int64_t ipix=0;
    double theta=0, phi=0, z=0, za=0, tt=0;

    hpix_eq2tp(ra, dec, &theta, &phi);

    z = cos(theta);
    za = fabs(z);

    // in [0,4)
    tt = fmod(phi, HPX_TWO_PI)/M_PI_2;

    if (za <= HPX_TWOTHIRD) {
        double temp1 = nside*(.5 + tt);
        double temp2 = nside*.75*z;

        int64_t jp = (int64_t)(temp1-temp2); // index of  ascending edge line
        int64_t jm = (int64_t)(temp1+temp2); // index of descending edge line
        int64_t ir = nside + 1 + jp - jm;  // in {1,2n+1} (ring number counted from z=2/3)
        int64_t kshift = 1 - (ir % 2);      // kshift=1 if ir even, 0 otherwise
        
        int64_t nl4 = 4*nside;
        int64_t ip = (int64_t)( ( jp+jm - nside + kshift + 1 ) / 2); // in {0,4n-1}

        ip = ip % nl4;

        ipix = hpix->ncap + nl4*(ir-1) + ip;

    } else { 
        // North & South polar caps
        double tp = tt - (int64_t)(tt);   // MODULO(tt,1.0_dp)

        double tmp = nside * sqrt( 3.0*(1.0 - za) );
        int64_t jp = (int64_t)(tp*tmp);              // increasing edge line index
        int64_t jm = (int64_t)((1.0 - tp) * tmp); // decreasing edge line index

        int64_t ir = jp + jm + 1;        // ring number counted from the closest pole
        int64_t ip = (int64_t)( tt * ir);     // in {0,4*ir-1}

        if (ip >= 4*ir) {
            ip = ip - 4*ir;
        }
        if (z>0.) {
            ipix = 2*ir*(ir-1) + ip;
        } else {
            ipix = hpix->npix - 2*ir*(ir+1) + ip;
        }

    }

    return ipix;
}


static int
PyHealPix_init(struct PyHealPix* self, PyObject *args, PyObject *kwds)
{
    long nside=0;
    if (!PyArg_ParseTuple(args, (char*)"l", &nside)) {
        return -1;
    }

    self->nside = (int64_t) nside;
    self->npix = hpix_npix(nside);
    self->area = hpix_area(nside);
    self->ncap = 2*nside*(nside-1); // number of pixels in the north polar cap

    return 0;
}

static void
PyHealPix_dealloc(struct PyHealPix* self)
{
#if ((PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION >= 6) || (PY_MAJOR_VERSION == 3))
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    // old way, removed in python 3
    self->ob_type->tp_free((PyObject*)self);
#endif
}

static PyObject *
PyHealPix_repr(struct PyHealPix* self) {
    char repr[255];
    sprintf(repr,
            "nside:   %ld\n"
            "npix:    %ld\n"
            "ncap:    %ld\n" 
            "area:    %g square degrees\n"
            "area:    %g square arcmin\n"
            "area:    %g square arcsec\n",
            self->nside,
            self->npix,
            self->ncap,
            self->area,
            self->area*3600.,
            self->area*3600.*3600.);

   return Py_BuildValue("s", repr);
}

/*
   getters
*/
static PyObject*
PyHealPix_get_nside(struct PyHealPix* self, PyObject* args)
{
   return Py_BuildValue("l", self->nside); 
}
static PyObject*
PyHealPix_get_npix(struct PyHealPix* self, PyObject* args)
{
   return Py_BuildValue("l", self->npix); 
}
static PyObject*
PyHealPix_get_ncap(struct PyHealPix* self, PyObject* args)
{
   return Py_BuildValue("l", self->ncap); 
}
static PyObject*
PyHealPix_get_area(struct PyHealPix* self, PyObject* args)
{
   return Py_BuildValue("d", self->area); 
}




/*
   convert the input ra,dec arrays to pixel numbers in the ring scheme.  no error checking
   done here

   ra,dec should be double arrays
   pixnum is int64 array
*/
static PyObject*
PyHealPix_eq2pix_ring(struct PyHealPix* self, PyObject* args)
{
    PyObject* ra_obj=NULL;
    PyObject* dec_obj=NULL;
    PyObject* pixnum_obj=NULL;

    double ra, dec;
    int64_t *pix_ptr=NULL;
    npy_intp i=0, num=0;

    if (!PyArg_ParseTuple(args, (char*)"OOO", 
                          &ra_obj, &dec_obj, &pixnum_obj)) {
        return NULL;
    }

    num=PyArray_SIZE(ra_obj);

    for (i=0; i<num; i++) {
        ra  = *(double *) PyArray_GETPTR1(ra_obj, i);
        dec = *(double *) PyArray_GETPTR1(dec_obj, i);
        pix_ptr = (int64_t *) PyArray_GETPTR1(pixnum_obj, i);
        (*pix_ptr) = hpix_eq2pix_ring(self, ra, dec);
    }

    Py_RETURN_NONE;
}


// stand alone methods
static PyMethodDef PyHealPix_methods[] = {

    {"get_nside", (PyCFunction)PyHealPix_get_nside, METH_VARARGS, "get nside\n"},
    {"get_npix", (PyCFunction)PyHealPix_get_npix, METH_VARARGS, "get the number of pixels at this resolution\n"},
    {"get_ncap", (PyCFunction)PyHealPix_get_ncap, METH_VARARGS, "get the number of pixels in the north polar cap at this resolution\n"},
    {"get_area", (PyCFunction)PyHealPix_get_area, METH_VARARGS, "get the area of a pixel at this resolution\n"},
    {"_fill_eq2pix_ring", (PyCFunction)PyHealPix_eq2pix_ring, METH_VARARGS, "convert ra,dec degrees to pixel number in ring scheme.  Don't call this method directly, since no error or type checking is performed\n"},
    {NULL}  /* Sentinel */
};


static PyTypeObject PyHealPixType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif

    "_healpix.HealPix",             /*tp_name*/
    sizeof(struct PyHealPix), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyHealPix_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    //0,                         /*tp_repr*/
    (reprfunc)PyHealPix_repr,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "HealPix class holding metadata",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyHealPix_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)PyHealPix_init,      /* tp_init */
    0,                         /* tp_alloc */
    //PyHealPix_new,                 /* tp_new */
    PyType_GenericNew,                 /* tp_new */
};



// stand alone methods
static PyMethodDef healpix_methods[] = {


    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_healpix",      /* m_name */
        "Defines the classes and some methods to work with healpix",  /* m_doc */
        -1,                  /* m_size */
        healpix_methods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit__healpix(void) 
#else
init_healpix(void) 
#endif
{
    PyObject* m;


    PyHealPixType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    if (PyType_Ready(&PyHealPixType) < 0) {
        return NULL;
    }
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else
    if (PyType_Ready(&PyHealPixType) < 0) {
        return;
    }
    m = Py_InitModule3("_healpix", healpix_methods, 
            "This module defines classis to work with HealPix\n"
            "and some generic functions.\n");
    if (m==NULL) {
        return;
    }
#endif

    Py_INCREF(&PyHealPixType);
    PyModule_AddObject(m, "HealPix", (PyObject *)&PyHealPixType);

    import_array();
#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
