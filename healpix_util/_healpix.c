/*

   Much of this code was adapted from various f90 code distributed with healpix
   under the GPL license

   The original license info is as follows

      This file is part of HEALPix.

      HEALPix is free software; you can redistribute it and/or modify
      it under the terms of the GNU General Public License as published by
      the Free Software Foundation; either version 2 of the License, or
      (at your option) any later version.

      HEALPix is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU General Public License for more details.

      You should have received a copy of the GNU General Public License
      along with HEALPix; if not, write to the Free Software
      Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <Python.h>
#include "numpy/arrayobject.h" 

#ifndef M_PI
# define M_E		2.7182818284590452354	/* e */
# define M_LOG2E	1.4426950408889634074	/* log_2 e */
# define M_LOG10E	0.43429448190325182765	/* log_10 e */
# define M_LN2		0.69314718055994530942	/* log_e 2 */
# define M_LN10		2.30258509299404568402	/* log_e 10 */
# define M_PI		3.14159265358979323846	/* pi */
# define M_PI_2		1.57079632679489661923	/* pi/2 */
# define M_PI_4		0.78539816339744830962	/* pi/4 */
# define M_1_PI		0.31830988618379067154	/* 1/pi */
# define M_2_PI		0.63661977236758134308	/* 2/pi */
# define M_2_SQRTPI	1.12837916709551257390	/* 2/sqrt(pi) */
# define M_SQRT2	1.41421356237309504880	/* sqrt(2) */
# define M_SQRT1_2	0.70710678118654752440	/* 1/sqrt(2) */
#endif

#define HPX_TWO_PI   6.28318530717958647693 /* 2*pi */
#define HPX_TWOTHIRD 0.66666666666666666666

#define HPX_D2R  0.017453292519943295
#define HPX_R2D  57.295779513082323

#define HPX_RING 1
#define HPX_NESTED 2

#define HPX_NS_MAX4 8192

// angular theta,phi in radians
#define HPX_SYSTEM_ANG 1
// equatorial ra,dec in degrees
#define HPX_SYSTEM_EQ 2

// this is 1<<29 for 64-bit
#define HPX_MAX_NSIDE 536870912L
// 12*max_nside^2
#define HPX_MAX_NPIX 3458764513820540928L


struct PyHealPix {
    PyObject_HEAD

    int scheme;  // HPX_RING or HPX_NESTED
    char scheme_name[7]; // "ring" or "nested"
    int64_t nside;
    int64_t npix;
    int64_t ncap;
    double area;

    // for healpy which often wants a nest= keyword
    int is_nested;
};
static int64_t pix2x[1024];
static int64_t pix2y[1024];

static int64_t x2pix[128];
static int64_t y2pix[128];

static int64_t jrll[]  = {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4}; // in unit of nside
static int64_t jpll[] =  {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7}; // in unit of nside/2


struct PyHealPixCap {
    double ra;
    double dec;
    double cosra;
    double sinra;
    double cosdec;
    double sindec;
};

static void PyHealPixCap_set(struct PyHealPixCap *cap,
                             double ra_degrees,
                             double dec_degrees)
{
    double ra,dec;
    ra=ra_degrees*HPX_D2R;
    dec=dec_degrees*HPX_D2R;

    cap->ra=ra_degrees;
    cap->dec=dec_degrees;
    cap->cosra=cos(ra);
    cap->sinra=sin(ra);
    cap->cosdec=cos(dec);
    cap->sindec=sin(dec);
}

#define STACK_PUSH_REALLOC_MULT 1
#define STACK_PUSH_REALLOC_MULTVAL 2
#define STACK_PUSH_INITSIZE 50

struct i64stack {
    size_t size;            // number of elements that are visible to the user
    size_t allocated_size;  // number of allocated elements in data vector
    size_t push_realloc_style; // Currently always STACK_PUSH_REALLOC_MULT, 
                               // which is reallocate to allocated_size*realloc_multval
    size_t push_initsize;      // default size on first push, default STACK_PUSH_INITSIZE 
    double realloc_multval; // when allocated size is exceeded while pushing, 
                            // reallocate to allocated_size*realloc_multval, default 
                            // STACK_PUSH_REALLOC_MULTVAL
                            // if allocated_size was zero, we allocate to push_initsize
    int64_t* data;
};

static struct i64stack* i64stack_new(size_t num) {
    struct i64stack* stack = malloc(sizeof(struct i64stack));
    if (stack == NULL) {
        printf("Could not allocate struct i64stack\n");
        exit(EXIT_FAILURE);
    }

    stack->size = 0;
    stack->allocated_size = num;
    stack->push_realloc_style = STACK_PUSH_REALLOC_MULT;
    stack->push_initsize = STACK_PUSH_INITSIZE;
    stack->realloc_multval = STACK_PUSH_REALLOC_MULTVAL;

    if (num == 0) {
        stack->data = NULL;
    } else {
        stack->data = calloc(num, sizeof(int64_t));
        if (stack->data == NULL) {
            printf("Could not allocate data in pixlist\n");
            exit(EXIT_FAILURE);
        }
    }

    return stack;
}

static void i64stack_realloc(struct i64stack* stack, size_t newsize) {

    size_t oldsize = stack->allocated_size;
    if (newsize != oldsize) {
        size_t elsize = sizeof(int64_t);

        int64_t* newdata = realloc(stack->data, newsize*elsize);
        if (newdata == NULL) {
            printf("failed to reallocate\n");
            exit(EXIT_FAILURE);
        }

        if (newsize > stack->allocated_size) {
            // the allocated size is larger.  make sure to initialize the new
            // memory region.  This is the area starting from index [oldsize]
            size_t num_new_bytes = (newsize-oldsize)*elsize;
            memset(&newdata[oldsize], 0, num_new_bytes);
        } else if (stack->size > newsize) {
            // The viewed size is larger than the allocated size in this case,
            // we must set the size to the maximum it can be, which is the
            // allocated size
            stack->size = newsize;
        }

        stack->data = newdata;
        stack->allocated_size = newsize;
    }

}

static void i64stack_resize(struct i64stack* stack, size_t newsize) {
   if (newsize > stack->allocated_size) {
       i64stack_realloc(stack, newsize);
   }

   stack->size = newsize;
}

static void i64stack_clear(struct i64stack* stack) {
    stack->size=0;
    stack->allocated_size=0;
    free(stack->data);
    stack->data=NULL;
}

static struct i64stack* i64stack_delete(struct i64stack* stack) {
    if (stack != NULL) {
        i64stack_clear(stack);
        free(stack);
    }
    return NULL;
}

static void i64stack_push(struct i64stack* stack, int64_t val) {
    // see if we have already filled the available data vector
    // if so, reallocate to larger storage
    if (stack->size == stack->allocated_size) {

        size_t newsize;
        if (stack->allocated_size == 0) {
            newsize=stack->push_initsize;
        } else {
            // currenly we always use the multiplier reallocation  method.
            if (stack->push_realloc_style != STACK_PUSH_REALLOC_MULT) {
                printf("Currently only support push realloc style STACK_PUSH_REALLOC_MULT\n");
                exit(EXIT_FAILURE);
            }
            // this will "floor" the size
            newsize = (size_t)(stack->allocated_size*stack->realloc_multval);
            // we want ceiling
            newsize++;
        }

        i64stack_realloc(stack, newsize);

    }

    stack->size++;
    stack->data[stack->size-1] = val;
}

static int64_t i64stack_pop(struct i64stack* stack) {
    if (stack->size == 0) {
        return INT64_MAX;
    }

    int64_t val=stack->data[stack->size-1];
    stack->size--;
    return val;
        
}

static int __i64stack_compare_el(const void *a, const void *b) {
    int64_t temp = 
        (  (int64_t) *( (int64_t*)a ) ) 
         -
        (  (int64_t) *( (int64_t*)b ) );
    if (temp > 0)
        return 1;
    else if (temp < 0)
        return -1;
    else
        return 0;
}


static void i64stack_sort(struct i64stack* stack) {
    qsort(stack->data, stack->size, sizeof(int64_t), __i64stack_compare_el);
}
static int64_t* i64stack_find(struct i64stack* stack, int64_t el) {
    return (int64_t*) bsearch(&el, stack->data, stack->size, sizeof(int64_t), __i64stack_compare_el);
}

/*
   helper functions
 */
static inline int64_t i64max(int64_t v1, int64_t v2) {
    return v1 > v2 ? v1 : v2;
}
static inline int64_t i64min(int64_t v1, int64_t v2) {
    return v1 < v2 ? v1 : v2;
}

static inline int64_t nint64(double x) {
    if (x >= 0.0) {
        return (int64_t) (x + 0.5);
    } else {
        return (int64_t) (x - 0.5);
    }
}

int64_t cheap_isqrt(int64_t in) {
    double din, dout;
    int64_t out, diff;

    din = (double) in;  // input integer number has ~19 significant digits (in base 10)
    dout = sqrt(din); // accurate to ~15 digits (base 10) , for ~10 needed
    out  = (int64_t) floor(dout); // limited accuracy creates round-off error

    // integer arithmetics solves round-off error
    diff = in - out*out;
    if (diff < 0) {
        out -= 1;
    } else if (diff > 2*out) {
        out += 1;
    }

    return out;
}


static void reset_bounds(double *angle,    /* MODIFIED -- the angle to bound in degrees*/
                         double min,       /* IN -- inclusive minimum value */
                         double max        /* IN -- exclusive maximum value */
                        )
{
    while (*angle<min) {
        *angle += 360.0;
    }
    while (*angle>=max) {
        *angle -= 360.0;
    }
    return;
}

void reset_bounds2(double *theta, /* MODIFIED -- the -90 to 90 angle */
                   double *phi  /* MODIFIED -- the 0 to 360 angle */
                  )
{
    reset_bounds(theta, -180.0, 180.0);
    if (fabs(*theta) > 90.0) {
        *theta = 180.0 - *theta;
        *phi += 180;
    }
    reset_bounds(theta, -180.0, 180.0);
    reset_bounds(phi, 0.0, 360.0);
    if (fabs(*theta)==90.0) *phi=0.;
    return;
}



/*
   verify validity of nside and npix
*/
static int nside_is_ok(long nside)
{
    return 
        (nside > 0)
        && (nside <= HPX_MAX_NSIDE)
        && ( (nside & (nside - 1)) == 0 );
}
static inline int npix_is_ok(int64_t npix)
{
    if (npix < 12 || npix > HPX_MAX_NPIX) {
        return 0;
    } else {
        return 1;
    }
}

/*
   number of pixels in the sky
 */
static inline int64_t nside2npix(int64_t nside) {
    return 12*nside*nside;
}

/*
   area of a pixel in sq degrees
 */
static inline double nside2area(int64_t nside) {
    int64_t npix;
    npix = nside2npix(nside);
    return 4.0*M_PI/npix;
}

/*
   number of pixels in the north polar cap
 */
static inline int64_t nside2ncap(int64_t nside) {
    return 2*nside*(nside-1);
}

/*
   infer nside from npix
*/
static inline int64_t npix2nside(int64_t npix)
{
    int64_t nside=0, npix_tmp=0;
    if (!npix_is_ok(npix)) {
        return -1;
    }

    nside = nint64( sqrt(npix/12.0) );

    npix_tmp = nside2npix(nside);
    if (npix_tmp != npix) {
        return -1;
    }
    return nside;
}

static double get_posangle_eq(const struct PyHealPixCap* cap,
                              double ra_degrees,
                              double dec_degrees)
{
    double ra, dec,
           cosra, sinra, cosdec, sindec,
           cosradiff, sinradiff, arg,
           posangle;

    ra=ra_degrees*HPX_D2R;
    dec=dec_degrees*HPX_D2R;
    cosra=cos(ra);
    sinra=sin(ra);
    cosdec=cos(dec);
    sindec=sin(dec);

    cosradiff = cosra*cap->cosra + sinra*cap->sinra;
    sinradiff = sinra*cap->cosra - cosra*cap->sinra;
    arg = cap->sindec*cosradiff - cap->cosdec*sindec/cosdec;

    // -pi,pi
    posangle = atan2(sinradiff, arg) - M_PI_2;

    posangle *= HPX_R2D;
    reset_bounds(&posangle, -180.0, 180.0);
    return posangle;
}

static int32_t get_quadrant_eq(const struct PyHealPixCap* cap,
                               double ra_degrees,
                               double dec_degrees)
{
    int32_t quadrant;
    double posangle;
    posangle=get_posangle_eq(cap, ra_degrees, dec_degrees);

    if (posangle < -90.) {
        quadrant=3;
    } else if (posangle < 0.0) {
        quadrant=4;
    } else if (posangle < 90.) {
        quadrant=1;
    } else {
        quadrant=2;
    }
    //quadrant = 1 + ( (int) (posangle/90.0) );
    //quadrant = 2 + ( (int) (posangle/90.0) );

    return quadrant;
}


static PyObject*
make_i64_array(npy_intp size, const char* name, int64_t** ptr)
{
    PyObject* array=NULL;
    npy_intp dims[1];
    int ndims=1;
    if (size < 0) {
        PyErr_Format(PyExc_ValueError, "size of %s array must be >= 0",name);
        return NULL;
    }

    dims[0] = size;
    array = PyArray_ZEROS(ndims, dims, NPY_INT64, 0);
    if (array==NULL) {
        PyErr_Format(PyExc_MemoryError, "could not create %s array",name);
        return NULL;
    }

    *ptr = PyArray_DATA((PyArrayObject*)array);
    return array;
}



/*
   convert equatorial ra,dec degrees to angular theta,phi radians
*/

static inline void eq2ang(double ra, double dec, double* theta, double* phi) {

    // make sure ra in [0,360] and dec within [-90,90]
    reset_bounds2(&dec, &ra);

    *phi = ra*HPX_D2R;
    *theta = M_PI_2 - dec*HPX_D2R;
}

/*
   convert angular theta,phi radians to equatorial ra,dec degrees
*/

static inline void ang2eq(double theta, double phi, double *ra, double *dec) {

    *ra = phi*HPX_R2D;
    *dec = (M_PI_2 - theta)*HPX_R2D;

    // make sure ra in [0,360] and dec within [-90,90]
    reset_bounds2(dec, ra);
}

/*
   theta,phi to standard x,y,z
*/
static inline void ang2xyz(double theta, double phi, double* x, double* y, double* z) {

    double sintheta=0;

    sintheta = sin(theta);
    *x = sintheta * cos(phi);
    *y = sintheta * sin(phi);
    *z = cos(theta);
}


/*
   ra,dec to standard x,y,z
*/
static inline void eq2xyz(double ra, double dec, double* x, double* y, double* z) {

    double theta=0, phi=0;

    eq2ang(ra, dec, &theta, &phi);
    ang2xyz(theta,phi,x,y,z);
}


/*
   make static variables
*/

/*
    fill the arrays giving x and y in the face from pixel number
    for the nested (quad-cube like) ordering of pixels
*/

static void mk_pix2xy(void)
{
    int64_t kpix, jpix, ix, iy, ip, id;

    for (kpix=0; kpix<1024; kpix++) {
        jpix = kpix;
        ix = 0;
        iy = 0;
        ip = 1;               // bit position (in x and y)
        while (jpix != 0) {  // go through all the bits


            id = jpix % 2; // bit value (in kpix), goes in ix
            jpix = jpix/2;
            ix = id*ip+ix;

            id = jpix % 2; //[ bit value (in kpix), goes in iy
            jpix = jpix/2;
            iy = id*ip+iy;

            ip = 2*ip;     // next bit (in x and y)
        }
        pix2x[kpix] = ix;     // in 0,31
        pix2y[kpix] = iy;     // in 0,31
    }

}

/*
    fill the arrays giving the number of the pixel lying in (x,y)
         x and y are in {1,128}
         the pixel number is in {0,128**2-1}
    
         if  i-1 = sum_p=0  b_p * 2^p
         then ix = sum_p=0  b_p * 4^p
              iy = 2*ix
         ix + iy in {0, 128**2 -1}
*/

static void mk_xy2pix(void) {
    int64_t k,ip,i,j,id;

    for (i=0; i<128; i++) {
        j  = i;
        k  = 0;
        ip = 1;

        while (1) {

            if (j==0) {
                x2pix[i] = k;
                y2pix[i] = 2*k;
                break;
            }

            id = j % 2;
            j  = j/2;
            k  = ip*id+k;
            ip = ip*4;

        }
    }
}

static int
PyHealPix_init(struct PyHealPix* self, PyObject *args, PyObject *kwds)
{
    int scheme=0;
    long nside=0;
    if (!PyArg_ParseTuple(args, (char*)"il", &scheme, &nside)) {
        return -1;
    }

    if (!nside_is_ok(nside)) {
        PyErr_Format(PyExc_ValueError, "bad nside: %ld", nside);
        return -1;
    }

    if (scheme != HPX_RING && scheme != HPX_NESTED) {
        PyErr_Format(PyExc_ValueError,
                     "scheme should be ring (%d) or nested (%d), got %d",
                     HPX_RING, HPX_NESTED, scheme);
        return -1;
    }
    self->scheme = scheme;
    if (scheme == HPX_RING) {
        sprintf(self->scheme_name,"RING");
    } else {
        sprintf(self->scheme_name,"NESTED");
    }
    self->nside  = (int64_t) nside;
    self->npix   = nside2npix(nside);
    self->area   = nside2area(nside);
    self->ncap   = nside2ncap(nside);

    self->is_nested = (scheme==HPX_NESTED) ? 1: 0;
    
    return 0;
}

static PyObject *
PyHealPix_repr(struct PyHealPix* self) {
    char repr[255];
    static const char* ring_name="RING";
    static const char* nest_name="NEST";

    const char *name=NULL;

    if (self->scheme == HPX_RING) {
        name=ring_name;
    } else {
        name=nest_name;
    }
    sprintf(repr,
            "scheme:       %d\n"
            "scheme_name:  %s\n"
            "nside:        %ld\n"
            "npix:         %ld\n"
            "ncap:         %ld\n" 
            "area:         %g square degrees\n"
            "area:         %g square arcmin\n"
            "area:         %g square arcsec\n",
            self->scheme,
            name,
            self->nside,
            self->npix,
            self->ncap,
            self->area,
            self->area*3600.,
            self->area*3600.*3600.);

   return Py_BuildValue("s", repr);
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

/*
   convert nest pixel to a ring pixel
*/

/*
static void nest2ring_arr(int64_t nside,
                          int64_t *ipnest,
                          int64_t *ipring,
                          int64_t n)
{
    int64_t npface, nl4, face_num, ipf,
            ix, iy, scale, ismax, i, ip_low,
            jrt, jpt, jr, jp, nr, n_before, kshift;
    int64_t npix;
    int64_t ai;

    npix=nside2npix(nside);
    for (ai=0; ai<n; ai++) {
        if (ipnest[ai] < 0 || ipnest[ai] > npix-1) {
            ipring[ai]=-9999;
            continue;
        }

        npface = nside*nside;
        nl4    = 4*nside;

        // finds the face, and the number in the face
        face_num = ipnest[ai]/npface;   // face number in [0,11]
        ipf = ipnest[ai] & (npface-1);  // pixel number in the face [0,npface-1]

        // finds the x,y on the face (starting from the lowest corner)
        // from the pixel number
        if (nside <= HPX_NS_MAX4) {
            int64_t ip_trunc, ip_med, ip_hi;
            ip_low   = ipf & 1023;       // content of the last 10 bits
            ip_trunc = ipf/1024;         // truncation of the last 10 bits
            ip_med   = ip_trunc & 1023;  // content of the next 10 bits
            ip_hi    = ip_trunc/1024;    // content of the high weight 10 bits

            ix = 1024*pix2x[ip_hi] + 32*pix2x[ip_med] + pix2x[ip_low];
            iy = 1024*pix2y[ip_hi] + 32*pix2y[ip_med] + pix2y[ip_low];
        } else {
            ix = 0;
            iy = 0;
            scale = 1;
            ismax = 4;
            for (i=0; i<= ismax; i++) {
                ip_low = ipf & 1023;
                ix = ix + scale * pix2x[ip_low];
                iy = iy + scale * pix2y[ip_low];
                scale = scale * 32;
                ipf   = ipf/1024;
            }
            ix = ix + scale * pix2x[ipf];
            iy = iy + scale * pix2y[ipf];
        }

        //     transforms this in (horizontal, vertical) coordinates
        jrt = ix + iy;  // 'vertical' in [0,2*(nside-1)]
        jpt = ix - iy;  // 'horizontal' in [-nside+1,nside-1]

        //     computes the z coordinate on the sphere
        jr =  jrll[face_num]*nside - jrt - 1;   // ring number in [1,4*nside-1]

        if (jr < nside) {     // north pole region
            nr = jr;
            n_before = 2 * nr * (nr - 1);
            kshift = 0;
        } else if (jr <= 3*nside) { // equatorial region (the most frequent)
            nr = nside;               
            n_before = 2 * nr * ( 2 * jr - nr - 1);
            kshift = (jr-nside) & 1;

        } else { //south pole region
            nr = nl4 - jr;
            n_before = npix - 2 * nr * (nr + 1);
            kshift = 0;
        }

        //     computes the phi coordinate on the sphere, in [0,2Pi]
        jp = (jpll[face_num]*nr + jpt + 1 + kshift)/2;  // 'phi' number in the ring in [1,4*nr]
        if (jp > nl4) {
            jp = jp - nl4;
        }
        if (jp < 1) {
            jp = jp + nl4;
        }

        ipring[ai] = n_before + jp - 1; // in [0, npix-1]

    } // over array
}

static int64_t nest2ring(int64_t nside, int64_t ipnest)
{
    int64_t ipring;
    nest2ring_arr(nside, &ipnest, &ipring, 1);
    return ipring;
}
*/

/*
   convert nest pixel to a ring pixel
*/

/*
static void ring2nest_arr(int64_t nside,
                          int64_t *ipring_arr,
                          int64_t *ipnest_arr,
                          int64_t n)
{
    int64_t nl2, nl4, face_num, ipf,
            ismax, i,
            irn, iphi, kshift, nr, 
            ip, ire, 
            irm, ifm, ifp, irs, irt, ipt,
            ix, iy, scale, scale_factor,
            ix_low, iy_low;
    int64_t ipring, npix, ncap;
    int64_t ai;

    npix=nside2npix(nside);
    ncap=nside2ncap(nside);
    for (ai=0; ai<n; ai++) {
        ipring = ipring_arr[ai];

        if (ipring < 0 || ipring > npix-1) {
            ipnest_arr[ai]=-9999;
            continue;
        }

        nl2 = 2*nside;
        nl4 = 4*nside;

        if (ipring < ncap) { 
            // north polar cap

            irn   = (cheap_isqrt(2*ipring+2) + 1) / 2;// counted from North pole
            iphi  = ipring - 2*irn*(irn - 1);

            kshift = 0;
            nr = irn;                 // 1/4 of the number of points on the current ring
            face_num = iphi / irn;    // in [0,3]

        } else if (ipring < npix - ncap) {
            // equatorial region

            ip    = ipring - ncap;
            irn   = ((int64_t) ( ip / nl4 )) + nside; // counted from North pole
            iphi  = ip & (nl4-1);

            kshift  = (irn+nside) & 1;  // MODULO(irn+nside,2), 1 if irn+nside is odd, 0 otherwise
            nr = nside;
            ire =  irn - nside + 1;               //! in [1, 2*nside +1]
            irm =  nl2 + 2 - ire;
            ifm = (iphi - ire/2 + nside) / nside; // face boundary
            ifp = (iphi - irm/2 + nside) / nside;

            if (ifp == ifm) { // faces 4 to 7
                face_num = (ifp & 3) + 4;
            } else if (ifp < ifm) { // (half-)faces 0 to 3
                face_num = ifp;
            } else { // (half-)faces 8 to 11
                face_num = ifp + 7;
            }

        } else { 
            // south polar cap

            ip    = npix - ipring;

            irs   = (cheap_isqrt(2*ip) +1)/2; // counted from South pole
            iphi  = 2*irs*(irs + 1) - ip;

            kshift = 0;
            nr = irs;
            irn   = nl4 - irs;
            face_num = iphi / irs + 8; // in [8,11]

        }

        //     finds the (x,y) on the face
        irt =   irn  - jrll[face_num]*nside + 1;          // in [-nside+1,0]
        ipt = 2*iphi - jpll[face_num]*nr - kshift + 1;    // ! in [-nside+1,nside-1]
        if (ipt >= nl2) {
            ipt = ipt - 8*nside; // for the face #4
        }

        ix =  (ipt - irt ) / 2;
        iy = -(ipt + irt ) / 2;

        if (nside <= HPX_NS_MAX4) {
            ix_low = ix & 127;
            iy_low = iy & 127;
            ipf = x2pix[ix_low] + y2pix[iy_low] + (x2pix[ix/128] + y2pix[iy/128]) * 16384;
        } else {
            scale = 1;
            scale_factor = 16384; // 128*128
            ipf = 0;
            ismax = 1;// for nside in [2^14, 2^20]
            if (nside >  1048576 ) {
                ismax = 3;
            }
            for (i=0; i<= ismax; i++) {
                ix_low = ix & 127; // last 7 bits
                iy_low = iy & 127; // last 7 bits
                ipf = ipf + (x2pix[ix_low]+y2pix[iy_low]) * scale;
                scale = scale * scale_factor;
                ix  = ix / 128;  // truncate out last 7 bits
                iy  = iy / 128;
            }
            ipf =  ipf + (x2pix[ix]+y2pix[iy]) * scale;
        }

        //ipnest_arr[ai] = ipf + face_num* (npix/12);   // in [0, 12*nside**2 - 1]
        ipnest_arr[ai] = ipf + face_num*nside*nside;   // in [0, 12*nside**2 - 1]

    } // over array
}

static int64_t ring2nest(int64_t nside, int64_t ipring)
{
    int64_t ipnest;
    ring2nest_arr(nside, &ipring, &ipnest, 1);
    return ipnest;
}
*/


/*




   ring scheme related routines




*/

/*
   convert angular theta,phi to pixel number in the ring scheme
*/
static int64_t ang2pix_ring(const struct PyHealPix* self,
                            double theta,
                            double phi) {
    int64_t nside=self->nside;
    int64_t ipix=0;
    double z=0, za=0, tt=0;

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

        ipix = self->ncap + nl4*(ir-1) + ip;

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
            ipix = self->npix - 2*ir*(ir+1) + ip;
        }
    }

    return ipix;
}

/*
   convert equatorial ra,dec to pixel number in the ring scheme
*/
static inline int64_t eq2pix_ring(const struct PyHealPix* hpix,
                                  double ra,
                                  double dec) {
    int64_t pixnum;
    double theta, phi;
    eq2ang(ra, dec, &theta, &phi);
    pixnum=ang2pix_ring(hpix, theta, phi);
    return pixnum;
}


/*
   get the nominal pixel center for the input theta phi
   in the ring scheme
*/
static void pix2ang_ring(const struct PyHealPix* self,
                         int64_t pixnum,
                         double *theta,
                         double *phi) {

    int64_t nl2, nl4, iring, iphi, ip;
    double dnside, fodd, arg;

    nl2  = 2*self->nside;
    dnside = (double) self->nside;

    if (pixnum < self->ncap) {
        // North Polar cap -------------
        //printf("north polar cap\n");

        iring = (cheap_isqrt(2*pixnum+2) + 1)/2;
        iphi  = pixnum - 2*iring*(iring - 1);

        (*theta) = 2.0 * asin(iring / (sqrt(6.0)*dnside));
        (*phi)   = ((double)iphi + 0.5) * M_PI_2/iring;

    } else if (pixnum < self->npix-self->ncap) { 
        // Equatorial region ------
        //printf("equatorial\n");

        ip    = pixnum - self->ncap;
        nl4   = 4*self->nside;
        iring = ( ip / nl4 ) + self->nside; // counted from North pole
        //iphi  = iand(ip, nl4-1_MKD)
        iphi  = ip & (nl4-1);

        // 0 if iring+nside is odd, 1/2 otherwise
        //fodd  = 0.5 * ( iand(iring+nside+1,1) )
        fodd = 0.5*(  (iring+self->nside+1) & 1  ); 

        arg =  (nl2 - iring) / (1.5*dnside); 
        (*theta) = acos(arg);
        (*phi)   = ((double) iphi + fodd) * M_PI_2/ dnside;

    } else {
        // South Polar cap -----------------------------------
        //printf("south polar cap\n");

        ip = self->npix - pixnum;

        iring = (cheap_isqrt(2*ip) + 1) / 2;
        iphi  = 2*iring*(iring + 1) - ip;

        (*theta) = M_PI - 2. * asin(iring / (sqrt(6.0)*dnside));
        (*phi)   = ((double)iphi + 0.5) * M_PI_2/iring;

    }

}

/*
   get the nominal pixel center for the input ra dec
   in the ring scheme
*/
static inline void pix2eq_ring(const struct PyHealPix* self,
                               int64_t pixnum,
                               double *ra,
                               double *dec) {
    double theta, phi;
    pix2ang_ring(self, pixnum, &theta, &phi);
    ang2eq(theta, phi, ra, dec);
}

/*
   returns the ring number in {1, 4*nside-1} from the z coordinate
   returns the ring closest to the z provided
*/

static int64_t get_ring_num(const struct PyHealPix *self, double z) {
    int64_t iring;

    // rounds double to nearest long long int
    iring = llrintl( self->nside*(2.-1.5*z) );

    // north cap
    if (z > HPX_TWOTHIRD) {
        iring = llrintl( self->nside* sqrt(3.*(1.-z)) );
        if (iring == 0) {
            iring = 1;
        }
    } else if (z < -HPX_TWOTHIRD) {
        iring = llrintl( self->nside* sqrt(3.*(1.+z)) );

        if (iring == 0) {
            iring = 1;
        }
        iring = 4*self->nside - iring;
    }

    return iring;
}

double ring2z(const struct PyHealPix *self, int64_t ir) {
    double fn, tmp, z;

    fn = (double) self->nside;

    if (ir < self->nside) {
        // polar cap (north)
        tmp = (double) ir;
        z = 1.0 - (tmp * tmp) / (3.0 * fn * fn);
    } else if (ir < 3*self->nside) {
        // tropical band
        z = ( (double)( 2*self->nside-ir) ) * 2.0 / (3.0 * fn);
    } else {
        // polar cap (south)
        tmp = (double) (4*self->nside - ir);
        z = -1.0 + (tmp * tmp) / (3.0 * fn * fn);
    }
    return z;
}

/*
 
  fill in the list of pixels in RING scheme. pixels are *appended* to plist so
  be sure to run i64stack_resize(plist, 0) or _clear or some such if necessary

*/

static void is_in_ring(
        const struct PyHealPix* self, 
        int64_t iz, 
        double phi0, 
        double dphi, 
        struct i64stack* plist) {

    int64_t nr, ir, ipix1, ipix2, i;
    double shift=0.5;
    int64_t nside = self->nside;

    if (iz<nside) {
        // north pole
        ir = iz;
        nr = ir*4;
        ipix1 = 2*ir*(ir-1);        //    lowest pixel number in the ring
    } else if (iz>(3*nside)) {
        // south pole
        ir = 4*nside - iz;
        nr = ir*4;
        ipix1 = self->npix - 2*ir*(ir+1); // lowest pixel number in the ring
    } else {
        // equatorial region
        ir = iz - nside + 1;           //    within {1, 2*nside + 1}
        nr = nside*4;
        if ((ir&1)==0) shift = 0;
        ipix1 = self->ncap + (ir-1)*nr; // lowest pixel number in the ring
    }

    ipix2 = ipix1 + nr - 1;  //    highest pixel number in the ring
 

    if (dphi > (M_PI-1e-7)) {
        for (i=ipix1; i<=ipix2; ++i) {
            i64stack_push(plist, i);
        }
    } else {
        int64_t ip_lo, ip_hi, pixnum;
        // M_1_PI is 1/pi
        ip_lo = (int64_t)( floor(nr*.5*M_1_PI*(phi0-dphi) - shift) )+1;
        ip_hi = (int64_t)( floor(nr*.5*M_1_PI*(phi0+dphi) - shift) );
        pixnum = ip_lo+ipix1;
        if (pixnum<ipix1) {
            pixnum += nr;
        }
        for (i=ip_lo; i<=ip_hi; ++i, ++pixnum) {
            if (pixnum>ipix2) {
                pixnum -= nr;
            }
            i64stack_push(plist, pixnum);
        }
    }

}


/*
   query_disc

   If inclusive==0, find the list of pixels whose centers are contained within
   the disc

   If inclusive==1, find the list of pixels whose centers are contained within
   the disc or if the pixel intersects the disc

   radius in radians
 */


/*
static void query_disc_xyz(const struct PyHealPix* self,
                           double x0, double y0, double z0,
                           double radius, 
                           int inclusive,
                           struct i64stack* listpix) {

    int64_t nside, irmin, irmax, iz;
    double cosang, dth1, dth2, phi0, cosphi0, a;
    double rlat0, rlat1, rlat2, zmax, zmin, tmp;

    if (inclusive) {
        // this number is acos(2/3)
        double fudge = 0.84106867056793033/self->nside; // 1.071* half pixel size
        radius += fudge;
    }

    //double vector0[3];
    nside=self->nside;
    cosang = cos(radius);

    // this does not alter the storage
    i64stack_resize(listpix, 0);

    dth1 = 1. / (3.0*nside*nside);
    dth2 = 2. / (3.0*nside);

    if ((x0 != 0.) || (y0 != 0.)) {
        // in (-Pi, Pi]
        phi0 = atan2(y0, x0);
    } else {
        phi0=0.;
    }
    cosphi0 = cos(phi0);
    a = x0*x0 + y0*y0;

    //     --- coordinate z of highest and lowest points in the disc ---
    rlat0  = asin(z0);    // latitude in RAD of the center
    rlat1  = rlat0 + radius;
    rlat2  = rlat0 - radius;

    if (rlat1 >=  M_PI_2) {
        zmax =  1.0;
    } else {
        zmax = sin(rlat1);
    }
    irmin = get_ring_num(self, zmax);
    irmin = i64max(1, irmin-1); // start from a higher point, to be safe

    if (rlat2 <= -M_PI_2) {
        zmin = -1.;
    } else {
        zmin = sin(rlat2);
    }
    irmax = get_ring_num(self, zmin);
    irmax = i64min(4*nside-1, irmax + 1); // go down to a lower point

    tmp=0;
    for (iz=irmin; iz<= irmax; iz++) {
        double z, b, c, dphi, cosdphi;
        if (iz <= nside-1) { // north polar cap
              z = 1.  - iz*iz*dth1;
        } else if (iz <= 3*nside) { // tropical band + equat.
            z = (2*nside-iz) * dth2;
        } else {
            tmp = 4*nside-iz;
            z = - 1. + tmp*tmp*dth1;
        }
        b = cosang - z*z0;
        c = 1. - z*z;

        if ((x0==0.) && (y0==0.)) {
            dphi=M_PI;
            if (b > 0.) {
                goto SKIP2; // out of the disc, 2008-03-30
            }
            goto SKIP1;
        } 

        cosdphi = b / sqrt(a*c);
        if (fabs(cosdphi) <= 1.) {
              dphi = acos(cosdphi); // in [0,Pi]
        } else {
            if (cosphi0 < cosdphi) {
                goto SKIP2; // out of the disc
            }
            dphi = M_PI; // all the pixels at this elevation are in the disc
        }
SKIP1:
        is_in_ring(self, iz, phi0, dphi, listpix);

SKIP2:
        // we have to put something here
        continue;

    }
}
*/

/*
   this is super slow because I'm converting the pixels from ring to nest.

   look into the stuff in the f90 program, see if faster
*/
/*
static void query_disc_nest(const struct PyHealPix* self,
                            double x, double y, double z,
                            double radius_degrees, 
                            int inclusive,
                            struct i64stack* listpix) {

    query_disc_xyz(self, x, y, z, radius_degrees,
                    inclusive, listpix);
    ring2nest_arr(self->nside,
                  listpix->data,
                  listpix->data,
                  listpix->size);
}
*/


/*
   radius in radians

   TODO check input radius range < PI
*/

/*
static double fudge_radius(int64_t nside, double radin, int quadratic) {
    double radout, fudge, factor;

    factor = sqrt( 0.55555555555555555555 + 1.29691115062192347448165712908*(1.0-0.5/nside) );
    fudge  = factor * M_PI / (4.0 * nside); //  increase effective radius

    if (quadratic) {
       radout = sqrt(radin*radin + fudge*fudge);
    } else {
       radout = radin + fudge;
    }
    if (radout > M_PI) {
        radout=M_PI;
    }

    return radout;
}


static void query_disc_xyz_new(const struct PyHealPix* self,
                               double x, double y, double z,
                               double radius, 
                               int inclusive,
                               struct i64stack* listpix) {

    double 
        norm_vect0, z0, rlat0, rlat1, rlat2, zmin, zmax;
    int64_t nside, irmin, irmax, nr, iz;

    nside=self->nside;
    i64stack_resize(listpix, 0);
    if (radius <= 0.) {
        return;
    }

    if (inclusive) {
        radius = fudge_radius(nside, radius, 0);
    }

    //     ---------- circle center -------------
    norm_vect0=sqrt(x*x + y*y + z*z);
    z0 = z/norm_vect0;

    //     --- coordinate z of highest and lowest points in the disc ---
    rlat0  = asin(z0);   // latitude in RAD of the center
    rlat1  = rlat0 + radius;
    rlat2  = rlat0 - radius;

    if (rlat1 >=  M_PI_2) {
       zmax =  1.0;
    } else {
       zmax = sin(rlat1);
    }
    irmin = get_ring_num(self, zmax);
    irmin = i64max(1, irmin - 1); // start from a higher point, to be safe

    if (rlat2 <= -M_PI_2) {
       zmin = -1.0;
    } else {
       zmin = sin(rlat2);
    }
    irmax = get_ring_num(self, zmin);
    irmax = i64min(4*nside-1, irmax + 1);  // go down to a lower point

    nr = irmax-irmin+1; // in [1, 4*Nside-1]

    for (iz=irmin; iz<= irmax; iz++) {
        //ztab[iz-irmin+1] = ring2z(self, iz);
    }
}
*/

/*


   nested scheme related routines


*/

/*
   convert angular theta,phi to pixel number in the nested scheme
*/
static int64_t ang2pix_nest(const struct PyHealPix* self,
                            double theta,
                            double phi) {

    int64_t nside=self->nside;
    int64_t i, ipix=0, face_num, 
            ix, iy, ix_low, iy_low,
            jp, jm, ifp, ifm, ntt, ipf,
            scale, scale_factor,
            ismax;;
    double z=0, za=0, tt=0, tp, tmp, temp1, temp2;

    z = cos(theta);
    za = fabs(z);

    // in [0,4)
    tt = fmod(phi, HPX_TWO_PI)/M_PI_2;

    if (za <= HPX_TWOTHIRD) { // equatorial region
        temp1=nside*(0.5 + tt - z*0.75);
        temp2=nside*(0.5 + tt + z*0.75);

        // (the index of edge lines increase when the longitude=phi goes up)
        jp = (int64_t) temp1; //  ascending edge line index
        jm = (int64_t) temp2; // descending edge line index

        //        finds the face
        ifp = jp / nside;  // in [0,4]
        ifm = jm / nside;

        if (ifp == ifm) {              // faces 4 to 7
            face_num = (ifp & 3)  + 4;
        } else if (ifp < ifm) {          // (half-)faces 0 to 3
            face_num = ifp & 3;
        } else {                         // (half-)faces 8 to 11
            face_num = (ifm & 3) + 8;
        }

        ix =          jm & (nside-1);
        iy = nside - (jp & (nside-1)) - 1;

    } else { // polar region, za > 2/3
        ntt = (int64_t) tt;
        if (ntt >= 4) {
            ntt = 3;
        }
        tp = tt - (double)ntt;

        if (z > 0.0) {
            tmp = sqrt(6.0) * sin( theta * 0.5);
        } else {
            tmp = sqrt(6.0) * cos( theta * 0.5);
        }

        // (the index of edge lines increase when distance from the closest pole goes up)
        jp = (int64_t) ( nside * tp * tmp ); // line going toward the pole as phi increases
        jm = (int64_t) ( nside * (1.0 - tp) * tmp );  // that one goes away of the closest pole
        jp = i64min(nside-1, jp); // for points too close to the boundary
        jm = i64min(nside-1, jm);

       // finds the face and pixel's (x,y)
       if (z >= 0) {
          face_num = ntt;       // in [0,3]
          ix = nside - jm - 1;
          iy = nside - jp - 1;
       } else {
          face_num = ntt + 8;   // in [8,11]
          ix =  jp;
          iy =  jm;
       }
    }

    if (nside <= HPX_NS_MAX4) {
       ix_low = ix & 127;
       iy_low = iy & 127;
       ipf =  x2pix[ix_low] + y2pix[iy_low] + (x2pix[ix/128] + y2pix[iy/128]) * 16384;
    } else {
        scale = 1;
        scale_factor = 16384;             // 128*128
        ipf = 0;
        ismax = 1;                        // for nside in [2^14, 2^20]
        if (nside >  1048576 ) {
            ismax = 3;
        }
        for (i=0; i <= ismax; i++) {
            ix_low = ix & 127;            // last 7 bits
            iy_low = iy & 127;            // last 7 bits
            ipf = ipf + (x2pix[ix_low]+y2pix[iy_low]) * scale;
            scale = scale * scale_factor;
            ix  =     ix / 128; // truncate out last 7 bits
            iy  =     iy / 128;
        }
        ipf =  ipf + (x2pix[ix]+y2pix[iy]) * scale;
    }

    ipix = ipf + face_num*nside*nside;   // in [0, 12*nside**2 - 1]

    return ipix;
}

/*
   convert equatorial ra,dec to pixel number in the nested scheme
*/
static inline int64_t eq2pix_nest(const struct PyHealPix* hpix,
                                  double ra,
                                  double dec) {
    int64_t pixnum;
    double theta, phi;
    eq2ang(ra, dec, &theta, &phi);
    pixnum=ang2pix_nest(hpix, theta, phi);
    return pixnum;
}

/*
   get the nominal pixel center for the input theta phi
   in the nested scheme
*/
static void pix2ang_nest(const struct PyHealPix* self,
                         int64_t pixnum,
                         double *theta,
                         double *phi) {

    double z, fn, fact1, fact2;
    int64_t i, nside, npface, nl4, face_num, ix, iy, scale, ismax,
            ip_low, ipf, jrt, jpt, jr, nr, jp;

    nside = self->nside;
    npface = nside * nside;
    nl4    = 4*nside;

    //     finds the face, and the number in the face
    face_num = pixnum/npface;    // face number in [0,11]
    ipf = pixnum % npface;       // pixel number in the face [0,npface-1]

    fn = (double) nside;

    //     finds the x,y on the face (starting from the lowest corner)
    //     from the pixel number
    ix = 0;
    iy = 0;
    scale = 1;
    ismax = 4;
    for (i=0; i <= ismax; i++) {
        ip_low = ipf & 1023;
        ix = ix + scale * pix2x[ip_low];
        iy = iy + scale * pix2y[ip_low];
        scale = scale * 32;
        ipf   = ipf/1024;
    }

    ix = ix + scale * pix2x[ipf];
    iy = iy + scale * pix2y[ipf];

    //     transforms this in (horizontal, vertical) coordinates
    jrt = ix + iy;  //  'vertical' in [0,2*(nside-1)]
    jpt = ix - iy;  // 'horizontal' in [-nside+1,nside-1]

    //     computes the z coordinate on the sphere
    jr =  jrll[face_num]*nside - jrt - 1;   // ring number in [1,4*nside-1]

    if (jr < nside) {
        // north pole region
        nr = jr;
        (*theta) = 2.0 * asin( nr / (sqrt(6.0) * fn) );
    } else if (jr <= 3*nside) {
        // equatorial region
        nr = nside;
        (*theta) = acos((2*nside-jr)* 2.0/(3.0*fn) );
    } else if (jr > 3*nside) {
        // south pole region
        nr = nl4 - jr;
        (*theta) = M_PI - 2.0 * asin( nr / (sqrt(6.0) * fn) );
    }

    //     computes the phi coordinate on the sphere, in [0,2Pi]
    jp = jpll[face_num]*nr + jpt;  // 'phi' number in the ring in [0,8*nr-1]
    if (jp < 0) {
        jp = jp + 2*nl4;
    }

    //(*phi) = jp  * (quartpi / nr);
    (*phi) = jp  * (M_PI / (4*nr));

}

/*
   get the nominal pixel center for the input ra dec
   in the nest scheme
*/
static inline void pix2eq_nest(const struct PyHealPix* self,
                               int64_t pixnum,
                               double *ra,
                               double *dec) {
    double theta, phi;
    pix2ang_nest(self, pixnum, &theta, &phi);
    ang2eq(theta, phi, ra, dec);
}


/*
   getters
*/
static PyObject*
PyHealPix_get_scheme_num(struct PyHealPix* self, PyObject* args)
{
   return Py_BuildValue("i", self->scheme); 
}
static PyObject*
PyHealPix_get_scheme(struct PyHealPix* self, PyObject* args)
{
   return Py_BuildValue("s", self->scheme_name); 
}
static PyObject*
PyHealPix_is_nested(struct PyHealPix* self, PyObject* args)
{
   return Py_BuildValue("i", self->is_nested); 
}



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
   convert the input theta,phi arrays to pixel numbers in the ring scheme.  no
   error checking done here

   theta,phi should be double arrays
   pixnum is int64 array
*/
static PyObject*
PyHealPix_fill_ang2pix(struct PyHealPix* self, PyObject* args)
{
    PyObject* theta_obj=NULL;
    PyObject* phi_obj=NULL;
    PyObject* pixnum_obj=NULL;

    double theta, phi;
    int64_t *pix_ptr=NULL;
    npy_intp i=0, num=0;

    if (!PyArg_ParseTuple(args, (char*)"OOO", 
                          &theta_obj, &phi_obj, &pixnum_obj)) {
        return NULL;
    }

    num=PyArray_SIZE(theta_obj);

    for (i=0; i<num; i++) {
        theta = *(double *) PyArray_GETPTR1(theta_obj, i);
        phi = *(double *) PyArray_GETPTR1(phi_obj, i);
        pix_ptr = (int64_t *) PyArray_GETPTR1(pixnum_obj, i);
        if (self->scheme == HPX_RING) {
            (*pix_ptr) = ang2pix_ring(self, theta, phi);
        } else {
            (*pix_ptr) = ang2pix_nest(self, theta, phi);
        }
    }

    Py_RETURN_NONE;
}

/*
   convert the input ra,dec arrays to pixel numbers in the ring scheme.  no error checking
   done here

   ra,dec should be double arrays
   pixnum is int64 array
*/
static PyObject*
PyHealPix_fill_eq2pix(struct PyHealPix* self, PyObject* args)
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

        if (self->scheme == HPX_RING) {
            (*pix_ptr) = eq2pix_ring(self, ra, dec);
        } else {
            (*pix_ptr) = eq2pix_nest(self, ra, dec);
        }
    }

    Py_RETURN_NONE;
}

/*

   convert the pixnums to the nominal pixel center in angular theta,phi
   coordinates

   noerror checking done here

   pixnum is int64 array
   theta,phi should be double arrays
*/
static PyObject*
PyHealPix_fill_pix2ang(struct PyHealPix* self, PyObject* args)
{
    PyObject* pixnum_obj=NULL;
    PyObject* theta_obj=NULL;
    PyObject* phi_obj=NULL;

    double pixnum;
    double *theta_ptr=NULL, *phi_ptr=NULL;
    npy_intp i=0, num=0;

    if (!PyArg_ParseTuple(args, (char*)"OOO", 
                          &pixnum_obj, &theta_obj, &phi_obj)) {
        return NULL;
    }

    num=PyArray_SIZE(theta_obj);

    for (i=0; i<num; i++) {
        pixnum = *(int64_t *) PyArray_GETPTR1(pixnum_obj, i);
        theta_ptr = (double *) PyArray_GETPTR1(theta_obj, i);
        phi_ptr = (double *) PyArray_GETPTR1(phi_obj, i);
        if (self->scheme == HPX_RING) {
            pix2ang_ring(self, pixnum, theta_ptr, phi_ptr);
        } else {
            pix2ang_nest(self, pixnum, theta_ptr, phi_ptr);
        }
    }

    Py_RETURN_NONE;
}

/*

   convert the pixnums to the nominal pixel center in equatorial
    ra,dec coordinates in degrees

   noerror checking done here

   pixnum is int64 array
   ra,dec should be double arrays
*/
static PyObject*
PyHealPix_fill_pix2eq(struct PyHealPix* self, PyObject* args)
{
    PyObject* pixnum_obj=NULL;
    PyObject* ra_obj=NULL;
    PyObject* dec_obj=NULL;

    double pixnum;
    double *ra_ptr=NULL, *dec_ptr=NULL;
    npy_intp i=0, num=0;

    if (!PyArg_ParseTuple(args, (char*)"OOO", 
                          &pixnum_obj, &ra_obj, &dec_obj)) {
        return NULL;
    }

    num=PyArray_SIZE(ra_obj);

    for (i=0; i<num; i++) {
        pixnum = *(int64_t *) PyArray_GETPTR1(pixnum_obj, i);
        ra_ptr = (double *) PyArray_GETPTR1(ra_obj, i);
        dec_ptr = (double *) PyArray_GETPTR1(dec_obj, i);
        if (self->scheme == HPX_RING) {
            pix2eq_ring(self, pixnum, ra_ptr, dec_ptr);
        } else {
            printf("nest is not yet supported!\n");
            pix2eq_nest(self, pixnum, ra_ptr, dec_ptr);
        }
    }

    Py_RETURN_NONE;
}



/*

   query_disc

   If inclusive==0, find the list of pixels whose centers are contained within
   the disc

   If inclusive==1, find the list of pixels whose centers are contained within
   the disc or if the pixel intersects the disc

   If system==HPX_SYSTEM_ANG then the input is
       theta,phi,radius_radians all in radians
   if system==HPX_SYSTEM_EQ then the input is
       ra,dec,radius_degrees all in degrees
*/
/*
static PyObject*
PyHealPix_query_disc(struct PyHealPix* self, PyObject* args)
{
    int system, inclusive;
    double ang1, ang2, radius;
    double x, y, z;
    struct i64stack* listpix=NULL;

    // output
    PyObject* pixnum_obj=NULL;
    int64_t *pixnum_ptr;
    int64_t i;

    if (!PyArg_ParseTuple(args, (char*)"dddii", 
                          &ang1, &ang2, &radius, &system,&inclusive)) {
        return NULL;
    }

    if (system==HPX_SYSTEM_ANG) {
        ang2xyz(ang1, ang2, &x, &y, &z);
    } else if (system==HPX_SYSTEM_EQ) {
        eq2xyz(ang1, ang2, &x, &y, &z);
        radius *= HPX_D2R;
    } else {
        PyErr_Format(PyExc_ValueError, 
                     "system should be %d or %d, got %d\n",
                     HPX_SYSTEM_ANG, HPX_SYSTEM_EQ, system);
        return NULL;
    }

    listpix=i64stack_new(0);
    if (self->scheme==HPX_RING) {
        query_disc_xyz(self,
                       x, y, z,
                       radius,
                       inclusive,
                       listpix);
    } else {
        query_disc_nest(self,
                        x, y, z, 
                        radius,
                        inclusive,
                        listpix);
    }

    pixnum_obj = make_i64_array(listpix->size, "pixnum", &pixnum_ptr);
    if (pixnum_obj != NULL) {
        for (i=0; i<listpix->size; i++) {
            pixnum_ptr[i] = listpix->data[i];
        }
    }

    listpix = i64stack_delete(listpix);

    return pixnum_obj;
}
*/


static PyMethodDef PyHealPix_methods[] = {

    {"get_scheme", (PyCFunction)PyHealPix_get_scheme, METH_VARARGS, "get scheme name\n"},
    {"get_scheme_num", (PyCFunction)PyHealPix_get_scheme_num, METH_VARARGS, "get scheme number\n"},
    {"is_nested", (PyCFunction)PyHealPix_is_nested, METH_VARARGS, "1 if is nested, else 0\n"},

    {"get_nside", (PyCFunction)PyHealPix_get_nside, METH_VARARGS, "get nside\n"},
    {"get_npix", (PyCFunction)PyHealPix_get_npix, METH_VARARGS, "get the number of pixels at this resolution\n"},
    {"get_ncap", (PyCFunction)PyHealPix_get_ncap, METH_VARARGS, "get the number of pixels in the north polar cap at this resolution\n"},
    {"get_area", (PyCFunction)PyHealPix_get_area, METH_VARARGS, "get the area of a pixel at this resolution\n"},

    /*
    {"_query_disc", (PyCFunction)PyHealPix_query_disc, METH_VARARGS, 
        "Find the list of pixels whose centers are contained within or intersect the disc.\n"},
        */

    {"_fill_ang2pix", (PyCFunction)PyHealPix_fill_ang2pix, METH_VARARGS, "convert theta,phi radians to pixel number.  Don't call this method directly, since no error or type checking is performed\n"},
    {"_fill_eq2pix", (PyCFunction)PyHealPix_fill_eq2pix, METH_VARARGS, "convert ra,dec degrees to pixel number.  Don't call this method directly, since no error or type checking is performed\n"},

    {"_fill_pix2ang", (PyCFunction)PyHealPix_fill_pix2ang, METH_VARARGS, "convert pixel number to angular theta,phi radians.  Don't call this method directly, since no error or type checking is performed\n"},
    {"_fill_pix2eq", (PyCFunction)PyHealPix_fill_pix2eq, METH_VARARGS, "convert pixel number to equatorial ra,dec degrees.  Don't call this method directly, since no error or type checking is performed\n"},


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



/*
   stand alone methods
*/

PyObject *PyHealPix_nside_is_ok(PyObject *self, PyObject *args)
{
    long nside;
    if (!PyArg_ParseTuple(args, (char*)"l", &nside)) {
        return NULL;
    }
    return Py_BuildValue("i", nside_is_ok(nside));
}
PyObject *PyHealPix_npix_is_ok(PyObject *self, PyObject *args)
{
    long npix;
    if (!PyArg_ParseTuple(args, (char*)"l", &npix)) {
        return NULL;
    }
    return Py_BuildValue("i", npix_is_ok(npix));
}


PyObject *PyHealPix_nside2npix(PyObject *self, PyObject *args)
{
    long nside, npix;
    if (!PyArg_ParseTuple(args, (char*)"l", &nside)) {
        return NULL;
    }

    if (!nside_is_ok(nside)) {
        PyErr_Format(PyExc_ValueError, "bad nside: %ld", nside);
        return NULL;
    }
    npix = nside2npix(nside);
    return Py_BuildValue("l", npix);
}
PyObject *PyHealPix_npix2nside(PyObject *self, PyObject *args)
{
    long nside, npix;
    if (!PyArg_ParseTuple(args, (char*)"l", &npix)) {
        return NULL;
    }

    if (!npix_is_ok(npix)) {
        PyErr_Format(PyExc_ValueError, "bad npix: %ld", npix);
        return NULL;
    }
    nside = npix2nside(npix);
    return Py_BuildValue("l", nside);
}


/*
   convert equatorial ra,dec degrees to angular theta,phi radians.
   
   No error checking here
*/
PyObject* PyHealPix_fill_eq2ang(PyObject *self, PyObject *args)
{
    PyObject* ra_obj=NULL;
    PyObject* dec_obj=NULL;
    PyObject* theta_obj=NULL;
    PyObject* phi_obj=NULL;

    double ra, dec;
    double *theta_ptr, *phi_ptr;
    npy_intp i, num;

    if (!PyArg_ParseTuple(args, (char*)"OOOO",
                          &ra_obj, &dec_obj, &theta_obj, &phi_obj)) {
        return NULL;
    }

    num=PyArray_SIZE(ra_obj);

    for (i=0; i<num; i++) {
        ra  = *(double *) PyArray_GETPTR1(ra_obj, i);
        dec = *(double *) PyArray_GETPTR1(dec_obj, i);
        theta_ptr = (double *) PyArray_GETPTR1(theta_obj, i);
        phi_ptr = (double *) PyArray_GETPTR1(phi_obj, i);

        eq2ang(ra, dec, theta_ptr, phi_ptr);
    }

    Py_RETURN_NONE;

}

/*
   convert angular theta,phi radians to equatorial ra,dec degrees
   
   No error checking here
*/
PyObject* PyHealPix_fill_ang2eq(PyObject *self, PyObject *args)
{
    PyObject* theta_obj=NULL;
    PyObject* phi_obj=NULL;
    PyObject* ra_obj=NULL;
    PyObject* dec_obj=NULL;

    double theta, phi;
    double *ra_ptr, *dec_ptr;
    npy_intp i, num;

    if (!PyArg_ParseTuple(args, (char*)"OOOO",
                          &theta_obj, &phi_obj, &ra_obj, &dec_obj)) {
        return NULL;
    }

    num=PyArray_SIZE(ra_obj);

    for (i=0; i<num; i++) {
        theta = *(double *) PyArray_GETPTR1(theta_obj, i);
        phi = *(double *) PyArray_GETPTR1(phi_obj, i);
        ra_ptr = (double *) PyArray_GETPTR1(ra_obj, i);
        dec_ptr = (double *) PyArray_GETPTR1(dec_obj, i);

        ang2eq(theta, phi, ra_ptr, dec_ptr);
    }

    Py_RETURN_NONE;

}


/*
   convert equatorial ra,dec degrees to x,y,z
   
   No error checking here
*/
PyObject* PyHealPix_fill_eq2xyz(PyObject *self, PyObject *args)
{
    PyObject* ra_obj=NULL;
    PyObject* dec_obj=NULL;
    PyObject* x_obj=NULL;
    PyObject* y_obj=NULL;
    PyObject* z_obj=NULL;

    double ra, dec;
    double *x_ptr, *y_ptr, *z_ptr;
    npy_intp i, num;

    if (!PyArg_ParseTuple(args, (char*)"OOOOO",
                          &ra_obj, &dec_obj, &x_obj, &y_obj, &z_obj)) {
        return NULL;
    }

    num=PyArray_SIZE(ra_obj);

    for (i=0; i<num; i++) {
        ra  = *(double *) PyArray_GETPTR1(ra_obj, i);
        dec = *(double *) PyArray_GETPTR1(dec_obj, i);
        x_ptr = (double *) PyArray_GETPTR1(x_obj, i);
        y_ptr = (double *) PyArray_GETPTR1(y_obj, i);
        z_ptr = (double *) PyArray_GETPTR1(z_obj, i);

        eq2xyz(ra, dec, x_ptr, y_ptr, z_ptr);
    }

    Py_RETURN_NONE;

}

/*
   convert angular theta,phi radians to x,y,z
   
   No error checking here
*/
PyObject* PyHealPix_fill_ang2xyz(PyObject *self, PyObject *args)
{
    PyObject* theta_obj=NULL;
    PyObject* phi_obj=NULL;
    PyObject* x_obj=NULL;
    PyObject* y_obj=NULL;
    PyObject* z_obj=NULL;

    double theta, phi;
    double *x_ptr, *y_ptr, *z_ptr;
    npy_intp i, num;

    if (!PyArg_ParseTuple(args, (char*)"OOOOO",
                          &theta_obj, &phi_obj, &x_obj, &y_obj, &z_obj)) {
        return NULL;
    }

    num=PyArray_SIZE(theta_obj);

    for (i=0; i<num; i++) {
        theta  = *(double *) PyArray_GETPTR1(theta_obj, i);
        phi = *(double *) PyArray_GETPTR1(phi_obj, i);
        x_ptr = (double *) PyArray_GETPTR1(x_obj, i);
        y_ptr = (double *) PyArray_GETPTR1(y_obj, i);
        z_ptr = (double *) PyArray_GETPTR1(z_obj, i);

        ang2xyz(theta, phi, x_ptr, y_ptr, z_ptr);
    }

    Py_RETURN_NONE;

}

/*
   convert nested pixnums to ring scheme pixnums
   
   No error checking here
*/
/*
PyObject* PyHealPix_fill_nest2ring(PyObject* self, PyObject *args)
{
    PyObject* ipnest_obj=NULL;
    PyObject* ipring_obj=NULL;

    long nside;
    int64_t *ipnest_ptr, *ipring_ptr;

    npy_intp num;
    if (!PyArg_ParseTuple(args, (char*)"lOO",
                          &nside,&ipnest_obj, &ipring_obj)) {
        return NULL;
    }

    num=PyArray_SIZE(ipnest_obj);

    ipnest_ptr = (int64_t *) PyArray_DATA(ipnest_obj);
    ipring_ptr = (int64_t *) PyArray_DATA(ipring_obj);

    nest2ring_arr(nside, ipnest_ptr, ipring_ptr, num);

    Py_RETURN_NONE;
}
*/

/*
   convert ring scheme to nested
   
   No error checking here
*/
/*
PyObject* PyHealPix_fill_ring2nest(PyObject* self, PyObject *args)
{
    PyObject* ipring_obj=NULL;
    PyObject* ipnest_obj=NULL;

    long nside;
    int64_t *ipring_ptr, *ipnest_ptr;

    npy_intp num;
    if (!PyArg_ParseTuple(args, (char*)"lOO",
                          &nside,&ipring_obj,&ipnest_obj)) {
        return NULL;
    }

    num=PyArray_SIZE(ipnest_obj);

    ipring_ptr = (int64_t *) PyArray_DATA(ipring_obj);
    ipnest_ptr = (int64_t *) PyArray_DATA(ipnest_obj);

    ring2nest_arr(nside, ipring_ptr, ipnest_ptr, num);

    Py_RETURN_NONE;
}
*/


/*
   get position angle round the central point.  No error checking
   here
*/
PyObject* PyHealPix_fill_posangle_eq(PyObject *self, PyObject *args)
{
    double ra_cen_degrees,dec_cen_degrees;
    PyObject* ra_obj=NULL;
    PyObject* dec_obj=NULL;
    PyObject* posangle_obj=NULL;

    double ra, dec;
    double *posangle_ptr;
    npy_intp i, num;
    struct PyHealPixCap cap;

    if (!PyArg_ParseTuple(args, (char*)"ddOOO",
                          &ra_cen_degrees, &dec_cen_degrees,
                          &ra_obj, &dec_obj, &posangle_obj)) {
        return NULL;
    }

    PyHealPixCap_set(&cap, ra_cen_degrees, dec_cen_degrees);

    num=PyArray_SIZE(ra_obj);

    for (i=0; i<num; i++) {
        ra  = *(double *) PyArray_GETPTR1(ra_obj, i);
        dec = *(double *) PyArray_GETPTR1(dec_obj, i);
        posangle_ptr = (double *) PyArray_GETPTR1(posangle_obj, i);

        (*posangle_ptr) = get_posangle_eq(&cap, ra, dec);
    }

    Py_RETURN_NONE;

}

/*
   get quadrant around the central point.  No error checking
   here
*/
PyObject* PyHealPix_fill_quadrant_eq(PyObject *self, PyObject *args)
{
    double ra_cen_degrees,dec_cen_degrees;
    PyObject* ra_obj=NULL;
    PyObject* dec_obj=NULL;
    PyObject* quadrant_obj=NULL;

    double ra, dec;
    int32_t *quadrant_ptr;
    npy_intp i, num;
    struct PyHealPixCap cap;

    if (!PyArg_ParseTuple(args, (char*)"ddOOO",
                          &ra_cen_degrees, &dec_cen_degrees,
                          &ra_obj, &dec_obj, &quadrant_obj)) {
        return NULL;
    }

    PyHealPixCap_set(&cap, ra_cen_degrees, dec_cen_degrees);

    num=PyArray_SIZE(ra_obj);

    for (i=0; i<num; i++) {
        ra  = *(double *) PyArray_GETPTR1(ra_obj, i);
        dec = *(double *) PyArray_GETPTR1(dec_obj, i);
        quadrant_ptr = (int32_t *) PyArray_GETPTR1(quadrant_obj, i);

        (*quadrant_ptr) = get_quadrant_eq(&cap, ra, dec);
    }

    Py_RETURN_NONE;

}


static PyMethodDef healpix_methods[] = {

    {"nside_is_ok", (PyCFunction)PyHealPix_nside_is_ok, METH_VARARGS, 
        "determine if input nside is valid\n"
        "\n"
        "parameters\n"
        "----------\n"
        "nside: integer\n"
        "    resolution of the healpix map\n"
        "\n"
        "returns\n"
        "-------\n"
        "validity: int\n"
        "    0 if nside is invalid, nonzero otherwise\n"
        },
    {"npix_is_ok", (PyCFunction)PyHealPix_npix_is_ok, METH_VARARGS, 
        "determine if input npix is valid\n"
        "\n"
        "parameters\n"
        "----------\n"
        "npix: integer\n"
        "    number of pixels in a healpix map\n"
        "\n"
        "returns\n"
        "-------\n"
        "validity: int\n"
        "    0 if npix is invalid, nonzero otherwise\n"
        },

    {"nside2npix", (PyCFunction)PyHealPix_nside2npix, METH_VARARGS, 
        "get npix for the input nside\n"
        "\n"
        "parameters\n"
        "----------\n"
        "nside: integer\n"
        "    resolution of the healpix map\n"
        "\n"
        "returns\n"
        "-------\n"
        "npix: integer\n"
        "    number of pixels a the map with the given nside\n"
        },
    {"npix2nside", (PyCFunction)PyHealPix_npix2nside, METH_VARARGS, 
        "get nside for the given npix\n"
        "\n"
        "parameters\n"
        "----------\n"
        "npix: integer\n"
        "    number of pixels in a map\n"
        "\n"
        "returns\n"
        "-------\n"
        "nside: integer\n"
        "    nside implied by the input npix\n"
        },

    {"_fill_eq2ang", (PyCFunction)PyHealPix_fill_eq2ang, METH_VARARGS, 
        "convert ra,dec to theta,phi.  no error checking performed\n"},
    {"_fill_ang2eq", (PyCFunction)PyHealPix_fill_ang2eq, METH_VARARGS, 
        "convert theta,phi to ra,dec.  no error checking performed\n"},
    {"_fill_eq2xyz", (PyCFunction)PyHealPix_fill_eq2xyz, METH_VARARGS, 
        "convert ra,dec to x,y,z.  no error checking performed\n"},
    {"_fill_ang2xyz", (PyCFunction)PyHealPix_fill_ang2xyz, METH_VARARGS, 
        "convert theta,phi to x,y,z.  no error checking performed\n"},

    /*
    {"_fill_nest2ring", (PyCFunction)PyHealPix_fill_nest2ring, METH_VARARGS, "convert nested pixnums to ring scheme pixnums. No error checking here\n"},
    {"_fill_ring2nest", (PyCFunction)PyHealPix_fill_ring2nest, METH_VARARGS, "convert ring to nested scheme. No error checking here\n"},
    */

    {"_fill_posangle_eq", (PyCFunction)PyHealPix_fill_posangle_eq, METH_VARARGS, 
        "get position angle around the input point.  no error checking performed\n"},
    {"_fill_quadrant_eq", (PyCFunction)PyHealPix_fill_quadrant_eq, METH_VARARGS, 
        "get quadrant around the input point.  no error checking performed\n"},

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

    // variables useful in many contexts
    mk_pix2xy();
    mk_xy2pix();

    import_array();
#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
