/*

   Much of this code was adapted from various C/C++/f90/idl code distributed
   with healpix and healpy, distributed under the GPL

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
#define HPX_NEST 2

struct PyHealPix {
    PyObject_HEAD

    int scheme;  // HPX_RING or HPX_NEST
    char scheme_name[5]; // "ring" or "nest"
    int64_t nside;
    int64_t npix;
    int64_t ncap;
    double area;
};

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
   helper functions
 */
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
   ra,dec to standard x,y,z
*/
static inline void eq2xyz(double ra, double dec, double* x, double* y, double* z) {

    double theta=0, phi=0, sintheta=0;

    eq2ang(ra, dec, &theta, &phi);

    sintheta = sin(theta);
    *x = sintheta * cos(phi);
    *y = sintheta * sin(phi);
    *z = cos(theta);
}

static int
PyHealPix_init(struct PyHealPix* self, PyObject *args, PyObject *kwds)
{
    int scheme=0;
    long nside=0;
    if (!PyArg_ParseTuple(args, (char*)"il", &scheme, &nside)) {
        return -1;
    }

    if (scheme != HPX_RING && scheme != HPX_NEST) {
        PyErr_Format(PyExc_ValueError,
                     "scheme should be ring (%d) or nest (%d), got %d",
                     HPX_RING, HPX_NEST, scheme);
        return -1;
    }
    self->scheme = scheme;
    if (scheme == HPX_RING) {
        sprintf(self->scheme_name,"ring");
    } else {
        sprintf(self->scheme_name,"nest");
    }
    self->nside  = (int64_t) nside;
    self->npix   = nside2npix(nside);
    self->area   = nside2area(nside);
    self->ncap   = nside2ncap(nside);

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

/*
static inline int64_t nint64(double x) {
    if (x >= 0.0) {
        return (int64_t) (x + 0.5);
    } else {
        return (int64_t) (x - 0.5);
    }
}
*/

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

static int64_t get_ring_num(const struct PyHealPix* self, double z) {
    int64_t nside, iring;
    nside=self->nside;

    // rounds double to nearest long long int
    iring = llrintl( nside*(2.-1.5*z) );

    // north cap
    if (z > HPX_TWOTHIRD) {
        iring = llrintl( nside* sqrt(3.*(1.-z)) );
        if (iring == 0) {
            iring = 1;
        }
    } else if (z < -HPX_TWOTHIRD) {
        iring = llrintl( nside* sqrt(3.*(1.+z)) );

        if (iring == 0) {
            iring = 1;
        }
        iring = 4*nside - iring;
    }

    return iring;
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
   fill listpix with list of all pixels with centers *within* radius r
   of the point

   If you want all pixels that are within or intersect the circle at radius
   r, use intersect_disc_ring

   ra,dec - degrees
   radius - degrees
 */


static void query_disc_ring(const struct PyHealPix* self,
                            double ra,
                            double dec,
                            double radius, 
                            struct i64stack* listpix) {

    int64_t nside, irmin, irmax, iz;
    double cosang, x0, y0, z0, dth1, dth2, phi0, cosphi0, a;
    double rlat0, rlat1, rlat2, zmax, zmin, tmp;

    //double vector0[3];
    nside=self->nside;
    cosang = cos(radius);

    // this does not alter the storage
    i64stack_resize(listpix, 0);

    eq2xyz(ra, dec, &x0, &y0, &z0);

    dth1 = 1. / (3.0*nside*nside);
    dth2 = 2. / (3.0*nside);

    phi0=0.0;
    if ((x0 != 0.) || (y0 != 0.)) {
        // in (-Pi, Pi]
        phi0 = atan2(y0, x0);
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

/*
   fill listpix with list of all pixels with centers within radius r
   or intersect the circle at radius r.  If you don't want to include
   those that intersect but with center outside, use query_disc_ring

   ra,dec - degrees
   radius - degrees
 */


static void intersect_disc_ring(const struct PyHealPix* self,
                                double ra,
                                double dec,
                                double radius, 
                                struct i64stack* listpix) {
    double fudge;

    // this is from the f90 code
    // this number is acos(2/3)
    fudge = 0.84106867056793033/self->nside; // 1.071* half pixel size

    // this is from the c++ code
    //double fudge = 1.362*M_PI/(4*hpix->nside);

    radius += fudge;
    query_disc_ring(self, ra, dec, radius, listpix);

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
   getters
*/
static PyObject*
PyHealPix_get_scheme(struct PyHealPix* self, PyObject* args)
{
   return Py_BuildValue("i", self->scheme); 
}
static PyObject*
PyHealPix_get_scheme_name(struct PyHealPix* self, PyObject* args)
{
   return Py_BuildValue("s", self->scheme_name); 
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
            //(*pix_ptr) = ang2pix_nest(self, ra, dec);
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
            //(*pix_ptr) = eq2pix_nest(self, ra, dec);
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
            //pix2ang_nest(self, pixnum, theta_ptr, phi_ptr);
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
            //pix2eq_nest(self, pixnum, ra_ptr, dec_ptr);
        }
    }

    Py_RETURN_NONE;
}


// stand alone methods
static PyMethodDef PyHealPix_methods[] = {

    {"get_scheme", (PyCFunction)PyHealPix_get_scheme, METH_VARARGS, "get scheme\n"},
    {"get_scheme_name", (PyCFunction)PyHealPix_get_scheme_name, METH_VARARGS, "get scheme name\n"},

    {"get_nside", (PyCFunction)PyHealPix_get_nside, METH_VARARGS, "get nside\n"},
    {"get_npix", (PyCFunction)PyHealPix_get_npix, METH_VARARGS, "get the number of pixels at this resolution\n"},
    {"get_ncap", (PyCFunction)PyHealPix_get_ncap, METH_VARARGS, "get the number of pixels in the north polar cap at this resolution\n"},
    {"get_area", (PyCFunction)PyHealPix_get_area, METH_VARARGS, "get the area of a pixel at this resolution\n"},

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
