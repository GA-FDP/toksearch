/*
Copyright 2024 General Atomics

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
 * ell1lib.c
 *
 * ell-1 regularized weighted least-squares
 * scalar signal reconstruction/filtering.
 *
 * Uses OpenBLAS / LAPACK and CBLAS calls internally.
 *
 *
 * NOTE: nthreads=2 seems to be faster than default.
 * NOTE: profiling suggests that about ~60% is spent on factorization and 2xsolves
 * NOTE: the O(n) behaviour has been checked: ~150 points/milliseconds on typical CPU.
 *
 * TODO: can the factorizer be made quicker by simple merge of M11/M11p loops?
 * TODO: simpler/faster ADMM version possible? closed-form for optimal rho?
 *
 */


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MEMALLOC malloc
#define MEMFREE free
#define THEPRINTF printf

/* OpenBLAS includes */
//#include "common.h"
#include "cblas.h"

#include "ell1.h"


/* Print array for checking during development */
/*void auxPrintArray(
  const double *A, int m, int n)
{
  if (A==NULL) {
    THEPRINTF("[null pointer; m=%i, n=%i]\n", m, n);
    return;
  }
  int rr,cc;
  for (rr=0;rr<m;rr++) {
    for (cc=0;cc<n;cc++) {
      THEPRINTF("%f ",A[cc*m+rr]);
    }
    THEPRINTF("\n");
  }
}
*/

void set_num_threads(int num_threads)
{
    openblas_set_num_threads(num_threads);
}


/* Deallocate buffers */
void releaseEll1ProgramBuffers(
  ell1ProgramData *dat,
  int verbose)
{
  if (dat->buf!=NULL) {
    MEMFREE(dat->buf);
    dat->buf = NULL;
    if (verbose>0)
      THEPRINTF("[%s]: bufsize=%i (%.2f kB)\n",
        __func__, dat->bufsize, (sizeof(double)*dat->bufsize)/1024.0);
  }
  
  if (dat->intbuf!=NULL) {
    MEMFREE(dat->intbuf);
    dat->intbuf = NULL;
    if (verbose>0)
      THEPRINTF("[%s]: intbufsize=%i (%.2f kB)\n",
        __func__, dat->intbufsize, (sizeof(int)*dat->intbufsize)/1024.0);
  }
}

/* Allocate buffers needed for length-n optimization problems and setup the
 * structurally fixed Ey representation which is a function of the 
 * boundary conditions provided by (yl,nyl) [left] and (yr,nyr) [right].
 */
int setupEll1ProgramBuffers(
  ell1ProgramData *dat,
  int n, int nyl, int nyr,
  int verbose)
{
  double *pd = NULL;
  int intbufsize = 0;
  int bufsize = 0;
  int ofs = 0;
  int ii, rr, cc;

  if (dat==NULL) return -1;  
  if (n<3 || nyl<0 || nyr<0) return -1;
  if (nyl>2 || nyr>2) return -1;
  
  dat->intbuf = NULL;
  dat->buf = NULL;
  
  /* The boundary conditions affect the required number of xi variables */
  dat->n = n;
  dat->nxi = (n-2) + nyl + nyr;
  dat->nyl = nyl;
  dat->nyr = nyr;
  
  /* magic numbers for mini-column(s) of Ey */
  pd = dat->cEyData;
  pd[0] = 1.0;
  pd[1] = -1.0;
  pd[2] = -2.0;
  pd[3] = 2.0;
  pd[4] = 1.0;
  pd[5] = -1.0;
  
  (dat->cEy).rows = 2*dat->nxi;
  (dat->cEy).cols = dat->n;
  (dat->cEy).colData = pd;
  
  intbufsize += 3*((dat->cEy).cols);
  
  /* magic numbers for mini-rows(s) of Ey */
  pd = dat->rEyData;
  pd[0] = 1.0;
  pd[1] = -2.0;
  pd[2] = 1.0;
  pd[3] = -1.0;
  pd[4] = 2.0;
  pd[5] = -1.0;
  
  (dat->rEy).rows = 2*dat->nxi;
  (dat->rEy).cols = dat->n;
  (dat->rEy).rowData = pd;
  
  intbufsize += 3*((dat->rEy).rows);
  
  /* meta-data for M21 matrix */
  (dat->cM21).rows = dat->nxi;
  (dat->cM21).cols = dat->n;
  (dat->cM21).colData = NULL; /* this is equated to dat->M21 below */
  
  /* allocate int arrays for M21 meta-data also */
  intbufsize += 3*((dat->cM21).cols);
  
  if (verbose>0) THEPRINTF("[%s]: n=%i, nxi=%i\n", __func__, dat->n, dat->nxi);
  
  dat->intbufsize = intbufsize;
  dat->intbuf = MEMALLOC(sizeof(int)*intbufsize);
  if (dat->intbuf==NULL) return -2;
  
  if (verbose>0) THEPRINTF("[%s]: intbufsize=%i\n", __func__, dat->intbufsize);
  
  ofs = 0;
  (dat->cEy).rowStart = &(dat->intbuf[ofs]); ofs += (dat->cEy).cols; /* recall .cols = n */
  (dat->cEy).numElems = &(dat->intbuf[ofs]); ofs += (dat->cEy).cols;
  (dat->cEy).idxFirst = &(dat->intbuf[ofs]); ofs += (dat->cEy).cols;
  
  (dat->rEy).colStart = &(dat->intbuf[ofs]); ofs += (dat->rEy).rows; /* recall .rows = 2*nxi */
  (dat->rEy).numElems = &(dat->intbuf[ofs]); ofs += (dat->rEy).rows;
  (dat->rEy).idxFirst = &(dat->intbuf[ofs]); ofs += (dat->rEy).rows;
  
  (dat->cM21).rowStart = &(dat->intbuf[ofs]); ofs += (dat->cM21).cols; /* recall .cols = n */
  (dat->cM21).numElems = &(dat->intbuf[ofs]); ofs += (dat->cM21).cols;
  (dat->cM21).idxFirst = &(dat->intbuf[ofs]); ofs += (dat->cM21).cols;

#ifdef __SCATTERED_ASSERTIONS__
  if (ofs!=intbufsize) {
    THEPRINTF("(ASSERTION FAIL): [%s]: intbufsize=%i != ofs=%i\n", __func__, dat->intbufsize, ofs);
    releaseEll1ProgramBuffers(dat, verbose);
    return -3;
  }
#endif
  /* The for-loops below set up the special representation of the matrix Ey */
  rr = -4 + 2*nyl; /* should be -4+2*nyl generally */
  for (cc=0;cc<n;cc++) {
    if (rr<0) {
      (dat->cEy).rowStart[cc] = 0;
      (dat->cEy).numElems[cc] = 6+rr;
      (dat->cEy).idxFirst[cc] = -rr;
    } else if (rr+6>2*(dat->nxi)) {
      (dat->cEy).rowStart[cc] = rr;
      (dat->cEy).numElems[cc] = 2*(dat->nxi)-rr;
      (dat->cEy).idxFirst[cc] = 0;
    } else {
      (dat->cEy).rowStart[cc] = rr;
      (dat->cEy).numElems[cc] = 6;
      (dat->cEy).idxFirst[cc] = 0;
    }
    rr += 2;
  }
#ifdef __SCATTERED_ASSERTIONS__
  if ( (dat->cEy).rowStart[n-1]+(dat->cEy).numElems[n-1] != 2*dat->nxi) {
    THEPRINTF("(ASSERTION FAIL): [%s]: rowstart(n)+nelms(n)=%i, 2*nxi=%i\n",
      __func__, (dat->cEy).rowStart[n-1]+(dat->cEy).numElems[n-1], 2*dat->nxi);
    releaseEll1ProgramBuffers(dat, verbose);
    return -4;
  }
#endif
  rr = 0;
  cc = -nyl;
  for (ii=0;ii<dat->nxi;ii++) { /* do rows rr and rr+1 (2 rows) */
    if (cc<0) {
      (dat->rEy).colStart[rr] = 0;
      (dat->rEy).numElems[rr] = 3+cc;
      (dat->rEy).idxFirst[rr] = -cc;
      (dat->rEy).colStart[rr+1] = 0;
      (dat->rEy).numElems[rr+1] = 3+cc;
      (dat->rEy).idxFirst[rr+1] = 3-cc;
    } else if (cc+3>n) {
      (dat->rEy).colStart[rr] = cc;
      (dat->rEy).numElems[rr] = n-cc;
      (dat->rEy).idxFirst[rr] = 0;
      (dat->rEy).colStart[rr+1] = cc;
      (dat->rEy).numElems[rr+1] = n-cc;
      (dat->rEy).idxFirst[rr+1] = 3;
    } else {
      (dat->rEy).colStart[rr] = cc;
      (dat->rEy).numElems[rr] = 3;
      (dat->rEy).idxFirst[rr] = 0;
      (dat->rEy).colStart[rr+1] = cc;
      (dat->rEy).numElems[rr+1] = 3;
      (dat->rEy).idxFirst[rr+1] = 3;
    }
    cc += 1;
    rr += 2;
  }
#ifdef __SCATTERED_ASSERTIONS__
  ii = 2*dat->nxi-1;
  if ( (dat->rEy).colStart[ii]+(dat->rEy).numElems[ii] != n) {
    THEPRINTF("(ASSERTION FAIL): [%s]: colstart(n)+nelms(n)=%i, n=%i\n",
      __func__, (dat->rEy).colStart[ii]+(dat->rEy).numElems[ii], n);
    releaseEll1ProgramBuffers(dat, verbose);
    return -5;
  }
#endif
  /* Determine how much memory is required for double-buffers */
  int nz = dat->n + dat->nxi;
  int nq = 2*dat->nxi;
  
  dat->uplo = 'L'; /* do not change ! */
  dat->ld11 = 3;   /* do not change ! */
  dat->kd11 = 2;   /* do not change ! */
  
  bufsize += 2*dat->nxi; /* rhs vector f */
  bufsize += dat->n;     /* weight vector w*/
  bufsize += nz;         /* gradient vector g */
  bufsize += nz;         /* diagonal Hessian H */
  bufsize += 3*dat->n;   /* band storage for M11 (ld11*n) */
  bufsize += 3*dat->n;   /* band storage for M21 */
  bufsize += dat->nxi;   /* diagonal of M22 */
  bufsize += dat->nxi;   /* diagonal iM22 */
  bufsize += 3*dat->n;   /* band storage for M11p (ld11*n) */
  
  /* Top off with a blob reserved for
     temporary vectors for the solver;
     create base pointer for it below */
     
  dat->scratchsize = 10*nq + 6*nz;
  bufsize += dat->scratchsize;
  
  dat->bufsize = bufsize;
  dat->buf = MEMALLOC(sizeof(double)*bufsize);
  if (dat->buf==NULL) return -6;
  
  if (verbose>0) THEPRINTF("[%s]: bufsize=%i\n", __func__, dat->bufsize);
  
  ofs = 0;
  dat->f = &(dat->buf[ofs]); ofs += 2*dat->nxi;
  dat->w = &(dat->buf[ofs]); ofs += dat->n;
  dat->g = &(dat->buf[ofs]); ofs += nz;
  dat->H = &(dat->buf[ofs]); ofs += nz;
  dat->M11 = &(dat->buf[ofs]); ofs += 3*dat->n;
  dat->M21 = &(dat->buf[ofs]); ofs += 3*dat->n;
  dat->M22 = &(dat->buf[ofs]); ofs += dat->nxi;
  dat->iM22 = &(dat->buf[ofs]); ofs += dat->nxi;
  dat->M11p = &(dat->buf[ofs]); ofs += 3*dat->n;
  
  dat->scratch = &(dat->buf[ofs]); ofs += dat->scratchsize;
  
#ifdef __SCATTERED_ASSERTIONS__
  if (ofs!=bufsize) {
    THEPRINTF("(ASSERTION FAIL): [%s]: bufsize=%i != ofs=%i\n", __func__, dat->bufsize, ofs);
    releaseEll1ProgramBuffers(dat, verbose);
    return -7;
  }
#endif
  
  (dat->cM21).colData = dat->M21; /* assign correctly (was NULL above) */
  
  /* Assemble basic meta-data for cM21 striped structure */
  rr = dat->nyl-2;
  for (cc=0;cc<dat->n;cc++) {
    if (rr<0) {
      (dat->cM21).rowStart[cc] = 0;
      (dat->cM21).numElems[cc] = 3+rr;
      (dat->cM21).idxFirst[cc] = 3*cc-rr;
    } else if (rr+3>dat->nxi) {
      (dat->cM21).rowStart[cc] = rr;
      (dat->cM21).numElems[cc] = dat->nxi-rr;
      (dat->cM21).idxFirst[cc] = 3*cc;
    } else {
      (dat->cM21).rowStart[cc] = rr;
      (dat->cM21).numElems[cc] = 3;
      (dat->cM21).idxFirst[cc] = 3*cc;
    }
    rr+=1;
  }
  
  /* Setup default unity sample weights */
  for (ii=0;ii<dat->n;ii++) dat->w[ii] = 1.0;
  
  /* Zero the RHS vector f; it is only nonzero for specific b.c:s */
  /* Actual setup of (edges of) f is done per instance in the solver routine; if nyl+nyr>0 */
  memset(dat->f, 0, sizeof(double)*2*(dat->nxi));
  memset(dat->M11, 0, sizeof(double)*(dat->ld11)*dat->n);  /* 3*dat->n */
  memset(dat->M11p, 0, sizeof(double)*(dat->ld11)*dat->n); /* 3*dat->n */
  memset(dat->M22, 0, sizeof(double)*dat->n);
  memset(dat->iM22, 0, sizeof(double)*dat->n);
  memset(dat->M21, 0, sizeof(double)*3*dat->n);
  
  return 0;
}

#ifdef __OUTPUT_TEST_EY_MATRICES__
/* X <- alpha*A + X, with rowStripe A */
void stripe_dxpra(
  const rowStripeMatrix *A, double alpha, double *X)
{
  int rr, cc, ii, nn;
  double *d = A->rowData;
  for (rr=0;rr<A->rows;rr++) {
    nn = A->numElems[rr];
    ii = A->idxFirst[rr];
    cc = A->colStart[rr];
    while (nn--)
      X[rr+(cc++)*(A->rows)] += alpha*d[ii++];
  }
}

/* X <- alpha*A + X, with colStripe A */
void stripe_dxpca(
  const colStripeMatrix *A, double alpha, double *X)
{
  int rr, cc, ii, nn;
  double *d = A->colData;
  for (cc=0;cc<A->cols;cc++) {
    nn = A->numElems[cc];
    ii = A->idxFirst[cc];
    rr = A->rowStart[cc];
    rr += cc*(A->rows);
    while (nn--)
      X[rr++] += alpha*d[ii++];
  }
}

void make_dense_exi(double *M, int n) {
  int ii;
  for (ii=0;ii<n;ii++) {
    M[2*ii+2*n*ii]=-1.0;
    M[2*ii+1+2*n*ii]=-1.0;
  }
}
#endif

#if defined(__OUTPUT_TEST_EY_MATRICES__) || defined(__RUN_BANDED_CHOLESKY_TEST__)
void make_runif(double *x, int n, double a, double b) {
  int ii;
  for (ii=0;ii<n;ii++)
    x[ii] = a+(b-a)*genrand_real3();
}
#endif

/*
 * Specially structured matrix-vector operations
 * needed in PDIPM algorithm loop
 */

/* H is a diagonal n-by-n matrix
 * (represented by n-vector): do y <- H*x + z
 */
/* NOTE: only first dat->n elements of H are nonzero if used correctly */
/* H[ii]==0 for ii=n..n+nxi-1 */
/*static inline void datHmultxplusz(
  const ell1ProgramData *dat,
  double *y, double *x, double *z)
{
  int ii;
  double *H = dat->H;
  int n = dat->n+dat->nxi;
  memcpy(y, z, sizeof(double)*n);
  for (ii=0;ii<dat->n;ii++)
    y[ii] += H[ii]*x[ii];
}
*/

/* y <- y + (H*x+h) exploiting that H is zero on diagonal for "last half" */
static inline void addGradient(
  const ell1ProgramData *dat,
  double *y, double *x)
{
  int ii;
  double *H = dat->H;
  double *h = dat->g;
  int n = dat->n+dat->nxi;
  for (ii=0;ii<dat->n;ii++) y[ii] += H[ii]*x[ii]+h[ii];
  for (ii=dat->n;ii<n;ii++) y[ii] += h[ii];
}

/* Here E = [Ey, Exi] has a special banded structure;
 * its representation is pre-initialized in the dat structure.
 * y <- E*x + z, achieve by: y <- Ey*x1 + z; y <- Exi*x2 + y
 * where x = [x1; x2]
 */
static inline void datEmultxplusz(
  const ell1ProgramData *dat,
  double *y, double *x, double *z)
{
  int q = 2*dat->nxi;
  int n = dat->n;
  double *x1 = x;
  double *x2 = &x[n];
  const rowStripeMatrix *rEy = &(dat->rEy);
  double *rey = rEy->rowData;
  int *rEyc = rEy->colStart;
  int *rEyn = rEy->numElems;
  int *rEyi = rEy->idxFirst;
  int ii,jj,ci,ni,xi;
  double e;
  for (ii=0;ii<q;ii++) {
    ci = rEyc[ii];
    ni = rEyn[ii];
    xi = rEyi[ii];
    e = z[ii];
    while (ni--) e += rey[xi++]*x1[ci++];
    y[ii] = e;
  }
  /* Do y <- Exi*x2 + y */
  for (jj=0,ii=0;jj<(q>>1);jj++,ii+=2) {
    y[ii] -= x2[jj];
    y[ii+1] -= x2[jj];
  }
}

/* y <- E'*x, E structured [Ey, Exi]
 * Split up into y = [y1; y2] and do y1 <- Ey'*x, y2 <- Exi'*x
 */
static inline void datEtmultx(
  const ell1ProgramData *dat,
  double *y, double *x)
{
  int q = 2*dat->nxi;
  int n = dat->n;
  double *y1 = y;
  double *y2 = &y[n];
  const colStripeMatrix *cEy = &(dat->cEy);
  double *cey = cEy->colData;
  int *cEyr = cEy->rowStart;
  int *cEyn = cEy->numElems;
  int *cEyi = cEy->idxFirst;
  int ii,jj,ci,ni,xi;
  double e;
  for (ii=0;ii<n;ii++) {
    ci = cEyr[ii];
    ni = cEyn[ii];
    xi = cEyi[ii];
    /*e = 0.0;
    while (ni--) e += cey[xi++]*x[ci++];*/
    e = cey[xi++]*x[ci++];
    while (--ni) e += cey[xi++]*x[ci++];
    y1[ii] = e;
  }
  for (ii=0,jj=0;ii<(q>>1);ii++,jj+=2) {
    y2[ii] = -x[jj]-x[jj+1];
  }
}

/* 
 * Main linear equation factorization step.
 *
 * Create  pos. def. factorization of the Schur complement
 * matrix M11p = M11 - M21'*inv(M22)*M21, which is symmetric
 * and pentadiagonal of size n.
 *
 * The quantities ( D=diag(d) )
 *   M11 = H + Ey'*D*Ey
 *   M21 = Exi'*D*Ey
 *   M22 = Exi'*D*Exi
 *
 * depends on d and are also stored in the ell1 struct
 * for later usage (needed by companion solver routine).
 *
 * TODO: attempt to make fewer passes over the arrays
 *       - merge loops
 *       - unroll loops mostly possible but boundaries are annoying
 */

/* These are declared here but the linker
 * needs to actually find them in LAPACK */

/*#define DPBTRF BLASFUNC(dpbtrf)*/
#define DPBTRF dpbtrf_
void dpbtrf_(
  char* uplo, int* n, int* kd,
  double* ab, int* ldab, int *info
);
/*#define DPBTRS BLASFUNC(dpbtrs)*/
#define DPBTRS dpbtrs_
void dpbtrs_(
  char* uplo, int* n, int* kd,
  int* nrhs, const double* ab, int* ldab,
  double* b, int* ldb, int *info
);
 
static int factorizeHpEtDE(ell1ProgramData *dat, double *d)
{
  if (dat==NULL) return -1;
  if (d==NULL) return -2;
  
  colStripeMatrix *cEy = &(dat->cEy);
  double *cey = cEy->colData;
  int *cEyn = cEy->numElems;
  int *cEyi = cEy->idxFirst;
  int *cEyr = cEy->rowStart;
  colStripeMatrix *cM21 = &(dat->cM21);
  int *cM21n = cM21->numElems;
  int *cM21i = cM21->idxFirst;
  int *cM21r = cM21->rowStart;
  double *m11 = dat->M11;
  double *m11p = dat->M11p;
  double *m21 = dat->M21;
  double *m22 = dat->M22;
  double *im22 = dat->iM22;
  int n = dat->n;
  int nxi = dat->nxi;
  int ii,jj,rr,kk;
  int ri,rj;
  int ni,nj;
  int xi,xj;
  double e;

  /* d has length 2*nxi (interpreted as diagonal matrix) */
  for (ii=0,rr=0;ii<nxi;ii++,rr+=2) {
    m22[ii] = d[rr]+d[rr+1]; /* yes M22 is this trivial */
    im22[ii] = 1.0/m22[ii];
  }
  
  /* Generate LAPACK band-storage buffer for M21
     (non-sym. tridiagonal) */
  rr = 2-dat->nyl; /* use as row offset for storage buffer -> rr+(ii-jj) in set {0,1,2} */
  for (jj=0;jj<n;jj++) {
    /* Need to evaluate three different diagonals;
     * can be either super-diagonals,
     * sub-diagonals, or both, but always nonsymmetric;
     * nyl=0 -> diagonal deltas: {-2,-1,0} (super)
     * nyl=1 -> deltas = {-1,0,1} (both)
     * nyl=2 -> deltas = {0,1,2} (sub)
     */
    nj = cEyn[jj];
    xj = cEyi[jj];
    rj = cEyr[jj];
    for (ii=(rj>>1);ii<=((rj+nj-2)>>1);ii++) {
      ri = ii<<1;
      kk = xj+(ri-rj);
      e = -d[ri]*cey[kk]-d[ri+1]*cey[kk+1];
      m21[3*jj+(rr+ii-jj)] = e;
    }
  }
  
  /* Generate LAPACK band-storage buffer for M11
     (sym. pentadiagonal; do not bother to assign upper elements) */
  for (jj=0;jj<n;jj++) {
    /* diagonal element (jj,jj) */
    nj = cEyn[jj];
    xj = cEyi[jj];
    rj = cEyr[jj];
    for (rr=rj,kk=0,e=dat->H[jj];kk<nj;kk++,xj++) {
      e += d[rr++]*cey[xj]*cey[xj];
    }
    m11[3*jj] = e;
  }
  
  for (jj=0;jj<n-1;jj++) {
    /* first subdiagonal (jj+1,jj) */
    nj = cEyn[jj];
    xj = cEyi[jj];
    rj = cEyr[jj];
    ni = cEyn[jj+1];
    xi = cEyi[jj+1];
    ri = cEyr[jj+1];
    xj += (ri-rj);
    for (rr=ri,e=0.0;rr<rj+nj;rr++) {
      e += d[rr]*cey[xi++]*cey[xj++];
    }
    m11[3*jj+1] = e;
  }
  
  for (jj=0;jj<n-2;jj++) {
    /* second subdiagonal (jj+2,jj) */
    nj = cEyn[jj];
    xj = cEyi[jj];
    rj = cEyr[jj];
    ni = cEyn[jj+2];
    xi = cEyi[jj+2];
    ri = cEyr[jj+2];
    xj += (ri-rj);
    for (rr=ri,e=0.0;rr<rj+nj;rr++) {
      e += d[rr]*cey[xi++]*cey[xj++];
    }
    m11[3*jj+2] = e;
  }
   
  memcpy(dat->M11p, m11, sizeof(double)*(dat->ld11)*dat->n); /* 3*dat->n */
  
  /* Generate LAPACK banded-storage for M11p = M11 - M21'*inv(M22)*M21
     (sym. pentadiagonal; only compute lower part */
  for (jj=0;jj<n;jj++) {
    rj = cM21r[jj];
    nj = cM21n[jj];
    xj = cM21i[jj];
    e = m11p[3*jj];
    for (rr=rj,kk=0;kk<nj;kk++,xj++) e -= im22[rr++]*m21[xj]*m21[xj];
    m11p[3*jj] = e;
  }
  
  /* downdate subdiagonal 1 */
  for (jj=0;jj<n-1;jj++) {
    rj = cM21r[jj];
    nj = cM21n[jj];
    xj = cM21i[jj];
    ri = cM21r[jj+1];
    ni = cM21n[jj+1];
    xi = cM21i[jj+1];
    xj += (ri-rj);
    e = m11p[3*jj+1];
    for (rr=ri;rr<rj+nj;rr++) e -= im22[rr]*m21[xi++]*m21[xj++];
    m11p[3*jj+1] = e;
  }
  
  /* downdate subdiagonal 2 */
  for (jj=0;jj<n-2;jj++) {
    rj = cM21r[jj];
    nj = cM21n[jj];
    xj = cM21i[jj];
    ri = cM21r[jj+2];
    ni = cM21n[jj+2];
    xi = cM21i[jj+2];
    xj += (ri-rj);
    e = m11p[3*jj+2];
    for (rr=ri;rr<rj+nj;rr++) e -= im22[rr]*m21[xi++]*m21[xj++];
    m11p[3*jj+2] = e;
  }
    
  /* in-place factorize M11p (banded Cholesky) */  
  int info = -1;
  
  DPBTRF(&(dat->uplo), &n, &(dat->kd11), dat->M11p, &(dat->ld11), &info);
  
  /* info=0 if OK, -i if param i had illegal value, i if not pos.def matrix */
  return info;
}

/* 
 * Solve the block equation for a new RHS b (n+nxi)-vector.
 *
 * M11 *x1 + M21'*x2 = b1
 * M21 *x1 + M22 *x2 = b2
 *
 * b = [b1;b2], and x=[x1;x2], x1,b1 lengths n; x2,b2 lengths nxi
 *
 * Solution x is overwritten to b.
 * Exploits the prefactorized banded Schur complement
 * from the companion routine above.
 *
 */
static int solveFactorizedEq(const ell1ProgramData *dat, double *b) {

  /* 1. replace b1: b1 <- b1 - M21'*inv(M22)*b2
   * 2. in-place solve M11p*x1 = b1 (so that b1 <- x1 = solution )
   * 3. replace b2 <- inv(M22)*(b2-M21*x1) = inv(M22)*(b2-M21*b1)
   *
   * TODO: optional iterative refinement (1-step)
   *
   */
   
  int n = dat->n;
  int nxi = dat->nxi;
  const colStripeMatrix *cM21 = &(dat->cM21);
  int *cM21r = cM21->rowStart;
  int *cM21n = cM21->numElems;
  int *cM21i = cM21->idxFirst;
  double *u1 = b;
  double *u2 = &b[n];
  double *m21 = dat->M21;
  double *im22 = dat->iM22;
  double e;
  int ii,ri,ni,xi,rr;
  
  /* step 1: edit u1 <- u1 - M21'*(im22.*u2) */
  for (ii=0;ii<n;ii++) {
    ri = cM21r[ii];
    ni = cM21n[ii];
    xi = cM21i[ii];
    for (e=0.0,rr=ri;rr<ri+ni;rr++) {
      e += m21[xi++]*(im22[rr]*u2[rr]);
    }
    u1[ii] -= e;
  }
    
  /* step 2: solve (L*L')*x1 = u1, inplace: u1 <- x1 */
  int nhrs = 1;
  int info = -1;
  
  /* NOTE: casting to remove warnings due to const *dat */
  DPBTRS(
    (char *)&(dat->uplo), &n, (int *)&(dat->kd11),
    &nhrs, dat->M11p, (int *)&(dat->ld11), u1, &n, &info);
  
  if (info!=0) return info;
  
  /* step 3: edit u2 : u2 <- im22.*(u2-M21*u1); call GBMV */
  
  /* M21 is nxi-by-n, kl = dat->nyl, ku = 2-dat->nyl */
  cblas_dgbmv(
    CblasColMajor, CblasNoTrans,
    nxi, n, dat->nyl, 2-dat->nyl,
    -1.0, m21, 3,
    u1, 1,
    1.0, u2, 1);
  
  for (ii=0;ii<nxi;ii++) u2[ii] *= im22[ii];

  return 0;
}

#ifdef __RUN_BANDED_CHOLESKY_TEST__
double maxInfDiff(double *x, double *y, int n) {
  double dii, s=0.0;
  int ii;
  for (ii=0;ii<n;ii++) {
    dii = fabs(x[ii]-y[ii]);
    if (dii>s) s=dii;
  }
  return s;
}

void AugmentedMultTest(ell1ProgramData *dat, double *x, double *y)
{
  /* y <- M*x, with M = [M11,M12;M21,M22], M symmetric; M12=M21' */
  if (dat==NULL || x==NULL || y==NULL) return;
  
  int ii;
  int n = dat->n;
  int nxi = dat->nxi;
  double *x1 = x;
  double *x2 = &x[n];
  double *y1 = y;
  double *y2 = &y[n];
  double *m22 = dat->M22;
  
  /* y1 <- M11*x1, y1 <- y1 + M21'*x2 */
  cblas_dsbmv(
    CblasColMajor, CblasLower,
    n, dat->kd11,
    1.0, dat->M11, dat->ld11,
    x1, 1,
    0.0, y1, 1);
  
  cblas_dgbmv(
    CblasColMajor, CblasTrans,
    nxi, n, dat->nyl, 2-dat->nyl,
    1.0, dat->M21, 3,
    x2, 1,
    1.0, y1, 1);
  
  /* y2 <- M22*x2, y2 <- y2 + M21*x1 */
  for (ii=0;ii<nxi;ii++)
    y2[ii] = m22[ii]*x2[ii];
    
  cblas_dgbmv(
    CblasColMajor, CblasNoTrans,
    nxi, n, dat->nyl, 2-dat->nyl,
    1.0, dat->M21, 3,
    x1, 1,
    1.0, y2, 1);
}

int randomFactorizeSolveTest(
  ell1ProgramData *dat, int numf, int rhsperf)
{
  THEPRINTF("[%s]: numf=%i, nrhs=%i\n", __func__, numf, rhsperf);
  
  int ff, bb;
  
  /* for 1..numf */
    /* Random H */
    /* Random d */
    /* Factorize: yields also M11, M21, M22 */
    /* for 1..rhsperf */
      /* solve eq. for x with random b */
      /* evaluate err=b-M*x ; report max(abs(err)) */
  
  double *d = dat->scratch;
  int nx = dat->n+dat->nxi;
  int nq = 2*dat->nxi;
  int info;
  
  double *x1 = &(dat->scratch[nq]);
  double *x2 = &(dat->scratch[nq+nx]);
  double *er = &(dat->scratch[nq+nx+nx]);
  
  for (ff=0;ff<numf;ff++){
    make_runif(dat->H, dat->n, 1.0, 2.0);
    make_runif(d, nq, 0.5, 1.5);
    info = factorizeHpEtDE(dat, d);
    if (info==0) {
      for (bb=0;bb<rhsperf;bb++) {
        /* random rhs; then solve, then check residual */
        make_runif(x1, nx, -1.0, 1.0);
        memcpy(x2, x1, sizeof(double)*nx);
        /* RHS x1 -> solution x1 (in-place) */
        info = solveFactorizedEq(dat, x1);
        if (info==0) {
          /* Now check that [M11,M12;M21,M22]*x1-x2=0 */
          AugmentedMultTest(dat, x1, er); /* er <- M*x1 */
          THEPRINTF("[%s]: max-abs-diff (%i:%i) = %e\n", __func__, ff, bb, maxInfDiff(er, x2, nx));
        } else {
          THEPRINTF("[%s]: failed to factorize (ff=%i:bb=%i).\n", __func__, ff, bb);
        }
      }
    } else {
      THEPRINTF("[%s]: failed to factorize (ff=%i).\n", __func__, ff);
    }
  }
      
  return 0;
}
#endif

/*
 * Basic utility functions needed in main PDIPM loop
 */

static inline double __dmaxFromPair(double a, double b) {
  if (a>=b) return a;
    else return b;
}

static inline double norminf(double *x, int n) {
  int i; double s=0.0; double axi;
  for (i=0;i<n;i++) {
    axi = fabs(x[i]);
    if (axi>s) s=axi;
  }
  return s;
}

/* daxpy: y <- y + a*x, so a=1.0 or -1.0 for add or sub */
static inline void blas_addvec(double *y,double *x,int n,double a) {
  cblas_daxpy(n, a, x, 1, y, 1);
}

static inline void flipsign(double *y,int n) {
  int i; for (i=0;i<n;i++) y[i]=-y[i];
}

static inline double vecmean(double *x,int n) {
  int i; double s=0.0; for (i=0;i<n;i++) s+=x[i]; return (s/n);
}

static inline double alpha1(double alpha0,double *x,double *dx,int n) {
  int i; double a1=alpha0; double tmp;
  for (i=0;i<n;i++) {
    if (dx[i]<0.0) {
      tmp=-x[i]/dx[i];
      if (tmp<a1) a1=tmp;
    }
  }
  return a1;
}

static inline double mu1(double a,double *z,double *dz,double *s,double *ds,int n) {
  int i; double aa=0.0;
  for (i=0;i<n;i++)
    aa+=(z[i]+a*dz[i])*(s[i]+a*ds[i]);
  return (aa/n);
}

static inline void ewprodxy(double *z,double *x,double *y,int n) {
  int i; for (i=0;i<n;i++) z[i]=x[i]*y[i];
}

static inline void ewdivxy(double *z,double *x,double *y,int n) {
  int i; for (i=0;i<n;i++) z[i]=x[i]/y[i];
}

static inline void ewmaccxyw(double *z,double *x,double *y,double *w,int n) {
  int i; for (i=0;i<n;i++) z[i]=x[i]*y[i]+w[i];
}

static inline void ewmaccnxyw(double *z,double *x,double *y,double *w,int n) {
  int i; for (i=0;i<n;i++) z[i]=-x[i]*y[i]+w[i];
}

static inline void scmaccxyw(double *z,double x,double *y,double *w,int n) {
  int i; for (i=0;i<n;i++) z[i]=x*y[i]+w[i];
}

/* Global variable for additional diagnostics timer (ugly) */
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
static double __global_clock_0 = 0.0;
static double __global_clock_1 = 0.0;
#endif

/*
 * Solve a new instance of the problem given data xsig and (if applicable boundary terms yl,yr).
 * Observation weights are specified by w (dat->w are default unity weights).
 * NOTE: the program size is fixed by the structure dat (pre-allocated buffers).
 *
 */
int ell1ProgramSolve(
  ell1ProgramData *dat,
  double *xsig,
  double lambda,
  double *w,
  double *yl,
  double *yr,
  double eta,
  double eps,
  int maxiters,
  int initopt,
  double *y,
  double *fy,
  double *xi,
  double *fxi,
  double *f0,
  int *iters,
  int *cholerr,
  double *inftuple)
{
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
  fclk_timespec _tic1, _toc1;
  __global_clock_0 = 0.0;
  /* the above clock value should contain
   * the total time spent in "factorizeHpEtDE"
   * at exit of this routine
   */
  __global_clock_1 = 0.0; /* and this one is to time triangular solves */
#endif

  /* check for basic misuse */
  if (dat==NULL) return -1;
  if (xsig==NULL) return -1;
  if (dat->w==NULL) return -1;
  if (dat->nyl>0 && yl==NULL) return -1;
  if (dat->nyr>0 && yr==NULL) return -1;
  
  if (maxiters<=0) return -1;
  if (eta<=0.0 || eps<=0.0) return -1;
  
  /* conform with notation for general inequality QP:
   *   min 0.5*x'*H*x + h'*x, s.t. E*x<=f
   * with x an n-vector and E has q rows.
   * So here: n = size of augmented decision variable x=[y;xi]
   * which is not equal to dat->n. NOTE: n != dat->n.
   */

  int n = dat->n + dat->nxi;
  int q = 2*dat->nxi;
  
  double
    *vx, *vz, *vs, *ve,
    *vrL, *vrs, *vrsz, *vrbar, *vhbar,
    *vdx, *vds, *vdz, *vtmp1, *vtmp2, *vtmp3, *vtmp4;
  
  int idx = 0;
  double *mbuf = dat->scratch;
  
  vx=&mbuf[idx]; idx+=n;
  vz=&mbuf[idx]; idx+=q;
  vs=&mbuf[idx]; idx+=q;
  ve=&mbuf[idx]; idx+=q;
  vrL=&mbuf[idx]; idx+=n;
  vrs=&mbuf[idx]; idx+=q;
  vrsz=&mbuf[idx]; idx+=q;
  vrbar=&mbuf[idx]; idx+=n;
  vhbar=&mbuf[idx]; idx+=n;
  vdx=&mbuf[idx]; idx+=n;
  vds=&mbuf[idx]; idx+=q;
  vdz=&mbuf[idx]; idx+=q;
  vtmp1=&mbuf[idx]; idx+=n;
  vtmp2=&mbuf[idx]; idx+=q;
  vtmp3=&mbuf[idx]; idx+=q;
  vtmp4=&mbuf[idx]; idx+=q;
  
  if (idx!=dat->scratchsize) {
    THEPRINTF("[%s]: ERROR %i=idx!=scratchsize=%i\n",
      __func__, idx, dat->scratchsize);
    return -1;
  }
  
  double *f = dat->f;
  /*memset(f, 0, q*sizeof(double));*/
  
  /* ASSUMES the "body" of f was already zeroed during initialization */
  f[0] = 0.0; f[1] = 0.0; f[2] = 0.0; f[3] = 0.0;
  f[q-4] = 0.0; f[q-3] = 0.0; f[q-2] = 0.0; f[q-1] = 0.0;
  
  if (dat->nyl>0) {
    double *pd = dat->rEyData;
    /* set boundaries of f correctly given yl and Ey */
    if (dat->nyl==1) {
      f[0] = -pd[0]*yl[0];
      f[1] = -pd[3]*yl[0];
    } else {
      f[0] = -pd[0]*yl[0] - pd[1]*yl[1];
      f[1] = -pd[3]*yl[0] - pd[4]*yl[1];
      f[2] = -pd[0]*yl[1];
      f[3] = -pd[3]*yl[1];
    }
  }
  
  if (dat->nyr>0) {
    double *pd = dat->rEyData;
    /* set boundaries of f correctly given yr and Ey */
    if (dat->nyr==1) {
      f[q-2] = -pd[2]*yr[0];
      f[q-1] = -pd[5]*yr[0];
    } else {
      f[q-4] = -pd[2]*yr[0];
      f[q-3] = -pd[5]*yr[0];
      f[q-2] = -pd[1]*yr[0] - pd[2]*yr[1];
      f[q-1] = -pd[4]*yr[0] - pd[5]*yr[1];
    }
  }
  
  /* Buffer pointers are initialized: proceed with PDIPM initialization */
  
  double *h = dat->g; /* setup h as if g = H*x + h always (QP) */
  double *H = dat->H; /* same comment applies (Huberization not yet supported) */
  
  memcpy(H, w, (dat->n)*sizeof(double)); /* copy weights to Hessian elements 0..nobs-1 */
  memset(&H[dat->n], 0, (dat->nxi)*sizeof(double)); /* make sure the xi part of H is zero */
  /* setup gradient data h (signal enters here) */
  for (idx=0;idx<dat->n;idx++) h[idx] = -w[idx]*xsig[idx];
  for (idx=dat->n;idx<n;idx++) h[idx] = lambda;

  int numiters = 0;
  int cholretval = 0;
  int solvretval = 0;
  int oktostop = 0;
  
  double thrL, thrs, thrmu;
  double mu, alphaa, mua, sigma, xia;
  double etainf, beta;
  double infL, infs;
  
  double hinf = norminf(h, n);
  double finf = norminf(dat->f, q);
  
  thrL = (1.0+hinf)*eps;
  thrs = (1.0+finf)*eps;
  thrmu = eps;
  
  for (idx=0;idx<n;idx++) vx[idx] = 0.0;
  for (idx=0;idx<q;idx++) ve[idx] = 1.0;
  
  /* Determine inf-norms of input data (h,f,H,E) to heuristically set an initial point */
  if (initopt>0) {
    etainf = __dmaxFromPair(hinf, finf);
    etainf = __dmaxFromPair(etainf, norminf(H, dat->n));
    etainf = __dmaxFromPair(etainf, 2.0); /* E matrix inf norm = 2 */
    beta = sqrt(etainf);
    for (idx=0;idx<q;idx++) {
      vz[idx] = beta;
      vs[idx] = beta;
    }
  } else {
    memcpy(vz, ve, q*sizeof(double));
    memcpy(vs, ve, q*sizeof(double));
  }
  
  /* Calculate first residual */
      
  /* rL=H*x+h+E'*z; note that H*x+h is the objective gradient vector */
  
/*  datEtmultx(dat, vtmp1, vz);
  blas_addvec(vtmp1, h, n, 1.0);
  datHmultxplusz(dat, vrL, vx, vtmp1);*/
  datEtmultx(dat, vrL, vz);
  addGradient(dat, vrL, vx);
  /* rs=s+E*x-f; */
  datEmultxplusz(dat, vrs, vx, vs);
  blas_addvec(vrs, f, q, -1.0);
  /* rsz=s.*z; */
  ewprodxy(vrsz, vs, vz, q);
  /* mu = sum(rsz)/q; */
  mu = vecmean(vrsz, q);
  
  /* TODO: assorted CBLAS-based vector operations/optimizations in the main loop */
  /* TODO: ewmacc, ewprod ops with ds[g]bmv ?! */
  
  infL = norminf(vrL, n);
  infs = norminf(vrs, q);
  oktostop = (infL<thrL && infs<thrs && mu<thrmu);
    
  /* Jump into algorithm main loop */
  while ( numiters<=maxiters && !oktostop  ) {
    ewdivxy(vtmp2, vz, vs, q); /* vtmp2=vz./vs; */
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_tic1);
#endif
    cholretval = factorizeHpEtDE(dat, vtmp2);
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_toc1);
    __global_clock_0 += fclk_delta_timestamps(&_tic1, &_toc1);
#endif
    if (cholretval!=0) break;
    ewmaccnxyw(vtmp4, vtmp2, vrs, vz, q); /* vtmp4=-vtmp2.*vrs+vz */
    datEtmultx(dat, vdx, vtmp4);
    blas_addvec(vdx, vrL, n, -1.0); /* vdx = E'*vtmp4 - vrL */
    /* Backsubstitution #1 using factorization above; store in vdx, rhs in vdx also */
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_tic1);
#endif
    solvretval = solveFactorizedEq(dat, vdx);
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_toc1);
    __global_clock_1 += fclk_delta_timestamps(&_tic1, &_toc1);
#endif
    if (solvretval!=0) break;
    datEmultxplusz(dat, vds, vdx, vrs);
    flipsign(vds, q);  /* vds=-(E*vdx+vrs); */ 
    ewmaccxyw(vdz, vtmp2, vds, vz, q);
    flipsign(vdz, q); /* vdz=-(z+vtmp2.*vds); */
    alphaa = 1.0;
    alphaa = alpha1(alphaa, vz, vdz, q);
    alphaa = alpha1(alphaa, vs, vds, q);
    mua = mu1(alphaa, vz, vdz, vs, vds, q);
    sigma = mua/mu;
    sigma *= sigma*sigma; /* sigma=(mua/mu)^3; */
    /* Update rhs for backsubstitution #2 */
    ewmaccxyw(vrsz, vds, vdz, vrsz, q);
    scmaccxyw(vrsz, -sigma*mu, ve, vrsz, q); /* vrsz=vrsz+ds.*dz-sigma*mu*ve; */
    ewdivxy(vtmp4, vrsz, vs, q);
    ewmaccnxyw(vtmp4, vtmp2, vrs, vtmp4, q); /* vtmp4=vrsz./vs-vtmp2.*vrs; */
    datEtmultx(dat, vdx, vtmp4);
    blas_addvec(vdx, vrL, n, -1.0); /* vdx = E'*vtmp4 - vrL */
    /* Backsubstitution #2 */
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_tic1);
#endif
    solvretval = solveFactorizedEq(dat, vdx);
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_toc1);
    __global_clock_1 += fclk_delta_timestamps(&_tic1, &_toc1);
#endif
    if (solvretval!=0) break;
    datEmultxplusz(dat, vds, vdx, vrs);
    flipsign(vds, q);  /* vds=-(E*vdx+vrs); */ 
    ewdivxy(vtmp4, vrsz, vs, q);
    ewmaccxyw(vdz, vtmp2, vds, vtmp4, q);
    flipsign(vdz, q); /* vdz=-(vrsz./vs+vtmp2.*vds); */
    /* Adjust search direction and take a step */
    alphaa = 1.0;
    alphaa = alpha1(alphaa, vz, vdz, q);
    alphaa = alpha1(alphaa, vs, vds, q);
    xia = alphaa*eta;
    blas_addvec(vx, vdx, n, xia);
    blas_addvec(vz, vdz, q, xia);
    blas_addvec(vs, vds, q, xia);
    /* Recalculate residuals */
    /*datEtmultx(dat, vtmp1, vz);
    blas_addvec(vtmp1, h, n, 1.0);
    datHmultxplusz(dat, vrL, vx, vtmp1);*/ /* rL=H*x+h+E'*z; */
    datEtmultx(dat, vrL, vz);
    addGradient(dat, vrL, vx);
    datEmultxplusz(dat, vrs, vx, vs);
    blas_addvec(vrs, f, q, -1.0); /* rs=s+E*x-f; */
    ewprodxy(vrsz, vs, vz, q);
    mu = vecmean(vrsz, q); /* rsz=s.*z; mu = sum(rsz)/q; */
    /* check if converged */
    infL = norminf(vrL, n);
    infs = norminf(vrs, q);
    oktostop = (infL<thrL && infs<thrs && mu<thrmu);
    numiters++;  /* Log the iteration and iterate again */
  }
    
  /* Return final vector, objective value, iteration count, and clean up */
  if (y!=NULL) memcpy(y, vx, (dat->n)*sizeof(double));
  if (xi!=NULL) memcpy(xi, &vx[dat->n], (dat->nxi)*sizeof(double));
  
  if (fy!=NULL) {
    double su=0.0;
    for (idx=0;idx<dat->n;idx++) su += w[idx]*vx[idx]*vx[idx];
    su = 0.5*su + cblas_ddot(dat->n, h, 1, vx, 1);
    *fy = su;
  }
  
  if (fxi!=NULL) {
    double su=0.0;
    for (idx=dat->n;idx<n;idx++) su += vx[idx];
    *fxi = lambda*su;
  }
  
  if (f0!=NULL) {
    double su=0.0;
    for (idx=0;idx<dat->n;idx++) su+= w[idx]*xsig[idx]*xsig[idx];
    *f0 = 0.5*su;
  }
  
  if (iters!=NULL) *iters = numiters;
  if (cholerr!=NULL) *cholerr = (cholretval==0 ? 0 : 1);
  
  /* return relative accuracy at final iteration */
  if (inftuple!=NULL) {
    inftuple[0] = infL/(1.0+hinf);
    inftuple[1] = infs/(1.0+finf);
    inftuple[2] = mu;
    /* all of the above are below eps, if converged */
  }
  
  return (oktostop ? 1 : 0);
}




