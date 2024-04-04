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

#ifndef _ELL1_H
#define _ELL1_H


#undef __COMPILE_WITH_INTERNAL_TICTOC__
#define __SCATTERED_ASSERTIONS__
#undef __OUTPUT_TEST_EY_MATRICES__
#undef __RUN_BANDED_CHOLESKY_TEST__

#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
#include "fastclock.h"
#endif

#if defined(__OUTPUT_TEST_EY_MATRICES__) || defined(__RUN_BANDED_CHOLESKY_TEST__)
  #include "mt19937ar.h"
#endif

/* Hard-coded restrictions */
#define MINIMUM_SAMPLES 5
#define MAXIMUM_ITERATIONS 250

/* Defaults */
#define DEFAULT_MAXITERS 50
#define DEFAULT_EPS 1.0e-6
#define DEFAULT_ETA 0.96

/* Special banded matrix representations */
typedef struct colStripeMatrix {
    /* This representation of M is convenient when performing transposed mults: M'*(.) */
    int rows;
    int cols;
    int *rowStart; /* cols integers */
    int *numElems; /* cols integers */
    int *idxFirst; /* cols integers */
    double *colData;
} colStripeMatrix;

typedef struct rowStripeMatrix {
    /* This representation of M is convenient when performing non-transposed mults: M*(.) */
    int rows;
    int cols;
    int *colStart; /* rows integers */
    int *numElems; /* rows integers */
    int *idxFirst; /* rows integers */
    double *rowData;
} rowStripeMatrix;

/* Optimization program structure representation + sized buffers */
typedef struct ell1ProgramData {
    /* Dimensions of optimization variables;
     * program decision variable has length n+nxi
     * and there are 2*nxi inequality constraints.
     */
    int n;
    int nxi; /* nxi = n-2 if no/free b.c. */
    int nyl;
    int nyr; /* in general: nxi = n-2 + nyl + nyr */

    double *w; /* n-vector weights for observation; if NULL, all are 1.0 implied */
    double *f; /* (2*nxi)-vector for RHS in inequality: [Ey,Exi]*[y;xi] <= f */

    double *g; /* n+nxi vector for gradient (at current y,xi) */
    double *H; /* n+nxi vector for Hessian (at current y,xi) */

    colStripeMatrix cEy; /* Ey is always 2*nxi-by-n */
    rowStripeMatrix rEy;
    double cEyData[6]; /* basic replications of very few unique elements */
    double rEyData[6];

    colStripeMatrix cM21; /* Useful meta-data for M21 banded pattern (used for M11p) */

    /* NOTE: Exi (2*nxi-by-nxi) matrix is too simple to deserve a special representation */

    /* In the notes: D is a diagonal matrix of size 2*nxi (with positive elements) */
    int ld11; /* will be initialized to 3: number of rows in M11[p] buffers */
    int kd11; /* will be init. to 2; # sub-diagonals specified for M11[p] buffers */
    char uplo; /* will be initialized to 'L' */
    double *M11;  /* standard LAPACK banded storage for M11 = H(y) + Ey'*D*Ey */
    double *M11p; /* banded storage for L*L' = M11p = M11 - M21'*inv(M22)*M21 (Cholesky factor L) */
    double *M21;  /* banded storage for M21 =  Exi'*D*Ey (nxi-by-n matrix) */
    double *M22;  /* storage for diagonal (of length nxi) matrix M22 = Exi'*D*Exi */
    double *iM22; /* reciprocal elements of M22 */

    /* NOTE: M11 and M11p are buffers with size ld11*n; M22 has size 1*nxi and M21 has size 3*n */

    double *scratch;
    int scratchsize;

    int* intbuf;
    int intbufsize; /* # ints allocated */
    double *buf;
    int bufsize;    /* # doubles allocated */

} ell1ProgramData;


void set_num_threads(int num_threads);

int setupEll1ProgramBuffers(ell1ProgramData *dat, int n, int nyl, int nyr, int verbose);

void releaseEll1ProgramBuffers(ell1ProgramData *dat, int verbose);

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
    double *inftuple
);

#endif // ifndef _ELL1_H
