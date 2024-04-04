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
 * ell1est1.c
 *
 * ell-1 regularized weighted least-squares
 * scalar signal reconstruction/filtering.
 *
 * Uses OpenBLAS / LAPACK and CBLAS calls internally.
 *
 * MATLAB Interface:
 *
 *   rep = ell1est1(
 *           x, lambda, yl, yr, w,          # problem args 
 *           eps, eta, maxiters, obthread)  # algorithm args
 *
 * Returns a report struct rep where the field rep.y is the solution
 * vector to the following optimization problem:
 *
 * (1)  min_{y(i)} sum_i loss_i(y(i), x(i)) + lambda*sum_j |y(j)-2*y(j+1)+y(j+2)|
 *
 * where i=1..n, and the range of j depends on whether boundary points are
 * provided with the arguments yl, yr (can be empty = []). See details below.
 *
 * If input x is a matrix, each column of x will be processed and the
 * output field rep.y will also be a matrix of the same size as x.
 *
 * The standard quadratic loss function loss_i(y(i), x(i)) is given by
 *
 * (2) loss_i(y(i), x(i)) = (1/2)*w(i)*(y(i)-x(i))^2
 *
 * where w(i)=1 for all i if w is empty. 
 *
 * ALGORITHM:
 *
 * Program (1)-(2) is a convex optimization problem.
 * It can be re-cast as a standard inequality constrained
 * quadratic program. The complexity to get the solution is O(n)
 * in both memory and time, where n=length(x).
 *
 * The bottleneck in the solver below is to solve a sym. pos.def. equation
 * (twice per main iteration; # iterations typically independent of n)
 *
 *   M*x = b; M = [M11, M12; M21, M22], x = [x1; x2], b = [b1; b2]
 *
 * with M11 = Hy + Ey'*D*Ey, M21 = Exi'*D*Ey, M22 = Exi'*D*Exi, D diagonal
 * with positive elements and M12 = M21'. The matrix E = [Ey, Exi] is 
 * a partitioning of the linear inequality constraint in the recast
 * formulation of (1)-(2).
 *
 * (3) min_{y(i),xi(j)} sum_i loss_i(y(i), x(i)) + lambda*sum_j xi(j)
 *       s.t. Ey*y + Exi*xi <= f
 *
 * In (3) y is the full column of y(i), i=1..n, and xi is the full column
 * of xi(j), j=[-1],[0],1,...,n-2,[n-1],[n] (brackets optional elements).
 *
 * COMPILE:
 *   mex -I"/u/olofsson/lib/OpenBLAS-0.2.19/" CFLAGS="\$CFLAGS -std=c99 -Wall -O2" ell1est1.c -lrt -lpthread -lgfortran /u/olofsson/lib/OpenBLAS-0.2.19/libopenblas.a
 *
 * NOTE: obthread=2 seems to be faster than default.
 * NOTE: profiling suggests that about ~60% is spent on factorization and 2xsolves
 * NOTE: the O(n) behaviour has been checked: ~150 points/milliseconds on typical CPU.
 *
 * TODO: can the factorizer be made quicker by simple merge of M11/M11p loops?
 * TODO: simpler/faster ADMM version possible? closed-form for optimal rho?
 *
 */

#include <string.h>

/* MATLAB includes */
#include "matrix.h"
#include "mex.h"

#include "ell1.h"


#undef __OUTPUT_TEST_EY_MATRICES__
#undef __RUN_BANDED_CHOLESKY_TEST__

#if defined(__OUTPUT_TEST_EY_MATRICES__) || defined(__RUN_BANDED_CHOLESKY_TEST__)
  #include "mt19937ar.h"
#endif


/* Input arguments */
#define ARG_x prhs[0]
#define ARG_lambda prhs[1]
#define ARG_yl prhs[2]
#define ARG_yr prhs[3]
#define ARG_weight prhs[4]
#define ARG_eps prhs[5]
#define ARG_eta prhs[6]
#define ARG_maxiters prhs[7]
#define ARG_obthread prhs[8]

/* Output report struct */
#define REP_OUT plhs[0]
#ifdef __OUTPUT_TEST_EY_MATRICES__
  #define REP_NUMFIELDS 14
#else
  #define REP_NUMFIELDS 11
#endif
#define REP_FIELD_Y 0
#define REP_FIELD_FY 1
#define REP_FIELD_XI 2
#define REP_FIELD_FXI 3
#define REP_FIELD_ITERS 4
#define REP_FIELD_ISCONVERGED 5
#define REP_FIELD_CHOLERROR 6
#define REP_FIELD_SOLVECLOCK 7
#define REP_FIELD_SCHURCLOCK 8
#define REP_FIELD_SLRHSCLOCK 9
#define REP_FIELD_INFTUPLE 10
#ifdef __OUTPUT_TEST_EY_MATRICES__
  #define REP_FIELD_CEY 11 /* TEST */
  #define REP_FIELD_REY 12
  #define REP_FIELD_EXI 13
#endif

const char *reportFieldNames[REP_NUMFIELDS]={
  "y",
  "fy",
  "xi",
  "fxi",
  "iters",
  "isconverged",
  "cholerror",
  "solveclock",
  "schurclock",
  "slrhsclock",
  "inftuple"
#ifdef __OUTPUT_TEST_EY_MATRICES__
  ,"cEy", /* TEST */
  "rEy",
  "Exi"
#endif
};



/*
 * MATLAB MAIN ENTRY POINT
 * 
 * NOTE: explicitly uses matlab's print function(s) (and should)
 */
 
void mexFunction(
  int nlhs,
  mxArray *plhs[],
  int nrhs,
  const mxArray *prhs[])
{
  int verbose = 0;
  int maxiters = DEFAULT_MAXITERS;
  int initopt = 1;
  int numOpenBlasThreads = 0; /* this means "default" or "do-nothing" */
  double eta = DEFAULT_ETA;
  double eps = DEFAULT_EPS;
  int nyl = 0;
  int nyr = 0;
  double *yl = NULL;
  double *yr = NULL;
  double *weights = NULL;
  double objy = 0.0, objxi = 0.0, obj0 = 0.0;
  
  ell1ProgramData optData;
  memset(&optData, 0, sizeof(optData));
  
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
  fclk_timespec _tic1, _toc1;
#endif

  /* Create an empty output struct */
  REP_OUT = mxCreateStructMatrix(1, 1, REP_NUMFIELDS, reportFieldNames);
    
  /* Must (always) provide 9 input arguments and either 0 or 1 output arguments */
  if (nrhs!=9 || (nlhs!=0 && nlhs!=1))
    mexErrMsgTxt("USAGE: rep=ell1est1(x,lambda,yl,yr,w,eps,eta,maxiters,obthread);");

  /* Check all inputs and determine problem dimensions; with consistency checks */
  if (mxIsComplex(ARG_x) || mxGetNumberOfDimensions(ARG_x)!=2
      || mxIsSparse(ARG_x) || !mxIsDouble(ARG_x))
    mexErrMsgTxt("Input x must be full real double.");

  /* So assume each column of X is a signal to be denoised with the same single lambda (to be loaded below) */
  int n = mxGetM(ARG_x);
  int numSignals = mxGetN(ARG_x);
  
  if (n<MINIMUM_SAMPLES)
    mexErrMsgTxt("Too few rows of data in input x (each column of x is a signal).");
  
  /* Grab pointer to vector or matrix x */
  double *x = mxGetPr(ARG_x);

  /* lambda should be a single positive scalar */
  if (mxIsComplex(ARG_lambda) || mxGetNumberOfDimensions(ARG_lambda)!=2
      || mxIsSparse(ARG_lambda) || !mxIsDouble(ARG_lambda))
    mexErrMsgTxt("Input lambda must be full real double.");
  if (mxGetM(ARG_lambda)!=1 || mxGetN(ARG_lambda)!=1)
    mexErrMsgTxt("Input lambda must be scalar.");
  double lambda = *(mxGetPr(ARG_lambda));
  if (lambda<=0.0)
    mexErrMsgTxt("Input lambda must be positive.");
  
  /* read weights w; must be consistent in size with x; accept either row or column vector w  */
  if (!mxIsEmpty(ARG_weight)) {
    if (mxIsComplex(ARG_weight) || mxGetNumberOfDimensions(ARG_weight)!=2
        || mxIsSparse(ARG_weight) || !mxIsDouble(ARG_weight))
      mexErrMsgTxt("Input w must be full real double (when not empty).");
    if (! (    (mxGetM(ARG_weight)==1 && mxGetN(ARG_weight)==n)
            || (mxGetM(ARG_weight)==n && mxGetN(ARG_weight)==1) ) )
      mexErrMsgTxt("Input w must be size compatible with x.");
    weights = mxGetPr(ARG_weight);
  }
  
  /* read epsilon */
  if (!mxIsEmpty(ARG_eps)) {
    if (mxIsComplex(ARG_eps) || mxGetNumberOfDimensions(ARG_eps)!=2
        || mxIsSparse(ARG_eps) || !mxIsDouble(ARG_eps))
      mexErrMsgTxt("Input eps must be full real double (when not empty).");
    if (mxGetM(ARG_eps)!=1 || mxGetN(ARG_eps)!=1)
      mexErrMsgTxt("Input eps must be scalar.");
    double tmp = *mxGetPr(ARG_eps);
    /* Silently ignore unreasonable epsilons; stay at default value */
    if (tmp>0.0 && tmp<=1.0) eps = tmp;
  }
  
  /* read eta */
  if (!mxIsEmpty(ARG_eta)) {
    if (mxIsComplex(ARG_eta) || mxGetNumberOfDimensions(ARG_eta)!=2
        || mxIsSparse(ARG_eta) || !mxIsDouble(ARG_eta))
      mexErrMsgTxt("Input eta must be full real double (when not empty).");
    if (mxGetM(ARG_eta)!=1 || mxGetN(ARG_eta)!=1)
      mexErrMsgTxt("Input eta must be scalar.");
    double tmp = *mxGetPr(ARG_eta);
    /* Silently ignore unreasonable etas; stay at default value */
    if (tmp>0.0 && tmp<=1.0) eta = tmp;
  }
  
  /* Read LHS boundary condition argument "yl"; can be a null, 1 or 2 component vector of doubles */
  if (!mxIsEmpty(ARG_yl)) {
    if (mxIsComplex(ARG_yl) || mxGetNumberOfDimensions(ARG_yl)!=2
        || mxIsSparse(ARG_yl) || !mxIsDouble(ARG_yl))
      mexErrMsgTxt("Input yl must be full real double (when not empty).");
    int numel = mxGetM(ARG_yl)*mxGetN(ARG_yl);
    if (!(numel==1 || numel==2))
      mexErrMsgTxt("Input yl must have either 1 or 2 elements (when not empty).");
    yl = mxGetPr(ARG_yl);
    nyl = numel;
  }
  
  /* Read RHS boundary condition argument "yr"; can be a null, 1 or 2 component vector of doubles */
  if (!mxIsEmpty(ARG_yr)) {
    if (mxIsComplex(ARG_yr) || mxGetNumberOfDimensions(ARG_yr)!=2
        || mxIsSparse(ARG_yr) || !mxIsDouble(ARG_yr))
      mexErrMsgTxt("Input yr must be full real double (when not empty).");
    int numel = mxGetM(ARG_yr)*mxGetN(ARG_yr);
    if (!(numel==1 || numel==2))
      mexErrMsgTxt("Input yr must have either 1 or 2 elements (when not empty).");
    yr = mxGetPr(ARG_yr);
    nyr = numel;
  }
  
  /* read maxiters option */
  if (!mxIsEmpty(ARG_maxiters)) {
    if (mxIsComplex(ARG_maxiters) || mxGetNumberOfDimensions(ARG_maxiters)!=2
        || mxIsSparse(ARG_maxiters) || !mxIsDouble(ARG_maxiters))
      mexErrMsgTxt("Input maxiters must be full real double (when not empty).");
    if (mxGetM(ARG_maxiters)!=1 || mxGetN(ARG_maxiters)!=1)
      mexErrMsgTxt("Input maxiters must be scalar.");
    int testmxi = (int)(*(mxGetPr(ARG_maxiters)));
    if (testmxi>0 && testmxi<=MAXIMUM_ITERATIONS)
      maxiters = testmxi;
  }
  
  if (!mxIsEmpty(ARG_obthread)) {
    if (mxIsComplex(ARG_obthread) || mxGetNumberOfDimensions(ARG_obthread)!=2
        || mxIsSparse(ARG_obthread) || !mxIsDouble(ARG_obthread))
      mexErrMsgTxt("Input obthread must be full real double (when not empty).");
    if (mxGetM(ARG_obthread)!=1 || mxGetN(ARG_obthread)!=1)
      mexErrMsgTxt("Input obthread must be scalar.");
    int testobt = (int)(*(mxGetPr(ARG_obthread)));
    if (testobt>=0) {
      /*mexPrintf("[%s]: obthread=%i from argument.\n", __func__, testobt);*/
      numOpenBlasThreads = testobt;
    }
  }
  
  if (numOpenBlasThreads>=1)
    set_num_threads(numOpenBlasThreads);

  /* 
   * Allocate and setup data buffers needed;
   * then solve a series of optimization problems
   * with identical structure; one for each column of x.
   *
   */
   
  int retcode =  setupEll1ProgramBuffers(&optData, n, nyl, nyr, verbose);
  
  if (retcode!=0)
    mexErrMsgTxt("Buffer allocation or initialization failed.");
    
#ifdef __RUN_BANDED_CHOLESKY_TEST__
  /* Run test suite: 10 factorizations; 2 RHSs for each eq. */
  randomFactorizeSolveTest(&optData, 10, 2);
#endif
  
#ifdef __OUTPUT_TEST_EY_MATRICES__
  mxArray *pmx = NULL;
  pmx = mxCreateDoubleMatrix(2*optData.nxi, optData.n, mxREAL);
  memset(mxGetPr(pmx), 0, sizeof(double)*(optData.rEy.rows)*(optData.rEy.cols));
  stripe_dxpra(&(optData.rEy), 1.0, mxGetPr(pmx)); /* add 1.0*A to X */
  mxSetFieldByNumber(REP_OUT, 0, REP_FIELD_REY, pmx);
  pmx = mxCreateDoubleMatrix(2*optData.nxi, optData.n, mxREAL);
  memset(mxGetPr(pmx), 0, sizeof(double)*(optData.cEy.rows)*(optData.cEy.cols));
  stripe_dxpca(&(optData.cEy), 1.0, mxGetPr(pmx)); /* add 1.0*A to X */
  mxSetFieldByNumber(REP_OUT, 0, REP_FIELD_CEY, pmx);
  pmx = mxCreateDoubleMatrix(2*optData.nxi, optData.nxi, mxREAL);
  memset(mxGetPr(pmx), 0, sizeof(double)*optData.nxi*2*optData.nxi);
  make_dense_exi(mxGetPr(pmx), optData.nxi);
  mxSetFieldByNumber(REP_OUT, 0, REP_FIELD_EXI, pmx);
#endif

  int iters=-1, cholerr=-1;
  
  mxArray *pmy = mxCreateDoubleMatrix(n, numSignals, mxREAL);
  double *y = mxGetPr(pmy);
  
  mxArray *pmfy = mxCreateDoubleMatrix(numSignals, 1, mxREAL);
  double *fy = mxGetPr(pmfy);
  
  mxArray *pmfxi = mxCreateDoubleMatrix(numSignals, 1, mxREAL);
  double *fxi = mxGetPr(pmfxi);
  
  mxArray *pmitr = mxCreateDoubleMatrix(numSignals, 1, mxREAL);
  double *itr = mxGetPr(pmitr);
  
  mxArray *pmcherr = mxCreateDoubleMatrix(numSignals, 1, mxREAL);
  double *cherr = mxGetPr(pmcherr);
  
  mxArray *pmcvg = mxCreateDoubleMatrix(numSignals, 1, mxREAL);
  double *cvg = mxGetPr(pmcvg);
  
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
  mxArray *pmclk = mxCreateDoubleMatrix(numSignals, 1, mxREAL);
  double *clk = mxGetPr(pmclk);
  mxArray *pmclkc = mxCreateDoubleMatrix(numSignals, 1, mxREAL);
  double *clkc = mxGetPr(pmclkc);
  mxArray *pmclkr = mxCreateDoubleMatrix(numSignals, 1, mxREAL);
  double *clkr = mxGetPr(pmclkr);
#endif

  mxArray *pminf = mxCreateDoubleMatrix(3, numSignals, mxREAL);
  double *inf = mxGetPr(pminf);

  /* Call solver once for each column of input data */
  int kk=0;
  while (kk<numSignals)
  {

#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_tic1);
#endif
  
    retcode = ell1ProgramSolve(
      &optData,
      &(x[kk*n]),
      lambda,
      (weights!=NULL ? weights : optData.w), /* NOTE: optData.w is a vector of default unity weights */
      yl,
      yr,  /* TODO: optionally transfer last solution as a boundary condition for next problem ?! ... */
      eta,
      eps,
      maxiters,
      initopt,
      &(y[kk*n]), /* y */
      &objy,      /* fy */
      NULL,       /* TODO: xi */
      &objxi,     /* fxi */
      &obj0,      /* f0 */
      &iters,
      &cholerr,
      &(inf[3*kk]));
      
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_toc1);
    clk[kk] = fclk_delta_timestamps(&_tic1, &_toc1);
    clkc[kk] = __global_clock_0;
    clkr[kk] = __global_clock_1;
#endif

    fy[kk] = objy+obj0; /* loss term */
    fxi[kk] = objxi;    /* regularization term */
    itr[kk] = (double)iters;
    cherr[kk] = (double)cholerr;
    cvg[kk] = (double)retcode;
    
    kk++;
  }
  
  mxSetFieldByNumber(REP_OUT, 0, REP_FIELD_Y, pmy);
  mxSetFieldByNumber(REP_OUT, 0, REP_FIELD_FY, pmfy);
  mxSetFieldByNumber(REP_OUT, 0, REP_FIELD_FXI, pmfxi);
  mxSetFieldByNumber(REP_OUT, 0, REP_FIELD_ITERS, pmitr);
  mxSetFieldByNumber(REP_OUT, 0, REP_FIELD_CHOLERROR, pmcherr);
  mxSetFieldByNumber(REP_OUT, 0, REP_FIELD_ISCONVERGED, pmcvg);
  mxSetFieldByNumber(REP_OUT, 0, REP_FIELD_INFTUPLE, pminf);
  
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
  mxSetFieldByNumber(REP_OUT, 0, REP_FIELD_SOLVECLOCK, pmclk);
  mxSetFieldByNumber(REP_OUT, 0, REP_FIELD_SCHURCLOCK, pmclkc);
  mxSetFieldByNumber(REP_OUT, 0, REP_FIELD_SLRHSCLOCK, pmclkr);
#endif

  releaseEll1ProgramBuffers(&optData, verbose);
  
  return;
}


