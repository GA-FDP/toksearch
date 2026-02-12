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
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include "numpy/arrayobject.h" 

#include "ell1.h"

/*
 * PYTHON MAIN ENTRY POINT
 * 
 */
 
static PyObject *
method_ell1(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int verbose = 0;
    int maxiters = DEFAULT_MAXITERS;
    int initopt = 1;
    int nthreads = 0; /* this means "default" or "do-nothing" */
    double eta = DEFAULT_ETA;
    double eps = DEFAULT_EPS;
    int nyl = 0;
    int nyr = 0;
    double *yl = NULL;
    double *yr = NULL;
    double *weights = NULL;
    double objy = 0.0, objxi = 0.0, obj0 = 0.0;
    double lambda;

    ell1ProgramData optData;

    // Input pyobjects
    PyObject *x_arg = NULL;
    PyObject *yl_arg = Py_None;
    PyObject *yr_arg = Py_None;
    PyObject *weights_arg = Py_None;

    // Internal pyobjects that need to be decref'd
    PyArrayObject *x_arr = NULL;
    PyArrayObject *yl_arr = NULL;
    PyArrayObject *yr_arr = NULL;
    PyArrayObject *w_arr = NULL;
    PyArrayObject *inf_arr = NULL;
    PyArrayObject *fy_arr = NULL;
    PyArrayObject *fxi_arr = NULL;
    PyArrayObject *itr_arr = NULL;
    PyArrayObject *cherr_arr = NULL;
    PyArrayObject *cvg_arr = NULL;
    PyArrayObject *y_arr = NULL;

    // Return object
    PyObject *rep_out = NULL;


    static char* argnames[] = {
        "x", "lamb", "yl", "yr", "w", "eps", "eta", "maxiters", "nthreads", NULL
    }; 

    int parse_status = PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "Od|$OOOddii",
        argnames,
        &x_arg,
        &lambda,
        &yl_arg,
        &yr_arg,
        &weights_arg,
        &eps,
        &eta,
        &maxiters,
        &nthreads
    );

    if(!parse_status) return NULL;

    // Convert x_arr object to numpy array
    x_arr = (PyArrayObject *)PyArray_FROM_OTF(x_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!x_arr) {
        goto cleanup;
    }

    int ndims = PyArray_NDIM(x_arr);
    int numSignals;

    if (ndims == 0) {
        PyErr_SetString(PyExc_RuntimeError, "x cannot be an empty array");
        goto cleanup;
    } else if (ndims == 1) {
        numSignals = 1;    
    } else if (ndims == 2) {
        numSignals = PyArray_DIM(x_arr, 1);
    } else {
        PyErr_SetString(PyExc_RuntimeError, "x cannot have more than 2 dimensions");
        goto cleanup;
    }

    int n = PyArray_DIM(x_arr, 0); // Num time series samples
    if(n < MINIMUM_SAMPLES) {
        PyErr_SetString(PyExc_RuntimeError, "Too few rows of data in input x (each column of x is a signal).");
        goto cleanup;
    }

    if(yl_arg != Py_None) {
        yl_arr = (PyArrayObject *)PyArray_FROM_OTF(yl_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if(!yl_arr) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert yl to numpy array");
            goto cleanup;
        }
        yl = (double *)PyArray_DATA(yl_arr);
        if(PyArray_NDIM(yl_arr) == 0) {
            nyl = 1;
        } else {
            nyl = PyArray_DIM(yl_arr, 0);
        }
    }

    if(yr_arg != Py_None) {
        yr_arr = (PyArrayObject *)PyArray_FROM_OTF(yr_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if(!yr_arr) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert yr to numpy array");
            goto cleanup;
        }
        yr = (double *)PyArray_DATA(yr_arr);
        if(PyArray_NDIM(yr_arr) == 0) {
            nyr = 1;
        } else {
            nyr = PyArray_DIM(yr_arr, 0);
        }
    }


    if(weights_arg != Py_None) {
        w_arr = (PyArrayObject *)PyArray_FROM_OTF(weights_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if(!w_arr) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert w to numpy array");
            goto cleanup;
        }
        weights = (double *)PyArray_DATA(w_arr);
    }

    /* Create an empty output dict */
    rep_out = PyDict_New();


    /* So assume each column of X is a signal to be denoised with the same single lambda (to be loaded below) */

    /* Grab pointer to vector or matrix x */
    double *x = (double *)PyArray_DATA(x_arr);

    if (lambda<=0.0) {
        PyErr_SetString(PyExc_RuntimeError, "Input lambda must be positive.");
        goto cleanup;
    }

    /* read weights w; must be consistent in size with x; accept either row or column vector w  */
    // TODO check that weights is same shape as x


    /* Silently ignore unreasonable epsilons and etas; stay at default value */
    if(eps < 0.0 || eps >= 1.0) eps = DEFAULT_EPS;
    if(eta < 0.0 || eta >= 1.0) eta = DEFAULT_ETA;


    if(maxiters <= 0) maxiters = DEFAULT_MAXITERS;
    if(maxiters >= MAXIMUM_ITERATIONS) maxiters = MAXIMUM_ITERATIONS;

    if (nthreads>=1) set_num_threads(nthreads);

    /* 
     * Allocate and setup data buffers needed;
     * then solve a series of optimization problems
     * with identical structure; one for each column of x.
     *
     */

    memset(&optData, 0, sizeof(optData));
    int retcode =  setupEll1ProgramBuffers(&optData, n, nyl, nyr, verbose);


    if (retcode!=0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to setup ell1 buffers");
        goto cleanup;
    }

    int iters=-1, cholerr=-1;

    y_arr = (PyArrayObject *)PyArray_NewLikeArray(x_arr, NPY_ANYORDER, NULL, 1);
    double *y = (double *)PyArray_DATA(y_arr); 

    //npy_intp const *eval_dims = {numSignals};

    const npy_intp num_sigs = numSignals;
    fy_arr = (PyArrayObject *)PyArray_SimpleNew(1, &num_sigs, NPY_DOUBLE);
    double *fy = (double *)PyArray_DATA(fy_arr); 

    fxi_arr = (PyArrayObject *)PyArray_SimpleNew(1, &num_sigs, NPY_DOUBLE);
    double *fxi = (double *)PyArray_DATA(fxi_arr); 

    itr_arr = (PyArrayObject *)PyArray_SimpleNew(1, &num_sigs, NPY_DOUBLE);
    double *itr = (double *)PyArray_DATA(itr_arr); 

    cherr_arr = (PyArrayObject *)PyArray_SimpleNew(1, &num_sigs, NPY_DOUBLE);
    double *cherr = (double *)PyArray_DATA(cherr_arr); 

    cvg_arr = (PyArrayObject *)PyArray_SimpleNew(1, &num_sigs, NPY_DOUBLE);
    double *cvg = (double *)PyArray_DATA(cvg_arr); 

    npy_intp inf_dims[] = {3, num_sigs};
    inf_arr = (PyArrayObject *)PyArray_SimpleNew(2, inf_dims, NPY_DOUBLE);
    double *inf = (double *)PyArray_DATA(inf_arr); 


    /* Call solver once for each column of input data */
    int kk=0;
    while (kk<numSignals)
    {

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
            &(inf[3*kk])
        );


        fy[kk] = objy+obj0; /* loss term */
        fxi[kk] = objxi;    /* regularization term */
        itr[kk] = (double)iters;
        cherr[kk] = (double)cholerr;
        cvg[kk] = (double)retcode;

        kk++;
    }

    PyDict_SetItemString(rep_out, "y", (PyObject *)y_arr);
    Py_XDECREF((PyObject *)y_arr);

    releaseEll1ProgramBuffers(&optData, verbose);

    cleanup:
        Py_XDECREF(x_arr);
        Py_XDECREF(yl_arr);
        Py_XDECREF(yr_arr);
        Py_XDECREF(w_arr);
        Py_XDECREF(inf_arr);
        Py_XDECREF(fy_arr);
        Py_XDECREF(fxi_arr);
        Py_XDECREF(itr_arr);
        Py_XDECREF(cherr_arr);
        Py_XDECREF(cvg_arr);

        if(PyErr_Occurred()) {
            Py_XDECREF(rep_out);
            return NULL;
        } else {
            return rep_out;
        }
}

/*****************************************************************************
 * Boilerplate python C API stuff
*****************************************************************************/

static PyMethodDef Ell1Methods[] = {
    {"_compute_ell1", (PyCFunction)method_ell1, METH_VARARGS|METH_KEYWORDS, "Calculate l1 filter"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef ell1module = {
    PyModuleDef_HEAD_INIT,
    "ell1module",
    "C extension for computing l1 trend filter",
    -1,
    Ell1Methods
};

PyMODINIT_FUNC PyInit_ell1module(void) {
    PyObject *module = PyModule_Create(&ell1module);
    import_array();
    return module;
}

