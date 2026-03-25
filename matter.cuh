#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include "spt.cuh"
#include "integrate.cuh"

// integrands
__host__ __device__ double integrand_1loop(double integration_vars[], size_t dim, void * userdata);
__host__ __device__ double integrand_2loop(double integration_vars[], size_t dim, void * userdata);
__host__ __device__ double  integrand_quad(double integration_vars[], size_t dim, void * userdata);

// integrand structs
struct fcn_1loop {
  int_data* idata;
  fcn_1loop(int_data* d) : idata(d) {} // constructor

  __device__ double operator()(double x1, double x2) const
  {
    double xl = log(1e-4);
    double xu = log(idata->cutoff);
    double vars[2] = {(xu-xl)*x1+xl,M_PI*x2};

    double res = (xu-xl)*M_PI*integrand_1loop(vars, 2, idata);
    return res;
  }
};

struct fcn_2loop {
  int_data* idata;
  fcn_2loop(int_data* d) : idata(d) {} // constructor

  __device__ double operator()
    (double x1, double x2, double x3, double x4, double x5) const
  {
    double xl = log(1e-4);
    double xu = log(idata->cutoff);
    double vars[5] = {
      (xu-xl)*x1+xl, M_PI*x2, 
      (xu-xl)*x3+xl, M_PI*x4, 2.0*M_PI*x5
    };

    double res = (xu-xl)*M_PI*(xu-xl)*M_PI*2*M_PI*integrand_2loop(vars, 5, idata);
    return res;
  }
};

struct fcn_quad {
  int_data* idata;
  fcn_quad(int_data* d) : idata(d) {} // constructor

  __device__ double operator()
    (double x1, double x2) const
  {
    double xl = log(1e-4);
    double xu = log(idata->cutoff);
    double vars[2] = {(xu-xl)*x1+xl, M_PI*x2};

    double res = (xu-xl)*M_PI*integrand_quad(vars, 2, idata);
    return res;
  }
};
