/* Holds all of the integration functions */

#include <cstdio>
#include "integrate.cuh"
#include "matter.cuh"

#include "cuda/mcubes/vegasT.cuh"

__host__ double integrate_1loop(int_data* idata, double epsrel, double epsabs,
  double ncall, cudaStream_t stream, bool print_result, int max_iter, int skip) 
{
  fcn_1loop fcn(idata);
  constexpr int ndim = 2;
  quad::Volume<double, ndim> vol;

  auto result = cuda_mcubes::integrate<fcn_1loop, ndim, 0>(
      fcn,
      epsrel,
      epsabs,
      ncall,
      &vol,
      max_iter,
      max_iter, // TBH I don't know what this does
      skip,
      stream);

  if (print_result) {
    printf("Result = %f +/- %f\n",result.estimate, result.errorest);
    printf("\t neval = %zd\n",result.neval);
    printf("\t status = %d\n", result.status);
    printf("\t chisq = %f\n", result.chi_sq);
    printf("\t iters = %zd\n", result.iters);
  }
  return result.estimate;
}

//--------------------------------------------------------------------------------------------------

__host__ double integrate_2loop(int_data* idata, double epsrel, double epsabs,
  double ncall, cudaStream_t stream, bool print_result, int max_iter, int skip) 
{
  fcn_2loop fcn(idata);
  constexpr int ndim = 5;
  quad::Volume<double, ndim> vol;

  auto result = cuda_mcubes::integrate<fcn_2loop, ndim, 0>(
      fcn,
      epsrel,
      epsabs,
      ncall,
      &vol,
      max_iter,
      max_iter, // TBH I don't know what this does
      skip,
      stream);

  if (print_result) {
    printf("Result = %f +/- %f\n",result.estimate, result.errorest);
    printf("\t neval = %zd\n",result.neval);
    printf("\t status = %d\n", result.status);
    printf("\t chisq = %f\n", result.chi_sq);
    printf("\t iters = %zd\n", result.iters);
  }
  return result.estimate;
}


