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

__host__ double integrate_quad(int_data* idata, double epsrel, double epsabs,
  double ncall, cudaStream_t stream, bool print_result, int max_iter, int skip) 
{
  fcn_quad fcn(idata);
  constexpr int ndim = 2;
  quad::Volume<double, ndim> vol;

  auto result = cuda_mcubes::integrate<fcn_quad, ndim, 0>(
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

/*------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--- Some functions to create and destroy integration data --------------------*/

int_data int_init(double logk[], double logp[], 
  const int size, const double lambda) 
{
  /* 
    creates an int_data on the host, but pointers are on device
    This allows us to update log keval and just copy
  */
    
  spt_data* d_sdata = spt_init(); // already on device

  const double step_size = (logk[size-1] - logk[0])/size;

  double slopes[size];
  for (int i=0; i<size-1; i++) {slopes[i] = (logp[i+1] - logp[i])/step_size;}

  // pointers for arrays
  double* d_logk;
  double* d_logp;
  double* d_slopes;

  cudaMalloc(&d_logk,   size*sizeof(double));
  cudaMalloc(&d_logp,   size*sizeof(double));
  cudaMalloc(&d_slopes, size*sizeof(double));

  cudaMemcpy(d_logk,   logk,   size*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_logp,   logp,   size*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_slopes, slopes, size*sizeof(double), cudaMemcpyHostToDevice);

  int_data h_idata;
  // int_data* d_idata;

  h_idata.logk = d_logk;
  h_idata.logp0= d_logp;
  h_idata.slopes = d_slopes;
  h_idata.kernel_data = d_sdata;

  h_idata.cutoff    = lambda;
  h_idata.dk        = step_size;
  h_idata.logk_eval = 0.0;
  h_idata.length    = size;

  // cudaMalloc(d_idata, sizeof(int_data));
  // cudaMemcpy(d_idata, &h_idata, sizeof(int_data), cudaMemcpyHostToDevice);

  return h_idata;
}

void int_free(int_data h_idata) 
{
  /* frees an int_data on the host with pointers on device */
  
  spt_free(h_idata.kernel_data);
  cudaFree((void*)h_idata.logk);
  cudaFree((void*)h_idata.logp0);
  cudaFree((void*)h_idata.slopes);

  // cudaFree(d_idata);
  return;
}








