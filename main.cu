/* Key problem: CUDA requires CUDA specific math */
/* Need to write ARMADILLO-free integrands       */

#include <cstdio>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>

#include "matter.cuh"
#include "spt.cuh"
#include "integrate.cuh"

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                                       \
{                                                                            \
  cudaError_t e = cudaGetLastError();                                        \
  if (e != cudaSuccess) {                                                    \
    printf("Cuda failure %s:%d: '%s'\n",                                     \
           __FILE__,                                                         \
           __LINE__,                                                         \
           cudaGetErrorString(e));                                           \
    exit(EXIT_FAILURE);                                                      \
  }                                                                          \
}

/*--------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------*/

int main() {
  cudaDeviceSetLimit(cudaLimitStackSize, 8192);

  const int evalsize = 10;

  cudaStream_t streams[evalsize];

  double logk_eval[evalsize];
  {
  double step_size = (log(1e1) - log(1e-2))/evalsize;
  for (int i=0; i<evalsize; i++) {logk_eval[i] = 1e-2 + i*step_size;}
  }
  
  double* results;
  cudaMallocManaged(&results, evalsize*sizeof(double));

  spt_data* sdata = spt_init();

  /* now integrate */
  
  auto StartTime = std::chrono::system_clock::now();
  // #pragma omp parallel for schedule(dynamic) // load balancing when integrals don't converge at same rate
  // for(int j=0; j<evalsize; j++) {
  //   cudaStreamCreate(&streams[j]);

  //   // init some things:
  //   const double lambda = 50;
  //   const int size = 100;
  //   const double step_size = (log(1e4) - log(1e-5))/size;

  //   double logk[size];
  //   double logp[size];
  //   double slopes[size];

  //   for (int i=0; i<size; i++) {
  //     logk[i] = log(1e-5) + i * step_size ;
  //     logp[i] = -2.2*logk[i];
  //   }
  //   for (int i=0; i<size-1; i++) {
  //     slopes[i] = (logp[i+1] - logp[i])/step_size;
  //   }

  //   double* d_logk;
  //   double* d_logp;
  //   double* d_slopes;

  //   cudaMallocAsync(&d_logk,   size*sizeof(double), streams[j]);
  //   cudaCheckError();
  //   cudaMallocAsync(&d_logp,   size*sizeof(double), streams[j]);
  //   cudaCheckError();
  //   cudaMallocAsync(&d_slopes, size*sizeof(double), streams[j]);
  //   cudaCheckError();

  //   cudaMemcpyAsync(d_logk, &logk, size*sizeof(double), cudaMemcpyHostToDevice, streams[j]);
  //   cudaCheckError();
  //   cudaMemcpyAsync(d_logp, &logp, size*sizeof(double), cudaMemcpyHostToDevice, streams[j]);
  //   cudaCheckError();
  //   cudaMemcpyAsync(d_slopes, &slopes, size*sizeof(double), cudaMemcpyHostToDevice, streams[j]);
  //   cudaCheckError();

  //   int_data idata;

  //   idata.logk_eval=logk_eval[j];
  //   idata.cutoff=lambda;
  //   idata.logk=d_logk;
  //   idata.logp0=d_logp;
  //   idata.dk=step_size;
  //   idata.kernel_data=sdata;
  //   idata.length=size;
  //   idata.slopes=d_slopes;

  //   int_data* d_idata;
  //   cudaMallocAsync(&d_idata, sizeof(int_data), streams[j]);
  //   cudaCheckError();
  //   cudaMemcpyAsync(d_idata, &idata, sizeof(int_data), cudaMemcpyHostToDevice, streams[j]);
  //   cudaCheckError();
    
  //   results[j] = integrate_1loop(d_idata, 1.0e-3, 1.0e-3, 50000, streams[j], false);

  //   cudaFreeAsync(d_logk, streams[j]);
  //   cudaFreeAsync(d_logp, streams[j]);
  //   cudaFreeAsync(d_slopes, streams[j]);
  //   cudaFreeAsync((void*)d_idata, streams[j]);
  // }
  // for (int i = 0; i < evalsize; i++) {
  //     cudaStreamSynchronize(streams[i]);
  //     cudaStreamDestroy(streams[i]);
  // }
  auto StopTime = std::chrono::system_clock::now();
  // std::cout << "streams: " << float(std::chrono::duration_cast <std::chrono::milliseconds> (StopTime - StartTime).count()) << std::endl;
  //printf("Spectrum results 1:\n");
  // for (int i=0; i<evalsize; i++) {
  //   printf("(logk, P_1L) = (%f, %f)\n", logk_eval[i], results[i]);
  // }

  // /* mCUBES */
  // {
  // double res = integrate_1loop(idata, 1.0e-3, 1.0e-3, 50000, true);
  // }
  // {
  // double res = integrate_2loop(idata, 1.0e-3, 1.0e-3, 100000, true);
  // }

  // Free memory and exit

  // cudaFree((void*)sdata->G);
  // cudaFree((void*)sdata->perm_table);
  // cudaFree((void*)sdata->perm_counts);
  // cudaFree((void*)sdata->perm_offsets);
  // cudaFree(sdata);

  // cudaFree(logk);
  // cudaFree(logp);
  // cudaFree(slopes);
  // cudaFree(idata);

  /* now integrate */
  cudaDeviceSynchronize();
  StartTime = std::chrono::system_clock::now();
  for(int j=0; j<evalsize; j++) {
    // init some things:
    double* logk;
    double* logp;
    double* slopes;

    //const double logk_eval = log(0.1);
    const double lambda = 50;
    const int size = 100;
    const double step_size = (log(1e4) - log(1e-5))/size;

    cudaMallocManaged(&logk,   size*sizeof(double));
    cudaMallocManaged(&logp,   size*sizeof(double));
    cudaMallocManaged(&slopes, size*sizeof(double));

    for (int i=0; i<size; i++) {
      logk[i] = log(1e-5) + i * step_size ;
      logp[i] = -2.2*logk[i];
    }
    for (int i=0; i<size-1; i++) {
      slopes[i] = (logp[i+1] - logp[i])/step_size;
    }

    spt_data* sdata = spt_init();
    int_data* idata;
    cudaMallocManaged(&idata, sizeof(int_data));
    
    idata->logk_eval=logk_eval[j];
    //printf("logk_eval = %f\n", idata->logk_eval);
    idata->cutoff=lambda;
    idata->logk=logk;
    idata->logp0=logp;
    idata->dk=step_size;
    idata->kernel_data=sdata;
    idata->length=size;
    idata->slopes=slopes;

    results[j] = integrate_2loop(idata, 1.0e-3, 1.0e-3, 100000, 0, true);

    cudaFree((void*)sdata->G);
    cudaFree((void*)sdata->perm_table);
    cudaFree((void*)sdata->perm_counts);
    cudaFree((void*)sdata->perm_offsets);
    cudaFree(sdata);

    cudaFree(logk);
    cudaFree(logp);
    cudaFree(slopes);
    cudaFree(idata);
  }
  cudaDeviceSynchronize();
  cudaFree((void*)sdata->G);
  cudaFree((void*)sdata->perm_table);
  cudaFree((void*)sdata->perm_counts);
  cudaFree((void*)sdata->perm_offsets);
  StopTime = std::chrono::system_clock::now();
  std::cout << "no streams: " << float(std::chrono::duration_cast <std::chrono::milliseconds> (StopTime - StartTime).count()) << std::endl;
  //printf("Spectrum results 2:\n");
  // for (int i=0; i<evalsize; i++) {
  //   printf("(logk, P_1L) = (%f, %f)\n", logk_eval[i], results[i]);
  // }
  cudaFree(results);

  return 0;
}