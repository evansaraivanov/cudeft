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

/*--------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------*/

int main() {
  cudaDeviceSetLimit(cudaLimitStackSize, 8192);

  // Set up the k-modes to evaluate
  const int evalsize = 10;
  double logk_eval[evalsize]; 

  {
  double step_size = (log(1e1) - log(1e-2))/evalsize;
  for (int i=0; i<evalsize; i++) {logk_eval[i] = 1e-2 + i*step_size;}
  }

  // allocate results array
  double results[evalsize];

  /* now integrate */
  StartTime = std::chrono::system_clock::now();

  #pragma omp parallel for schedule(dynamic) // load balancing when integrals don't converge at same rate
  for(int j=0; j<evalsize; j++) {
    // init some things: 
    // lots of this can be done outside the for loop to be most optimal
    double* logk;
    double* logp;
    double* slopes;

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

    cudaFree(logk);
    cudaFree(logp);
    cudaFree(slopes);
    cudaFree(idata);
  }
  cudaFree((void*)sdata->G);
  cudaFree((void*)sdata->perm_table);
  cudaFree((void*)sdata->perm_counts);
  cudaFree((void*)sdata->perm_offsets);
  cudaFree(sdata);

  auto StopTime = std::chrono::system_clock::now();

  std::cout << "runtime:" << float(std::chrono::duration_cast <std::chrono::milliseconds> (StopTime - StartTime).count()) << std::endl;

  printf("Spectrum results:\n");
  for (int i=0; i<evalsize; i++) {
    printf("(logk, P_2L) = (%f, %f)\n", logk_eval[i], results[i]);
  }

  return 0;
}