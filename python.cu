/* python bindings */

#include <string>
#include <omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "integrate.cuh"

//------------------------------------------------------------------------------

// from m-CUBES
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

//------------------------------------------------------------------------------

py::array_t<double> p_mm(
py::array_t<double>& logk_eval, const int len_logk_eval,
py::array_t<double>& log_plin, py::array_t<double>& log_klin, const int size,
double lambda, std::string term) {

	// computes N-loop power spectrum
  std::vector<double> results(len_logk_eval);

  int num_gpus;
  cudaGetDeviceCount(&num_gpus);

  auto buf_log_klin  =  log_klin.unchecked<1>();
  auto buf_log_plin  =  log_plin.unchecked<1>();
  auto buf_logk_eval = logk_eval.unchecked<1>();

  // make a stack here so we can return the numpy array
  {
  py::gil_scoped_release release;

  #pragma omp parallel
  {
  // threads are initialized, now we can set the data on the GPUs
  int thread_id = omp_get_thread_num(); // get OMP thread number
  int gpu_id = thread_id % num_gpus;    // map thread number to GPU number
  cudaSetDevice(gpu_id);                // set the active GPU to the GPU number

  cudaDeviceSetLimit(cudaLimitStackSize, 2048); // required to be set on all devices
  //printf("thread %d is running on GPU %d\n", thread_id, gpu_id);

  double* logk = (double*)malloc(size*sizeof(double));
  double* logp = (double*)malloc(size*sizeof(double));

  for (int i = 0; i<size; i++) {
    logk[i] = buf_log_klin(i);
    logp[i] = buf_log_plin(i);
  }

  int_data  h_idata = int_init(logk, logp, size, lambda);
  int_data* d_idata;
  cudaMalloc(&d_idata, sizeof(int_data));

  // stream that the integration will run on.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

	#pragma omp for schedule(dynamic)
  for (int i=0; i<len_logk_eval; i++) {
    h_idata.logk_eval=buf_logk_eval(i);

    cudaMemcpyAsync(d_idata, &h_idata, sizeof(int_data), cudaMemcpyHostToDevice, stream);
    cudaCheckError();

    if ("1"==term) {
    	results[i] = integrate_1loop(d_idata, 1.0e-3, 1.0e-7, 25000, stream, false);
    }
    else if ("2"==term) {
      results[i] = integrate_2loop(d_idata, 1.0e-3, 1.0e-7, 25000, stream, false);
    }
    else if ("quad"==term) {
      results[i] =  integrate_quad(d_idata, 1.0e-3, 1.0e-7, 25000, stream, false);
    }
    else{printf("Term not allowed\n");}
  }
  cudaStreamSynchronize(stream);
  cudaFree(d_idata);
  cudaStreamDestroy(stream);

  int_free(h_idata);
  free(logk);
  free(logp);
  }
  }

  return py::array_t<double>(len_logk_eval, results.data());
}

PYBIND11_MODULE(pycudeft, m) {
  m.def("p_mm", &p_mm, "Computes matter-matter power spectrum", py::return_value_policy::move);
}