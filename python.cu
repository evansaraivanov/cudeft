/* python bindings */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "matter.cuh"
#include "spt.cuh"
#include "integrate.cuh"

py::array_t<double> p_mm(
py::array_t<double>& logk_eval, const int len_logk_eval,
py::array_t<double>& log_plin, py::array_t<double>& log_klin, const int size,
double lambda, int loop_order) {

	// computes N-loop power spectrum
	cudaDeviceSetLimit(cudaLimitStackSize, 4096);

	spt_data* sdata = spt_init();

	double* logk;
  double* logp;
  double* slopes;

	//const int size = 100; // length of plin and klin
	auto buf_log_klin  =  log_klin.unchecked<1>();
	auto buf_log_plin  =  log_plin.unchecked<1>();
  auto buf_logk_eval = logk_eval.unchecked<1>();

	const double step_size = (buf_log_klin(size-1) - buf_log_klin(0))/size;

	cudaMallocManaged(&logk,   size*sizeof(double));
  cudaMallocManaged(&logp,   size*sizeof(double));
  cudaMallocManaged(&slopes, size*sizeof(double));

  for (int i = 0; i<size; i++) {
    logk[i] = buf_log_klin(i);
    logp[i] = buf_log_plin(i);
	}

  for (int i=0; i<size-1; i++) {slopes[i] = (logp[i+1] - logp[i])/step_size;}

  std::vector<double> results(len_logk_eval);

	#pragma omp parallel for schedule(dynamic)
  for (int i=0; i<len_logk_eval; i++) {
    int_data* idata;
    cudaMallocManaged(&idata, sizeof(int_data));
    
    idata->logk_eval=buf_logk_eval(i);
    idata->cutoff=lambda;
    idata->logk=logk;
    idata->logp0=logp;
    idata->dk=step_size;
    idata->kernel_data=sdata;
    idata->length=size;
    idata->slopes=slopes;

    if (1==loop_order) {
    	results[i] = integrate_1loop(idata, 1.0e-3, 1.0e-7, 50000, 0, true);
    }
    if (2==loop_order) {
      results[i] = integrate_2loop(idata, 1.0e-3, 1.0e-7, 50000, 0, true);
    }

    cudaFree(idata);
  }

	cudaFree((void*)sdata->G);
	cudaFree((void*)sdata->perm_table);
	cudaFree((void*)sdata->perm_counts);
	cudaFree((void*)sdata->perm_offsets);
	cudaFree(sdata);

	cudaFree(logk);
  cudaFree(logp);
  cudaFree(slopes);

  return py::array_t<double>(len_logk_eval, results.data());
}

PYBIND11_MODULE(pycudeft, m) {
  m.def("p_mm", &p_mm, "Computes matter-matter power spectrum", py::return_value_policy::move);
}