#pragma once

#include "spt.cuh"

// struct holding static integration data (e.g P_linear, SPT permutations, etc.)
struct int_data{
  double* logk;
  double* logp0;
  double* slopes;
  spt_data* kernel_data;
  double logk_eval;
  double cutoff;
  double dk;
  int length;
};

// Integration routines --------------------------------------------------------

__host__ double integrate_1loop(int_data* idata, double epsrel, double epsabs,
  double ncall, cudaStream_t stream=0, bool print_result=false, int max_iter=2000, int skip=5);

__host__ double integrate_2loop(int_data* idata, double epsrel, double epsabs,
  double ncall, cudaStream_t stream=0, bool print_result=false, int max_iter=2000, int skip=5);

__host__ double integrate_quad(int_data* idata, double epsrel, double epsabs,
  double ncall, cudaStream_t stream=0, bool print_result=false, int max_iter=2000, int skip=5);

// Helper functions
void int_free(int_data h_idata);
int_data int_init(double logk[], double logp[], const int size, const double lambda);