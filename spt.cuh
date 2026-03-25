#pragma once

struct spt_data {
  // struct of all static data needed for SPT kernels
  const double* G;
  const int* perm_table;
  const int* perm_counts;
  const int* perm_offsets;
  int max_order;
  int ROWS;
  int COLS;
};

spt_data* spt_init();
void spt_free(spt_data* d_data);

__host__ __device__ double2 spt_kernels_cuda(
	int order, const double* Q, const int* active_idxs, spt_data* sdata);