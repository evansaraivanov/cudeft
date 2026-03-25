#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "spt.cuh"

const int MAX_ORDER = 5; // global variable for max order, sets malloc sizes.
const int ROWS = MAX_ORDER+1;
const int COLS = MAX_ORDER/2;

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
/* SPT combinatorics ---------------------------------------------------------*/

int binomial(int n, int k) {
  int num  = 1;
  int den1 = 1;
  int den2 = 1;
  for (int i=1; i<=n; i++) {num *= i;}
  for (int i=1; i<=k; i++) {den1 *= i;}
  for (int i=1; i<=(n-k); i++) {den2 *= i;}

  return num/(den1*den2);
}

void get_unique_subsets(int* permutations, int* A, int size, int order, int n_permutations) {
  // first, we generate all permutations //
  int count = 1;
  for (int i=1; i<size; i++) {count *= i;}

  for (int i=0; i<size; i++) {permutations[i] = A[i];}
  int current_permutations = 1;

  // malloc arrays for comparisons:
  int* a1 = (int*)malloc(order*sizeof(int));
  int* b1 = (int*)malloc((size-order)*sizeof(int));

  do {
    // store current and sort
    for (int j=0; j<order; j++) {a1[j] = A[j];}
    for (int j=order; j<size; j++) {b1[j-order] = A[j];}

    std::sort(a1, a1+order);
    std::sort(b1, b1+(size-order));

    // iterate over the currently found permutation
    //want to check if the new permutation is equivalent to one of the previous
    bool any_equal = false;
    for(int i=0; i<current_permutations; i++) {
      bool this_equal = true;
      if (order != size-order) {
        for (int j=0; j<order; j++) {if (a1[j] != permutations[size*i+j]) {this_equal = false;}}
        for (int j=order; j<size; j++) {if (b1[j-order] != permutations[size*i+j]) {this_equal = false;}}
      }
      // handle the special case:
      else {
        bool this_equal2 = true;
        for (int j=0; j<order; j++) {
          //printf("a1[%d] = %d, b1[%d] = %d, perm[%d] = %d\n", j, a1[j], j, b1[j], j, permutations[size*i+j]);
          if (a1[j] != permutations[size*i+j]) {this_equal  = false;}
          if (b1[j] != permutations[size*i+j]) {this_equal2 = false;}
        }
        this_equal = this_equal || this_equal2;
      }
      if (this_equal == true) {any_equal = true;}
    }

    if (any_equal == false) {
      if (current_permutations >= n_permutations) {
        printf("Found more permutations than expected: Additional memory not allocated\n");
        free(a1);
        free(b1);
        free(permutations);
        exit(1);
      }

      for (int i=0; i<size; i++) {permutations[size*(current_permutations)+i] = A[i];}
      current_permutations += 1;
    }
  } while (std::next_permutation(A, A + size));
  //std::cout << "Found " << current_permutations << " unique permutations." << std::endl;

  free(a1);
  free(b1);
}

spt_data* spt_init() {
  /* Creates constants in memory */

  /* create cuda managed spt_data */
  spt_data data;
  //cudaMallocManaged(&data, sizeof(spt_data));

  // starting with green fcn ---------------------------------------------------
  //double* G = new double[4*(MAX_ORDER+1)];
  double G[4*(MAX_ORDER+1)];
  // cudaMallocManaged(&G, 4*(MAX_ORDER+1)*sizeof(double));

  for (int i=0; i<2*MAX_ORDER; i++) {G[i] = 0.0;}
  for (int i=2; i<=MAX_ORDER; i++) {
    G[4*i+0] = 0.2 * (1.0/(i-1.0) * 3.0) - 0.2 * (1.0/(i+1.5) * (-2.0) );
    G[4*i+1] = 0.2 * (1.0/(i-1.0) * 2.0) - 0.2 * (1.0/(i+1.5) * ( 2.0) );
    G[4*i+2] = 0.2 * (1.0/(i-1.0) * 3.0) - 0.2 * (1.0/(i+1.5) * ( 3.0) );
    G[4*i+3] = 0.2 * (1.0/(i-1.0) * 2.0) - 0.2 * (1.0/(i+1.5) * (-3.0) );
  }

  // Permutation info //
  //int* n_perms = new int[ROWS*COLS];
  int n_perms[ROWS*COLS];
  //cudaMallocManaged(&n_perms, ROWS*COLS*sizeof(int));

  for (int i=0; i<ROWS*COLS; i++) {n_perms[i] = 0.0;}
  for (int N=2; N<ROWS; N++) {
    for (int i=1; i<=N/2; i++) {
      int count = 1;
      if (N-i == i) {
        for (int m=1; m<=i-1; m++) {count += binomial(i-1,m)*binomial(N-i,m);}
      }
      else {
        for (int m=1; m<=i; m++) {count += binomial(i,m)*binomial(N-i,m);}
      }
      n_perms[COLS*N+i-1] = count;
    }
  }

  //int* offsets = new int[ROWS*COLS];
  int offsets[ROWS*COLS];
  //cudaMallocManaged(&offsets, ROWS*COLS*sizeof(int));

  int offset = 0;
  for (int N=0; N<ROWS; N++) {
    for (int i=1; i<=N/2; i++) {
      int count = n_perms[COLS*N+i-1];
      if (0 == count) {offsets[COLS*N+i-1] = -1;}
      else {offsets[COLS*N+i-1] = offset; offset += count*N;}
    }
  }

  // Permutation table //
  int count = 0;
  for (int i=2; i<ROWS; i++) {
    for (int j=1; j<=i/2; j++) {
      count += i*n_perms[COLS*i+j-1];
    }
  }

  // permutations table
  int* perm_table = (int*)malloc(count*sizeof(int));
  // int* perm_table;
  // cudaMallocManaged(&perm_table, count * sizeof(int));

  for (int i=2; i<ROWS; i++) {
    for (int j=1; j<=i/2; j++) {
      int N = n_perms[COLS*i+j-1];

      int* temp_idxs = (int*)malloc(i*sizeof(int));
      int* permutations = (int*)malloc(N*i*sizeof(int));

      for (int k=0; k<i; k++) {
        temp_idxs[k] = k;
      }

      get_unique_subsets(permutations, temp_idxs, i, j, N);
      int offset = offsets[COLS*i+j-1];
      for (int k=0; k<i*N; k++) {
        perm_table[offset+k] = permutations[k];
      }
      free(temp_idxs);
      free(permutations);
    }
  }
  data.G = G;
  data.perm_table=perm_table;
  data.perm_counts=n_perms;
  data.perm_offsets=offsets;
  data.max_order=MAX_ORDER;
  data.ROWS=ROWS;
  data.COLS=COLS;

  // Now we create the GPU copy:
  spt_data* d_data;
  double* d_G;
  int* d_n_perms;
  int* d_offsets;
  int* d_perm_table;

  cudaMalloc(&d_data, sizeof(spt_data));
  cudaMalloc(&d_G, 4*(MAX_ORDER+1)*sizeof(double));
  cudaMalloc(&d_n_perms, ROWS*COLS*sizeof(int));
  cudaMalloc(&d_offsets, ROWS*COLS*sizeof(int));
  cudaMalloc(&d_perm_table, count*sizeof(int));

  cudaMemcpy(d_G, data.G, 4*(MAX_ORDER+1)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_n_perms, data.perm_counts, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, data.perm_offsets, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_perm_table, data.perm_table, count*sizeof(int), cudaMemcpyHostToDevice);

  // replace data with device pointers
  data.G = d_G;
  data.perm_counts = d_n_perms;
  data.perm_offsets = d_offsets;
  data.perm_table = d_perm_table;

  // copy full struct
  cudaMemcpy(d_data, &data, sizeof(spt_data), cudaMemcpyHostToDevice);

  // free the standard malloc 
  free(perm_table);

  return d_data;
}

void spt_free(spt_data* d_data) {
  // easy to call function to free the malloc'd data inside sdata.
  spt_data h_data;

  cudaMemcpy(&h_data, d_data, sizeof(spt_data), cudaMemcpyDeviceToHost);

  cudaFree((void*)h_data.G);
  cudaFree((void*)h_data.perm_table);
  cudaFree((void*)h_data.perm_counts);
  cudaFree((void*)h_data.perm_offsets);

  cudaFree(d_data);

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
/* SPT kernels ---------------------------------------------------------------*/

__host__ __device__ double2 spt_kernels_cuda(
  int order, const double* Q, const int* active_idxs, spt_data* sdata) 
{
  //printf("Order = %d\n", order);
  if (1 == order) {return make_double2(1.0, 1.0);}

  // tadpole check
  double sum_check_x = 0.0;
  double sum_check_y = 0.0;
  double sum_check_z = 0.0;

  for (int j=0; j<order; j++) {
    sum_check_x += Q[3*j+0];
    sum_check_y += Q[3*j+1];
    sum_check_z += Q[3*j+2];
  }
  if (0.0 == sum_check_x && 0.0 == sum_check_y && 0.0 == sum_check_z) {
    return make_double2(0.0, 0.0);
  } 

  // int n_permutations;
  double2 K = make_double2(0.0, 0.0);

  // compute the kernels recursively
  double3 k1;
  double3 k2;

  // init some quantities for permutaions
  double factor;
  int n_permutations;
  int offset;
  const int* permutations;

  // green fcn part
  double g00 = sdata->G[4*order+0];
  double g01 = sdata->G[4*order+1];
  double g10 = sdata->G[4*order+2];
  double g11 = sdata->G[4*order+3];

  // init some numbers
  double d11, d12, d22, alpha_1, alpha_2, beta;
  double2 K_m, K_m_n;

  for (int i=1; i<=0.5*order; i++) {
    // handle the permutaions
    n_permutations = sdata->perm_counts[sdata->COLS*order+i-1];
    offset = sdata->perm_offsets[sdata->COLS*order+i-1];
    permutations = &sdata->perm_table[offset];

    // combinatorial factor
    factor = 1.0/(n_permutations);
    if (i != order-i ) {
      factor *= 2.0;
    }

    // loop over permutations to get symmetrized kernel
    for (int n=0; n<n_permutations; n++) {

      // compute vectors and active indices in recursion
      k1 = make_double3(0.0, 0.0, 0.0);
      k2 = make_double3(0.0, 0.0, 0.0);

      int active_idxs_1[MAX_ORDER];
      int active_idxs_2[MAX_ORDER];
      
      for (int j=0; j<i; j++) {
        active_idxs_1[j] = active_idxs[permutations[order*n+j]];
        k1.x += Q[3*active_idxs[permutations[order*n+j]] + 0];
        k1.y += Q[3*active_idxs[permutations[order*n+j]] + 1];
        k1.z += Q[3*active_idxs[permutations[order*n+j]] + 2];
      }
      for (int j=i; j<order; j++) {
        active_idxs_2[j-i] = active_idxs[permutations[order*n+j]];
        k2.x += Q[3*active_idxs[permutations[order*n+j]] + 0];
        k2.y += Q[3*active_idxs[permutations[order*n+j]] + 1];
        k2.z += Q[3*active_idxs[permutations[order*n+j]] + 2];
      }

      // dot products
      d11 = k1.x*k1.x + k1.y*k1.y + k1.z*k1.z;
      d12 = k1.x*k2.x + k1.y*k2.y + k1.z*k2.z;
      d22 = k2.x*k2.x + k2.y*k2.y + k2.z*k2.z;

      // start computations
      if (d11 > 1e-12 && d22 > 1e-12 ) {
        // symmetrized alpha and beta
        alpha_1 = 0.5*(1+d12/d11);
        alpha_2 = 0.5*(1+d12/d22);
        beta  = d12*(d11+d22+2.0*d12)/(2.0*d11*d22);

        // kernels recursion
        K_m   = spt_kernels_cuda(i,       Q, active_idxs_1, sdata);
        K_m_n = spt_kernels_cuda(order-i, Q, active_idxs_2, sdata);

        K.x += factor*(K_m.y*(g00*alpha_1*K_m_n.x + g01*beta*K_m_n.y) + K_m.x*g00*alpha_2*K_m_n.y);
        K.y += factor*(K_m.y*(g10*alpha_1*K_m_n.x + g11*beta*K_m_n.y) + K_m.x*g10*alpha_2*K_m_n.y);
      }
    }
  }
  return K;
}