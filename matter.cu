/* Key problem: CUDA requires CUDA specific math */
/* Need to write ARMADILLO-free integrands       */

#include <cuda_runtime.h>
#include <vector_types.h>
#include <cstdio>
#include <iostream>

#include "spt.cuh"
#include "interpolator.cuh"
#include "matter.cuh"

/*------------------------------------------------------------------------------
--------------------------------------------------------------------------------
    1-Loop  ------------------------------------------------------------------*/


__host__ __device__ double integrand_1loop(
  double integration_vars[], 
  size_t dim, 
  void * userdata
) {
 	// read the static integration information (k-mode, cutoff, etc)
 	struct int_data * data = (struct int_data *)userdata;

 	double k = exp(data->logk_eval);

 	// read the integration variables and construct the vectors
 	double q = exp(integration_vars[0]);
 	double qtheta1 = integration_vars[1];
	
 	// jacobian for spherical coords
 	const double fourier_factor = 1/pow(2*M_PI,3);
 	double jac = 2.0 * M_PI * q*q*q * sin(qtheta1); 

  // compute dot products
  double kmq = sqrt(k*k + q*q - 2.0*k*q*cos(qtheta1));
  double kpq = sqrt(k*k + q*q + 2.0*k*q*cos(qtheta1));

  // compute the SPT kernels;
  double Q2_1[6] = {
    -q*sin(qtheta1), 0, k-q*cos(qtheta1), 
     q*sin(qtheta1), 0,   q*cos(qtheta1)
  };
  double Q2_2[6] = {
    q*sin(qtheta1), 0, k+q*cos(qtheta1), 
   -q*sin(qtheta1), 0,  -q*cos(qtheta1)
  };
  double Q3[9]   = {
     0,              0,  k,
     q*sin(qtheta1), 0,  q*cos(qtheta1),
    -q*sin(qtheta1), 0, -q*cos(qtheta1)
  }; 

  // get the interpolated power spectrum values we need
  double P_k   = exp(lin_interp(data, log(k)));
  double P_q   = exp(lin_interp(data, log(q)));
  double P_kmq = 0.0;
  double P_kpq = 0.0;


	// compute the kernels in IR safe
	double2 K2_1 = make_double2(0.0, 0.0);
	double2 K2_2 = make_double2(0.0, 0.0);

  const int idxs2[2] = {0,1};
  const int idxs3[3] = {0,1,2};

	if ( (kmq - q) > 0 ) {
    K2_1 = spt_kernels_cuda(2, Q2_1, idxs2, data->kernel_data);\
    P_kmq = exp(lin_interp(data, log(kmq)));
  }
  if ( (kpq - q) > 0 ) {
    K2_2 = spt_kernels_cuda(2, Q2_2, idxs2, data->kernel_data);
    P_kpq = exp(lin_interp(data, log(kpq)));
  }

	double2 K3 = spt_kernels_cuda(3, Q3, idxs3, data->kernel_data);

  // printf("-----------------------------------------\n");
  // printf("k = %f, q=%f, theta=%f\n",k, q, qtheta1);
  // printf("_k = %f, P_q = %f, P_kpq = %f, P_kmq = %f\n",  P_k, P_q, P_kmq, P_kpq);
  // printf("F2_1 = %f, F2_2 = %f, F3 = %f\n", K2_1.x, K2_2.x, K3.x);
  // printf("-----------------------------------------\n");

	return fourier_factor*jac*
    (   2.0*K2_1.x * K2_1.x * P_q * P_kmq
    	+ 2.0*K2_2.x * K2_2.x * P_q * P_kpq
    	+ 6.0*K3.x            * P_k * P_q   );
}

/*------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--- 2-Loop integrand ---------------------------------------------------------*/

__host__ __device__ double integrand_2loop(
  double integration_vars[], 
  size_t dim, 
  void * userdata
) {
  /* 
    NOTE: I "simplified" the step functions to reduce the number of ifs 
    -> increase speed but less easy to read mathematically 
  */
  struct int_data * data = (struct int_data *)userdata;

  double k = exp(data->logk_eval);

  // read the integration variables
  double q1 = exp(integration_vars[0]);
  double qtheta1 = integration_vars[1];
  double q2 = exp(integration_vars[2]);
  double qtheta2 = integration_vars[3];
  double qphi2 = integration_vars[4];

  double sqtheta1 = sin(qtheta1);
  double sqtheta2 = sin(qtheta2);

  double q1x = q1*sqtheta1;
  double q1z = q1*cos(qtheta1);
  double q2x = q2*sqtheta2*cos(qphi2);
  double q2y = q2*sqtheta2*sin(qphi2);
  double q2z = q2*cos(qtheta2);
  
  // jacobian for spherical coords
  const double fourier_factor = 1/pow(2*M_PI,6);
  double jac = 2.0 * M_PI * q1*q1*q1 * sqtheta1 * q2*q2*q2 * sqtheta2; 

  double res = 0.0;

  if (q2>q1) {
    double2 K5, K3_1, K3_2;

    double Q5[15] = {
         0,    0,  k,
       q1x,    0,  q1z,
      -q1x,    0, -q1z,
       q2x,  q2y,  q2z,
      -q2x, -q2y, -q2z
    };

    const int idxs5[5]   = {0, 1, 2, 3, 4};
    const int idxs3_1[3] = {0, 1, 2};
    const int idxs3_2[3] = {0, 3, 4};

    K5   = spt_kernels_cuda(5, Q5, idxs5,   data->kernel_data); // F5(k,q1,-q1, q2, -q2)
    K3_1 = spt_kernels_cuda(3, Q5, idxs3_1, data->kernel_data); // F3(k,q1,-q1)
    K3_2 = spt_kernels_cuda(3, Q5, idxs3_2, data->kernel_data); // F3(k,q2,-q2)

    double logp_k  = lin_interp(data, log(k));
    double logp_q1 = lin_interp(data, log(q1));
    double logp_q2 = lin_interp(data, log(q2));

    res += (60.0*K5.x + 18.0*K3_1.x*K3_2.x) * exp(logp_k + logp_q1 + logp_q2);

    //==============================================================================================
    // F2*F4 terms
    //--------------------------------------------------------------------------

    const int idxs4[4] = {0, 1, 2, 3};
    const int idxs2[2] = {0, 1};
    double2 K2;
    double2 K4;

    double kmq1_squared = k*k + q1*q1 - 2.0*k*q1z;
    double kpq1_squared = k*k + q1*q1 + 2.0*k*q1z;
    double kmq2_squared = k*k + q2*q2 - 2.0*k*q2z;
    double kpq2_squared = k*k + q2*q2 + 2.0*k*q2z;

    double kmq1 = sqrt(max(0.0, kmq1_squared));
    double kpq1 = sqrt(max(0.0, kpq1_squared));
    double kmq2 = sqrt(max(0.0, kmq2_squared));
    double kpq2 = sqrt(max(0.0, kpq2_squared));

    double logp_x;
    double Q4[12];

    //--------------------------------------------------------------------------
    if (kmq1 - q1 > 0) {
      // double Q4[12] = {
      //    q1x,    0,   q1z,
      //   -q1x,    0, k-q1z,
      //    q2x,  q2y,   q2z,
      //   -q2x, -q2y,  -q2z
      // };
      Q4[0] =  q1x; Q4[1]  =    0; Q4[2]  =   q1z;
      Q4[3] = -q1x; Q4[4]  =    0; Q4[5]  = k-q1z;
      Q4[6] =  q2x; Q4[7]  =  q2y; Q4[8]  =   q2z;
      Q4[9] = -q2x; Q4[10] = -q2y; Q4[11] =  -q2z;

      K2 = spt_kernels_cuda(2, Q4, idxs2, data->kernel_data);
      K4 = spt_kernels_cuda(4, Q4, idxs4, data->kernel_data);

      logp_x = lin_interp(data, log(kmq1)); //exp(lin_interp(data, log(kmq1)));

      res += 24.0 * K2.x*K4.x * exp(logp_x + logp_q1 + logp_q2); //p_kmq1*p_q1*p_q2;
    }
    //--------------------------------------------------------------------------
    if (kpq1 - q1 > 0) {
      // double Q4[12] = {
      //   -q1x,    0,  -q1z,
      //    q1x,    0, k+q1z,
      //    q2x,  q2y,   q2z,
      //   -q2x, -q2y,  -q2z
      // };
      Q4[0] = -q1x; Q4[1]  =    0; Q4[2]  =  -q1z;
      Q4[3] =  q1x; Q4[4]  =    0; Q4[5]  = k+q1z;
      Q4[6] =  q2x; Q4[7]  =  q2y; Q4[8]  =   q2z;
      Q4[9] = -q2x; Q4[10] = -q2y; Q4[11] =  -q2z;

      K2 = spt_kernels_cuda(2, Q4, idxs2, data->kernel_data);
      K4 = spt_kernels_cuda(4, Q4, idxs4, data->kernel_data);

      logp_x = lin_interp(data, log(kpq1));

      res += 24.0 * K2.x*K4.x * exp(logp_x + logp_q1 + logp_q2); //p_kpq1*p_q1*p_q2;
    }
    //--------------------------------------------------------------------------
    if (kmq2 - q2 > 0) {
      // double Q4[12] = {
      //    q2x,  q2y,   q2z,
      //   -q2x, -q2y, k-q2z,
      //    q1x,    0,   q1z,
      //   -q1x,    0,  -q1z
      // };
      Q4[0] =  q2x; Q4[1]  =  q2y; Q4[2]  =   q2z;
      Q4[3] = -q2x; Q4[4]  = -q2y; Q4[5]  = k-q2z;
      Q4[6] =  q1x; Q4[7]  =    0; Q4[8]  =   q1z;
      Q4[9] = -q1x; Q4[10] =    0; Q4[11] =  -q1z;

      K2 = spt_kernels_cuda(2, Q4, idxs2, data->kernel_data);
      K4 = spt_kernels_cuda(4, Q4, idxs4, data->kernel_data);

      logp_x = lin_interp(data, log(kmq2));

      res += 24.0 * K2.x*K4.x * exp(logp_x + logp_q1 + logp_q2); //p_kmq2*p_q1*p_q2;
    }
    //--------------------------------------------------------------------------
    if (kpq2 - q2 > 0) {
      // double Q4[12] = {
      //   -q2x, -q2y,  -q2z,
      //    q2x,  q2y, k+q2z,
      //    q1x,    0,   q1z,
      //   -q1x,    0,  -q1z
      // };
      Q4[0] = -q2x; Q4[1]  = -q2y; Q4[2]  =  -q2z;
      Q4[3] =  q2x; Q4[4]  =  q2y; Q4[5]  = k+q2z;
      Q4[6] =  q1x; Q4[7]  =    0; Q4[8]  =   q1z;
      Q4[9] = -q1x; Q4[10] =    0; Q4[11] =  -q1z;

      K2 = spt_kernels_cuda(2, Q4, idxs2, data->kernel_data);
      K4 = spt_kernels_cuda(4, Q4, idxs4, data->kernel_data);

      logp_x = lin_interp(data, log(kpq2));

      res += 24.0 * K2.x*K4.x * exp(logp_x + logp_q1 + logp_q2); //p_kpq2*p_q1*p_q2;
    }
    //==============================================================================================
    // F3^2 terms
    //--------------------------------------------------------------------------
    const int idxs3[3] = {0, 1, 2};
    double2 K3;

    double kmq1mq2_squared = k*k + q1*q1 + q2*q2 - 2.0*k*q1z - 2.0*k*q2z + 2.0*(q1x*q2x + q1z*q2z);
    double kpq1mq2_squared = k*k + q1*q1 + q2*q2 + 2.0*k*q1z - 2.0*k*q2z - 2.0*(q1x*q2x + q1z*q2z);
    double kmq1pq2_squared = k*k + q1*q1 + q2*q2 - 2.0*k*q1z + 2.0*k*q2z - 2.0*(q1x*q2x + q1z*q2z);
    double kpq1pq2_squared = k*k + q1*q1 + q2*q2 + 2.0*k*q1z + 2.0*k*q2z + 2.0*(q1x*q2x + q1z*q2z);

    double kmq1mq2 = sqrt(max(0.0, kmq1mq2_squared));
    double kpq1mq2 = sqrt(max(0.0, kpq1mq2_squared));
    double kmq1pq2 = sqrt(max(0.0, kmq1pq2_squared));
    double kpq1pq2 = sqrt(max(0.0, kpq1pq2_squared));

    double Q3[9];

    //--------------------------------------------------------------------------
    if (kmq1mq2 - q2 > 0) {
      // double Q3[9] = {
      //   q1x,        0,       q1z,
      //       q2x,  q2y,   q2z,
      //  -q1x-q2x, -q2y, k-q2z-q1z
      // };
      Q3[0] =  q1x;     Q3[1] =    0; Q3[2] =   q1z;
      Q3[3] =      q2x; Q3[4] =  q2y; Q3[5] =       q2z;
      Q3[6] = -q1x-q2x; Q3[7] = -q2y; Q3[8] = k-q1z-q2z;

      K3 = spt_kernels_cuda(3, Q3, idxs3, data->kernel_data);

      logp_x = lin_interp(data, log(kmq1mq2));

      res += 9.0 * K3.x*K3.x * exp(logp_x + logp_q1 + logp_q2); //p_q1*p_q2*p_kmq1mq2;
      //printf("Here 4.2\n");
    }
    //--------------------------------------------------------------------------
    if (kpq1mq2 - q2 > 0) {
      // double Q3[9] = {
      //  -q1x,        0,      -q1z,
      //       q2x,  q2y,   q2z,
      //   q1x-q2x, -q2y, k-q2z+q1z
      // };
      Q3[0] = -q1x;     Q3[1] =    0; Q3[2] =  -q1z;
      Q3[3] =      q2x; Q3[4] =  q2y; Q3[5] =       q2z;
      Q3[6] =  q1x-q2x; Q3[7] = -q2y; Q3[8] = k+q1z-q2z;

      K3 = spt_kernels_cuda(3, Q3, idxs3, data->kernel_data);

      logp_x = lin_interp(data, log(kpq1mq2));

      res += 9.0 * K3.x*K3.x * exp(logp_x + logp_q1 + logp_q2); //p_q1*p_q2*p_kpq1mq2;
      //printf("Here 4.3\n");
    }
    //--------------------------------------------------------------------------
    if (kmq1pq2 - q2 > 0) {
      // double Q3[9] = {
      //   q1x,        0,       q1z,
      //      -q2x, -q2y,  -q2z,
      //  -q1x+q2x,  q2y, k+q2z-q1z
      // };
      Q3[0] =  q1x;     Q3[1] =    0; Q3[2] =   q1z;
      Q3[3] =     -q2x; Q3[4] = -q2y; Q3[5] =      -q2z;
      Q3[6] = -q1x+q2x; Q3[7] =  q2y; Q3[8] = k-q1z+q2z;

      K3 = spt_kernels_cuda(3, Q3, idxs3, data->kernel_data);

      logp_x = lin_interp(data, log(kmq1pq2));

      res += 9.0 * K3.x*K3.x * exp(logp_x + logp_q1 + logp_q2); //p_q1*p_q2*p_kmq1pq2;
      //printf("Here 4.4\n");
    }
    //--------------------------------------------------------------------------
    if (kpq1pq2 - q2 > 0) {
      // double Q3[9] = {
      //  -q1x,        0,      -q1z,
      //      -q2x, -q2y,  -q2z,
      //   q1x+q2x,  q2y, k+q2z+q1z
      // };
      Q3[0] = -q1x;     Q3[1] =    0; Q3[2] =  -q1z;
      Q3[3] =     -q2x; Q3[4] = -q2y; Q3[5] =      -q2z;
      Q3[6] =  q1x+q2x; Q3[7] =  q2y; Q3[8] = k+q1z+q2z;

      K3 = spt_kernels_cuda(3, Q3, idxs3, data->kernel_data);

      logp_x = lin_interp(data, log(kpq1pq2));

      res += 9.0 * K3.x*K3.x * exp(logp_x + logp_q1 + logp_q2); //p_q1*p_q2*p_kpq1pq2;
      //printf("Here 4.5\n");
    }
    //printf("Here 5\n");
  }
  //printf("%f, %f, %f\n",fourier_factor, jac, res);
  return fourier_factor*jac*res;
}

