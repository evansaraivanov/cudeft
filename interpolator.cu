/* Linear interpolator for the matter power spectrum */
/* Two templates:
	1. For use in integrands - takes entire integration data
	2. For use outside integrads - takes a pofk struct
*/

#include <fstream>
#include <tuple>
#include "integrate.cuh"

__host__ __device__ double lin_interp(int_data* data, double logk) {
	/* evaluate p(k) using linear interpolation */
	int idx = floor((logk-data->logk[0])/data->dk);

	// if we are outside of interpolation range, do linear extrapolation
	//idx = max(0, min(idx, data->length - 2));
	idx = max(0, min(idx, data->length-2));

	//if (idx<0 || idx>data->length-2) {printf("idx = %d",idx);}

	return data->logp0[idx] + (logk-data->logk[idx]) * data->slopes[idx];
}






