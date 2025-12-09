// tile size (magic number)
#define TS 32

__kernel void linear_layer(
	__global const float* g_input,
	__global const float* g_weight,
	__global const float* g_bias,
	__global float* g_output,
	const int input_row,
	const int input_col,
	const int output_col
) {
	const size_t o_grp = get_group_id(0);
	const size_t t_grp = get_group_id(1);
	const size_t o_lid = get_local_id(0);
	const size_t t_lid = get_local_id(1);

	const size_t o = o_grp * TS + o_lid;
	const size_t t = t_grp * TS + t_lid;

	// input and weight lcoal cache
	__local float Isub[TS][TS];
	__local float Wsub[TS][TS];

	// set default value 
	float sum = (o < output_col) ? g_bias[o] : 0.0f;

	const size_t n_tiles = (input_col + TS - 1) / TS;
	for (int tile=0; tile<n_tiles; ++tile) {
		// load Isub
		const size_t ic = tile * TS + o_lid;
		if (t < input_row && ic < input_col) {
			Isub[t_lid][o_lid] = g_input[t * input_col + ic];
		} else {
			Isub[t_lid][o_lid] = 0.0f;
		}

		// load Wsub (transpose for cache)
		const size_t wc = tile * TS + t_lid;
		if (o < output_col && wc < input_col) {
			Wsub[o_lid][t_lid] = g_weight[o * input_col + wc];
		} else {
			Wsub[o_lid][t_lid] = 0.0f;
		}

		barrier(CLK_LOCAL_MEM_FENCE);


		// acuumalte
		for (int k=0; k<TS; ++k) {
			sum += Isub[t_lid][k] * Wsub[o_lid][k];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// store result
	if (t < input_row && o < output_col) {
		g_output[t * output_col + o] = sum;
	}
}
