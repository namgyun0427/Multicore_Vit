__kernel void linear(
	__global const float* g_input,
	__global const float* g_weight,
	__global const float* g_bias,
	__global float* g_output,
	const int input_row,
	const int input_col,
	const int output_col
) {
	int t = get_global_id(0);
	int o = get_global_id(1);

	float sum = g_bias[o];

	for (int i = 0; i < input_col; i++) {
		sum += g_input[t * input_col + i] * g_weight[o * input_col + i];
	}

	g_output[t * output_col + o] = sum;
}
