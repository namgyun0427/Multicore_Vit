__kernel void linear(
	__global const float* g_input,
	__global const float* g_weight,
	__global const float* g_bias,
	__global float* g_output,
	const int tokens,
	const int in_features,
	const int out_features
) {
	int t = get_global_id(0);
	int o = get_global_id(1);

	float sum = g_bias[o];

	for (int i = 0; i < in_features; i++) {
		sum += g_input[t * in_features + i] * g_weight[o * in_features + i];
	}

	g_output[t * out_features + o] = sum;
}