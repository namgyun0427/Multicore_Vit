float v_gelu(float x) {
    return 0.5f * x * (1.0f + erf(x / rootn(2.0f, 2)));
}

__kernel void mlp(
	__global const float* g_input,
	__global const float* g_weight1,
	__global const float* g_bias1,
	__global const float* g_weight2,
	__global const float* g_bias2,
	__global float* g_output,
	const int tokens,
	const int in_features,
	const int out_features
) {
	int t = get_global_id(0);
	int o = get_global_id(1);

	float sum0 = g_bias1[o];

	for (int i = 0; i < in_features; i++) {
		sum0 += g_input[t * in_features + i] * g_weight1[o * in_features + i];
	}

	g_output[t * out_features + o] = sum0;



	float sum1 = g_bias1[o];

	for (int i = 0; i < in_features; i++) {
		sum1 += g_input[t * in_features + i] * g_weight1[o * in_features + i];
	}

	g_output[t * out_features + o] = sum1;
}
