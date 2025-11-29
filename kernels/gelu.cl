float f(float x) {
    return 0.5f * x * (1.0f + erf(x / rootn(2.0f, 2)));
}

__kernel void gelu(
	__global float* g_data
) {
	int idx = get_global_id(0);

	g_data[idx] = f(g_data[idx]);
}
