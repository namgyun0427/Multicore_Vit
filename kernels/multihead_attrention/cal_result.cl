__kernel void cal_result(
	__global float* g_scores,
	__global float* g_V,
	__global float* g_attn_output,
	int EMBED_DIM,
	int head_offset
) {
	int i = get_global_id(0);
	int d = get_global_id(1);
    size_t N_TOTAL_TOKEN = get_global_size(0);

	float sum = 0.0f;

	for (int j = 0; j < N_TOTAL_TOKEN; j++) {
		sum += g_scores[i * N_TOTAL_TOKEN + j] * g_V[j * EMBED_DIM + head_offset + d];
	}

	g_attn_output[i * EMBED_DIM + head_offset + d] = sum;
}

// calculate result
// for (int i = 0; i < n_tokens; i++) {
// 	for (int d = 0; d < head_dim; d++) {
// 		float sum = 0.0f;
// 		for (int j = 0; j < n_tokens; j++) {
// 			sum += scores[i * n_tokens + j] * V[j * EMBED_DIM + head_offset + d];
// 		}

// 		attn_output[i * EMBED_DIM + head_offset + d] = sum;
// 	}
// }
