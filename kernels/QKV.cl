__kernel void QKV(
	__global const float* input,
	__global const float* in_weight,
	__global const float* in_bias,
	__global float* Q,
	__global float* K,
	__global float* V,
	const int n_tokens,
	const int embed_dim
){
	int t = get_global_id(0);
	int i = get_global_id(1);

	if (t < n_tokens && i < embed_dim){
		float sum_q = in_bias[i];
		float sum_k = in_bias[embed_dim + i];
		float sum_v = in_bias[2 * embed_dim + i];

		for ( int j = 0; j < embed_dim; j++){
			float x = input[t * embed_dim + j];
			sum_q += x * in_weight[i * embed_dim + j];
			sum_k += x * in_weight[(embed_dim + i) * embed_dim + j];
			sum_v += x * in_weight[(2 * embed_dim + i) * embed_dim + j];
		}

		Q[t * embed_dim + i] = sum_q;
		K[t * embed_dim + i] = sum_k;
		V[t * embed_dim + i] = sum_v;
	}
}