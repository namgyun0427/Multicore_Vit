__kernel void normalize_score (
    __global float* g_scores,
    __global float* g_sum_exp
) {
    size_t i = get_global_id(0);    // row
    size_t j = get_global_id(1);    // col
    size_t N_TOTAL_TOKEN = get_global_size(0);
    
    g_scores[i * N_TOTAL_TOKEN + j] /= g_sum_exp[i];
}

            