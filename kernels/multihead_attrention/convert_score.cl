__kernel void convert_score (
    __global float* g_scores,
    __global float* g_max_val
) {
    size_t i = get_global_id(0);    // row
    size_t j = get_global_id(1);    // col
    size_t N_TOTAL_TOKEN = get_global_size(0);
    
    g_scores[i * N_TOTAL_TOKEN + j] = exp(g_scores[i * N_TOTAL_TOKEN + j] - g_max_val[i]);
}


// for (int i = 0; i < N_TOTAL_TOKEN; i++) {
//     for (int j = 0; j < N_TOTAL_TOKEN; j++) {
//         scores[i * N_TOTAL_TOKEN + j] = expf(scores[i * N_TOTAL_TOKEN + j] - p_max_val[i]);
//         p_sum_exp[i] += scores[i * N_TOTAL_TOKEN + j];
//     }
// }
            