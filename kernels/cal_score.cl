__kernel void cal_score (
    __global float* g_Q,
    __global float* g_K,
    __global float* g_scores,
    int N_TOTAL_TOKEN,
    int EMBED_DIM,
    int HEAD_DIM,
    int head_offset,
    float scale
) {
    size_t i = get_global_id(0);    // row
    size_t j = get_global_id(1);    // col

    float score = 0.0f;

    for (int d=0; d < HEAD_DIM; ++d) {
        float q = g_Q[i * EMBED_DIM + head_offset + d];
        float k = g_K[j * EMBED_DIM + head_offset + d];
        score += q * k;
    }

    g_scores[i * N_TOTAL_TOKEN + j] = score / scale;
}


// for (int i = 0; i < n_tokens; i++) {
//     for (int j = 0; j < n_tokens; j++) {
//         float score = 0.0f;

//         for (int d = 0; d < HEAD_DIM; d++) {
//             float q = Q[i * EMBED_DIM + head_offset + d];
//             float k = K[j * EMBED_DIM + head_offset + d];
//             score += q * k;
//         }

//         scores[i * n_tokens + j] = score / sqrtf((float)HEAD_DIM);
//     }
// }