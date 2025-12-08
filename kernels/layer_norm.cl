__kernel void layer_norm(
    __global float* g_input,
    __global float* g_weight,
    __global float* g_bias,
    __global float* g_output,
    const int EMBED_DIM,
    const int EPSILON,
    __local float* l_plain_sum,
    __local float* l_sum_of_square
) {
    size_t gid = get_global_id(0);
    size_t lid = get_local_id(0);
    size_t local_work_size = get_local_size(0);

    // get original value
    float origin_data = g_input[gid];

    // make local data cache
    l_plain_sum[lid] = origin_data;
    l_sum_of_square[lid] = origin_data * origin_data;
    barrier(CLK_LOCAL_MEM_FENCE);

    // cal plain_sum and sum_of_sqare with reducing
    // EMBED_DIM = 786 이므로 3일 때까지 (매직넘버 쩔 수 없...)
    for (int offset=local_work_size / 2; offset >= 3; offset >>= 1) {
        if (lid < offset) {
            l_plain_sum[lid] += l_plain_sum[lid + offset];
            l_sum_of_square[lid] += l_sum_of_square[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        l_plain_sum[0] = l_plain_sum[0] + l_plain_sum[1] + l_plain_sum[2];
        l_sum_of_square[0] = l_sum_of_square[0] + l_sum_of_square[1] + l_sum_of_square[2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    // calculate mean, var, inv_std
    float mean = l_plain_sum[0] / EMBED_DIM;
    float var = l_sum_of_square[0] / EMBED_DIM - mean * mean;
    float inv_std = 1.0f / rootn(var + EPSILON, 2);

    // normalize data
    g_output[gid] = ((origin_data - mean) * inv_std) * g_weight[lid] + g_bias[lid];
}
