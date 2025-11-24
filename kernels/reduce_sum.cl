__kernel void reduce_sum (
    __global float* const g_input,
    __global float* const g_output,
    int n_data,
    __local float* l_sum
) {
    size_t global_id = get_global_id(0);
    size_t group_id = get_group_id(0);
    size_t local_id = get_local_id(0);
    size_t group_size = get_local_size(0);

    l_sum[local_id] = (global_id < n_data) ? g_input[global_id] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = group_size/2; i >= 1; i /=2) {
        if (local_id < i) {
            l_sum[local_id] += l_sum[local_id + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        g_output[group_id] = l_sum[0];
    }
}
