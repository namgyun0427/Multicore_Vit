__kernel void my_normalize(
    __global float* g_input,
    __global const float* g_weight,
    __global const float* g_bias,
    const float MEAN,
    const float INV_STD
) {
   size_t globla_id = get_global_id(0);

   g_input[globla_id] = ((g_input[globla_id] - MEAN) * INV_STD) * g_weight[globla_id] + g_bias[globla_id];
}
