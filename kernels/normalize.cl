__kernel void my_normalize(
    __global float* g_input,
    __global const float* g_weight,
    __global const float* g_bias,
    __global float* g_MEAN,
    __global float* g_INV_STD
) {
   size_t globla_id = get_global_id(0);

   g_input[globla_id] = ((g_input[globla_id] - g_MEAN[0]) * g_INV_STD[0]) * g_weight[globla_id] + g_bias[globla_id];
}
