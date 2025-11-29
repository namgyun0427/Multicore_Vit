__kernel void cal_mean_and_inv_std (
    __global float* g_sum,
    __global float* g_sum_of_square,
    __global float* g_output_mean,
    __global float* g_output_inv_std,
    int EMBED_DIM,
    float EPSILON
) {
    float mean = g_sum[0] / EMBED_DIM;
    float var = g_sum_of_square[0] / EMBED_DIM - mean * mean;
    float sqrt = rootn(var + EPSILON, 2);
    float inv_std = 1.0f / sqrt;

    g_output_mean[0] = mean;
    g_output_inv_std[0] = inv_std;
}
