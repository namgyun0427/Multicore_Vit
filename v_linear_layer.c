#include "ViT_cl.h"

// tile size = magic number
#define TS 32

// do linear transform with input matrix(network)
void v_linear_layer(
    cl_mem m_input, cl_mem m_weight, cl_mem m_bias, cl_mem m_output,
    int input_row, int input_col, int output_col,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
) {
    const cl_kernel k = container.kernels[__linear_layer];

    // set kenrl args
    // __kernel void linear(
    // 	__global const float* g_input,
    // 	__global const float* g_weight,
    // 	__global const float* g_bias,
    // 	__global float* g_output,
    // 	const int input_row,           // M
    // 	const int input_col,      // N
    // 	const int output_col      // K
    // ) {
    KernelArg args[] = {
        {.size = sizeof(cl_mem), .addr = &m_input },
        {.size = sizeof(cl_mem), .addr = &m_weight },
        {.size = sizeof(cl_mem), .addr = &m_bias },
        {.size = sizeof(cl_mem), .addr = &m_output },
        {.size = sizeof(int), .addr = &input_row },
        {.size = sizeof(int), .addr = &input_col },
        {.size = sizeof(int), .addr = &output_col },
    };

    for (int i = 0; i < 7; ++i) {
        err = clSetKernelArg(k, i, args[i].size, args[i].addr);
        CHECK_CL_ERROR(err);
    }

    // run kernel
    const size_t tile_num_output_col = (output_col + TS - 1) / TS;
    const size_t tile_num_input_row = (input_row + TS - 1) / TS;
    const size_t global_dim[] = { tile_num_output_col * TS, tile_num_input_row * TS };
    const size_t local_dim[] = { TS, TS };
    err = clEnqueueNDRangeKernel(
        container.queue, k,
        2, NULL, global_dim, local_dim,
        e_num_waiting, e_waiting_arr, e_out
    );
    CHECK_CL_ERROR(err);
}
