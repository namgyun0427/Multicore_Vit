#include "ViT_cl.h"

// normalize
// input[total_num_patches + 1][EMBED_DIM] => output[total_num_patches + 1][EMBED_DIM]
void v_layer_norm(
    cl_mem m_input, cl_mem m_output,
    cl_mem m_weight, cl_mem m_bias,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
) {
    cl_kernel k = container.kernels[__layer_norm];

    // set kernel args
    // __kernel void layer_norm(
    //     __global float* g_input,
    //     __global float* g_weight,
    //     __global float* g_bias,
    //     __global float* g_output,
    //     const int EMBED_DIM,
    //     const int EPSILON,
    //     __local float* l_plain_sum,
    //     __local float* l_sum_of_square
    // ) {
    int embed_dim = EMBED_DIM;
    int epsilon = EPSILON;
    KernelArg args[] = {
        { .size = sizeof(cl_mem), .addr = &m_input },
        { .size = sizeof(cl_mem), .addr = &m_weight },
        { .size = sizeof(cl_mem), .addr = &m_bias },
        { .size = sizeof(cl_mem), .addr = &m_output },
        { .size = sizeof(int), .addr = &embed_dim },
        { .size = sizeof(int), .addr = &epsilon },
        { .size = EMBED_DIM * sizeof(float), .addr = NULL },
        { .size = EMBED_DIM * sizeof(float), .addr = NULL },
    };
    for (int i=0; i<8; ++i) {
        err = clSetKernelArg(k, i, args[i].size, args[i].addr);
        CHECK_CL_ERROR(err);
    }

    // run kernel
    const size_t global_config[] = { N_TOTAL_TOKEN * EMBED_DIM };
    const size_t local_config[] = { EMBED_DIM };
    err = clEnqueueNDRangeKernel(
        container.queue, k, 
        1, 0, global_config, local_config, 
        e_num_waiting, e_waiting_arr, e_out
    );
    CHECK_CL_ERROR(err);
}
