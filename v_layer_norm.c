#include "ViT_cl.h"

// normalize
// input[total_num_patches + 1][EMBED_DIM] => output[total_num_patches + 1][EMBED_DIM]
// TODO: cl_mem 받는 함수로

void v_layer_norm(
    float* input, float* output,
    Network weight, Network bias
) {
    cl_kernel k = container.kernels[__layer_norm];

    // create and write mem obj
    const size_t data_size = N_TOTAL_TOKEN * EMBED_DIM * sizeof(float);
    cl_mem m_input = clCreateBuffer(container.context, CL_MEM_READ_WRITE, data_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_input, CL_TRUE, 0, data_size, input, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    
    const size_t weight_size = weight.size * sizeof(float);
    cl_mem m_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, weight_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_weight, CL_TRUE, 0, weight_size, weight.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    
    const size_t bias_size = bias.size * sizeof(float);
    cl_mem m_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, bias_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_bias, CL_TRUE, 0, bias_size, bias.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    cl_mem m_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, data_size, NULL, &err);
    CHECK_CL_ERROR(err);

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
        0, NULL, NULL
    );
    CHECK_CL_ERROR(err);

    // read result
    err = clEnqueueReadBuffer(container.queue, m_output, CL_TRUE, 0, data_size, output, 0, NULL, NULL);
    CHECK_CL_ERROR(err);


    // release
    err = clReleaseMemObject(m_input);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_weight);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_bias);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_output);
    CHECK_CL_ERROR(err);
}
