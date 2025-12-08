#include "ViT_cl.h"


// image patch embedding (convolution)
// input[IN_CAHNS][PATCH_SIZE][PATCH_SIZE] => output[EMBED_DIM][n_patch_per_image][n_patch_per_image]
void v_Conv2d(
    cl_mem m_input, cl_mem m_output,
    cl_mem m_weight, cl_mem m_bias
) {
    const int OUTPUT_SIZE = IMG_SIZE / PATCH_SIZE;

    cl_kernel k = container.kernels[__patch_embed];

    // create and write mem obj
    // const size_t input_size = IN_CAHNS * IMG_SIZE * IMG_SIZE * sizeof(float);
    // cl_mem m_input = clCreateBuffer(container.context, CL_MEM_READ_WRITE, input_size, NULL, &err);
    // CHECK_CL_ERROR(err);
    // err = clEnqueueWriteBuffer(container.queue, m_input, CL_TRUE, 0, input_size, input, 0, NULL, NULL);
    // CHECK_CL_ERROR(err);

    // const size_t weight_size = weight.size * sizeof(float);
    // cl_mem m_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, weight_size, NULL, &err);
    // CHECK_CL_ERROR(err);
    // err = clEnqueueWriteBuffer(container.queue, m_weight, CL_TRUE, 0, weight_size, weight.data, 0, NULL, NULL);
    // CHECK_CL_ERROR(err);

    // const size_t bias_size = bias.size * sizeof(float);
    // cl_mem m_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, bias_size, NULL, &err);
    // CHECK_CL_ERROR(err);
    // err = clEnqueueWriteBuffer(container.queue, m_bias, CL_TRUE, 0, bias_size, bias.data, 0, NULL, NULL);
    // CHECK_CL_ERROR(err);
    
    // const size_t output_size = EMBED_DIM * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float);
    // cl_mem m_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, output_size, NULL, &err);
    // CHECK_CL_ERROR(err);
    

    // set kernel args
    // __kernel void patch_embed (
    //     __global float* g_input,
    //     __global float* g_weight,
    //     __global float* g_bias,
    //     __global float* g_output,
    //     int IN_CAHNS,
    //     int PATCH_SIZE,
    //     int IMG_SIZE,
    //     int OUTPUT_SIZE
    // ) {
    int in_chans = IN_CAHNS;
    int patchsize = PATCH_SIZE;
    int imgsize = IMG_SIZE;
    KernelArg args[] = {
        { .size = sizeof(cl_mem), .addr = &m_input },
        { .size = sizeof(cl_mem), .addr = &m_weight },
        { .size = sizeof(cl_mem), .addr = &m_bias },
        { .size = sizeof(cl_mem), .addr = &m_output },
        { .size = sizeof(int), .addr = &in_chans },
        { .size = sizeof(int), .addr = &patchsize },
        { .size = sizeof(int), .addr = &imgsize },
        { .size = sizeof(int), .addr = &OUTPUT_SIZE },
    };
    for (int i=0; i<8; ++i) {
        err = clSetKernelArg(k, i, args[i].size, args[i].addr);
        CHECK_CL_ERROR(err);
    }

    // run kernel
    const size_t global_dim[] = { EMBED_DIM, OUTPUT_SIZE, OUTPUT_SIZE };
    err = clEnqueueNDRangeKernel(container.queue, k, 3, 0, global_dim, NULL, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // // read result
    // err = clEnqueueReadBuffer(container.queue, m_output, CL_TRUE, 0, output_size, output, 0, NULL, NULL);
    // CHECK_CL_ERROR(err);

    // // release
    // err = clReleaseMemObject(m_input);
    // CHECK_CL_ERROR(err);
    // err = clReleaseMemObject(m_weight);
    // CHECK_CL_ERROR(err);
    // err = clReleaseMemObject(m_bias);
    // CHECK_CL_ERROR(err);
    // err = clReleaseMemObject(m_output);
    // CHECK_CL_ERROR(err);
}
