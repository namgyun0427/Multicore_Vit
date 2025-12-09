#include "ViT_cl.h"

// image patch embedding (convolution)
// input[IN_CAHNS][PATCH_SIZE][PATCH_SIZE] => output[EMBED_DIM][n_patch_per_image][n_patch_per_image]
void v_Conv2d(
    cl_mem m_input, cl_mem m_output,
    cl_mem m_weight, cl_mem m_bias
    ) {
    const int OUTPUT_SIZE = IMG_SIZE / PATCH_SIZE;

    cl_kernel k = container.kernels[__patch_embed];

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
    int outsize = OUTPUT_SIZE;
    KernelArg args[] = {
        {.size = sizeof(cl_mem), .addr = &m_input },
        {.size = sizeof(cl_mem), .addr = &m_weight },
        {.size = sizeof(cl_mem), .addr = &m_bias },
        {.size = sizeof(cl_mem), .addr = &m_output },
        {.size = sizeof(int), .addr = &in_chans },
        {.size = sizeof(int), .addr = &patchsize },
        {.size = sizeof(int), .addr = &imgsize },
        {.size = sizeof(int), .addr = &outsize },
    };
    for (int i = 0; i < 8; ++i) {
        err = clSetKernelArg(k, i, args[i].size, args[i].addr);
        CHECK_CL_ERROR(err);
    }

    // run kernel
    const size_t global_dim[] = { OUTPUT_SIZE, OUTPUT_SIZE, EMBED_DIM };
    err = clEnqueueNDRangeKernel(container.queue, k, 3, 0, global_dim, NULL, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
}
