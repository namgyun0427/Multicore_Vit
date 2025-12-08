#include "ViT_cl.h"

// transpose + flat
// input[EMBED_DIM][n_patch_per_image][n_patch_per_image] => output[total_num_patches][EMBED_DIM]
void v_flatten_transpose(float* input, float* output) {
    int output_size = IMG_SIZE / PATCH_SIZE;

    cl_kernel k = container.kernels[__flatten_transpose];

    // create and write mem obj
    const size_t input_size = EMBED_DIM * output_size * output_size * sizeof(float);
    cl_mem m_input = clCreateBuffer(container.context, CL_MEM_READ_WRITE, input_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_input, CL_TRUE, 0, input_size, input, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    
    const size_t final_output_size = EMBED_DIM * output_size * output_size * sizeof(float);
    cl_mem m_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, final_output_size, NULL, &err);
    CHECK_CL_ERROR(err);


    // set kenrl args
    // __kernel void flatten_transpose (
    //     __global float* g_input,
    //     __global float* g_output,
    //     int OUTPUT_SIZE,
    //     int EMBED_DIM
    // ) {
    int embeddim = EMBED_DIM;
    KernelArg args[] = {
        { .size = sizeof(cl_mem), .addr = &m_input },
        { .size = sizeof(cl_mem), .addr = &m_output },
        { .size = sizeof(int), .addr = &output_size },
        { .size = sizeof(int), .addr = &embeddim },
    };
    for (int i=0; i<4; ++i) {
        err = clSetKernelArg(k, i, args[i].size, args[i].addr);
        CHECK_CL_ERROR(err);
    }

    // run kernel
    const size_t global_dim[] = { output_size, output_size, EMBED_DIM };
    err = clEnqueueNDRangeKernel(container.queue, k, 3, 0, global_dim, NULL, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // read result
    err = clEnqueueReadBuffer(container.queue, m_output, CL_TRUE, 0, final_output_size, output, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // release mem obj
    err = clReleaseMemObject(m_input);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_output);
    CHECK_CL_ERROR(err);
}
