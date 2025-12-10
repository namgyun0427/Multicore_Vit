#include "ViT_cl.h"

// transpose + flat
// input[EMBED_DIM][n_patch_per_image][n_patch_per_image] => output[total_num_patches][EMBED_DIM]
void v_flatten_transpose(
    cl_mem m_input, cl_mem m_output,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
) {
    int output_size = IMG_SIZE / PATCH_SIZE;

    cl_kernel k = container.kernels[__flatten_transpose];

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
    err = clEnqueueNDRangeKernel(container.queue, k, 3, 0, global_dim, NULL, e_num_waiting, e_waiting_arr, e_out);
    CHECK_CL_ERROR(err);
}
