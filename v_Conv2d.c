#include "ViT_cl.h"

void v_Conv2d(
    cl_mem m_input, cl_mem m_output,
    cl_mem m_weight, cl_mem m_bias
    ) {
    const int OUTPUT_SIZE = IMG_SIZE / PATCH_SIZE;

    cl_kernel k = container.kernels[__conv2d];

    int imgsize = IMG_SIZE;
    int outsize = OUTPUT_SIZE;

    KernelArg args[] = {
        {.size = sizeof(cl_mem), .addr = &m_input },   // 0: Input Image
        {.size = sizeof(cl_mem), .addr = &m_weight },  // 1: Weight (Filter)
        {.size = sizeof(cl_mem), .addr = &m_bias },    // 2: Bias
        {.size = sizeof(cl_mem), .addr = &m_output },  // 3: Output Feature Map
        {.size = sizeof(int), .addr = &imgsize },      // 4: Image Size (224)
        {.size = sizeof(int), .addr = &outsize },      // 5: Output Grid Size (14)
    };

    for (int i = 0; i < 6; ++i) {
        err = clSetKernelArg(k, i, args[i].size, args[i].addr);
        CHECK_CL_ERROR(err);
    }

    const size_t global_dim[] = {
        (size_t)OUTPUT_SIZE,  // dim 0: ow
        (size_t)OUTPUT_SIZE,  // dim 1: oh
        (size_t)EMBED_DIM     // dim 2: oc (768)
    };

    err = clEnqueueNDRangeKernel(container.queue, k, 3, 0, global_dim, NULL, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
}
