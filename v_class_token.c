#include "ViT_cl.h"

void v_class_token(
    cl_mem m_input, cl_mem m_output,
    cl_mem m_class_token
) {
    const size_t class_token_size = EMBED_DIM * sizeof(float);
    err = clEnqueueCopyBuffer(container.queue, m_class_token, m_output, 0, 0, class_token_size, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    
    int output_size = IMG_SIZE / PATCH_SIZE;
    int num_patches = output_size * output_size;
    const size_t input_size = num_patches * EMBED_DIM * sizeof(float);
    err = clEnqueueCopyBuffer(container.queue, m_input, m_output, 0, class_token_size, input_size, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
}
