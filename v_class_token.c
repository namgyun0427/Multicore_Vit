#include "ViT_cl.h"

// prepend class tokens in front
// input[total_num_patches][EMBED_DIM] => output[total_num_patches + 1][EMBED_DIM]
// EMBEDDIM크기의 class token을 기존 데이터 앞에 붙임
void v_class_token(
    cl_mem m_input, cl_mem m_output,
    cl_mem m_class_token
) {
    // class 토큰 앞에 붙이고
    const size_t class_token_size = EMBED_DIM * sizeof(float);
    err = clEnqueueCopyBuffer(container.queue, m_class_token, m_output, 0, 0, class_token_size, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    
    // 뒤에 기존내용 붙이기
    int output_size = IMG_SIZE / PATCH_SIZE;
    int num_patches = output_size * output_size;
    const size_t input_size = num_patches * EMBED_DIM * sizeof(float);
    err = clEnqueueCopyBuffer(container.queue, m_input, m_output, 0, class_token_size, input_size, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
}
