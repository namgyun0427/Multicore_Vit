#include "ViT_cl.h"

// add position embedding data
// input[total_num_patches + 1][EMBED_DIM] => output[total_num_patches + 1][EMBED_DIM]
void v_pos_emb(
    float* input, float* output,
    Network pos_emb
) {
    // create and write mem obj
    const size_t data_size = N_TOTAL_TOKEN * EMBED_DIM * sizeof(float);
    cl_mem m_input = clCreateBuffer(container.context, CL_MEM_READ_WRITE, data_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_input, CL_TRUE, 0, data_size, input, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    cl_mem m_pos_emb = clCreateBuffer(container.context, CL_MEM_READ_WRITE, data_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_pos_emb, CL_TRUE, 0, data_size, pos_emb.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    cl_mem m_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, data_size, NULL, &err);
    CHECK_CL_ERROR(err);

    // run kernel
    v_matrix_plus(m_input, m_pos_emb, m_output, N_TOTAL_TOKEN * EMBED_DIM, 0, NULL, NULL);

    // read result
    err = clEnqueueReadBuffer(container.queue, m_output, CL_TRUE, 0, data_size, output, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // release
    err = clReleaseMemObject(m_input);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_pos_emb);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_output);
    CHECK_CL_ERROR(err);
}
