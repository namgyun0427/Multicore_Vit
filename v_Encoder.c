#include "ViT_cl.h"

// encode input
// input[total_num_patches + 1][EMBED_DIM]
void v_Encoder(
    float* input, float* output,
    Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
    Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b
) {
    printf(">> [v_Encoder] started\n");

    int n_tokens = ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE)) + 1;
    size_t buffer_size = sizeof(float) * n_tokens * EMBED_DIM;






    /* ------------------------------------------------------------------------------------------- */
    // create and write input mem obj
    cl_mem m_input = clCreateBuffer(container.context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_input, CL_TRUE, 0, buffer_size, input, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    
    const size_t ln1_weight_size = ln1_w.size * sizeof(float);
    cl_mem m_ln1_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, ln1_weight_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_ln1_weight, CL_TRUE, 0, ln1_weight_size, ln1_w.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    const size_t ln1_bias_size = ln1_b.size * sizeof(float);
    cl_mem m_ln1_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, ln1_bias_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_ln1_bias, CL_TRUE, 0, ln1_bias_size, ln1_b.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    
    const size_t ln2_weight_size = ln2_w.size * sizeof(float);
    cl_mem m_ln2_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, ln2_weight_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_ln2_weight, CL_TRUE, 0, ln2_weight_size, ln2_w.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    const size_t ln2_bias_size = ln2_b.size * sizeof(float);
    cl_mem m_ln2_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, ln2_bias_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_ln2_bias, CL_TRUE, 0, ln2_bias_size, ln2_b.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    cl_mem m_attn_in_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, attn_w.size * sizeof(float), NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_attn_in_weight, CL_TRUE, 0, attn_w.size * sizeof(float), attn_w.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    
    cl_mem m_attn_in_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, attn_b.size * sizeof(float), NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_attn_in_bias, CL_TRUE, 0, attn_b.size * sizeof(float), attn_b.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    cl_mem m_attn_out_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, attn_out_w.size * sizeof(float), NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_attn_out_weight, CL_TRUE, 0, attn_out_w.size * sizeof(float), attn_out_w.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    
    cl_mem m_attn_out_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, attn_out_b.size * sizeof(float), NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_attn_out_bias, CL_TRUE, 0, attn_out_b.size * sizeof(float), attn_out_b.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    const size_t mlp1_weight_size = HIDDEN_DIM * EMBED_DIM * sizeof(float);
    cl_mem m_mlp1_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, mlp1_weight_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_mlp1_weight, CL_TRUE, 0, mlp1_weight_size, mlp1_w.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    const size_t mlp1_bias_size = HIDDEN_DIM * sizeof(float);
    cl_mem m_mlp1_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, mlp1_bias_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_mlp1_bias, CL_TRUE, 0, mlp1_bias_size, mlp1_b.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    const size_t mlp2_weight_size = EMBED_DIM * HIDDEN_DIM * sizeof(float);
    cl_mem m_mlp2_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, mlp2_weight_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_mlp2_weight, CL_TRUE, 0, mlp2_weight_size, mlp2_w.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    const size_t mlp2_bias_size = EMBED_DIM * sizeof(float);
    cl_mem m_mlp2_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, mlp2_bias_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_mlp2_bias, CL_TRUE, 0, mlp2_bias_size, mlp2_b.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    
    // create output mem obj
    cl_mem m_input_normalized = clCreateBuffer(container.context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);
    CHECK_CL_ERROR(err);

    cl_mem m_attn_out = clCreateBuffer(container.context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);
    CHECK_CL_ERROR(err);

    cl_mem m_residual = clCreateBuffer(container.context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);
    CHECK_CL_ERROR(err);
    
    cl_mem m_residual_normalized = clCreateBuffer(container.context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);
    CHECK_CL_ERROR(err);

    const size_t mlp_out_size = N_TOTAL_TOKEN * EMBED_DIM * sizeof(float);
    cl_mem m_mlp_out = clCreateBuffer(container.context, CL_MEM_READ_WRITE, mlp_out_size, NULL, &err);
    CHECK_CL_ERROR(err);

    cl_mem m_final_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);
    CHECK_CL_ERROR(err);



    /* ------------------------------------------------------------------------------------------- */
    // normalize input
    LOG("input_normalized", v_layer_norm(m_input, m_input_normalized, m_ln1_weight, m_ln1_bias));

    // multi-head self attention
    LOG(
        "multi-head self attention", 
        v_multihead_attn(m_input_normalized, m_attn_out, m_attn_in_weight, m_attn_in_bias, m_attn_out_weight, m_attn_out_bias, 0, NULL, NULL)
    );

    /* Residual1 */
    // skip-connection
    LOG(
        "1st Residual1", 
        v_matrix_plus(m_input, m_attn_out, m_residual, N_TOTAL_TOKEN * EMBED_DIM, 0, NULL, NULL)
    );

    // normalize again
    LOG("residual_normalized", v_layer_norm(m_residual, m_residual_normalized, m_ln2_weight, m_ln2_bias));

    /* MLP */
    LOG("MLP", v_mlp_block(m_residual_normalized, m_mlp_out, m_mlp1_weight, m_mlp1_bias, m_mlp2_weight, m_mlp2_bias));

    /* Residual2 */
    // skip connection again
    LOG("2nd Residual1", v_matrix_plus(m_residual, m_mlp_out, m_final_output, n_tokens * EMBED_DIM, 0, NULL, NULL));






    
    /* ------------------------------------------------------------------------------------------- */
    // read result
    err = clEnqueueReadBuffer(container.queue, m_final_output, CL_TRUE, 0, buffer_size, output, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    
    
    
    
    /* ------------------------------------------------------------------------------------------- */
    // release
    err = clReleaseMemObject(m_input);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_ln1_weight);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_ln1_bias);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_ln2_weight);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_ln2_bias);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_attn_in_weight);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_attn_in_bias);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_attn_out_weight);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_attn_out_bias);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_mlp1_weight);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_mlp1_bias);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_mlp2_weight);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_mlp2_bias);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_input_normalized);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_attn_out);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_residual);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_residual_normalized);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_mlp_out);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_final_output);
    CHECK_CL_ERROR(err);

    printf(">> [v_Encoder] ended\n");
}