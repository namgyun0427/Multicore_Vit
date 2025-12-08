#include "ViT_cl.h"

// encode input
// input[total_num_patches + 1][EMBED_DIM]
void v_Encoder(
    cl_mem m_input, cl_mem m_final_output,
    cl_mem m_ln1_weight, cl_mem m_ln1_bias,
    cl_mem m_attn_in_weight, cl_mem m_attn_in_bias, cl_mem m_attn_out_weight, cl_mem m_attn_out_bias, 
    cl_mem m_ln2_weight, cl_mem m_ln2_bias, 
    cl_mem m_mlp1_weight, cl_mem m_mlp1_bias, cl_mem m_mlp2_weight, cl_mem m_mlp2_bias
) {
    printf(">> [v_Encoder] started\n");
    
    // create output mem obj
    size_t buffer_size = sizeof(float) * N_TOTAL_TOKEN * EMBED_DIM;
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
    LOG("2nd Residual1", v_matrix_plus(m_residual, m_mlp_out, m_final_output, N_TOTAL_TOKEN * EMBED_DIM, 0, NULL, NULL));



    
    /* ------------------------------------------------------------------------------------------- */
    // release
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


    printf(">> [v_Encoder] ended\n");
}