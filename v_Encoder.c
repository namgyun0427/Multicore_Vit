#include "ViT_cl.h"

// encode input
// input[total_num_patches + 1][EMBED_DIM]
void v_Encoder(
    cl_mem m_input, cl_mem m_final_output,
    cl_mem m_ln1_weight, cl_mem m_ln1_bias,
    cl_mem m_attn_in_weight, cl_mem m_attn_in_bias, cl_mem m_attn_out_weight, cl_mem m_attn_out_bias, 
    cl_mem m_ln2_weight, cl_mem m_ln2_bias, 
    cl_mem m_mlp1_weight, cl_mem m_mlp1_bias, cl_mem m_mlp2_weight, cl_mem m_mlp2_bias,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
) {
    LOG(">> [v_Encoder] started\n", {});
    
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




    // normalize input
    cl_event e_input_normalized;
    LOG("input_normalized", v_layer_norm(m_input, m_input_normalized, m_ln1_weight, m_ln1_bias, e_num_waiting, e_waiting_arr, &e_input_normalized));

    // multi-head self attention
    cl_event e_attention;
    LOG(
        "multi-head self attention", 
        v_multihead_attn(m_input_normalized, m_attn_out, m_attn_in_weight, m_attn_in_bias, m_attn_out_weight, m_attn_out_bias, 1, &e_input_normalized, &e_attention)
    );

    /* Residual1 */
    // skip-connection
    cl_event e_residual_1;
    LOG(
        "1st Residual1", 
        v_matrix_plus(m_input, m_attn_out, m_residual, N_TOTAL_TOKEN * EMBED_DIM, 1, &e_attention, &e_residual_1)
    );

    // normalize again
    cl_event e_residual_normalized;
    LOG("residual_normalized", v_layer_norm(m_residual, m_residual_normalized, m_ln2_weight, m_ln2_bias, 1, &e_residual_1, &e_residual_normalized));

    /* MLP */
    cl_event e_mlp;
    LOG("MLP", v_mlp_block(m_residual_normalized, m_mlp_out, m_mlp1_weight, m_mlp1_bias, m_mlp2_weight, m_mlp2_bias, 1, &e_residual_normalized, &e_mlp));

    /* Residual2 */
    // skip connection again
    // cl_event e_residual_2;
    LOG("2nd Residual1", v_matrix_plus(m_residual, m_mlp_out, m_final_output, N_TOTAL_TOKEN * EMBED_DIM, 1, &e_mlp, e_out));



    
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


    LOG(">> [v_Encoder] ended\n", {});
}