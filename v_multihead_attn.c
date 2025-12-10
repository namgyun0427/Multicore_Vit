#include "ViT_cl.h"

static void cal_scores(
    cl_mem m_q_output, cl_mem m_k_output, 
    cl_mem m_scores, 
    int n_total_tokens, int embed_dim, int head_dim, int head_offset,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
);

static void cal_attn_output (
    cl_mem m_scores, cl_mem m_v_output, 
    cl_mem m_attn_output, 
    int embed_dim, int head_offset,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
);

static void softmax_scores (
    cl_mem m_scores, int n_tokens, 
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
);



// multi-head self attention
// input[total_num_patches + 1][EMBED_DIM]
void v_multihead_attn(
    cl_mem m_input, cl_mem m_fianl_output,
    cl_mem m_in_weight, cl_mem m_in_bias,
    cl_mem m_out_weight, cl_mem m_out_bias,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
) {
    const int Q_dim = 0;
    const int K_dim = EMBED_DIM;
    const int V_dim = EMBED_DIM * 2;

    // create and write input memory obj
    // weight
    const size_t weight_size = EMBED_DIM * EMBED_DIM * sizeof(float);
    cl_mem m_q_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, weight_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_k_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, weight_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_v_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, weight_size, NULL, &err);
    CHECK_CL_ERROR(err);
    // bias
    const size_t bias_size = 1 * EMBED_DIM * sizeof(float);
    cl_mem m_q_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, bias_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_k_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, bias_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_v_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, bias_size, NULL, &err);
    CHECK_CL_ERROR(err);
    
    // create output memory objs
    // qkv
    const size_t qkv_size = N_TOTAL_TOKEN * EMBED_DIM * sizeof(float);
    cl_mem m_q_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, qkv_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_k_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, qkv_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_v_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, qkv_size, NULL, &err);
    CHECK_CL_ERROR(err);
    // att output
    const size_t attn_output_size = sizeof(float) * N_TOTAL_TOKEN * EMBED_DIM;
    cl_mem m_attn_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, attn_output_size, NULL, &err);
    CHECK_CL_ERROR(err);



    
    // write mem obj
    cl_event e_write[6];
    err = clEnqueueCopyBuffer(container.queue, m_in_weight, m_q_weight, Q_dim * EMBED_DIM * sizeof(float), 0, weight_size, e_num_waiting, e_waiting_arr, &e_write[0]);
    CHECK_CL_ERROR(err);
    err = clEnqueueCopyBuffer(container.queue, m_in_weight, m_k_weight, K_dim * EMBED_DIM * sizeof(float), 0, weight_size, e_num_waiting, e_waiting_arr, &e_write[1]);
    CHECK_CL_ERROR(err);
    err = clEnqueueCopyBuffer(container.queue, m_in_weight, m_v_weight, V_dim * EMBED_DIM * sizeof(float), 0, weight_size, e_num_waiting, e_waiting_arr, &e_write[2]);
    CHECK_CL_ERROR(err);
    err = clEnqueueCopyBuffer(container.queue, m_in_bias, m_q_bias, Q_dim * sizeof(float), 0, bias_size, e_num_waiting, e_waiting_arr, &e_write[3]);
    CHECK_CL_ERROR(err);
    err = clEnqueueCopyBuffer(container.queue, m_in_bias, m_k_bias, K_dim * sizeof(float), 0, bias_size, e_num_waiting, e_waiting_arr, &e_write[4]);
    CHECK_CL_ERROR(err);
    err = clEnqueueCopyBuffer(container.queue, m_in_bias, m_v_bias, V_dim * sizeof(float), 0, bias_size, e_num_waiting, e_waiting_arr, &e_write[5]);
    CHECK_CL_ERROR(err);



    
    // calculate Q, K, V
    cl_event e_qkv[3];
    v_linear_layer(m_input, m_q_weight, m_q_bias, m_q_output, N_TOTAL_TOKEN, EMBED_DIM, EMBED_DIM, 6, e_write, &e_qkv[0]);
    v_linear_layer(m_input, m_k_weight, m_k_bias, m_k_output, N_TOTAL_TOKEN, EMBED_DIM, EMBED_DIM, 6, e_write, &e_qkv[1]);
    v_linear_layer(m_input, m_v_weight, m_v_bias, m_v_output, N_TOTAL_TOKEN, EMBED_DIM, EMBED_DIM, 6, e_write, &e_qkv[2]);



    // multi head attention
    cl_event e_multihead_end[NUM_HEADS];
    for (int h = 0; h < NUM_HEADS; h++) {
        int head_offset = h * HEAD_DIM;
        cl_event e_multihead_mid[3];

        // create score
        const size_t socre_size = N_TOTAL_TOKEN * N_TOTAL_TOKEN * sizeof(float);
        cl_mem m_scores = clCreateBuffer(container.context, CL_MEM_READ_WRITE, socre_size, NULL, &err);
        CHECK_CL_ERROR(err);

        // calculate socres
        cal_scores(m_q_output, m_k_output, m_scores, N_TOTAL_TOKEN, EMBED_DIM, HEAD_DIM, head_offset, 3, e_qkv, &e_multihead_mid[0]);

        // v_Softmax scores
        softmax_scores(m_scores, N_TOTAL_TOKEN, 1, &e_multihead_mid[0], &e_multihead_mid[1]);
       
        // calculate result
        cal_attn_output(m_scores, m_v_output, m_attn_output, EMBED_DIM, head_offset, 1, &e_multihead_mid[1], &e_multihead_end[h]);

        // release score
        err = clReleaseMemObject(m_scores);
        CHECK_CL_ERROR(err);
    }

    // calcualte final output
    v_linear_layer(m_attn_output, m_out_weight, m_out_bias, m_fianl_output, N_TOTAL_TOKEN, EMBED_DIM, EMBED_DIM, NUM_HEADS, e_multihead_end, e_out);




    // wrap up
    err = clReleaseMemObject(m_attn_output);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_q_weight);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_k_weight);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_v_weight);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_q_bias);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_k_bias);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_v_bias);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_q_output);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_k_output);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_v_output);
    CHECK_CL_ERROR(err);
}


static void cal_scores(
    cl_mem m_q_output, cl_mem m_k_output, 
    cl_mem m_scores, 
    int n_total_tokens, int embed_dim, int head_dim, int head_offset,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
) {
    cl_kernel k = container.kernels[__cal_score];

    // set kernel args
    // __kernel void cal_score (
    //     __global float* g_Q,
    //     __global float* g_K,
    //     __global float* g_scores,
    //     int N_TOTAL_TOKEN,
    //     int EMBED_DIM,
    //     int HEAD_DIM,
    //     int head_offset,
    //     float scale
    // ) {
    float scale = sqrtf((float)head_dim);
    KernelArg args[] = {
        { .size = sizeof(cl_mem), .addr = &m_q_output },
        { .size = sizeof(cl_mem), .addr = &m_k_output },
        { .size = sizeof(cl_mem), .addr = &m_scores },
        { .size = sizeof(int), .addr = &n_total_tokens },
        { .size = sizeof(int), .addr = &embed_dim },
        { .size = sizeof(int), .addr = &head_dim },
        { .size = sizeof(int), .addr = &head_offset },
        { .size = sizeof(float), .addr = &scale },
    };
    for (int i=0; i< 8; ++i) {
        err = clSetKernelArg(k, i, args[i].size, args[i].addr);
        CHECK_CL_ERROR(err);
    }

    // run kernels
    const size_t dim_config[] = { N_TOTAL_TOKEN, N_TOTAL_TOKEN };
    err = clEnqueueNDRangeKernel(container.queue, k, 2, NULL, dim_config, NULL, e_num_waiting, e_waiting_arr, e_out);
    CHECK_CL_ERROR(err);
}


static void cal_attn_output (
    cl_mem m_scores, cl_mem m_v_output, 
    cl_mem m_attn_output, 
    int embed_dim, int head_offset,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
) {
    cl_kernel k = container.kernels[__cal_result];

    // set kernel args
    // __kernel void cal_result(
    // 	__global float* g_scores,
    // 	__global float* g_V,
    // 	__global float* g_attn_output,
    // 	int EMBED_DIM,
    // 	int head_offset
    // ) {
    KernelArg args[] = {
        { .size = sizeof(cl_mem), .addr = &m_scores },
        { .size = sizeof(cl_mem), .addr = &m_v_output },
        { .size = sizeof(cl_mem), .addr = &m_attn_output },
        { .size = sizeof(int), .addr = &embed_dim },
        { .size = sizeof(int), .addr = &head_offset },
    };
    for (int i=0; i<5; ++i) {
        err = clSetKernelArg(k, i, args[i].size, args[i].addr);
        CHECK_CL_ERROR(err);
    }

    // run kernel
    const size_t dim_config[] = { N_TOTAL_TOKEN, HEAD_DIM };
    err = clEnqueueNDRangeKernel(container.queue, k, 2, NULL, dim_config, NULL, e_num_waiting, e_waiting_arr, e_out);
    CHECK_CL_ERROR(err);
}


static void softmax_scores (
    cl_mem m_scores, int n_tokens, 
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
) {
    cl_kernel k = container.kernels[__softmax_score];
    int cols = n_tokens; 
    size_t local_mem_size = 256 * sizeof(float);

    int arg_idx = 0;
    clSetKernelArg(k, arg_idx++, sizeof(cl_mem), &m_scores);
    clSetKernelArg(k, arg_idx++, sizeof(int), &cols);
    clSetKernelArg(k, arg_idx++, local_mem_size, NULL);
    size_t local_work_size[] = { 256 };
    size_t global_work_size[] = { (size_t)n_tokens * 256 };

    clEnqueueNDRangeKernel(container.queue, k, 1, NULL, global_work_size, local_work_size, e_num_waiting, e_waiting_arr, e_out);    
}
