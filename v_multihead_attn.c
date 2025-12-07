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

    UNUSED(e_num_waiting);
    UNUSED(e_waiting_arr);
    UNUSED(e_out);

    // create and write input memory obj
    // const size_t input_size = N_TOTAL_TOKEN * EMBED_DIM * sizeof(float);
    // cl_mem m_input = clCreateBuffer(container.context, CL_TRUE, input_size, NULL, &err);
    // CHECK_CL_ERROR(err);
    // err = clEnqueueWriteBuffer(container.queue, m_input, CL_TRUE, 0, input_size, input, 0, NULL, NULL);
    // CHECK_CL_ERROR(err);

    const size_t weight_size = EMBED_DIM * EMBED_DIM * sizeof(float);
    cl_mem m_q_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, weight_size, NULL, &err);
    CHECK_CL_ERROR(err);
    // err = clEnqueueWriteBuffer(container.queue, m_q_weight, CL_TRUE, 0, weight_size, in_weight.data + Q_dim * EMBED_DIM, 0, NULL, NULL);
    err = clEnqueueCopyBuffer(container.queue, m_in_weight, m_q_weight, Q_dim * EMBED_DIM * sizeof(float), 0, weight_size, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    cl_mem m_k_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, weight_size, NULL, &err);
    CHECK_CL_ERROR(err);
    // err = clEnqueueWriteBuffer(container.queue, m_k_weight, CL_TRUE, 0, weight_size, in_weight.data + K_dim * EMBED_DIM, 0, NULL, NULL);
    err = clEnqueueCopyBuffer(container.queue, m_in_weight, m_k_weight, K_dim * EMBED_DIM * sizeof(float), 0, weight_size, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    cl_mem m_v_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, weight_size, NULL, &err);
    CHECK_CL_ERROR(err);
    // err = clEnqueueWriteBuffer(container.queue, m_v_weight, CL_TRUE, 0, weight_size, in_weight.data + V_dim * EMBED_DIM, 0, NULL, NULL);
    err = clEnqueueCopyBuffer(container.queue, m_in_weight, m_v_weight, V_dim * EMBED_DIM * sizeof(float), 0, weight_size, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    const size_t bias_size = 1 * EMBED_DIM * sizeof(float);
    cl_mem m_q_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, bias_size, NULL, &err);
    CHECK_CL_ERROR(err);
    // err = clEnqueueWriteBuffer(container.queue, m_q_bias, CL_TRUE, 0, bias_size, in_bias.data + Q_dim, 0, NULL, NULL);
    err = clEnqueueCopyBuffer(container.queue, m_in_bias, m_q_bias, Q_dim * sizeof(float), 0, bias_size, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    cl_mem m_k_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, bias_size, NULL, &err);
    CHECK_CL_ERROR(err);
    // err = clEnqueueWriteBuffer(container.queue, m_k_bias, CL_TRUE, 0, bias_size, in_bias.data + K_dim, 0, NULL, NULL);
    err = clEnqueueCopyBuffer(container.queue, m_in_bias, m_k_bias, K_dim * sizeof(float), 0, bias_size, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    cl_mem m_v_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, bias_size, NULL, &err);
    CHECK_CL_ERROR(err);
    // err = clEnqueueWriteBuffer(container.queue, m_v_bias, CL_TRUE, 0, bias_size, in_bias.data + V_dim, 0, NULL, NULL);
    err = clEnqueueCopyBuffer(container.queue, m_in_bias, m_v_bias, V_dim * sizeof(float), 0, bias_size, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // create output memory objs
    const size_t qkv_size = N_TOTAL_TOKEN * EMBED_DIM * sizeof(float);
    cl_mem m_q_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, qkv_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_k_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, qkv_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_v_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, qkv_size, NULL, &err);
    CHECK_CL_ERROR(err);

    const size_t socre_size = N_TOTAL_TOKEN * N_TOTAL_TOKEN * sizeof(float);
    cl_mem m_scores = clCreateBuffer(container.context, CL_MEM_READ_WRITE, socre_size, NULL, &err);
    CHECK_CL_ERROR(err);

    const size_t attn_output_size = sizeof(float) * N_TOTAL_TOKEN * EMBED_DIM;
    cl_mem m_attn_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, attn_output_size, NULL, &err);
    CHECK_CL_ERROR(err);

    // const size_t out_bias_size = out_bias.size * sizeof(float);
    // cl_mem m_out_bias = clCreateBuffer(container.context, CL_TRUE, out_bias_size, NULL, &err);
    // CHECK_CL_ERROR(err);
    // err = clEnqueueWriteBuffer(container.queue, m_out_bias, CL_TRUE, 0, out_bias_size, out_bias.data, 0, NULL, NULL);
    // CHECK_CL_ERROR(err);
    
    // const size_t out_weight_size = out_weight.size * sizeof(float);
    // cl_mem m_out_weight = clCreateBuffer(container.context, CL_TRUE, out_weight_size, NULL, &err);
    // CHECK_CL_ERROR(err);
    // err = clEnqueueWriteBuffer(container.queue, m_out_weight, CL_TRUE, 0, out_weight_size, out_weight.data, 0, NULL, NULL);
    // CHECK_CL_ERROR(err);
    
    // const size_t fianl_output_size = N_TOTAL_TOKEN * EMBED_DIM* sizeof(float);
    // cl_mem m_fianl_output = clCreateBuffer(container.context, CL_TRUE, fianl_output_size, NULL, &err);
    // CHECK_CL_ERROR(err);



    
    // calculate Q, K, V
    v_linear_layer(m_input, m_q_weight, m_q_bias, m_q_output, N_TOTAL_TOKEN, EMBED_DIM, EMBED_DIM, 0, NULL, NULL);
    v_linear_layer(m_input, m_k_weight, m_k_bias, m_k_output, N_TOTAL_TOKEN, EMBED_DIM, EMBED_DIM, 0, NULL, NULL);
    v_linear_layer(m_input, m_v_weight, m_v_bias, m_v_output, N_TOTAL_TOKEN, EMBED_DIM, EMBED_DIM, 0, NULL, NULL);

    // multi head attention
    for (int h = 0; h < NUM_HEADS; h++) {
        int head_offset = h * HEAD_DIM;

        // calculate socres
        cal_scores(m_q_output, m_k_output, m_scores, N_TOTAL_TOKEN, EMBED_DIM, HEAD_DIM, head_offset, 0, NULL, NULL);

        // v_Softmax scores
        softmax_scores(m_scores, N_TOTAL_TOKEN, 0, NULL, NULL);
       
        // calculate result
        cal_attn_output(m_scores, m_v_output, m_attn_output, EMBED_DIM, head_offset, 0, NULL, NULL);
    }

    // calcualte final output
    v_linear_layer(m_attn_output, m_out_weight, m_out_bias, m_fianl_output, N_TOTAL_TOKEN, EMBED_DIM, EMBED_DIM, 0, NULL, NULL);

    // read result
    // err = clEnqueueReadBuffer(container.queue, m_fianl_output, CL_TRUE, 0, fianl_output_size, output, 0, NULL, NULL);
    // CHECK_CL_ERROR(err);




    // wrap up
    err = clReleaseMemObject(m_scores);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_attn_output);
    CHECK_CL_ERROR(err);
    // err = clReleaseMemObject(m_out_bias);
    // CHECK_CL_ERROR(err);
    // err = clReleaseMemObject(m_out_weight);
    // CHECK_CL_ERROR(err);
    // err = clReleaseMemObject(m_fianl_output);
    // CHECK_CL_ERROR(err);
    // err = clReleaseMemObject(m_input);
    // CHECK_CL_ERROR(err);
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
    UNUSED(e_num_waiting);
    UNUSED(e_waiting_arr);
    UNUSED(e_out);

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
    err = clEnqueueNDRangeKernel(container.queue, k, 2, NULL, dim_config, NULL, 0, NULL, NULL);
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
