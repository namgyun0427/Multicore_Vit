#include "ViT_cl.h"

// multi-head self attention
// input[total_num_patches + 1][EMBED_DIM]
void v_multihead_attn(
    float* input, float* output,
    Network in_weight, Network in_bias,
    Network out_weight, Network out_bias
    ) {
    const int n_tokens = ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE)) + 1;
    const int Q_dim = 0;
    const int K_dim = EMBED_DIM;
    const int V_dim = EMBED_DIM * 2;
    const int n_total_tokens = N_TOTAL_TOKEN;
    const int embed_dim = EMBED_DIM;

    float* Q = (float*)malloc(sizeof(float) * n_tokens * EMBED_DIM);
    float* K = (float*)malloc(sizeof(float) * n_tokens * EMBED_DIM);
    float* V = (float*)malloc(sizeof(float) * n_tokens * EMBED_DIM);

    // calculate QKV
    // create memory obj
    const size_t input_size = n_tokens * EMBED_DIM * sizeof(float);
    cl_mem m_input = clCreateBuffer(container.context, CL_TRUE, input_size, NULL, &err);
    CHECK_CL_ERROR(err);

    const size_t weight_size = EMBED_DIM * EMBED_DIM * sizeof(float);
    cl_mem m_q_weight = clCreateBuffer(container.context, CL_TRUE, weight_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_k_weight = clCreateBuffer(container.context, CL_TRUE, weight_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_v_weight = clCreateBuffer(container.context, CL_TRUE, weight_size, NULL, &err);
    CHECK_CL_ERROR(err);

    const size_t bias_size = 1 * EMBED_DIM * sizeof(float);
    cl_mem m_q_bias = clCreateBuffer(container.context, CL_TRUE, bias_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_k_bias = clCreateBuffer(container.context, CL_TRUE, bias_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_v_bias = clCreateBuffer(container.context, CL_TRUE, bias_size, NULL, &err);
    CHECK_CL_ERROR(err);

    const size_t qkv_size = n_tokens * EMBED_DIM * sizeof(float);
    cl_mem m_q_output = clCreateBuffer(container.context, CL_TRUE, qkv_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_k_output = clCreateBuffer(container.context, CL_TRUE, qkv_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_v_output = clCreateBuffer(container.context, CL_TRUE, qkv_size, NULL, &err);
    CHECK_CL_ERROR(err);

    const size_t socre_size = N_TOTAL_TOKEN * N_TOTAL_TOKEN * sizeof(float);
    cl_mem m_scores = clCreateBuffer(container.context, CL_TRUE, socre_size, NULL, &err);
    CHECK_CL_ERROR(err);

    const size_t attn_output_size = sizeof(float) * N_TOTAL_TOKEN * EMBED_DIM;
    cl_mem m_attn_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, attn_output_size, NULL, &err);
    CHECK_CL_ERROR(err);


    const size_t out_bias_size = out_bias.size * sizeof(float);
    cl_mem m_out_bias = clCreateBuffer(container.context, CL_TRUE, out_bias_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_out_bias, CL_TRUE, 0, out_bias_size, out_bias.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    
    const size_t out_weight_size = out_weight.size * sizeof(float);
    cl_mem m_out_weight = clCreateBuffer(container.context, CL_TRUE, out_weight_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_out_weight, CL_TRUE, 0, out_weight_size, out_weight.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    
    const size_t fianl_output_size = N_TOTAL_TOKEN * EMBED_DIM* sizeof(float);
    cl_mem m_fianl_output = clCreateBuffer(container.context, CL_TRUE, fianl_output_size, NULL, &err);
    CHECK_CL_ERROR(err);






    // write
    err = clEnqueueWriteBuffer(container.queue, m_input, CL_TRUE, 0, input_size, input, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    err = clEnqueueWriteBuffer(container.queue, m_q_weight, CL_TRUE, 0, weight_size, in_weight.data + Q_dim * EMBED_DIM, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_k_weight, CL_TRUE, 0, weight_size, in_weight.data + K_dim * EMBED_DIM, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_v_weight, CL_TRUE, 0, weight_size, in_weight.data + V_dim * EMBED_DIM, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    err = clEnqueueWriteBuffer(container.queue, m_q_bias, CL_TRUE, 0, bias_size, in_bias.data + Q_dim, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_k_bias, CL_TRUE, 0, bias_size, in_bias.data + K_dim, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_v_bias, CL_TRUE, 0, bias_size, in_bias.data + V_dim, 0, NULL, NULL);
    CHECK_CL_ERROR(err);


    



    // calculate Q, K, V
    v_linear_layer(m_input, m_q_weight, m_q_bias, m_q_output, n_tokens, EMBED_DIM, EMBED_DIM, 0, NULL, NULL);
    v_linear_layer(m_input, m_k_weight, m_k_bias, m_k_output, n_tokens, EMBED_DIM, EMBED_DIM, 0, NULL, NULL);
    v_linear_layer(m_input, m_v_weight, m_v_bias, m_v_output, n_tokens, EMBED_DIM, EMBED_DIM, 0, NULL, NULL);



    /* ================================================================================ */





    int head_dim = EMBED_DIM / NUM_HEADS;
    float* scores = (float*)malloc(sizeof(float) * n_tokens * n_tokens);
    float* head_out = (float*)malloc(sizeof(float) * n_tokens * head_dim);
    float* attn_output = (float*)malloc(sizeof(float) * n_tokens * EMBED_DIM);

    for (int h = 0; h < NUM_HEADS; h++) {
        int head_offset = h * head_dim;


        {
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



        // v_Softmax scores
        {
            cl_kernel k = container.kernels[__softmax_score];
            int cols = n_tokens; 
            size_t local_mem_size = 256 * sizeof(float);

            int arg_idx = 0;
            clSetKernelArg(k, arg_idx++, sizeof(cl_mem), &m_scores);
            clSetKernelArg(k, arg_idx++, sizeof(int), &cols);
            clSetKernelArg(k, arg_idx++, local_mem_size, NULL);
            size_t local_work_size[] = { 256 };
            size_t global_work_size[] = { (size_t)n_tokens * 256 };

            clEnqueueNDRangeKernel(container.queue, k, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);    
        }
       
        // calculate result
        {
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
            err = clEnqueueNDRangeKernel(container.queue, k, 2, NULL, dim_config, NULL, 0, NULL, NULL);
            CHECK_CL_ERROR(err);
        }
    }


    
    

    // calcualte finaloutput
    v_linear_layer(m_attn_output, m_out_weight, m_out_bias, m_fianl_output, N_TOTAL_TOKEN, EMBED_DIM, EMBED_DIM, 0, NULL, NULL);




    // read
    err = clEnqueueReadBuffer(container.queue, m_fianl_output, CL_TRUE, 0, fianl_output_size, output, 0, NULL, NULL);
    CHECK_CL_ERROR(err);



    // wrap up
    free(scores); free(head_out); free(attn_output); free(Q); free(K); free(V);







    clReleaseMemObject(m_scores);


    err = clReleaseMemObject(m_attn_output);
    err = clReleaseMemObject(m_out_bias);
    err = clReleaseMemObject(m_out_weight);
    err = clReleaseMemObject(m_fianl_output);


    err = clReleaseMemObject(m_input);
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