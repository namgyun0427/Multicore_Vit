#include "ViT_cl.h"

// multi-layer perceptron
void v_mlp_block(
    float* input, float* output,
    Network fc1_weight, Network fc1_bias,
    Network fc2_weight, Network fc2_bias
) {
    // create and write mem obj
    // input data
    const size_t input_size = N_TOTAL_TOKEN * EMBED_DIM * sizeof(float);
    cl_mem m_input = clCreateBuffer(container.context, CL_MEM_READ_WRITE, input_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_input, CL_TRUE, 0, input_size, input, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // weight1
    const size_t weight1_size = HIDDEN_DIM * EMBED_DIM * sizeof(float);
    cl_mem m_weight1 = clCreateBuffer(container.context, CL_MEM_READ_WRITE, weight1_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_weight1, CL_TRUE, 0, weight1_size, fc1_weight.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // bias1
    const size_t bias_size1 = HIDDEN_DIM * sizeof(float);
    cl_mem m_bias1 = clCreateBuffer(container.context, CL_MEM_READ_WRITE, bias_size1, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_bias1, CL_TRUE, 0, bias_size1, fc1_bias.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // mid data -> no write
    const size_t mid_n_data = N_TOTAL_TOKEN * HIDDEN_DIM;
    const size_t mid_size = N_TOTAL_TOKEN * HIDDEN_DIM * sizeof(float);
    cl_mem m_mid = clCreateBuffer(container.context, CL_MEM_READ_WRITE, mid_size, NULL, &err);
    CHECK_CL_ERROR(err);

    // weight2
    const size_t weight2_size = EMBED_DIM * HIDDEN_DIM * sizeof(float);
    cl_mem m_weight2 = clCreateBuffer(container.context, CL_MEM_READ_WRITE, weight2_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_weight2, CL_TRUE, 0, weight2_size, fc2_weight.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // bias2
    const size_t bias2_size = EMBED_DIM * sizeof(float);
    cl_mem m_bias2 = clCreateBuffer(container.context, CL_MEM_READ_WRITE, bias2_size, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_bias2, CL_TRUE, 0, bias2_size, fc2_bias.data, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // output -> no write
    const size_t output_size = N_TOTAL_TOKEN * EMBED_DIM * sizeof(float);
    cl_mem m_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, output_size, NULL, &err);
    CHECK_CL_ERROR(err);



    // first layer
    v_linear_layer(
        m_input, m_weight1, m_bias1, m_mid, 
        N_TOTAL_TOKEN, EMBED_DIM, HIDDEN_DIM,
        0, NULL, NULL
        );

    // gelu
    v_gelu(m_mid, mid_n_data);

    // second layer
    v_linear_layer(
        m_mid, m_weight2, m_bias2, m_output, 
        N_TOTAL_TOKEN, HIDDEN_DIM, EMBED_DIM,
        0, NULL, NULL
        );



    // read result
    err = clEnqueueReadBuffer(container.queue, m_output, CL_TRUE, 0, output_size, output, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // releaase mem obj
    err = clReleaseMemObject(m_input);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_weight1);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_bias1);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_mid);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_weight2);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_bias2);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_output);
    CHECK_CL_ERROR(err);
}

void v_gelu (cl_mem m_data, size_t n_data) {
    cl_kernel k = container.kernels[__gelu];

    // set kernel args
    // __kernel void gelu(
    //     __global float* g_data
    // ) {
    KernelArg args[] = {
        { .size = sizeof(cl_mem), .addr = &m_data },
    };

    for (int i=0; i<1; ++i) {
        err = clSetKernelArg(k, i, args[i].size, args[i].addr);
        CHECK_CL_ERROR(err);
    }

    // run kerenl
    size_t global_work_size[] = { n_data };
    err = clEnqueueNDRangeKernel(
        container.queue, k, 
        1, NULL, global_work_size, NULL, 
        0, NULL, NULL
    );
    CHECK_CL_ERROR(err);
}