#include "ViT_cl.h"

// multi-layer perceptron
void v_mlp_block(
    cl_mem m_input, cl_mem m_output,
    cl_mem m_weight1, cl_mem m_bias1,
    cl_mem m_weight2, cl_mem m_bias2,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
) {
    // create mem obj
    const size_t mid_n_data = N_TOTAL_TOKEN * HIDDEN_DIM;
    const size_t mid_size = N_TOTAL_TOKEN * HIDDEN_DIM * sizeof(float);
    cl_mem m_mid = clCreateBuffer(container.context, CL_MEM_READ_WRITE, mid_size, NULL, &err);
    CHECK_CL_ERROR(err);

    cl_event e[2];

    // first layer
    v_linear_layer(
        m_input, m_weight1, m_bias1, m_mid, 
        N_TOTAL_TOKEN, EMBED_DIM, HIDDEN_DIM,
        e_num_waiting, e_waiting_arr, &e[0]
    );

    // gelu
    v_gelu(m_mid, mid_n_data, 1, &e[0], &e[1]);

    // second layer
    v_linear_layer(
        m_mid, m_weight2, m_bias2, m_output, 
        N_TOTAL_TOKEN, HIDDEN_DIM, EMBED_DIM,
        1, &e[1], e_out
    );
}

void v_gelu (
    cl_mem m_data, size_t n_data,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
) {
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
        e_num_waiting,  e_waiting_arr, e_out
    );
    CHECK_CL_ERROR(err);
}
