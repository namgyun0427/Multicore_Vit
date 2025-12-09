#include "ViT_cl.h"

void v_matrix_plus(
    cl_mem m_input1, cl_mem m_input2, 
    cl_mem m_output, 
    size_t n_data,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
) {
    assert(n_data % 4 == 0);

    cl_kernel k = container.kernels[__matrix_plus];

    // set kernel args
    // __kernel void matrix_plus (
    //     __global float* g_A,
    //     __global float* g_B,
    //     __global float* g_C
    // ) {
    KernelArg args[] = {
        {.size = sizeof(cl_mem), .addr = &m_input1 },
        {.size = sizeof(cl_mem), .addr = &m_input2 },
        {.size = sizeof(cl_mem), .addr = &m_output }
    };

    for (int i = 0; i < 3; ++i) {
        err = clSetKernelArg(k, i, args[i].size, args[i].addr);
        CHECK_CL_ERROR(err);
    }

    // run kernel
    size_t dim_config[] = { n_data / 4 };
    err = clEnqueueNDRangeKernel(
        container.queue, k,
        1, NULL, dim_config, NULL,
        e_num_waiting, e_waiting_arr, e_out
    );
    CHECK_CL_ERROR(err);
}
