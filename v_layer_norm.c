#include "ViT_cl.h"

// normalize
// input[total_num_patches + 1][EMBED_DIM] => output[total_num_patches + 1][EMBED_DIM]
// TODO: cl_mem 받는 함수로

void v_layer_norm(
    float* input, float* output,
    Network weight, Network bias
    ) {
    const int token = N_TOTAL_TOKEN;
    const int total_num_data = token * EMBED_DIM;

    memcpy(output, input, total_num_data * sizeof(float));

    const size_t work_group_size = 1024;
    for (int t = 0; t < token; t++) {
        float* p_data = output + t * EMBED_DIM;
        const size_t n_data = EMBED_DIM;

        // create and write mem obj
        size_t data_size = n_data * sizeof(float);
        cl_mem m_data = clCreateBuffer(container.context, CL_MEM_READ_WRITE, data_size, NULL, &err);
        err = clEnqueueWriteBuffer(container.queue, m_data, CL_TRUE, 0, data_size, p_data, 0, NULL, NULL);

        size_t weight_size = weight.size * sizeof(float);
        cl_mem m_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, weight_size, NULL, &err);
        err = clEnqueueWriteBuffer(container.queue, m_weight, CL_TRUE, 0, weight_size, weight.data, 0, NULL, NULL);

        size_t bias_size = bias.size * sizeof(float);
        cl_mem m_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, bias_size, NULL, &err);
        err = clEnqueueWriteBuffer(container.queue, m_bias, CL_TRUE, 0, bias_size, bias.data, 0, NULL, NULL);

        size_t just_float1_size = sizeof(float);
        cl_mem m_sum = clCreateBuffer(container.context, CL_MEM_READ_WRITE, just_float1_size, NULL, &err);
        cl_mem m_sum_of_square = clCreateBuffer(container.context, CL_MEM_READ_WRITE, just_float1_size, NULL, &err);
        cl_mem m_mean = clCreateBuffer(container.context, CL_MEM_READ_WRITE, just_float1_size, NULL, &err);
        cl_mem m_inv_std = clCreateBuffer(container.context, CL_MEM_READ_WRITE, just_float1_size, NULL, &err);



        // run kernels
        v_reduce_sum(m_data, m_sum, n_data, work_group_size, 0, NULL, NULL);
        v_reduce_sum_of_square(m_data, m_sum_of_square, n_data, work_group_size, 0, NULL, NULL);
        v_cal_mean_and_inv_std(m_sum, m_sum_of_square, m_mean, m_inv_std, 0, NULL, NULL);
        v_normalize(m_data, m_weight, m_bias, m_mean, m_inv_std, total_num_data, 0, NULL, NULL);



        // read result
        err = clEnqueueReadBuffer(container.queue, m_data, CL_TRUE, 0, data_size, p_data, 0, NULL, NULL);


        // release mem objs
        err = clReleaseMemObject(m_data);
        err = clReleaseMemObject(m_weight);
        err = clReleaseMemObject(m_bias);
        err = clReleaseMemObject(m_sum);
        err = clReleaseMemObject(m_sum_of_square);
        err = clReleaseMemObject(m_mean);
        err = clReleaseMemObject(m_inv_std);
    }
}

void v_cal_mean_and_inv_std(
    cl_mem m_sum, cl_mem m_sum_of_square,
    cl_mem m_mean, cl_mem m_inv_std,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
    ) {
    cl_kernel k = container.kernels[__cal_mean_and_inv_std];

    // set kernel args
    // __kernel void cal_mean_and_inv_std (
    //     __global float* g_sum,
    //     __global float* g_sum_of_square,
    //     __global float* g_output_mean,
    //     __global float* g_output_inv_std,
    //     int EMBED_DIM,
    //     float EPSILON
    // ) {
    const int embed_dim = EMBED_DIM;
    const float epsilon = EPSILON;
    KernelArg args[] = {
        {.size = sizeof(cl_mem), .addr = &m_sum },
        {.size = sizeof(cl_mem), .addr = &m_sum_of_square },
        {.size = sizeof(cl_mem), .addr = &m_mean },
        {.size = sizeof(cl_mem), .addr = &m_inv_std },
        {.size = sizeof(int), .addr = &embed_dim },
        {.size = sizeof(float), .addr = &epsilon },
    };

    for (int i = 0; i < 6; ++i) {
        err = clSetKernelArg(k, i, args[i].size, args[i].addr);
        CHECK_CL_ERROR(err);
    }

    // run kernel
    size_t global_work_size[] = { 1 };
    err = clEnqueueNDRangeKernel(
        container.queue, k,
        1, NULL, global_work_size, NULL,
        e_num_waiting, e_waiting_arr, e_out
        );
    CHECK_CL_ERROR(err);
}

void v_reduce_sum(
    cl_mem m_data,
    cl_mem m_output,
    size_t total_num_data,
    size_t work_group_size,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
    ) {
    // work_group_size는 2의 거듭제곱수여야 함
    assert(((work_group_size & (work_group_size - 1)) == 0));

    // create memory object
    const size_t buf_size = total_num_data * sizeof(float);
    cl_mem m_from = clCreateBuffer(container.context, CL_MEM_READ_WRITE, buf_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_to = clCreateBuffer(container.context, CL_MEM_READ_WRITE, buf_size, NULL, &err);
    CHECK_CL_ERROR(err);

    // copy inpupt
    err = clEnqueueCopyBuffer(container.queue, m_data, m_from, 0, 0, buf_size, e_num_waiting, e_waiting_arr, NULL);
    CHECK_CL_ERROR(err);

    size_t n_data = total_num_data;
    while (n_data > 1) {
        // n_grouup = ceil(n_data / work_group_size)
        size_t n_group = (n_data + work_group_size - 1) / work_group_size;

        // set kernel args
        // __kernel void reduce_sum (
        //     __global float* const g_input,
        //     __global float* const g_output,
        //     int n_data,
        //     __local float* l_sum
        // ) {
        KernelArg args[] = {
            {.size = sizeof(cl_mem), .addr = &m_from},
            {.size = sizeof(cl_mem), .addr = &m_to},
            {.size = sizeof(int), .addr = &n_data},
            {.size = sizeof(float) * work_group_size, .addr = NULL}
        };

        for (int i = 0; i < 4; ++i) {
            err = clSetKernelArg(container.kernels[__reduce_sum], i, args[i].size, args[i].addr);
            CHECK_CL_ERROR(err);
        }

        // run kernel
        const size_t global_work_size = n_group * work_group_size;
        const size_t local_work_size = work_group_size;
        err = clEnqueueNDRangeKernel(
            container.queue, container.kernels[__reduce_sum],
            1, NULL, &global_work_size, &local_work_size,
            0, NULL, NULL);
        CHECK_CL_ERROR(err);

        // swap input and output buffers
        cl_mem tmp = m_from;
        m_from = m_to;
        m_to = tmp;

        // set next value
        n_data = n_group;
    }

    // write result to m_output
    // m_from[0]에 최종결과가 남음
    err = clEnqueueCopyBuffer(container.queue, m_from, m_output, 0, 0, sizeof(float), 0, NULL, e_out);

    // release mem obj
    err = clReleaseMemObject(m_from);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_to);
    CHECK_CL_ERROR(err);
}


void v_reduce_sum_of_square(
    cl_mem m_data,
    cl_mem m_output,
    size_t total_num_data,
    size_t work_group_size,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
    ) {
    // work_group_size는 2의 거듭제곱수여야 함
    assert(((work_group_size & (work_group_size - 1)) == 0));

    // 흐음 얘 어떻게 해야하지...?
    UNUSED(e_out);

    // create memory objects
    size_t data_size = total_num_data * sizeof(float);
    cl_mem m_tmp = clCreateBuffer(container.context, CL_MEM_READ_WRITE, data_size, NULL, &err);
    CHECK_CL_ERROR(err);

    // copy data to m_tmp
    err = clEnqueueCopyBuffer(container.queue, m_data, m_tmp, 0, 0, data_size, e_num_waiting, e_waiting_arr, NULL);
    CHECK_CL_ERROR(err);

    // square m_tmp matrix
    {
        // set args
        // __kernel void load_square (
        //     __global float* g_input
        // ) {
        //     size_t global_id = get_global_id(0);
        //     g_input[global_id] = g_input[global_id] * g_input[global_id];
        // }
        KernelArg args[] = {
            {.size = sizeof(cl_mem), .addr = &m_tmp},
        };

        for (int i = 0; i < 1; ++i) {
            err = clSetKernelArg(container.kernels[__load_square], i, args[i].size, args[i].addr);
            CHECK_CL_ERROR(err);
        }

        // run kernel
        const size_t global_work_size = total_num_data;
        err = clEnqueueNDRangeKernel(
            container.queue, container.kernels[__load_square],
            1, NULL, &global_work_size, NULL,
            0, NULL, NULL);
        CHECK_CL_ERROR(err);
    }

    // process reduce sum with squared data matrix
    v_reduce_sum(m_tmp, m_output, total_num_data, work_group_size, 0, NULL, NULL);

    // release memory objects
    err = clReleaseMemObject(m_tmp);
    CHECK_CL_ERROR(err);
}


void v_normalize(
    cl_mem m_input,
    cl_mem m_weight, cl_mem m_bias,
    cl_mem m_mean, cl_mem m_inv_std,
    int total_num_data,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
    ) {
    // set kernel args
    // __kernel void my_normalize(
    //     __global float* g_input,
    //     __global const float* g_weight,
    //     __global const float* g_bias,
    //     __global float* g_MEAN,
    //     __global float* g_INV_STD
    // ) {
    KernelArg args[] = {
        {.size = sizeof(cl_mem), .addr = &m_input},
        {.size = sizeof(cl_mem), .addr = &m_weight},
        {.size = sizeof(cl_mem), .addr = &m_bias},
        {.size = sizeof(cl_mem), .addr = &m_mean},
        {.size = sizeof(cl_mem), .addr = &m_inv_std}
    };
    for (int i = 0; i < 5; ++i) {
        err = clSetKernelArg(container.kernels[__normalize], i, args[i].size, args[i].addr);
        CHECK_CL_ERROR(err);
    }

    // run kernel
    const size_t global_work_size = total_num_data;
    err = clEnqueueNDRangeKernel(
        container.queue, container.kernels[__normalize],
        1, NULL, &global_work_size, NULL,
        e_num_waiting, e_waiting_arr, e_out);
    CHECK_CL_ERROR(err);
}


