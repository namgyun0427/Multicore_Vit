#pragma once
#ifndef _ViT_cl_H
#define _ViT_cl_H

#ifdef _WIN32
    #pragma warning(disable : 4996)
#endif

/////////////////////////////////////////////////////////////////////////////
// includes

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include "constants.h"
#include "log.h"
#include "Network.h"


/////////////////////////////////////////////////////////////////////////////
// structs

typedef struct Kernel_config {
    const char* const kernel_name;
    const char* const file_path;
} Kernel_config;

typedef struct CL_container{
    // default data
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    // kernels
    size_t n_kernels;

    Kernel_config* kernel_configs;
    cl_kernel* kernels;
    char** src_arr;
    size_t* len_arr;
} CL_container;

typedef struct KernelArg {
    size_t size;
    const void* addr;
} KernelArg;


/////////////////////////////////////////////////////////////////////////////
// macro function

#define UNUSED(var) \
    (void)(var);

#define CHECK_CL_ERROR(err) \
    if (err != CL_SUCCESS) {    \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err);   \
        exit(EXIT_FAILURE); \
    }   \

/////////////////////////////////////////////////////////////////////////////
// core function

void ViT_cl(ImageData* image, Network* networks, float** prb);


/////////////////////////////////////////////////////////////////////////////
// sub functions

void init();

void cleanup();


char* v_get_source_code(const char* file_name, size_t* len);

void v_build_error(cl_program program, cl_device_id device, cl_int err);

void v_Conv2d (
    float* input, float* output, 
    Network weight, Network bias
);

void v_flatten_transpose (float* input, float* output);

void v_class_token (
    float* patch_tokens, float* final_tokens, 
    Network cls_tk
);

void v_pos_emb (
    float* input, float* output, 
    Network pos_emb
);

void v_Encoder(
    float* input, float* output,
    Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
    Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b
);

void v_multihead_attn(
    float* input, float* output,
    Network in_weight, Network in_bias, 
    Network out_weight, Network out_bias
);

void v_mlp_block (
    float* input, float* output, 
    Network fc1_weight, Network fc1_bias, 
    Network fc2_weight, Network fc2_bias
);

void v_gelu (cl_mem m_data, size_t n_data);

void v_Softmax(float* logits, float* probabilities, int length);

void v_layer_norm (
    float* input, float* output, 
    Network weight, Network bias
);

void matrix_plus (
    float* input1, float* input2, float* output, 
    size_t n_data
);

// void cl_matrix_plus (
//     cl_mem m_lvalue, cl_mem m_rvalue, 
//     size_t n_data, 
//     cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
// );

void v_linear_layer (
    cl_mem m_input, cl_mem m_weight, cl_mem m_bias, cl_mem m_output, 
    int tokens, int in_features, int out_features,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
);

void v_reduce_sum_of_square (
    cl_mem m_data, 
    cl_mem m_output, 
    size_t total_num_data,
    size_t work_group_size,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
);

void v_reduce_sum (
    cl_mem m_data, 
    cl_mem m_output, 
    size_t total_num_data,
    size_t work_group_size,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
);

void v_cal_mean_and_inv_std (
    cl_mem m_sum, cl_mem m_sum_of_square,
    cl_mem m_mean, cl_mem m_inv_std,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
);

void v_normalize (
    cl_mem m_input, 
    cl_mem m_weight, cl_mem m_bias,
    cl_mem m_mean, cl_mem m_inv_std,
    int total_num_data,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
);




#endif // _ViT_cl_H
