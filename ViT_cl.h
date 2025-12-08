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
    cl_kernel* kernels;     // TODO: 생각해보니 이거 malloc할 필요가 없는데?
    char** src_arr;
    size_t* len_arr;

    cl_mem m_networks[NETWORK_NUM];
} CL_container;

typedef struct KernelArg {
    size_t size;
    const void* addr;
} KernelArg;

/////////////////////////////////////////////////////////////////////////////
// export global variables

extern CL_container container;
extern cl_int err;


/////////////////////////////////////////////////////////////////////////////
// kernel configuration
// 커널 개수 알맞게 바꾸고, enum 및 필요 정보 추가
// Kernels_idxs와 kernel_configs의 순서가 맞아야 함
// kernel_configs 를 순회해서 각 file_path 별로 소스 코드를 뽑아서 빌드함

#define N_KERNEL 10

enum Kernels_idxs {
    __matrix_plus = 0,
    __linear,
    __gelu,
    __cal_score,
    __cal_result,
    __softmax,
    __softmax_score,
    __layer_norm,
    __patch_embed,
    __flatten_transpose,
};


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

// void v_Conv2d (
//     float* input, float* output, 
//     Network weight, Network bias
// );
void v_Conv2d(
    cl_mem m_input, cl_mem m_output,
    cl_mem m_weight, cl_mem m_bias
);

// void v_flatten_transpose (float* input, float* output);
void v_flatten_transpose(cl_mem m_input, cl_mem m_output);

// void v_class_token (
//     float* patch_tokens, float* final_tokens, 
//     Network cls_tk
// );
void v_class_token(
    cl_mem m_input, cl_mem m_output,
    cl_mem m_class_token
);

// void v_pos_emb (
//     float* input, float* output, 
//     Network pos_emb
// );
void v_pos_emb(
    cl_mem m_input, cl_mem m_output,
    cl_mem m_pos_emb
);

// void v_Encoder(
//     float* input, float* output,
//     Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
//     Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b
// );
void v_Encoder(
    cl_mem m_input, cl_mem m_final_output,
    cl_mem m_ln1_weight, cl_mem m_ln1_bias,
    cl_mem m_attn_in_weight, cl_mem m_attn_in_bias, cl_mem m_attn_out_weight, cl_mem m_attn_out_bias, 
    cl_mem m_ln2_weight, cl_mem m_ln2_bias, 
    cl_mem m_mlp1_weight, cl_mem m_mlp1_bias, cl_mem m_mlp2_weight, cl_mem m_mlp2_bias
);

void v_multihead_attn(
    cl_mem m_input, cl_mem m_fianl_output,
    cl_mem m_in_weight, cl_mem m_in_bias,
    cl_mem m_out_weight, cl_mem m_out_bias,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
);

void v_mlp_block(
    cl_mem m_input, cl_mem m_output,
    cl_mem m_weight1, cl_mem m_bias1,
    cl_mem m_weight2, cl_mem m_bias2
);

void v_gelu (cl_mem m_data, size_t n_data);

void v_Softmax(float* logits, float* probabilities, int length);

void v_layer_norm(
    cl_mem m_input, cl_mem m_output,
    cl_mem m_weight, cl_mem m_bias
);


void v_matrix_plus(
    cl_mem m_input1, cl_mem m_input2, 
    cl_mem m_output, 
    size_t n_data,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
);

void v_linear_layer (
    cl_mem m_input, cl_mem m_weight, cl_mem m_bias, cl_mem m_output, 
    int tokens, int in_features, int out_features,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
);


#endif // _ViT_cl_H
