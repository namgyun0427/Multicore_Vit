#pragma once
#ifndef _ViT_cl_H
#define _ViT_cl_H

#ifdef _WIN32
    #pragma warning(disable:4996)
#endif


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

    cl_mem m_networks[NETWORK_NUM];
} CL_container;

typedef struct KernelArg {
    size_t size;
    const void* addr;
} KernelArg;

extern CL_container container;
extern cl_int err;


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



#define UNUSED(var) \
    (void)(var);

#define CHECK_CL_ERROR(err) \
    if (err != CL_SUCCESS) {    \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err);   \
        exit(EXIT_FAILURE); \
    }   \


void ViT_cl(ImageData* image, Network* networks, float** prb);



void init();

void cleanup();


char* v_get_source_code(const char* file_name, size_t* len);

void v_build_error(cl_program program, cl_device_id device, cl_int err);


void v_Conv2d(
    cl_mem m_input, cl_mem m_output,
    cl_mem m_weight, cl_mem m_bias
);

void v_flatten_transpose(cl_mem m_input, cl_mem m_output);

void v_class_token(
    cl_mem m_input, cl_mem m_output,
    cl_mem m_class_token
);

void v_pos_emb(
    cl_mem m_input, cl_mem m_output,
    cl_mem m_pos_emb
);

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

// void v_Softmax(float* logits, float* probabilities, int length);
void v_Softmax(cl_mem m_input, cl_mem m_output, int length);

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
