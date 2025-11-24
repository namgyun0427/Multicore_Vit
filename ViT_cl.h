#pragma once
#ifndef _ViT_cl_H
#define _ViT_cl_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include "Network.h"


typedef struct KernelArg {
    size_t size;
    const void* addr;
} KernelArg;

typedef struct CL_container{
    // default data
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    // kernels
    cl_kernel __normalize;
    cl_kernel __reduce_sum;
    cl_kernel __load_square;
} CL_container;

void ViT_cl(ImageData* image, Network* networks, float** prb);

static char* get_source_code(const char* file_name, size_t* len);

static void build_error(cl_program program, cl_device_id device, cl_int err);

void init();

void cleanup();

float reduce_sum (
    float* input, 
    int total_num_data,
    size_t work_group_size
);

float reduce_sum_of_square (
    float* input, 
    int total_num_data,
    size_t work_group_size
);

void normalize (
    float* input, 
    Network weight, 
    Network bias, 
    int total_num_data,
    const float mean,
    const float inv_std
);










static void Conv2d (
    float* input, float* output, 
    Network weight, Network bias
);

static void flatten_transpose (float* input, float* output);

static void class_token (
    float* patch_tokens, float* final_tokens, 
    Network cls_tk
);

static void pos_emb (
    float* input, float* output, 
    Network pos_emb
);

static void Encoder(
    float* input, float* output,
    Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
    Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b
);

static void multihead_attn(
    float* input, float* output,
    Network in_weight, Network in_bias, 
    Network out_weight, Network out_bias
);

static void mlp_block (
    float* input, float* output, 
    Network fc1_weight, Network fc1_bias, 
    Network fc2_weight, Network fc2_bias
);

static float gelu(float x);

static void Softmax(float* logits, float* probabilities, int length);

static void layer_norm (
    float* input, float* output, 
    Network weight, Network bias
);

static void linear_layer (
    float* input, float* output, 
    int tokens, int in_features, int out_features, 
    Network weight, Network bias
);

#endif // _ViT_cl_H
