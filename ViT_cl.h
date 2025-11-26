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

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

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
// sub functions: 테스트를 위해 외부로 노출

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

#endif // _ViT_cl_H
