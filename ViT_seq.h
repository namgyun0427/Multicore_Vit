#pragma once

#ifndef _ViT_seq_H
#define _ViT_seq_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

// #include <CL/cl.h>

#include "Network.h"


void ViT_seq(ImageData* image, Network* networks, float** prb);

void Conv2d (
    float* input, float* output, 
    Network weight, Network bias
);

void flatten_transpose (float* input, float* output);

void class_token (
    float* patch_tokens, float* final_tokens, 
    Network cls_tk
);

void pos_emb (
    float* input, float* output, 
    Network pos_emb
);

void Encoder(
    float* input, float* output,
    Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
    Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b
);

void multihead_attn(
    float* input, float* output,
    Network in_weight, Network in_bias, 
    Network out_weight, Network out_bias
);

void mlp_block (
    float* input, float* output, 
    Network fc1_weight, Network fc1_bias, 
    Network fc2_weight, Network fc2_bias
);

float gelu(float x);

void Softmax(float* logits, float* probabilities, int length);

void layer_norm (
    float* input, float* output, 
    Network weight, Network bias
);

void linear_layer (
    float* input, float* output, 
    int tokens, int in_features, int out_features, 
    Network weight, Network bias
);

#endif // _ViT_seq_H
