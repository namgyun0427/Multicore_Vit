#pragma once

#ifndef _Network_H
#define _Network_H

#include <time.h>

typedef struct {
    int n;      // �̹��� ����
    int c;      // ä�� ��
    int h;      // ����
    int w;      // �ʺ�
    float* data; // ��� �̹��� �����͸� ���ӵ� �޸� ������ ���� (N x C x H x W)
} ImageData;

ImageData* load_image_data(const char* filename);

// Network �ε忡 ���� ����
typedef struct {
    float* data;
    size_t size;
} Network;

static double conv2d_t, pos_emb_t, ln1_t, mha_t, ln2_t, mlp_t;
static double mlp_read, mlp_write, mlp_compute;
static double mlp1, mlp2;
static double attn1, attn2, attn3;
static double attn2_1, attn2_2, attn2_3;
static time_t start, end;
static time_t start_t, end_t;
static time_t start_mlp, end_mlp;
static time_t start_attn, end_attn;
static time_t start_2, end_2;

void load_weights(const char* directory, Network network[], int count);

#endif // _Network_H
