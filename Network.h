#pragma once

#ifndef _Network_H
#define _Network_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef struct {
    int n;      // �̹��� ����
    int c;      // ä�� ��
    int h;      // ����
    int w;      // �ʺ�
    float* data; // ��� �̹��� �����͸� ���ӵ� �޸� ������ ���� (N x C x H x W)
} ImageData;

// Network �ε忡 ���� ����
typedef struct {
    float* data;
    size_t size;
} Network;

ImageData* load_image_data(const char* filename);
void load_weights(const char* directory, Network network[], int count);

#endif // _Network_H
