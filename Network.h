#pragma once

#ifndef _Network_H
#define _Network_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// for linux and mac
#ifdef _WIN32
    #include "dirent.h"
#else
    #include <dirent.h>
    #include "posix_port.h"
#endif

typedef struct {
    int n;
    int c;
    int h;
    int w;
    float* data;
} ImageData;


typedef struct {
    float* data;
    size_t size;
} Network;

ImageData* load_image_data(const char* filename);
void load_weights(const char* directory, Network network[], int count);

#endif // _Network_H
