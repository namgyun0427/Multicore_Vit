#pragma once
#ifndef _CONSTANTS_H
#define _CONSTANTS_H

    #define IMG_SIZE 224
    #define PATCH_SIZE 16
    #define IN_CAHNS 3
    #define NUM_CLASSES 1000
    #define EMBED_DIM 768
    #define NUM_HEADS 12
    #define MLP_RATIO 4.0
    #define EPSILON 1e-6

    
    #define IMAGE_COUNT 100
    #define IMG_FILE_PATH "./Data/input-100.bin" 
    #define ANSER_FILE_PATH "./Data/answer_result.txt"
    
    // #define IMAGE_COUNT 1
    // #define IMG_FILE_PATH "./Data/input-1.bin" 
    // #define ANSER_FILE_PATH "./Data/answer_result_1.txt"

#endif // _CONSTANTS_H
