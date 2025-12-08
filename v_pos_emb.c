#include "ViT_cl.h"

// add position embedding data
// input[total_num_patches + 1][EMBED_DIM] => output[total_num_patches + 1][EMBED_DIM]
void v_pos_emb(
    float* input, float* output,
    Network pos_emb
) {
    int output_size = IMG_SIZE / PATCH_SIZE;
    int num_patches = output_size * output_size;
    int total_tokens = num_patches + 1;
    int total_elements = total_tokens * EMBED_DIM;

    // add position embedding data
    for (int i = 0; i < total_elements; i++) {
        output[i] = input[i] + pos_emb.data[i];
    }
}
