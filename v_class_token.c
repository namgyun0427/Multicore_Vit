#include "ViT_cl.h"

// prepend class tokens in front
// input[total_num_patches][EMBED_DIM] => output[total_num_patches + 1][EMBED_DIM]
// TODO
void v_class_token(
    float* patch_tokens, float* final_tokens,
    Network cls_tk
    ) {
    int output_size = IMG_SIZE / PATCH_SIZE;
    int num_patches = output_size * output_size;

    // attach class token in front
    for (int j = 0; j < EMBED_DIM; j++) {
        final_tokens[j] = cls_tk.data[j];
    }

    // copy rest of data
    memcpy(final_tokens + EMBED_DIM, patch_tokens, sizeof(float) * EMBED_DIM * num_patches);
}
