#include "ViT_cl.h"

// add position embedding data
// input[total_num_patches + 1][EMBED_DIM] => output[total_num_patches + 1][EMBED_DIM]
void v_pos_emb(
    cl_mem m_input, cl_mem m_output,
    cl_mem m_pos_emb
) {
    // run kernel
    v_matrix_plus(m_input, m_pos_emb, m_output, N_TOTAL_TOKEN * EMBED_DIM, 0, NULL, NULL);
}
