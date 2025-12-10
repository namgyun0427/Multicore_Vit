#include "ViT_cl.h"

// add position embedding data
// input[total_num_patches + 1][EMBED_DIM] => output[total_num_patches + 1][EMBED_DIM]
void v_pos_emb(
    cl_mem m_input, cl_mem m_output,
    cl_mem m_pos_emb,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
) {
    // run kernel
    v_matrix_plus(m_input, m_pos_emb, m_output, N_TOTAL_TOKEN * EMBED_DIM, e_num_waiting, e_waiting_arr, e_out);
}
