__kernel void flatten_transpose (
    __global float* g_input,
    __global float* g_output,
    int OUTPUT_SIZE,
    int EMBED_DIM
) {
    size_t oh = get_global_id(0);
    size_t ow = get_global_id(1);
    size_t oc = get_global_id(2);

    int patch_idx = oh * OUTPUT_SIZE + ow;
    int idx_input = (oc * OUTPUT_SIZE + oh) * OUTPUT_SIZE + ow;
    int idx_output = patch_idx * EMBED_DIM + oc;

    g_output[idx_output] = g_input[idx_input];
}
