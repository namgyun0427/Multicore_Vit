__kernel void patch_embed (
    __global float* g_input,
    __global float* g_weight,
    __global float* g_bias,
    __global float* g_output,
    int IN_CAHNS,
    int PATCH_SIZE,
    int IMG_SIZE,
    int OUTPUT_SIZE
) {
    size_t oc = get_global_id(0);
    size_t oh = get_global_id(1);
    size_t ow = get_global_id(2);

    float sum = g_bias[oc];

    // itr for input_channel
    for (int ic = 0; ic < IN_CAHNS; ++ic) {
        // itr for PATCH_SIZE^2
        for (int kh = 0; kh < PATCH_SIZE; ++kh) {
            for (int kw = 0; kw < PATCH_SIZE; ++kw) {
                int ih = oh * PATCH_SIZE + kh;
                int iw = ow * PATCH_SIZE + kw;
                int input_idx = (ic * IMG_SIZE + ih) * IMG_SIZE + iw;
                int kernel_idx = ((oc * IN_CAHNS + ic) * PATCH_SIZE + kh) * PATCH_SIZE + kw;

                sum += g_input[input_idx] * g_weight[kernel_idx];
            }
        }
    }

    g_output[(oc * OUTPUT_SIZE + oh) * OUTPUT_SIZE + ow] = sum;
}
