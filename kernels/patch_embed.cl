__kernel void patch_embed (
    __global float* input,
    __global float* weight,
    __global float* bias,
    __global float* output
) {


    float sum = bias.data[oc];

    // itr for input_channel
    for (int ic = 0; ic < IN_CAHNS; ++ic) {
        // itr for PATCH_SIZE^2
        for (int kh = 0; kh < PATCH_SIZE; ++kh) {
            for (int kw = 0; kw < PATCH_SIZE; ++kw) {
                int ih = oh * PATCH_SIZE + kh;
                int iw = ow * PATCH_SIZE + kw;
                int input_idx = (ic * IMG_SIZE + ih) * IMG_SIZE + iw;
                int kernel_idx = ((oc * IN_CAHNS + ic) * PATCH_SIZE + kh) * PATCH_SIZE + kw;

                sum += input[input_idx] * weight.data[kernel_idx];
            }
        }
    }

    // input => 3 * 
    // output[EMBED_DIM][n_patch][n_patch]

    // output[oc][oh][ow] = sum
    output[(oc * OUTPUT_SIZE + oh) * OUTPUT_SIZE + ow] = sum;

}

// itr for output_channel
// for (int oc = 0; oc < EMBED_DIM; ++oc) {
//     // itr for output_height * output_width
//     for (int oh = 0; oh < OUTPUT_SIZE; ++oh) {
//         for (int ow = 0; ow < OUTPUT_SIZE; ++ow) {
//             // set bias as initial value
//             float sum = bias.data[oc];

//             // itr for input_channel
//             for (int ic = 0; ic < IN_CAHNS; ++ic) {
//                 // itr for PATCH_SIZE^2
//                 for (int kh = 0; kh < PATCH_SIZE; ++kh) {
//                     for (int kw = 0; kw < PATCH_SIZE; ++kw) {
//                         int ih = oh * PATCH_SIZE + kh;
//                         int iw = ow * PATCH_SIZE + kw;
//                         int input_idx = (ic * IMG_SIZE + ih) * IMG_SIZE + iw;
//                         int kernel_idx = ((oc * IN_CAHNS + ic) * PATCH_SIZE + kh) * PATCH_SIZE + kw;

//                         // EMBED_DIM = IN_CAHNS * PATCH_SIZE * PATCH_SIZE 임
//                         // input[IN_CAHNS][PATCH_SIZE][PATCH_SIZE] 로 해석
//                         // weight.data[EMBED_DIM][IN_CAHNS][PATCH_SIZE][PATCH_SIZE]라고 해석
//                         // 밑 범위를 전부 계산
//                         // sum += input[ic][ih][iw] * weight.data[oc][ic][0 ~ PATCH_SIZE-1][0 ~ PATCH_SIZE-1]
//                         sum += input[input_idx] * weight.data[kernel_idx];
//                     }
//                 }
//             }

//             // input => 3 * 
//             // output[EMBED_DIM][n_patch][n_patch]

//             // output[oc][oh][ow] = sum
//             output[(oc * OUTPUT_SIZE + oh) * OUTPUT_SIZE + ow] = sum;
//         }
//     }
// }