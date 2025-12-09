#define PATCH_SIZE 16
#define IN_CHANNELS 3

__kernel void patch_embed(
    __global const float* input,      // [C_in, H, W]
    __global const float* weight,     // [C_out, C_in, K, K]
    __global const float* bias,       // [C_out]
    __global float* output,           // [C_out, Out_H, Out_W]
    const int img_size,               // 224
    const int out_size                // 14
) {
    const int ow = get_global_id(0); 
    const int oh = get_global_id(1); 
    const int oc = get_global_id(2); 

    if (ow >= out_size || oh >= out_size) {
        return;
    }

    float sum = 0.0f;

    const int in_start_x = ow * PATCH_SIZE;
    const int in_start_y = oh * PATCH_SIZE;

    const int img_channel_stride = img_size * img_size;               // H * W
    const int weight_filter_stride = IN_CHANNELS * PATCH_SIZE * PATCH_SIZE; // 3 * 16 * 16
    const int weight_channel_stride = PATCH_SIZE * PATCH_SIZE;        // 16 * 16

    int weight_base_idx = oc * weight_filter_stride;

    for (int ic = 0; ic < IN_CHANNELS; ++ic) {
        
        int input_offset = (ic * img_channel_stride) + (in_start_y * img_size) + in_start_x;
        int weight_offset = weight_base_idx + (ic * weight_channel_stride);

        for (int kh = 0; kh < PATCH_SIZE; ++kh) {
            
            // 16개의 픽셀을 4개씩 묶어서 처리 (4번 반복)
            for (int kw = 0; kw < PATCH_SIZE; kw += 4) {
                
                // float4 (16바이트)
                // input: (current_row + kw) 위치에서 4개 로드
                float4 in_vec = vload4(0, input + input_offset + (kh * img_size) + kw);
                
                // weight: (current_row + kw) 위치에서 4개 로드
                float4 w_vec  = vload4(0, weight + weight_offset + (kh * PATCH_SIZE) + kw);
                
                // sum += dot(in_vec, w_vec)
                sum = fma(in_vec.x, w_vec.x, sum);
                sum = fma(in_vec.y, w_vec.y, sum);
                sum = fma(in_vec.z, w_vec.z, sum);
                sum = fma(in_vec.w, w_vec.w, sum);
            }
        }
    }

    // 3. Bias 더하기
    sum += bias[oc];

    // 4. 결과 저장
    // Output Index: (Channel * H * W) + (Row * W) + Col
    int output_idx = (oc * out_size * out_size) + (oh * out_size) + ow;
    output[output_idx] = sum;
}
