kk1
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

    const int layer_stride = img_size * img_size;       // H * W
    const int weight_stride = PATCH_SIZE * PATCH_SIZE;  // 16 * 16

    int input_offset_base = (in_start_y * img_size) + in_start_x;
    int weight_offset_base = oc * IN_CHANNELS * weight_stride;

    {
        int img_offset = input_offset_base; 
        int w_offset = weight_offset_base;

        #pragma unroll 
        for (int kh = 0; kh < PATCH_SIZE; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < PATCH_SIZE; kw += 4) {
                float4 in_vec = vload4(0, input + img_offset + (kh * img_size) + kw);
                float4 w_vec  = vload4(0, weight + w_offset + (kh * PATCH_SIZE) + kw);
                
                sum = fma(in_vec.x, w_vec.x, sum);
                sum = fma(in_vec.y, w_vec.y, sum);
                sum = fma(in_vec.z, w_vec.z, sum);
                sum = fma(in_vec.w, w_vec.w, sum);
            }
        }
    }

    {
        int img_offset = input_offset_base + layer_stride; 
        int w_offset = weight_offset_base + weight_stride; 

        #pragma unroll
        for (int kh = 0; kh < PATCH_SIZE; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < PATCH_SIZE; kw += 4) {
                float4 in_vec = vload4(0, input + img_offset + (kh * img_size) + kw);
                float4 w_vec  = vload4(0, weight + w_offset + (kh * PATCH_SIZE) + kw);
                
                sum = fma(in_vec.x, w_vec.x, sum);
                sum = fma(in_vec.y, w_vec.y, sum);
                sum = fma(in_vec.z, w_vec.z, sum);
                sum = fma(in_vec.w, w_vec.w, sum);
            }
        }
    }

    {
        int img_offset = input_offset_base + (2 * layer_stride); 
        int w_offset = weight_offset_base + (2 * weight_stride); 

        #pragma unroll
        for (int kh = 0; kh < PATCH_SIZE; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < PATCH_SIZE; kw += 4) {
                float4 in_vec = vload4(0, input + img_offset + (kh * img_size) + kw);
                float4 w_vec  = vload4(0, weight + w_offset + (kh * PATCH_SIZE) + kw);
                
                sum = fma(in_vec.x, w_vec.x, sum);
                sum = fma(in_vec.y, w_vec.y, sum);
                sum = fma(in_vec.z, w_vec.z, sum);
                sum = fma(in_vec.w, w_vec.w, sum);
            }
        }
    }

    sum += bias[oc];

    int output_idx = (oc * out_size * out_size) + (oh * out_size) + ow;
    output[output_idx] = sum;
}
