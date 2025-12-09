__kernel void patch_embed(
    __global const float* input,      // [C_in, H, W]
    __global const float* weight,     // [C_out, C_in, K, K]
    __global const float* bias,       // [C_out]
    __global float* output,           // [C_out, Out_H, Out_W]
    const int in_channels,            // 3
    const int patch_size,             // 16
    const int img_size,               
    const int out_size                
) {
    const int ow = get_global_id(0); 
    const int oh = get_global_id(1); 
    const int oc = get_global_id(2); 

    if (ow >= out_size || oh >= out_size) {
        return;
    }

    float sum = 0.0f;

    const int in_start_x = ow * patch_size;
    const int in_start_y = oh * patch_size;

    for (int ic = 0; ic < in_channels; ++ic) {
        
        int img_offset = (ic * img_size * img_size) + (in_start_y * img_size) + in_start_x;

        int weight_offset = (oc * in_channels * patch_size * patch_size) + (ic * patch_size * patch_size);
     for (int kh = 0; kh < patch_size; ++kh) {
            
	    //float4 사용
            for (int kw = 0; kw < patch_size; kw += 4) {
                
                float4 in_vec = vload4(0, input + img_offset + (kh * img_size) + kw);

                float4 w_vec = vload4(0, weight + weight_offset + (kh * patch_size) + kw);

                // sum += dot(in_vec, w_vec); 
                sum = fma(in_vec.x, w_vec.x, sum);
                sum = fma(in_vec.y, w_vec.y, sum);
                sum = fma(in_vec.z, w_vec.z, sum);
                sum = fma(in_vec.w, w_vec.w, sum);
            }
        }
    }

    //편향
    sum += bias[oc];

    // output [C_out, Out_H, Out_W]
    // Index = (oc * Out_H * Out_W) + (oh * Out_W) + ow
    int output_idx = (oc * out_size * out_size) + (oh * out_size) + ow;
    output[output_idx] = sum;
}
