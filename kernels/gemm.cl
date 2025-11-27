__kernel void gemm(
    __global float* A, 
    __global float* B, 
    __global float* C, 
    int ROW_A, 
    int COL_A, 
    int COL_B
) {

    size_t global_row = get_global_id(0);
    size_t global_col = get_global_id(1);
    
    float acc = 0.0f;
    for(int i=0; i<COL_A; ++i) {
        acc += A[COL_A * global_row + i] * B[COL_B * i + global_col];
    }
    C[COL_B * global_row + global_col] = acc;
}


// [row x col] => [col x row]
__kernel void transpose(
    __global const float* g_input, 
    __global float* g_output
) {
    size_t global_row = get_global_id(0);
    size_t global_col = get_global_id(1);
    size_t row = get_global_size(0);
    size_t col = get_global_size(1);
    
    g_output[global_col * row + global_row] = g_input[global_row * col + global_col];
}

