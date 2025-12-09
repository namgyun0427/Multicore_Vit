__kernel void matrix_plus (
    __global float4* g_A,
    __global float4* g_B,
    __global float4* g_C
) {
    size_t global_id = get_global_id(0);

    g_C[global_id] = g_A[global_id] + g_B[global_id];
}
