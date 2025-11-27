__kernel void cl_matrix_plus (
    __global float* g_lvalue,
    __global float* g_rvalue
) {
    size_t global_id = get_global_id(0);

    g_lvalue[global_id] += g_rvalue[global_id];
}
