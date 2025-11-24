__kernel void load_square (
    __global float* g_input
) {
    size_t global_id = get_global_id(0);

    g_input[global_id] = g_input[global_id] * g_input[global_id];
}
