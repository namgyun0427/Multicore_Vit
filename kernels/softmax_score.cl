// - Logic: 각 행(Row)별로 독립적인 Softmax 수행(197x197)

__kernel void softmax_score_kernel(
    __global float* data,       
    const int N,                
    __local float* local_cache 
) {
    int row_idx = get_group_id(0);
    
    int offset = row_idx * N;
    __global float* row_data = data + offset;

    int tid = get_local_id(0);
    int lsize = get_local_size(0);

    //max값 찾기
    float thread_max = -INFINITY;

    if (tid < N) {
        thread_max = row_data[tid];
    }
    local_cache[tid] = thread_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = lsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (local_cache[tid + s] > local_cache[tid]) {
                local_cache[tid] = local_cache[tid + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_max = local_cache[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Exponential & Sum 계산
    float thread_sum = 0.0f;

    if (tid < N) {
        // Max를 빼고 exp 계산 (오버플로우 방지)
        float val = exp(row_data[tid] - row_max);
        row_data[tid] = val;
        thread_sum = val;
    } else {
        thread_sum = 0.0f; 
    }
    
    local_cache[tid] = thread_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = lsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            local_cache[tid] += local_cache[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_sum = local_cache[0];
    barrier(CLK_LOCAL_MEM_FENCE);

     // Normalize 
     if (tid < N) {
        row_data[tid] /= row_sum;
    }
}
