__kernel void softmax_kernel(
    __global const float* input,
    __global float* output,
    const int N,
    __local float* local_cache // 워크그룹 내 공유 메모리
) {
    int tid = get_local_id(0);
    int lsize = get_local_size(0);

    // Exponential, Sum 계산 (Grid-Stride Loop)
    float thread_sum = 0.0f;

    for (int i = tid; i < N; i += lsize) {//스레드가 데이터 나눠 처리
        float val = exp(input[i]); 
        output[i] = val;           
        thread_sum += val;       
    }

    // 각 스레드의 부분 합을 로컬 메모리에 기록
    local_cache[tid] = thread_sum;
    barrier(CLK_LOCAL_MEM_FENCE); 

    // 병렬 리덕션-> local_cache[0]에 전체 합을 만듦
    for (int s = lsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            local_cache[tid] += local_cache[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE); 
    }
   //최종합계: 0번방 
    float sum_val = local_cache[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 구해진 전체 합으로 나누기->  최종 확률 계산
    for (int i = tid; i < N; i += lsize) {
        output[i] = output[i] / sum_val;
    }
}
