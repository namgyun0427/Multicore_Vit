#include "../ViT_cl.h"

int main(void) {
    srand(time(NULL));
    size_t n_data = rand() % 1000 + 1;
    size_t divider = rand() + 1;

    // alloc and calculate sequal result
    float* input = (float*)calloc(n_data, sizeof(float));
    float result_seq = 0.0f;
    for (size_t i=0; i<n_data; ++i) {
        float divident = rand();
        input[i] = divident / divider;
        result_seq += input[i];
    }

    // calculate cl result
    init();

    size_t work_group_size = 2;
    size_t power = rand() % 5 + 1;
    for (size_t i=0; i<power; ++i) {
        work_group_size *= 2;
    }

    float result_cl = reduce_sum(input, n_data, work_group_size);

    cleanup();

    // compare
    const float rtol = 1e-4f;
    const float atol = 1e-5f;
    float diff = fabsf(result_seq - result_cl);
    float tol = fmaxf(rtol * fmaxf(fabsf(result_seq), fabsf(result_cl)), atol);
    if(diff > tol) {
        printf("[test failed: reduce_sum]\n");
        printf("result_seq != result_cl\n");
        printf("result_seq = %f\n", result_seq);
        printf("result_cl = %f\n", result_cl);

        return 1;
    } else {
        printf("[test passed: reduce_sum]\n");

        return 0;
    }
}