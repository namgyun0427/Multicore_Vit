#include "../ViT_cl.h"

int main(void) {
    srand(time(NULL));
    size_t n_data = rand() % 1000 + 1;
    size_t divider = rand() + 1;

    // alloc and calculate sequal result
    float* input = (float*)calloc(n_data, sizeof(float));
    float result_seq = 0.0f;
    for (int i=0; i<n_data; ++i) {
        float divident = rand();
        input[i] = divident / divider;
        result_seq += input[i] * input[i];
    }

    // calculate cl result
    init();

    size_t work_group_size = 2;
    size_t power = rand() % 5 + 1;
    for (int i=0; i<power; ++i) {
        work_group_size *= 2;
    }

    float result_cl = reduce_sum_of_square(input, n_data, work_group_size);

    cleanup();

    const float eps = 1e-5;
    if(fabsf(result_seq - result_cl) > eps) {
        printf("[reduce_sum_of_square] result_seq != result_cl\n");
        printf("result_seq = %f\n", result_seq);
        printf("result_cl = %f\n", result_cl);
        return 1;
    } else {
        return 0;
    }
}