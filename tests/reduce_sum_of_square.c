#include "../ViT_cl.h"

int main(void) {
    srand(time(NULL));
    size_t n_data = rand() % 1000 + 1;
    size_t divider = rand() + 1;

    // alloc and calculate sequal result
    float* input = (float*)calloc(n_data, sizeof(float));
    float sum_seq = 0.0f;
    for (int i=0; i<n_data; ++i) {
        float divident = rand();
        input[i] = divident / divider;
        sum_seq += input[i] * input[i];
    }

    init();

    size_t work_group_size = 2;
    size_t power = rand() % 5 + 1;
    for (int i=0; i<power; ++i) {
        work_group_size *= 2;
    }

    float sum_cl = reduce_sum_of_square(input, n_data, work_group_size);

    cleanup();

    // TODO: compare
    printf("seq_sum = %f\n", sum_seq);
    printf("sum_cl = %f\n", sum_cl);


    return 0;
}