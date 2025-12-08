#include "ViT_cl.h"

// Softmax 
void v_Softmax(float* logits, float* probabilities, int length) {
    // 1. 메모리 객체 생성 ( logits -> probabilities )
    // logits은 입력, probabilities는 출력입니다.
    // logits, probabilities는 Host 메모리 포인터-> Device 버퍼를 만들어 사용

    size_t data_size = length * sizeof(float);

    cl_mem m_input = clCreateBuffer(container.context, CL_MEM_READ_ONLY, data_size, NULL, &err);
    CHECK_CL_ERROR(err);

    cl_mem m_output = clCreateBuffer(container.context, CL_MEM_WRITE_ONLY, data_size, NULL, &err);
    CHECK_CL_ERROR(err);

    //Host -> Device
    err = clEnqueueWriteBuffer(container.queue, m_input, CL_TRUE, 0, data_size, logits, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // 커널 인자 설정
    // __kernel void softmax_kernel(in, out, N, local_cache)

    cl_kernel k = container.kernels[__softmax];
    int n_data = length;

    size_t local_mem_size = 256 * sizeof(float);

    int arg_idx = 0;
    err = clSetKernelArg(k, arg_idx++, sizeof(cl_mem), &m_input);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(k, arg_idx++, sizeof(cl_mem), &m_output);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(k, arg_idx++, sizeof(int), &n_data);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(k, arg_idx++, local_mem_size, NULL);
    CHECK_CL_ERROR(err);

    // 데이터가 1000개 정도이므로 1개의 워크그룹(256 스레드)만 띄워서 처리
    // 커널 내부에서 반복문으로 1000개를 처리하도록 설계됨
    //제한된 스레드 사용하면 속도 올라감

    size_t local_work_size[] = { 256 };
    size_t global_work_size[] = { 256 }; // 1개 워크그룹

    err = clEnqueueNDRangeKernel(container.queue, k, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // 5. 결과 읽기 (Device -> Host)
    err = clEnqueueReadBuffer(container.queue, m_output, CL_TRUE, 0, data_size, probabilities, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // 6. 자원 해제
    clReleaseMemObject(m_input);
    clReleaseMemObject(m_output);
}
