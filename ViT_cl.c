#include "ViT_cl.h"

/*
[잡생각]
커널 이름 컨벤션 정해두면 좋을듯 => 일단 지금은 '__'로 시작하는 걸로 통일
v_ 로 시작하게끔 함수이름 변경
cl 메모리 객체 관련해서도 => 일단 지금은 "m_"으로 시작하는 경로 통일
커널을 쓰는 함수에 대해서도 네이밍 컨벤션?


커널 하나당 wrppaer 함수 같이 만들어서 시그너쳐를 이용해서 매개변수 정보를 제공하는 쪽이 좋은 듯

work_group_size 최대 크기 가져오기?

reducing 을 2차원으로??

테스트 어떻게 하지...? 그냥 static 말고 깡 전역으로 conatiner 선언하고 extern 으로 받아서 써야하나

*/

// TODO: matrix_plus 함수 cl_mem 받도록
// TODO: position embedding 함수 matrix_plus쓰는 편으로 변경
// TODO: Network 담을 buffer init에서 생성 및 초기화



////////////////////////////////////////////////////////////////////////////////////
// constants

static const int size[] = {
    EMBED_DIM * (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE), // conv2D
    EMBED_DIM * (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE), // flatten and transpose
    EMBED_DIM * ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE) + 1), // class token
    EMBED_DIM * ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE) + 1) // position embedding
};

static const int ENC_SIZE = EMBED_DIM * ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE) + 1);


/////////////////////////////////////////////////////////////////////////////
// global variables

CL_container container;
cl_int err;

/////////////////////////////////////////////////////////////////////////////
// Kernel configs


Kernel_config kernel_configs[N_KERNEL] = {
    {.kernel_name = "reduce_sum", .file_path = "./kernels/reduce_sum.cl" },
    {.kernel_name = "load_square", .file_path = "./kernels/load_square.cl" },
    {.kernel_name = "my_normalize", .file_path = "./kernels/normalize.cl" },
    {.kernel_name = "matrix_plus", .file_path = "./kernels/matrix_plus.cl" },
    {.kernel_name = "cl_matrix_plus", .file_path = "./kernels/cl_matrix_plus.cl" },
    {.kernel_name = "linear", .file_path = "./kernels/linear.cl" },
    {.kernel_name = "cal_mean_and_inv_std", .file_path = "./kernels/cal_mean_and_inv_std.cl" },
    {.kernel_name = "gelu", .file_path = "./kernels/gelu.cl" },
    { .kernel_name = "cal_score", .file_path = "./kernels/multihead_attrention/cal_score.cl" },
    { .kernel_name = "convert_score", .file_path = "./kernels/multihead_attrention/convert_score.cl" },
    { .kernel_name = "normalize_score", .file_path = "./kernels/multihead_attrention/normalize_score.cl" },
    { .kernel_name = "cal_result", .file_path = "./kernels/multihead_attrention/cal_result.cl" },
    {.kernel_name = "softmax_kernel", .file_path = "./kernels/softmax.cl" },
    {.kernel_name = "softmax_score_kernel", .file_path = "./kernels/softmax_score.cl" },
};


/////////////////////////////////////////////////////////////////////////////
// cl configuration functions

char* v_get_source_code(const char* file_name, size_t* len) {
    int unused_ret;

    FILE* file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    size_t length = (size_t)ftell(file);
    rewind(file);

    char* source_code = (char*)malloc(length + 1);
    unused_ret = fread(source_code, length, 1, file);
    source_code[length] = '\0';
    fclose(file);
    *len = length;

    (void)unused_ret;

    return source_code;
}

void v_build_error(cl_program program, cl_device_id device, cl_int err) {
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char* log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_CL_ERROR(err);

        log = (char*)malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        CHECK_CL_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    };
}

void init() {
    printf(">> [init] : start\n");

    // get platform
    err = clGetPlatformIDs(1, &container.platform, NULL);
    CHECK_CL_ERROR(err);

    // get device id
    err = clGetDeviceIDs(container.platform, CL_DEVICE_TYPE_GPU, 1, &container.device, NULL);
    CHECK_CL_ERROR(err);

    // create context
    container.context = clCreateContext(NULL, 1, &container.device, NULL, NULL, &err);
    CHECK_CL_ERROR(err);

    // create command queue
    // MEMO: set queue as out-of-ordef later
    container.queue = clCreateCommandQueueWithProperties(container.context, container.device, NULL, &err);
    CHECK_CL_ERROR(err);

    // set kernel configuration
    container.n_kernels = N_KERNEL;
    container.kernel_configs = kernel_configs;
    container.kernels = (cl_kernel*)calloc(container.n_kernels, sizeof(cl_kernel));
    container.src_arr = (char**)calloc(container.n_kernels, sizeof(char*));
    container.len_arr = (size_t*)calloc(container.n_kernels, sizeof(size_t));

    // get sources
    for (size_t i = 0; i < container.n_kernels; ++i) {
        container.src_arr[i] = v_get_source_code(container.kernel_configs[i].file_path, &container.len_arr[i]);
    }

    // create program
    container.program = clCreateProgramWithSource(container.context, container.n_kernels, (const char**)container.src_arr, container.len_arr, &err);
    CHECK_CL_ERROR(err);

    // build program
    err = clBuildProgram(container.program, 1, &container.device, NULL, NULL, NULL);
    v_build_error(container.program, container.device, err);

    // create kernels
    for (size_t i = 0; i < container.n_kernels; ++i) {
        container.kernels[i] = clCreateKernel(container.program, container.kernel_configs[i].kernel_name, &err);
        CHECK_CL_ERROR(err);
    }

    // free sources
    for (int i = 0; i < 1; ++i) {
        free(container.src_arr[i]);
        container.src_arr[i] = NULL;
    }
    free(container.src_arr);
    free(container.len_arr);

    printf(">> [init] : ended\n");
}

void cleanup() {
    printf(">> [cleanup] : start\n");

    // release kernels
    for (size_t i = 0; i < container.n_kernels; ++i) {
        err = clReleaseKernel(container.kernels[i]);
        CHECK_CL_ERROR(err);
    }

    // release cl objects
    err = clReleaseProgram(container.program);
    CHECK_CL_ERROR(err);
    err = clReleaseCommandQueue(container.queue);
    CHECK_CL_ERROR(err);
    err = clReleaseContext(container.context);
    CHECK_CL_ERROR(err);
    err = clReleaseDevice(container.device);
    CHECK_CL_ERROR(err);

    // free heap memory
    free(container.kernels);
    container.kernels = NULL;

    printf(">> [cleanup] : ended\n");
}


////////////////////////////////////////////////////////////////////////////////////
// core funciton

/*
Network networks[152]
    0: ????
    1: patch-embedding weight
    2: patch-embedding bias
    3: position embedding
    4~147: encoding
    148, 149: normalize
    150, 151: v_Softmax
*/

void ViT_cl(
    ImageData* image,
    Network* networks,
    float** probabilities
    ) {
    printf(">> [ViT_cl] : start\n");

    init();

    float* layer[4];
    for (int i = 0; i < 4; i++) {
        layer[i] = (float*)malloc(sizeof(float) * size[i]);
    }

    // encoding layer
    float* enc_layer[12];
    for (int i = 0; i < 12; i++) {
        enc_layer[i] = (float*)malloc(sizeof(float) * ENC_SIZE);
    }

    // encoding output
    float* enc_output;
    enc_output = (float*)malloc(sizeof(float) * ENC_SIZE);


    // process per image
    for (int i = 0; i < image->n; i++) {
        printf("============= processing %d-th iamge =============\n", i);

        /*patch embedding*/
        float* patch_embedded = layer[0];
        LOG("patch_embedded", v_Conv2d(image[i].data, patch_embedded, networks[1], networks[2]));

        /*flatten and transpose*/
        float* flatten_transposed = layer[1];
        LOG("flatten_transposed", v_flatten_transpose(patch_embedded, flatten_transposed));

        /*prepend class token*/
        float* clas_token_prepended = layer[2];
        LOG("clas_token_prepended", v_class_token(flatten_transposed, clas_token_prepended, networks[0]));

        /*position embedding*/
        float* position_embeded = layer[3];
        LOG("position_embeded", v_pos_emb(clas_token_prepended, position_embeded, networks[3]));

        /*v_Encoder - 12 Layers*/
        {
            v_Encoder(position_embeded, enc_layer[0],
                networks[4], networks[5], networks[6], networks[7],
                networks[8], networks[9], networks[10], networks[11],
                networks[12], networks[13], networks[14], networks[15]);

            v_Encoder(enc_layer[0], enc_layer[1],
                networks[16], networks[17], networks[18], networks[19],
                networks[20], networks[21], networks[22], networks[23],
                networks[24], networks[25], networks[26], networks[27]);

            v_Encoder(enc_layer[1], enc_layer[2],
                networks[28], networks[29], networks[30], networks[31],
                networks[32], networks[33], networks[34], networks[35],
                networks[36], networks[37], networks[38], networks[39]);

            v_Encoder(enc_layer[2], enc_layer[3],
                networks[40], networks[41], networks[42], networks[43],
                networks[44], networks[45], networks[46], networks[47],
                networks[48], networks[49], networks[50], networks[51]);

            v_Encoder(enc_layer[3], enc_layer[4],
                networks[52], networks[53], networks[54], networks[55],
                networks[56], networks[57], networks[58], networks[59],
                networks[60], networks[61], networks[62], networks[63]);

            v_Encoder(enc_layer[4], enc_layer[5],
                networks[64], networks[65], networks[66], networks[67],
                networks[68], networks[69], networks[70], networks[71],
                networks[72], networks[73], networks[74], networks[75]);

            v_Encoder(enc_layer[5], enc_layer[6],
                networks[76], networks[77], networks[78], networks[79],
                networks[80], networks[81], networks[82], networks[83],
                networks[84], networks[85], networks[86], networks[87]);

            v_Encoder(enc_layer[6], enc_layer[7],
                networks[88], networks[89], networks[90], networks[91],
                networks[92], networks[93], networks[94], networks[95],
                networks[96], networks[97], networks[98], networks[99]);

            v_Encoder(enc_layer[7], enc_layer[8],
                networks[100], networks[101], networks[102], networks[103],
                networks[104], networks[105], networks[106], networks[107],
                networks[108], networks[109], networks[110], networks[111]);

            v_Encoder(enc_layer[8], enc_layer[9],
                networks[112], networks[113], networks[114], networks[115],
                networks[116], networks[117], networks[118], networks[119],
                networks[120], networks[121], networks[122], networks[123]);

            v_Encoder(enc_layer[9], enc_layer[10],
                networks[124], networks[125], networks[126], networks[127],
                networks[128], networks[129], networks[130], networks[131],
                networks[132], networks[133], networks[134], networks[135]);

            v_Encoder(enc_layer[10], enc_layer[11],
                networks[136], networks[137], networks[138], networks[139],
                networks[140], networks[141], networks[142], networks[143],
                networks[144], networks[145], networks[146], networks[147]);
        }


        // normalize
        LOG("normalize", v_layer_norm(enc_layer[11], enc_output, networks[148], networks[149]));


        // load class token
        float* cls_token = (float*)malloc(sizeof(float) * EMBED_DIM);
        memcpy(cls_token, enc_output, sizeof(float) * EMBED_DIM);

        // convert class token into output
        float* cls_output = (float*)malloc(sizeof(float) * NUM_CLASSES);

        LOG("cls_output", {
            // create mem obj

            // input = [tokens x in_features]
            const size_t input_size = 1 * EMBED_DIM * sizeof(float);
            cl_mem m_input = clCreateBuffer(container.context, CL_MEM_READ_WRITE, input_size, NULL, &err);
            CHECK_CL_ERROR(err);

            // weight = [out_features x in_features]
            const size_t weight_size = NUM_CLASSES * EMBED_DIM * sizeof(float);
            cl_mem m_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, weight_size, NULL, &err);
            CHECK_CL_ERROR(err);

            // m_bias = [1 x out_features]
            const size_t bias_size = NUM_CLASSES * sizeof(float);
            cl_mem m_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, bias_size, NULL, &err);
            CHECK_CL_ERROR(err);

            // output = [tokens x out_features]
            const size_t output_size = 1 * NUM_CLASSES * sizeof(float);
            cl_mem m_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, output_size, NULL, &err);
            CHECK_CL_ERROR(err);

            // write
            err = clEnqueueWriteBuffer(container.queue, m_input, CL_TRUE, 0, input_size, cls_token, 0, NULL, NULL);
            CHECK_CL_ERROR(err);
            err = clEnqueueWriteBuffer(container.queue, m_weight, CL_TRUE, 0, weight_size, networks[150].data, 0, NULL, NULL);
            CHECK_CL_ERROR(err);
            err = clEnqueueWriteBuffer(container.queue, m_bias, CL_TRUE, 0, bias_size, networks[151].data, 0, NULL, NULL);
            CHECK_CL_ERROR(err);

            v_linear_layer(
                m_input, m_weight, m_bias, m_output,
                1, EMBED_DIM, NUM_CLASSES,
                0, NULL, NULL
            );

            // read result
            err = clEnqueueReadBuffer(container.queue, m_output, CL_TRUE, 0, output_size, cls_output, 0, NULL, NULL);
            CHECK_CL_ERROR(err);

            err = clReleaseMemObject(m_input);
            CHECK_CL_ERROR(err);
            err = clReleaseMemObject(m_weight);
            CHECK_CL_ERROR(err);
            err = clReleaseMemObject(m_bias);
            CHECK_CL_ERROR(err);
            err = clReleaseMemObject(m_output);
            CHECK_CL_ERROR(err);
            });


        // sofemax
        LOG("sofemax", v_Softmax(cls_output, probabilities[i], NUM_CLASSES));
        free(cls_token);
        free(cls_output);
    }
    for (int i = 0; i < 4; i++) free(layer[i]);
    for (int i = 0; i < 12; i++) free(enc_layer[i]);
    free(enc_output);

    cleanup();

    printf(">> [ViT_cl] : ended\n");
}



////////////////////////////////////////////////////////////////////////////////////
// sub funcitons

// image patch embedding (convolution)
// input[IN_CAHNS][PATCH_SIZE][PATCH_SIZE] => output[EMBED_DIM][n_patch_per_image][n_patch_per_image]
void v_Conv2d(
    float* input, float* output,
    Network weight, Network bias
    ) {
    const int OUTPUT_SIZE = IMG_SIZE / PATCH_SIZE;

    // itr for output_channel
    for (int oc = 0; oc < EMBED_DIM; ++oc) {
        // itr for output_height * output_width
        for (int oh = 0; oh < OUTPUT_SIZE; ++oh) {
            for (int ow = 0; ow < OUTPUT_SIZE; ++ow) {
                // set bias as initial value
                float sum = bias.data[oc];

                // itr for input_channel
                for (int ic = 0; ic < IN_CAHNS; ++ic) {
                    // itr for PATCH_SIZE^2
                    for (int kh = 0; kh < PATCH_SIZE; ++kh) {
                        for (int kw = 0; kw < PATCH_SIZE; ++kw) {
                            int ih = oh * PATCH_SIZE + kh;
                            int iw = ow * PATCH_SIZE + kw;
                            int input_idx = (ic * IMG_SIZE + ih) * IMG_SIZE + iw;
                            int kernel_idx = ((oc * IN_CAHNS + ic) * PATCH_SIZE + kh) * PATCH_SIZE + kw;

                            // EMBED_DIM = IN_CAHNS * PATCH_SIZE * PATCH_SIZE 임
                            // input[IN_CAHNS][PATCH_SIZE][PATCH_SIZE] 로 해석
                            // weight.data[EMBED_DIM][IN_CAHNS][PATCH_SIZE][PATCH_SIZE]라고 해석
                            // 밑 범위를 전부 계산
                            // sum += input[ic][ih][iw] * weight.data[oc][ic][0 ~ PATCH_SIZE-1][0 ~ PATCH_SIZE-1]
                            sum += input[input_idx] * weight.data[kernel_idx];
                        }
                    }
                }

                // input => 3 * 
                // output[EMBED_DIM][n_patch][n_patch]

                // output[oc][oh][ow] = sum
                output[(oc * OUTPUT_SIZE + oh) * OUTPUT_SIZE + ow] = sum;
            }
        }
    }
}



/* ------------------------------------------------------------------------------ */
// transpose + flat
// input[EMBED_DIM][n_patch_per_image][n_patch_per_image] => output[total_num_patches][EMBED_DIM]
void v_flatten_transpose(float* input, float* output) {
    int output_size = IMG_SIZE / PATCH_SIZE;

    for (int oh = 0; oh < output_size; oh++) {
        for (int ow = 0; ow < output_size; ow++) {
            int patch_idx = oh * output_size + ow;
            for (int oc = 0; oc < EMBED_DIM; oc++) {
                int idx_input = (oc * output_size + oh) * output_size + ow;
                int idx_output = patch_idx * EMBED_DIM + oc;

                // output[patch_idx][EMBED_DIM] == output[oh][ow][EMBED_DIM]
                // input[oc][oh][ow]
                output[idx_output] = input[idx_input];
            }
        }
    }
}



/* ------------------------------------------------------------------------------ */
// prepend class tokens in front
// input[total_num_patches][EMBED_DIM] => output[total_num_patches + 1][EMBED_DIM]
void v_class_token(
    float* patch_tokens, float* final_tokens,
    Network cls_tk
    ) {
    int output_size = IMG_SIZE / PATCH_SIZE;
    int num_patches = output_size * output_size;

    // attach class token in front
    for (int j = 0; j < EMBED_DIM; j++) {
        final_tokens[j] = cls_tk.data[j];
    }

    // copy rest of data
    memcpy(final_tokens + EMBED_DIM, patch_tokens, sizeof(float) * EMBED_DIM * num_patches);
}



/* ------------------------------------------------------------------------------ */
// add position embedding data
// input[total_num_patches + 1][EMBED_DIM] => output[total_num_patches + 1][EMBED_DIM]
void v_pos_emb(
    float* input, float* output,
    Network pos_emb
    ) {
    int output_size = IMG_SIZE / PATCH_SIZE;
    int num_patches = output_size * output_size;
    int total_tokens = num_patches + 1;
    int total_elements = total_tokens * EMBED_DIM;

    // add position embedding data
    for (int i = 0; i < total_elements; i++) {
        output[i] = input[i] + pos_emb.data[i];
    }
}



/* ------------------------------------------------------------------------------ */
// encode input
// input[total_num_patches + 1][EMBED_DIM]
void v_Encoder(
    float* input, float* output,
    Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
    Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b
    ) {
    printf(">> [v_Encoder] started\n");

    int n_tokens = ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE)) + 1;
    size_t buffer_size = sizeof(float) * n_tokens * EMBED_DIM;

    // normalize input
    float* input_normalized = (float*)malloc(buffer_size);
    LOG("input_normalized", v_layer_norm(input, input_normalized, ln1_w, ln1_b));

    // multi-head self attention
    float* attn_out = (float*)malloc(buffer_size);
    LOG("multi-head self attention", v_multihead_attn(input_normalized, attn_out, attn_w, attn_b, attn_out_w, attn_out_b));

    /*Residual1*/
    // skip-connection
    float* residual = (float*)malloc(buffer_size);
    LOG("1st Residual1", matrix_plus(input, attn_out, residual, n_tokens * EMBED_DIM));


    // normalize again
    float* residual_normalized = (float*)malloc(buffer_size);
    LOG("residual_normalized", v_layer_norm(residual, residual_normalized, ln2_w, ln2_b));

    /* MLP */
    float* mlp_out = (float*)malloc(buffer_size);
    LOG("MLP", v_mlp_block(residual_normalized, mlp_out, mlp1_w, mlp1_b, mlp2_w, mlp2_b));

    /*Residual2*/
    // skip connection again
    LOG("2nd Residual1",
        for (int i = 0; i < n_tokens * EMBED_DIM; i++) {
            output[i] = residual[i] + mlp_out[i];
        }
    )

    // wrap up
    free(input_normalized);
    free(attn_out);
    free(residual);
    free(residual_normalized);
    free(mlp_out);

    printf(">> [v_Encoder] ended\n");
}











/* ------------------------------------------------------------------------------ */
//
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




///////////////////////////////////////////////////////////////////////////////////////////////////



// TODO: cl_mem 으로
void matrix_plus(
    float* input1, float* input2, float* output,
    size_t n_data
    ) {
    // create memory objects
    size_t buf_size = n_data * sizeof(float);
    cl_mem m_input1 = clCreateBuffer(container.context, CL_MEM_READ_WRITE, buf_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_input2 = clCreateBuffer(container.context, CL_MEM_READ_WRITE, buf_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, buf_size, NULL, &err);
    CHECK_CL_ERROR(err);

    // write inputs
    err = clEnqueueWriteBuffer(container.queue, m_input1, CL_TRUE, 0, buf_size, input1, 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    err = clEnqueueWriteBuffer(container.queue, m_input2, CL_TRUE, 0, buf_size, input2, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // set kernel args
    // __kernel void matrix_plus (
    //     __global float* g_A,
    //     __global float* g_B,
    //     __global float* g_C
    // ) {
    KernelArg args[] = {
        {.size = sizeof(cl_mem), .addr = &m_input1 },
        {.size = sizeof(cl_mem), .addr = &m_input2 },
        {.size = sizeof(cl_mem), .addr = &m_output }
    };

    for (int i = 0; i < 3; ++i) {
        err = clSetKernelArg(container.kernels[__matrix_plus], i, args[i].size, args[i].addr);
        CHECK_CL_ERROR(err);
    }

    // run kernel
    size_t gloabl_work_size[] = { n_data };
    err = clEnqueueNDRangeKernel(
        container.queue, container.kernels[__matrix_plus],
        1, NULL, gloabl_work_size, NULL,
        0, NULL, NULL
        );
    CHECK_CL_ERROR(err);

    // read reuslt
    err = clEnqueueReadBuffer(container.queue, m_output, CL_TRUE, 0, buf_size, output, 0, NULL, NULL);
    CHECK_CL_ERROR(err);


    err = clReleaseMemObject(m_input1);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_input2);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_output);
    CHECK_CL_ERROR(err);
}

