#include "ViT_cl.h"

/*
[잡생각]
커널 이름 컨벤션 정해두면 좋을듯 => 일단 지금은 '__'로 시작하는 걸로 통일
v_ 로 시작하게끔 함수이름 변경
cl 메모리 객체 관련해서도 => 일단 지금은 "m_"으로 시작하는 경로 통일
커널을 쓰는 함수에 대해서도 네이밍 컨벤션?

커널 setArg하는 부분에 kenrl 정의부만 복붙?

커널 인자 세팅도 함수로 뺄 수 있나?

work_group_size 최대 크기 가져오기?

완전히 CL로만 바뀌면 인자를 cl_mem으로 서로 넘겨받는게 좋을 듯
확장성 고려하면 이벤트 등등도 인자로 하는게 좋을지도?
-> 큐 in-order면 차피 무시되니까 그냥 초장부터 이렇게 하는 게 좋을 듯

*/

// TODO: matrix_plus 함수 cl_mem 받도록
// TODO: position embedding 함수 matrix_plus쓰는 편으로 변경
// TODO: n_token은 상수로 뺄까? -> 뺐고 대채하기

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
// kernel configuration
// 커널 개수 알맞게 바꾸고, enum 및 필요 정보 추가
// Kernels_idxs와 kernel_configs의 순서가 맞아야 함
// kernel_configs 를 순회해서 각 file_path 별로 소스 코드를 뽑아서 빌드함

#define N_KERNEL 7

enum Kernels_idxs{
    __reduce_sum = 0,
    __load_square,
    __normalize,
    __matrix_plus,
    __cl_matrix_plus,
    __linear,
    __cal_mean_and_inv_std,
};

static Kernel_config kernel_configs[N_KERNEL] = {
    { .kernel_name = "reduce_sum", .file_path = "./kernels/reduce_sum.cl" },
    { .kernel_name = "load_square", .file_path = "./kernels/load_square.cl" },
    { .kernel_name = "my_normalize", .file_path = "./kernels/normalize.cl" },
    { .kernel_name = "matrix_plus", .file_path = "./kernels/matrix_plus.cl" },
    { .kernel_name = "cl_matrix_plus", .file_path = "./kernels/cl_matrix_plus.cl" },
    { .kernel_name = "linear", .file_path = "./kernels/linear.cl" },
    { .kernel_name = "cal_mean_and_inv_std", .file_path = "./kernels/cal_mean_and_inv_std.cl" },
};



/////////////////////////////////////////////////////////////////////////////
// global variables

static CL_container container;
static cl_int err;

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
    for (size_t i=0; i<container.n_kernels; ++i) {
        container.src_arr[i] = v_get_source_code(container.kernel_configs[i].file_path, &container.len_arr[i]);
    }

    // create program
    container.program = clCreateProgramWithSource(container.context, container.n_kernels, (const char**)container.src_arr, container.len_arr, &err);
    CHECK_CL_ERROR(err);
    
    // build program
    err = clBuildProgram(container.program, 1, &container.device, NULL, NULL, NULL);
    v_build_error(container.program, container.device, err);

    // create kernels
    for (size_t i=0; i<container.n_kernels; ++i) {
        container.kernels[i] = clCreateKernel(container.program, container.kernel_configs[i].kernel_name, &err);
        CHECK_CL_ERROR(err);
    }

    // free sources
    for (int i=0; i<1; ++i) {
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
    for (size_t i=0; i<container.n_kernels; ++i) {
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

void ViT_cl (
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
    }

    cleanup();

    printf(">> [ViT_cl] : ended\n");
}



////////////////////////////////////////////////////////////////////////////////////
// sub funcitons

// image patch embedding (convolution)
// input[IN_CAHNS][PATCH_SIZE][PATCH_SIZE] => output[EMBED_DIM][n_patch_per_image][n_patch_per_image]
void v_Conv2d (
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
void v_flatten_transpose (float* input, float* output) {
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
void v_class_token (
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
void v_pos_emb (
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


// multi-head self attention
// input[total_num_patches + 1][EMBED_DIM]
void v_multihead_attn(
    float* input, float* output,
    Network in_weight, Network in_bias, 
    Network out_weight, Network out_bias
) {
    const int n_tokens = ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE)) + 1;
    const int Q_dim = 0;
    const int K_dim = EMBED_DIM;
    const int V_dim = EMBED_DIM * 2;

    float* Q = (float*)malloc(sizeof(float) * n_tokens * EMBED_DIM);
    float* K = (float*)malloc(sizeof(float) * n_tokens * EMBED_DIM);
    float* V = (float*)malloc(sizeof(float) * n_tokens * EMBED_DIM);

    // calculate QKV
    {
        // create memory obj
        const size_t input_size = n_tokens * EMBED_DIM * sizeof(float);
        cl_mem m_input = clCreateBuffer(container.context, CL_TRUE, input_size, NULL, &err);
        CHECK_CL_ERROR(err);

        const size_t weight_size = EMBED_DIM * EMBED_DIM * sizeof(float);
        cl_mem m_q_weight = clCreateBuffer(container.context, CL_TRUE, weight_size, NULL, &err);
        CHECK_CL_ERROR(err);    
        cl_mem m_k_weight = clCreateBuffer(container.context, CL_TRUE, weight_size, NULL, &err);
        CHECK_CL_ERROR(err);
        cl_mem m_v_weight = clCreateBuffer(container.context, CL_TRUE, weight_size, NULL, &err);
        CHECK_CL_ERROR(err);
        
        const size_t bias_size = 1 * EMBED_DIM * sizeof(float);
        cl_mem m_q_bias = clCreateBuffer(container.context, CL_TRUE, bias_size, NULL, &err);
        CHECK_CL_ERROR(err);
        cl_mem m_k_bias = clCreateBuffer(container.context, CL_TRUE, bias_size, NULL, &err);
        CHECK_CL_ERROR(err);
        cl_mem m_v_bias = clCreateBuffer(container.context, CL_TRUE, bias_size, NULL, &err);
        CHECK_CL_ERROR(err);
        
        const size_t output_size = n_tokens * EMBED_DIM * sizeof(float);
        cl_mem m_q_output = clCreateBuffer(container.context, CL_TRUE, output_size, NULL, &err);
        CHECK_CL_ERROR(err);
        cl_mem m_k_output = clCreateBuffer(container.context, CL_TRUE, output_size, NULL, &err);
        CHECK_CL_ERROR(err);
        cl_mem m_v_output = clCreateBuffer(container.context, CL_TRUE, output_size, NULL, &err);
        CHECK_CL_ERROR(err);

    
        // write
        err = clEnqueueWriteBuffer(container.queue, m_input, CL_TRUE, 0, input_size, input, 0, NULL, NULL);
        CHECK_CL_ERROR(err);
        
        err = clEnqueueWriteBuffer(container.queue, m_q_weight, CL_TRUE, 0, weight_size, in_weight.data + Q_dim * EMBED_DIM, 0, NULL, NULL);
        CHECK_CL_ERROR(err);
        err = clEnqueueWriteBuffer(container.queue, m_k_weight, CL_TRUE, 0, weight_size, in_weight.data + K_dim * EMBED_DIM, 0, NULL, NULL);
        CHECK_CL_ERROR(err);
        err = clEnqueueWriteBuffer(container.queue, m_v_weight, CL_TRUE, 0, weight_size, in_weight.data + V_dim * EMBED_DIM, 0, NULL, NULL);
        CHECK_CL_ERROR(err);
        
        err = clEnqueueWriteBuffer(container.queue, m_q_bias, CL_TRUE, 0, bias_size, in_bias.data + Q_dim, 0, NULL, NULL);
        CHECK_CL_ERROR(err);
        err = clEnqueueWriteBuffer(container.queue, m_k_bias, CL_TRUE, 0, bias_size, in_bias.data + K_dim, 0, NULL, NULL);
        CHECK_CL_ERROR(err);
        err = clEnqueueWriteBuffer(container.queue, m_v_bias, CL_TRUE, 0, bias_size, in_bias.data + V_dim, 0, NULL, NULL);
        CHECK_CL_ERROR(err);


        // set kernel args
        // run kernel
        v_linear_layer(m_input, m_q_weight, m_q_bias, m_q_output, n_tokens, EMBED_DIM, EMBED_DIM, 0, NULL, NULL);
        v_linear_layer(m_input, m_k_weight, m_k_bias, m_k_output, n_tokens, EMBED_DIM, EMBED_DIM, 0, NULL, NULL);
        v_linear_layer(m_input, m_v_weight, m_v_bias, m_v_output, n_tokens, EMBED_DIM, EMBED_DIM, 0, NULL, NULL);

        // read
        err = clEnqueueReadBuffer(container.queue, m_q_output, CL_TRUE, 0, output_size, Q, 0, NULL, NULL);
        CHECK_CL_ERROR(err);
        err = clEnqueueReadBuffer(container.queue, m_k_output, CL_TRUE, 0, output_size, K, 0, NULL, NULL);
        CHECK_CL_ERROR(err);
        err = clEnqueueReadBuffer(container.queue, m_v_output, CL_TRUE, 0, output_size, V, 0, NULL, NULL);
        CHECK_CL_ERROR(err);
        
        // relase mem obj
        err = clReleaseMemObject(m_input);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_q_weight);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_k_weight);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_v_weight);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_q_bias);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_k_bias);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_v_bias);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_q_output);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_k_output);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_v_output);
        CHECK_CL_ERROR(err);
    }



    // attn_output[n_tokens][EMBED_DIM]
    float* attn_output = (float*)malloc(sizeof(float) * n_tokens * EMBED_DIM);
    for (int i = 0; i < n_tokens * EMBED_DIM; i++) {
        attn_output[i] = 0.0f;
    }

    // calculate multi-head self attention
    int head_dim = EMBED_DIM / NUM_HEADS;
    for (int h = 0; h < NUM_HEADS; h++) {
        int head_offset = h * head_dim;

        // scores[tokens][tokens]
        float* scores = (float*)malloc(sizeof(float) * n_tokens * n_tokens);

        // calcuate socres: scaled-dot-product Q and K
        for (int i = 0; i < n_tokens; i++) {
            for (int j = 0; j < n_tokens; j++) {
                float score = 0.0f;

                for (int d = 0; d < head_dim; d++) {
                    float q = Q[i * EMBED_DIM + head_offset + d];
                    float k = K[j * EMBED_DIM + head_offset + d];
                    score += q * k;
                }

                scores[i * n_tokens + j] = score / sqrtf((float)head_dim);
            }
        }

        // v_Softmax scores
        for (int i = 0; i < n_tokens; i++) {
            // get max value of 1 score row
            float max_val = scores[i * n_tokens];
            for (int j = 1; j < n_tokens; j++) {
                if (scores[i * n_tokens + j] > max_val) {
                    max_val = scores[i * n_tokens + j];
                }
            }

            // score[i][j] = e^(score[i][j] - max_score)
            // cal sum of them
            float sum_exp = 0.0f;
            for (int j = 0; j < n_tokens; j++) {
                scores[i * n_tokens + j] = expf(scores[i * n_tokens + j] - max_val);
                sum_exp += scores[i * n_tokens + j];
            }

            // normalzie
            for (int j = 0; j < n_tokens; j++) {
                scores[i * n_tokens + j] /= sum_exp;
            }
        }

        // calculate result
        float* head_out = (float*)malloc(sizeof(float) * n_tokens * head_dim);
        for (int i = 0; i < n_tokens; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j < n_tokens; j++) {
                    sum += scores[i * n_tokens + j] * V[j * EMBED_DIM + head_offset + d];
                }

                head_out[i * head_dim + d] = sum;
            }
        }

        // load result
        for (int i = 0; i < n_tokens; i++) {
            for (int d = 0; d < head_dim; d++) {
                // attn_output[i][head_offset + d] = head_out[i][d]
                attn_output[i * EMBED_DIM + head_offset + d] = head_out[i * head_dim + d];
            }
        }

        free(scores);
        free(head_out);
    }

    free(Q); free(K); free(V);



    // convert attn_output into output space
    // TODO: convert below using v_lyneaer_
    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < EMBED_DIM; i++) {
            float sum = out_bias.data[i];
            for (int j = 0; j < EMBED_DIM; j++) {
                sum += attn_output[t * EMBED_DIM + j] * out_weight.data[i * EMBED_DIM + j];
            }

            output[t * EMBED_DIM + i] = sum;
        }
    }

    // wrap up
    free(attn_output);
}


// multi-layer perceptron
void v_mlp_block (
    float* input, float* output, 
    Network fc1_weight, Network fc1_bias, 
    Network fc2_weight, Network fc2_bias
) {
    int tokens = ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE)) + 1; //197
    int Embed_dim = EMBED_DIM; //768
    int hidden_dim = ((int)(EMBED_DIM * MLP_RATIO)); //3072

    UNUSED(Embed_dim);

    float* fc1_out = (float*)malloc(sizeof(float) * tokens * hidden_dim);
    {
        // create mem obj

        // input = [tokens x in_features]
        const size_t input_size = tokens * EMBED_DIM * sizeof(float);
        cl_mem m_input = clCreateBuffer(container.context, CL_MEM_READ_WRITE, input_size, NULL, &err);
        CHECK_CL_ERROR(err);
        
        // weight = [out_features x in_features]
        const size_t weight_size = hidden_dim * EMBED_DIM * sizeof(float);
        cl_mem m_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, weight_size, NULL, &err);
        CHECK_CL_ERROR(err);

        // m_bias = [1 x out_features]
        const size_t bias_size = hidden_dim * sizeof(float);
        cl_mem m_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, bias_size, NULL, &err);
        CHECK_CL_ERROR(err);
        
        // output = [tokens x out_features]
        const size_t output_size = tokens * hidden_dim * sizeof(float);
        cl_mem m_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, output_size, NULL, &err);
        CHECK_CL_ERROR(err);

        // write
        err = clEnqueueWriteBuffer(container.queue, m_input, CL_TRUE, 0, input_size, input, 0, NULL, NULL);
        CHECK_CL_ERROR(err);
        err = clEnqueueWriteBuffer(container.queue, m_weight, CL_TRUE, 0, weight_size, fc1_weight.data, 0, NULL, NULL);
        CHECK_CL_ERROR(err);
        err = clEnqueueWriteBuffer(container.queue, m_bias, CL_TRUE, 0, bias_size, fc1_bias.data, 0, NULL, NULL);
        CHECK_CL_ERROR(err);

        v_linear_layer(
            m_input, m_weight, m_bias, m_output, 
            tokens, EMBED_DIM, hidden_dim,
            0, NULL, NULL
        );

        // read result
        err = clEnqueueReadBuffer(container.queue, m_output, CL_TRUE, 0, output_size, fc1_out, 0, NULL, NULL);
        CHECK_CL_ERROR(err);
        
        // release mem obj
        err = clReleaseMemObject(m_input);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_weight);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_bias);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_output);
        CHECK_CL_ERROR(err);
    }

    // TODO: 여기도 해버려서 v_mlp_block 그냥 하나로 묶어버리자
    // apply GELU
    for (int i = 0; i < tokens * hidden_dim; i++) {
        fc1_out[i] = v_gelu(fc1_out[i]);
    }

    // fc2: (tokens, in_dim)
    {
        // create mem obj

        // input = [tokens x in_features]
        const size_t input_size = tokens * hidden_dim * sizeof(float);
        cl_mem m_input = clCreateBuffer(container.context, CL_MEM_READ_WRITE, input_size, NULL, &err);
        CHECK_CL_ERROR(err);
        
        // weight = [out_features x in_features]
        const size_t weight_size = EMBED_DIM * hidden_dim * sizeof(float);
        cl_mem m_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, weight_size, NULL, &err);
        CHECK_CL_ERROR(err);

        // m_bias = [1 x out_features]
        const size_t bias_size = EMBED_DIM * sizeof(float);
        cl_mem m_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, bias_size, NULL, &err);
        CHECK_CL_ERROR(err);
        
        // output = [tokens x out_features]
        const size_t output_size = tokens * EMBED_DIM * sizeof(float);
        cl_mem m_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, output_size, NULL, &err);
        CHECK_CL_ERROR(err);

        // write
        err = clEnqueueWriteBuffer(container.queue, m_input, CL_TRUE, 0, input_size, fc1_out, 0, NULL, NULL);
        CHECK_CL_ERROR(err);
        err = clEnqueueWriteBuffer(container.queue, m_weight, CL_TRUE, 0, weight_size, fc2_weight.data, 0, NULL, NULL);
        CHECK_CL_ERROR(err);
        err = clEnqueueWriteBuffer(container.queue, m_bias, CL_TRUE, 0, bias_size, fc2_bias.data, 0, NULL, NULL);
        CHECK_CL_ERROR(err);

        v_linear_layer(
            m_input, m_weight, m_bias, m_output, 
            tokens, hidden_dim, EMBED_DIM,
            0, NULL, NULL
        );

        // read result
        err = clEnqueueReadBuffer(container.queue, m_output, CL_TRUE, 0, output_size, output, 0, NULL, NULL);
        CHECK_CL_ERROR(err);

        // release mem obj
        err = clReleaseMemObject(m_input);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_weight);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_bias);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_output);
        CHECK_CL_ERROR(err);
    }

    free(fc1_out);
}


// GELU function
float v_gelu(float x) {
    return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
}




/* ------------------------------------------------------------------------------ */
//
void v_Softmax(float* logits, float* probabilities, int length) {
    float max_val = logits[0];
    for (int i = 1; i < length; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < length; i++) {
        probabilities[i] = expf(logits[i] - max_val);
        sum_exp += probabilities[i];
    }

    for (int i = 0; i < length; i++) {
        probabilities[i] /= sum_exp;
    }
}



////////////////////////////////////////////////////////////////////////////////////
// rather common functions

// normalize
// input[total_num_patches + 1][EMBED_DIM] => output[total_num_patches + 1][EMBED_DIM]
void v_layer_norm (
    float* input, float* output, 
    Network weight, Network bias
) {
    const int token = N_TOTAL_TOKEN;
    const int total_num_data = token * EMBED_DIM;

    memcpy(output, input, total_num_data * sizeof(float));

    const size_t work_group_size = 1024;
    for (int t = 0; t < token; t++) {
        float* p_data = output + t * EMBED_DIM;
        const size_t n_data = EMBED_DIM;

        // create and write mem obj
        size_t data_size = n_data * sizeof(float);
        cl_mem m_data = clCreateBuffer(container.context, CL_MEM_READ_WRITE, data_size, NULL, &err);
        err = clEnqueueWriteBuffer(container.queue, m_data, CL_TRUE, 0, data_size, p_data, 0, NULL, NULL);

        size_t weight_size = weight.size * sizeof(float);
        cl_mem m_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, weight_size, NULL, &err);
        err = clEnqueueWriteBuffer(container.queue, m_weight, CL_TRUE, 0, weight_size, weight.data, 0, NULL, NULL);

        size_t bias_size = bias.size * sizeof(float);
        cl_mem m_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, bias_size, NULL, &err);
        err = clEnqueueWriteBuffer(container.queue, m_bias, CL_TRUE, 0, bias_size, bias.data, 0, NULL, NULL);

        size_t just_float1_size = sizeof(float);
        cl_mem m_sum = clCreateBuffer(container.context, CL_MEM_READ_WRITE, just_float1_size, NULL, &err);
        cl_mem m_sum_of_square = clCreateBuffer(container.context, CL_MEM_READ_WRITE, just_float1_size, NULL, &err);
        cl_mem m_mean = clCreateBuffer(container.context, CL_MEM_READ_WRITE, just_float1_size, NULL, &err);
        cl_mem m_inv_std = clCreateBuffer(container.context, CL_MEM_READ_WRITE, just_float1_size, NULL, &err);



        // run kernels
        v_reduce_sum(m_data, m_sum, n_data, work_group_size, 0, NULL, NULL);
        v_reduce_sum_of_square(m_data, m_sum_of_square, n_data, work_group_size, 0, NULL, NULL);
        v_cal_mean_and_inv_std(m_sum, m_sum_of_square, m_mean, m_inv_std, 0, NULL, NULL);
        v_normalize(m_data, m_weight, m_bias, m_mean, m_inv_std, total_num_data, 0, NULL, NULL);



        // read result
        err = clEnqueueReadBuffer(container.queue, m_data, CL_TRUE, 0, data_size, p_data, 0, NULL, NULL);


        // release mem objs
        err = clReleaseMemObject(m_data);
        err = clReleaseMemObject(m_weight);
        err = clReleaseMemObject(m_bias);
        err = clReleaseMemObject(m_sum);
        err = clReleaseMemObject(m_sum_of_square);
        err = clReleaseMemObject(m_mean);
        err = clReleaseMemObject(m_inv_std);
    }
}

void v_cal_mean_and_inv_std (
    cl_mem m_sum, cl_mem m_sum_of_square,
    cl_mem m_mean, cl_mem m_inv_std,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
) {
    cl_kernel k = container.kernels[__cal_mean_and_inv_std];

    // set kernel args
    // __kernel void cal_mean_and_inv_std (
    //     __global float* g_sum,
    //     __global float* g_sum_of_square,
    //     __global float* g_output_mean,
    //     __global float* g_output_inv_std,
    //     int EMBED_DIM,
    //     float EPSILON
    // ) {
    const int embed_dim = EMBED_DIM;
    const float epsilon = EPSILON;
    KernelArg args[] = {
        { .size = sizeof(cl_mem), .addr = &m_sum },
        { .size = sizeof(cl_mem), .addr = &m_sum_of_square },
        { .size = sizeof(cl_mem), .addr = &m_mean },
        { .size = sizeof(cl_mem), .addr = &m_inv_std },
        { .size = sizeof(int), .addr = &embed_dim },
        { .size = sizeof(float), .addr = &epsilon },
    };

    for (int i=0; i<6; ++i) {
        err = clSetKernelArg(k, i, args[i].size, args[i].addr);
        CHECK_CL_ERROR(err);
    }

    // run kernel
    size_t global_work_size[] = { 1 };
    err = clEnqueueNDRangeKernel(
        container.queue, k, 
        1, NULL, global_work_size, NULL, 
        e_num_waiting, e_waiting_arr, e_out
    );
    CHECK_CL_ERROR(err);
}

void v_reduce_sum (
    cl_mem m_data, 
    cl_mem m_output, 
    size_t total_num_data,
    size_t work_group_size,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
) {
    // work_group_size는 2의 거듭제곱수여야 함
    assert(((work_group_size & (work_group_size - 1)) == 0));
    
    // create memory object
    const size_t buf_size = total_num_data * sizeof(float);
    cl_mem m_from = clCreateBuffer(container.context, CL_MEM_READ_WRITE, buf_size, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem m_to = clCreateBuffer(container.context, CL_MEM_READ_WRITE, buf_size, NULL, &err);
    CHECK_CL_ERROR(err);
    
    // copy inpupt
    err = clEnqueueCopyBuffer(container.queue, m_data, m_from, 0, 0, buf_size, e_num_waiting, e_waiting_arr, NULL);
    CHECK_CL_ERROR(err);

    size_t n_data = total_num_data;
    while (n_data > 1) {
        // n_grouup = ceil(n_data / work_group_size)
        size_t n_group = (n_data + work_group_size - 1) / work_group_size;

        // set kernel args
        // __kernel void reduce_sum (
        //     __global float* const g_input,
        //     __global float* const g_output,
        //     int n_data,
        //     __local float* l_sum
        // ) {
        KernelArg args[] = {
            { .size = sizeof(cl_mem), .addr = &m_from},
            { .size = sizeof(cl_mem), .addr = &m_to},
            { .size = sizeof(int), .addr = &n_data},
            { .size = sizeof(float) * work_group_size, .addr = NULL}
        };

        for (int i=0; i<4; ++i) {
            err = clSetKernelArg(container.kernels[__reduce_sum], i, args[i].size, args[i].addr);
            CHECK_CL_ERROR(err);
        }

        // run kernel
        const size_t global_work_size = n_group * work_group_size;
        const size_t local_work_size = work_group_size;
        err = clEnqueueNDRangeKernel(
            container.queue, container.kernels[__reduce_sum], 
            1, NULL, &global_work_size, &local_work_size, 
            0, NULL, NULL);
        CHECK_CL_ERROR(err);

        // swap input and output buffers
        cl_mem tmp = m_from;
        m_from = m_to;
        m_to = tmp;

        // set next value
        n_data = n_group;
    }

    // write result to m_output
    // m_from[0]에 최종결과가 남음
    err = clEnqueueCopyBuffer(container.queue, m_from, m_output, 0, 0, sizeof(float), 0, NULL, e_out);

    // release mem obj
    err = clReleaseMemObject(m_from);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_to);
    CHECK_CL_ERROR(err);
}


void v_reduce_sum_of_square (
    cl_mem m_data, 
    cl_mem m_output, 
    size_t total_num_data,
    size_t work_group_size,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
) {
    // work_group_size는 2의 거듭제곱수여야 함
    assert(((work_group_size & (work_group_size - 1)) == 0));

    // 흐음 얘 어떻게 해야하지...?
    UNUSED(e_out);

    // create memory objects
    size_t data_size = total_num_data * sizeof(float);
    cl_mem m_tmp = clCreateBuffer(container.context, CL_MEM_READ_WRITE, data_size, NULL, &err);
    CHECK_CL_ERROR(err);
    
    // copy data to m_tmp
    err = clEnqueueCopyBuffer(container.queue, m_data, m_tmp, 0, 0, data_size, e_num_waiting, e_waiting_arr, NULL);
    CHECK_CL_ERROR(err);

    // square m_tmp matrix
    {
        // set args
        // __kernel void load_square (
        //     __global float* g_input
        // ) {
        //     size_t global_id = get_global_id(0);
        //     g_input[global_id] = g_input[global_id] * g_input[global_id];
        // }
        KernelArg args[] = {
            { .size = sizeof(cl_mem), .addr = &m_tmp},
        };

        for (int i=0; i<1; ++i) {
            err = clSetKernelArg(container.kernels[__load_square], i, args[i].size, args[i].addr);
            CHECK_CL_ERROR(err);
        }

        // run kernel
        const size_t global_work_size = total_num_data;
        err = clEnqueueNDRangeKernel(
            container.queue, container.kernels[__load_square], 
            1, NULL, &global_work_size, NULL, 
            0, NULL, NULL);
        CHECK_CL_ERROR(err);
    }

    // process reduce sum with squared data matrix
    v_reduce_sum(m_tmp, m_output, total_num_data, work_group_size, 0, NULL, NULL);

    // release memory objects
    err = clReleaseMemObject(m_tmp);
    CHECK_CL_ERROR(err);
}


void v_normalize (
    cl_mem m_input, 
    cl_mem m_weight, cl_mem m_bias,
    cl_mem m_mean, cl_mem m_inv_std,
    int total_num_data,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
) {
    // set kernel args
    // __kernel void my_normalize(
    //     __global float* g_input,
    //     __global const float* g_weight,
    //     __global const float* g_bias,
    //     __global float* g_MEAN,
    //     __global float* g_INV_STD
    // ) {
    KernelArg args[] = {
        { .size = sizeof(cl_mem), .addr = &m_input},
        { .size = sizeof(cl_mem), .addr = &m_weight},
        { .size = sizeof(cl_mem), .addr = &m_bias},
        { .size = sizeof(cl_mem), .addr = &m_mean},
        { .size = sizeof(cl_mem), .addr = &m_inv_std}
    };
    for (int i=0; i<5; ++i) {
        err = clSetKernelArg(container.kernels[__normalize], i, args[i].size, args[i].addr);
        CHECK_CL_ERROR(err);
    }

    // run kernel
    const size_t global_work_size = total_num_data;
    err = clEnqueueNDRangeKernel(
        container.queue, container.kernels[__normalize], 
        1, NULL, &global_work_size, NULL, 
        e_num_waiting, e_waiting_arr, e_out);
    CHECK_CL_ERROR(err);
}





///////////////////////////////////////////////////////////////////////////////////////////////////

// do linear transform with input matrix(network)
void v_linear_layer (
    cl_mem m_input, cl_mem m_weight, cl_mem m_bias, cl_mem m_output, 
    int tokens, int in_features, int out_features,
    cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
) {
    const cl_kernel k = container.kernels[__linear];

    // set kenrl args
    // __kernel void linear(
    // 	__global const float* g_input,
    // 	__global const float* g_weight,
    // 	__global const float* g_bias,
    // 	__global float* g_output,
    // 	const int tokens,           // M
    // 	const int in_features,      // N
    // 	const int out_features      // K
    // ) {
    KernelArg args[] = {
        { .size = sizeof(cl_mem), .addr = &m_input },
        { .size = sizeof(cl_mem), .addr = &m_weight },
        { .size = sizeof(cl_mem), .addr = &m_bias },
        { .size = sizeof(cl_mem), .addr = &m_output },
        { .size = sizeof(int), .addr = &tokens },
        { .size = sizeof(int), .addr = &in_features },
        { .size = sizeof(int), .addr = &out_features },
    };

    for (int i=0; i<7; ++i) {
        err = clSetKernelArg(k, i, args[i].size, args[i].addr);
        CHECK_CL_ERROR(err);
    }

    // run kernel
    const size_t gloabal_work_size[] = { tokens, out_features };
    err = clEnqueueNDRangeKernel(
        container.queue, k,
        2, NULL, gloabal_work_size, NULL,
        e_num_waiting, e_waiting_arr, e_out
    );
    CHECK_CL_ERROR(err);
}

void matrix_plus (
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
        { .size = sizeof(cl_mem), .addr = &m_input1 },
        { .size = sizeof(cl_mem), .addr = &m_input2 },
        { .size = sizeof(cl_mem), .addr = &m_output }
    };

    for (int i=0; i<3; ++i) {
        err = clSetKernelArg(container.kernels[__matrix_plus], i, args[i].size, args[i].addr);
        CHECK_CL_ERROR(err);
    }

    // run kernel
    size_t gloabl_work_size[] = { n_data };
    err = clEnqueueNDRangeKernel(
        container.queue, container.kernels[__matrix_plus], 
        1, NULL, gloabl_work_size, NULL, 
        0,NULL, NULL
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







///////////////////////////////////////////////////////////////////////////////////////////////////


// 행렬덧셈: m_lvalue += m_rvalue
// void cl_matrix_plus (
//     cl_mem m_lvalue, cl_mem m_rvalue, 
//     size_t n_data,
//     cl_uint e_num_waiting, const cl_event* e_waiting_arr, cl_event* e_out
// ) {
//     cl_kernel target_kernel = container.kernels[__cl_matrix_plus];

//     // set kernel args
//     // __kernel void cl_matrix_plus (
//     //     __global float* g_lvalue,
//     //     __global float* g_rvalue
//     // ) {
//     KernelArg args[] = {
//         { .size = sizeof(cl_mem), .addr = &m_lvalue },
//         { .size = sizeof(cl_mem), .addr = &m_rvalue },
//     };

//     for (int i=0; i<2; ++i) {
//         err = clSetKernelArg(target_kernel, i, args[i].size, args[i].addr);
//         CHECK_CL_ERROR(err);
//     }

//     // run kernel
//     size_t gloabl_work_size[] = { n_data };
//     err = clEnqueueNDRangeKernel(
//         container.queue, target_kernel, 
//         1, NULL, gloabl_work_size, NULL, 
//         e_num_waiting, e_waiting_arr, e_out
//     );
//     CHECK_CL_ERROR(err);
// }


