#include "ViT_cl.h"

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
    {.kernel_name = "matrix_plus", .file_path = "./kernels/matrix_plus.cl" },
    {.kernel_name = "linear", .file_path = "./kernels/linear.cl" },
    {.kernel_name = "gelu", .file_path = "./kernels/gelu.cl" },
    {.kernel_name = "cal_score", .file_path = "./kernels/multihead_attrention/cal_score.cl" },
    {.kernel_name = "cal_result", .file_path = "./kernels/multihead_attrention/cal_result.cl" },
    {.kernel_name = "softmax_kernel", .file_path = "./kernels/softmax.cl" },
    {.kernel_name = "softmax_score_kernel", .file_path = "./kernels/softmax_score.cl" },
    {.kernel_name = "layer_norm", .file_path = "./kernels/layer_norm.cl" },
    {.kernel_name = "patch_embed", .file_path = "./kernels/patch_embed.cl" },
    {.kernel_name = "flatten_transpose", .file_path = "./kernels/flatten_transpose.cl" },
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


        

        /* ------------------------------------------------------------------------------------ */
        // normalize

        // create ans write mem obj
        const size_t data_size = N_TOTAL_TOKEN * EMBED_DIM * sizeof(float);
        cl_mem m_encoded = clCreateBuffer(container.context, CL_MEM_READ_WRITE, data_size, NULL, &err);
        CHECK_CL_ERROR(err);
        err = clEnqueueWriteBuffer(container.queue, m_encoded, CL_TRUE, 0, data_size, enc_layer[11], 0, NULL, NULL);
        CHECK_CL_ERROR(err);
        
        const size_t weight_size = networks[148].size * sizeof(float);
        cl_mem m_weight = clCreateBuffer(container.context, CL_MEM_READ_WRITE, weight_size, NULL, &err);
        CHECK_CL_ERROR(err);
        err = clEnqueueWriteBuffer(container.queue, m_weight, CL_TRUE, 0, weight_size, networks[148].data, 0, NULL, NULL);
        CHECK_CL_ERROR(err);
        
        const size_t bias_size = networks[149].size * sizeof(float);
        cl_mem m_bias = clCreateBuffer(container.context, CL_MEM_READ_WRITE, bias_size, NULL, &err);
        CHECK_CL_ERROR(err);
        err = clEnqueueWriteBuffer(container.queue, m_bias, CL_TRUE, 0, bias_size, networks[149].data, 0, NULL, NULL);
        CHECK_CL_ERROR(err);

        cl_mem m_normalized = clCreateBuffer(container.context, CL_MEM_READ_WRITE, data_size, NULL, &err);
        CHECK_CL_ERROR(err);

        // layer_norm
        LOG("normalize", v_layer_norm(m_encoded, m_normalized, m_weight, m_bias));

        // read result
        err = clEnqueueReadBuffer(container.queue, m_normalized, CL_TRUE, 0, data_size, enc_output, 0, NULL, NULL);
        CHECK_CL_ERROR(err);

        // release
        err = clReleaseMemObject(m_encoded);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_weight);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_bias);
        CHECK_CL_ERROR(err);
        err = clReleaseMemObject(m_normalized);
        CHECK_CL_ERROR(err);


        /* ------------------------------------------------------------------------------------ */

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

            // release
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

        // wrap up
        free(cls_token);
        free(cls_output);
    }


    for (int i = 0; i < 4; i++) free(layer[i]);
    for (int i = 0; i < 12; i++) free(enc_layer[i]);
    free(enc_output);

    cleanup();

    printf(">> [ViT_cl] : ended\n");
}

