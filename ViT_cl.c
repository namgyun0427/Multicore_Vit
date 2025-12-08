#include "ViT_cl.h"

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

void init(Network* networks) {
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

    // create network mem objs
    for (int i=0; i<NETWORK_NUM; ++i) {
        const size_t network_size = networks[i].size * sizeof(float);
        container.m_networks[i] = clCreateBuffer(container.context, CL_MEM_READ_WRITE, network_size, NULL, &err);
        CHECK_CL_ERROR(err);
        err = clEnqueueWriteBuffer(container.queue, container.m_networks[i], CL_TRUE, 0, network_size, networks[i].data, 0, NULL, NULL);
        CHECK_CL_ERROR(err);
    }


    printf(">> [init] : ended\n");
}

void cleanup() {
    printf(">> [cleanup] : start\n");

    // release newtork mem objs
    for (int i=0; i<NETWORK_NUM; ++i) {
        err = clReleaseMemObject(container.m_networks[i]);
        CHECK_CL_ERROR(err);
    }

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

    init(networks);

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
            // input mem obj
            const size_t position_embeded_size = size[3] * sizeof(float);
            cl_mem m_position_embeded = clCreateBuffer(container.context, CL_MEM_READ_WRITE, position_embeded_size, NULL, &err);
            CHECK_CL_ERROR(err);
            err = clEnqueueWriteBuffer(container.queue, m_position_embeded, CL_TRUE, 0, position_embeded_size, position_embeded, 0, NULL, NULL);
            CHECK_CL_ERROR(err);

            // create output mem obj
            const size_t enc_output_size = ENC_SIZE * sizeof(float);
            cl_mem m_enc_output_arr[12];
            for (int i=0; i<12; ++i) {
                m_enc_output_arr[i] = clCreateBuffer(container.context, CL_MEM_READ_WRITE, enc_output_size, NULL, &err);
                CHECK_CL_ERROR(err);
            }

            // run encoder kernels
            v_Encoder(
                m_position_embeded, m_enc_output_arr[0],
                container.m_networks[4], container.m_networks[5], container.m_networks[6], container.m_networks[7],
                container.m_networks[8], container.m_networks[9], container.m_networks[10], container.m_networks[11],
                container.m_networks[12], container.m_networks[13], container.m_networks[14], container.m_networks[15]
            );

            v_Encoder(m_enc_output_arr[0], m_enc_output_arr[1],
                container.m_networks[16], container.m_networks[17], container.m_networks[18], container.m_networks[19],
                container.m_networks[20], container.m_networks[21], container.m_networks[22], container.m_networks[23],
                container.m_networks[24], container.m_networks[25], container.m_networks[26], container.m_networks[27]);

            v_Encoder(m_enc_output_arr[1], m_enc_output_arr[2],
                container.m_networks[28], container.m_networks[29], container.m_networks[30], container.m_networks[31],
                container.m_networks[32], container.m_networks[33], container.m_networks[34], container.m_networks[35],
                container.m_networks[36], container.m_networks[37], container.m_networks[38], container.m_networks[39]);

            v_Encoder(m_enc_output_arr[2], m_enc_output_arr[3],
                container.m_networks[40], container.m_networks[41], container.m_networks[42], container.m_networks[43],
                container.m_networks[44], container.m_networks[45], container.m_networks[46], container.m_networks[47],
                container.m_networks[48], container.m_networks[49], container.m_networks[50], container.m_networks[51]);

            v_Encoder(m_enc_output_arr[3], m_enc_output_arr[4],
                container.m_networks[52], container.m_networks[53], container.m_networks[54], container.m_networks[55],
                container.m_networks[56], container.m_networks[57], container.m_networks[58], container.m_networks[59],
                container.m_networks[60], container.m_networks[61], container.m_networks[62], container.m_networks[63]);

            v_Encoder(m_enc_output_arr[4], m_enc_output_arr[5],
                container.m_networks[64], container.m_networks[65], container.m_networks[66], container.m_networks[67],
                container.m_networks[68], container.m_networks[69], container.m_networks[70], container.m_networks[71],
                container.m_networks[72], container.m_networks[73], container.m_networks[74], container.m_networks[75]);

            v_Encoder(m_enc_output_arr[5], m_enc_output_arr[6],
                container.m_networks[76], container.m_networks[77], container.m_networks[78], container.m_networks[79],
                container.m_networks[80], container.m_networks[81], container.m_networks[82], container.m_networks[83],
                container.m_networks[84], container.m_networks[85], container.m_networks[86], container.m_networks[87]);

            v_Encoder(m_enc_output_arr[6], m_enc_output_arr[7],
                container.m_networks[88], container.m_networks[89], container.m_networks[90], container.m_networks[91],
                container.m_networks[92], container.m_networks[93], container.m_networks[94], container.m_networks[95],
                container.m_networks[96], container.m_networks[97], container.m_networks[98], container.m_networks[99]);

            v_Encoder(m_enc_output_arr[7], m_enc_output_arr[8],
                container.m_networks[100], container.m_networks[101], container.m_networks[102], container.m_networks[103],
                container.m_networks[104], container.m_networks[105], container.m_networks[106], container.m_networks[107],
                container.m_networks[108], container.m_networks[109], container.m_networks[110], container.m_networks[111]);

            v_Encoder(m_enc_output_arr[8], m_enc_output_arr[9],
                container.m_networks[112], container.m_networks[113], container.m_networks[114], container.m_networks[115],
                container.m_networks[116], container.m_networks[117], container.m_networks[118], container.m_networks[119],
                container.m_networks[120], container.m_networks[121], container.m_networks[122], container.m_networks[123]);

            v_Encoder(m_enc_output_arr[9], m_enc_output_arr[10],
                container.m_networks[124], container.m_networks[125], container.m_networks[126], container.m_networks[127],
                container.m_networks[128], container.m_networks[129], container.m_networks[130], container.m_networks[131],
                container.m_networks[132], container.m_networks[133], container.m_networks[134], container.m_networks[135]);

            v_Encoder(m_enc_output_arr[10], m_enc_output_arr[11],
                container.m_networks[136], container.m_networks[137], container.m_networks[138], container.m_networks[139],
                container.m_networks[140], container.m_networks[141], container.m_networks[142], container.m_networks[143],
                container.m_networks[144], container.m_networks[145], container.m_networks[146], container.m_networks[147]);

            // read result
            err = clEnqueueReadBuffer(container.queue, m_enc_output_arr[11], CL_TRUE, 0, enc_output_size, enc_layer[11], 0, NULL, NULL);
            CHECK_CL_ERROR(err);

            // release mem obj
            err = clReleaseMemObject(m_position_embeded);
            CHECK_CL_ERROR(err);
            for (int i=0; i<12; ++i) {
                err = clReleaseMemObject(m_enc_output_arr[i]);
                CHECK_CL_ERROR(err);
            }





            //////////////////////////

            // v_Encoder(position_embeded, enc_layer[0],
            //     networks[4], networks[5], networks[6], networks[7],
            //     networks[8], networks[9], networks[10], networks[11],
            //     networks[12], networks[13], networks[14], networks[15]);

            // v_Encoder(enc_layer[0], enc_layer[1],
            //     networks[16], networks[17], networks[18], networks[19],
            //     networks[20], networks[21], networks[22], networks[23],
            //     networks[24], networks[25], networks[26], networks[27]);

            // v_Encoder(enc_layer[1], enc_layer[2],
            //     networks[28], networks[29], networks[30], networks[31],
            //     networks[32], networks[33], networks[34], networks[35],
            //     networks[36], networks[37], networks[38], networks[39]);

            // v_Encoder(enc_layer[2], enc_layer[3],
            //     networks[40], networks[41], networks[42], networks[43],
            //     networks[44], networks[45], networks[46], networks[47],
            //     networks[48], networks[49], networks[50], networks[51]);

            // v_Encoder(enc_layer[3], enc_layer[4],
            //     networks[52], networks[53], networks[54], networks[55],
            //     networks[56], networks[57], networks[58], networks[59],
            //     networks[60], networks[61], networks[62], networks[63]);

            // v_Encoder(enc_layer[4], enc_layer[5],
            //     networks[64], networks[65], networks[66], networks[67],
            //     networks[68], networks[69], networks[70], networks[71],
            //     networks[72], networks[73], networks[74], networks[75]);

            // v_Encoder(enc_layer[5], enc_layer[6],
            //     networks[76], networks[77], networks[78], networks[79],
            //     networks[80], networks[81], networks[82], networks[83],
            //     networks[84], networks[85], networks[86], networks[87]);

            // v_Encoder(enc_layer[6], enc_layer[7],
            //     networks[88], networks[89], networks[90], networks[91],
            //     networks[92], networks[93], networks[94], networks[95],
            //     networks[96], networks[97], networks[98], networks[99]);

            // v_Encoder(enc_layer[7], enc_layer[8],
            //     networks[100], networks[101], networks[102], networks[103],
            //     networks[104], networks[105], networks[106], networks[107],
            //     networks[108], networks[109], networks[110], networks[111]);

            // v_Encoder(enc_layer[8], enc_layer[9],
            //     networks[112], networks[113], networks[114], networks[115],
            //     networks[116], networks[117], networks[118], networks[119],
            //     networks[120], networks[121], networks[122], networks[123]);

            // v_Encoder(enc_layer[9], enc_layer[10],
            //     networks[124], networks[125], networks[126], networks[127],
            //     networks[128], networks[129], networks[130], networks[131],
            //     networks[132], networks[133], networks[134], networks[135]);

            // v_Encoder(enc_layer[10], enc_layer[11],
            //     networks[136], networks[137], networks[138], networks[139],
            //     networks[140], networks[141], networks[142], networks[143],
            //     networks[144], networks[145], networks[146], networks[147]);
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

