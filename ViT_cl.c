#include "ViT_cl.h"


// constants

static const int size[] = {
    EMBED_DIM * (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE), // conv2D
    EMBED_DIM * (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE), // flatten and transpose
    EMBED_DIM * ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE) + 1), // class token
    EMBED_DIM * ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE) + 1) // position embedding
};

static const int ENC_SIZE = EMBED_DIM * ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE) + 1);


// global variables

CL_container container;
cl_int err;

// Kernel configs


Kernel_config kernel_configs[N_KERNEL] = {
    {.kernel_name = "matrix_plus", .file_path = "./kernels/matrix_plus.cl" },
    {.kernel_name = "linear_layer", .file_path = "./kernels/linear_layer.cl" },
    {.kernel_name = "gelu", .file_path = "./kernels/gelu.cl" },
    {.kernel_name = "cal_score", .file_path = "./kernels/multihead_attrention/cal_score.cl" },
    {.kernel_name = "cal_result", .file_path = "./kernels/multihead_attrention/cal_result.cl" },
    {.kernel_name = "softmax_kernel", .file_path = "./kernels/softmax.cl" },
    {.kernel_name = "softmax_score_kernel", .file_path = "./kernels/softmax_score.cl" },
    {.kernel_name = "layer_norm", .file_path = "./kernels/layer_norm.cl" },
    {.kernel_name = "conv2d", .file_path = "./kernels/conv2d.cl" },
    {.kernel_name = "flatten_transpose", .file_path = "./kernels/flatten_transpose.cl" },
};


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
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES,
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
        0
    };
    container.queue = clCreateCommandQueueWithProperties(container.context, container.device, props, &err);
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
    for (int i = 0; i < NETWORK_NUM; ++i) {
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
    for (int i = 0; i < NETWORK_NUM; ++i) {
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


    // create and write input mem obj = image
    const size_t img_size = IN_CAHNS * IMG_SIZE * IMG_SIZE * sizeof(float);
    cl_mem m_img = clCreateBuffer(container.context, CL_MEM_READ_WRITE, img_size, NULL, &err);
    CHECK_CL_ERROR(err);

    // create output mem obj
    const size_t patch_embedded_size = size[0] * sizeof(float);
    cl_mem m_patch_embedded = clCreateBuffer(container.context, CL_MEM_READ_WRITE, patch_embedded_size, NULL, &err);
    CHECK_CL_ERROR(err);

    const size_t flatten_transposed_size = size[1] * sizeof(float);
    cl_mem m_flatten_transposed = clCreateBuffer(container.context, CL_MEM_READ_WRITE, flatten_transposed_size, NULL, &err);
    CHECK_CL_ERROR(err);

    const size_t class_token_prepended_size = size[2] * sizeof(float);
    cl_mem m_class_token_prepended = clCreateBuffer(container.context, CL_MEM_READ_WRITE, class_token_prepended_size, NULL, &err);
    CHECK_CL_ERROR(err);

    const size_t position_embeded_size = size[3] * sizeof(float);
    cl_mem m_position_embeded = clCreateBuffer(container.context, CL_MEM_READ_WRITE, position_embeded_size, NULL, &err);
    CHECK_CL_ERROR(err);

    const size_t enc_output_size = ENC_SIZE * sizeof(float);
    cl_mem m_enc_output_arr[12];
    for (int i = 0; i < 12; ++i) {
        m_enc_output_arr[i] = clCreateBuffer(container.context, CL_MEM_READ_WRITE, enc_output_size, NULL, &err);
        CHECK_CL_ERROR(err);
    }

    const size_t normalized_size = N_TOTAL_TOKEN * EMBED_DIM * sizeof(float);
    cl_mem m_normalized = clCreateBuffer(container.context, CL_MEM_READ_WRITE, normalized_size, NULL, &err);
    CHECK_CL_ERROR(err);

    const size_t class_token_size = EMBED_DIM * sizeof(float);
    cl_mem m_class_token = clCreateBuffer(container.context, CL_MEM_READ_WRITE, class_token_size, NULL, &err);
    CHECK_CL_ERROR(err);

    const size_t class_output_size = NUM_CLASSES * sizeof(float);
    cl_mem m_class_output = clCreateBuffer(container.context, CL_MEM_READ_WRITE, class_output_size, NULL, &err);
    CHECK_CL_ERROR(err);

    const size_t probabilities_size = NUM_CLASSES * sizeof(float);
    cl_mem m_probabilities = clCreateBuffer(container.context, CL_MEM_READ_WRITE, probabilities_size, NULL, &err);
    CHECK_CL_ERROR(err);





    // process per image
    cl_event e_process_end[IMAGE_COUNT];
    for (int i = 0; i < image->n; i++) {
        printf("============= processing %d-th iamge =============\n", i);

        // write image obj
        cl_event e_img_write;
        err = clEnqueueWriteBuffer(container.queue, m_img, CL_TRUE, 0, img_size, image[i].data, 0, NULL, &e_img_write);
        CHECK_CL_ERROR(err);

        /*patch embedding*/
        cl_event e_conv2d;
        LOG("patch_embedded", v_Conv2d(m_img, m_patch_embedded, container.m_networks[1], container.m_networks[2], 1, &e_img_write, &e_conv2d));

        /*flatten and transpose*/
        cl_event e_flatten_transpose;
        LOG("flatten_transposed", v_flatten_transpose(m_patch_embedded, m_flatten_transposed, 1, &e_conv2d, &e_flatten_transpose));

        /*prepend class token*/
        cl_event e_class_prepend;
        LOG("class_token_prepended", v_class_token(m_flatten_transposed, m_class_token_prepended, container.m_networks[0], 1, &e_flatten_transpose, &e_class_prepend));

        /* position embedding */
        cl_event e_pos_embed;
        LOG("position_embeded", v_pos_emb(m_class_token_prepended, m_position_embeded, container.m_networks[3], 1, &e_class_prepend, &e_pos_embed));

        /*v_Encoder - 12 Layers*/
        cl_event e_encoder[12];
        {
            v_Encoder(
                m_position_embeded, m_enc_output_arr[0],
                container.m_networks[4], container.m_networks[5], container.m_networks[6], container.m_networks[7],
                container.m_networks[8], container.m_networks[9], container.m_networks[10], container.m_networks[11],
                container.m_networks[12], container.m_networks[13], container.m_networks[14], container.m_networks[15],
                1, &e_pos_embed, &e_encoder[0]
            );

            v_Encoder(
                m_enc_output_arr[0], m_enc_output_arr[1],
                container.m_networks[16], container.m_networks[17], container.m_networks[18], container.m_networks[19],
                container.m_networks[20], container.m_networks[21], container.m_networks[22], container.m_networks[23],
                container.m_networks[24], container.m_networks[25], container.m_networks[26], container.m_networks[27],
                1, &e_encoder[0], &e_encoder[1]
            );

            v_Encoder(
                m_enc_output_arr[1], m_enc_output_arr[2],
                container.m_networks[28], container.m_networks[29], container.m_networks[30], container.m_networks[31],
                container.m_networks[32], container.m_networks[33], container.m_networks[34], container.m_networks[35],
                container.m_networks[36], container.m_networks[37], container.m_networks[38], container.m_networks[39],
                1, &e_encoder[1], &e_encoder[2]
            );

            v_Encoder(
                m_enc_output_arr[2], m_enc_output_arr[3],
                container.m_networks[40], container.m_networks[41], container.m_networks[42], container.m_networks[43],
                container.m_networks[44], container.m_networks[45], container.m_networks[46], container.m_networks[47],
                container.m_networks[48], container.m_networks[49], container.m_networks[50], container.m_networks[51],
                1, &e_encoder[2], &e_encoder[3]
            );

            v_Encoder(
                m_enc_output_arr[3], m_enc_output_arr[4],
                container.m_networks[52], container.m_networks[53], container.m_networks[54], container.m_networks[55],
                container.m_networks[56], container.m_networks[57], container.m_networks[58], container.m_networks[59],
                container.m_networks[60], container.m_networks[61], container.m_networks[62], container.m_networks[63],
                1, &e_encoder[3], &e_encoder[4]
                );

            v_Encoder(
                m_enc_output_arr[4], m_enc_output_arr[5],
                container.m_networks[64], container.m_networks[65], container.m_networks[66], container.m_networks[67],
                container.m_networks[68], container.m_networks[69], container.m_networks[70], container.m_networks[71],
                container.m_networks[72], container.m_networks[73], container.m_networks[74], container.m_networks[75],
                1, &e_encoder[4], &e_encoder[5]
                );

            v_Encoder(
                m_enc_output_arr[5], m_enc_output_arr[6],
                container.m_networks[76], container.m_networks[77], container.m_networks[78], container.m_networks[79],
                container.m_networks[80], container.m_networks[81], container.m_networks[82], container.m_networks[83],
                container.m_networks[84], container.m_networks[85], container.m_networks[86], container.m_networks[87],
                1, &e_encoder[5], &e_encoder[6]
                );

            v_Encoder(
                m_enc_output_arr[6], m_enc_output_arr[7],
                container.m_networks[88], container.m_networks[89], container.m_networks[90], container.m_networks[91],
                container.m_networks[92], container.m_networks[93], container.m_networks[94], container.m_networks[95],
                container.m_networks[96], container.m_networks[97], container.m_networks[98], container.m_networks[99],
                1, &e_encoder[6], &e_encoder[7]
                );

            v_Encoder(
                m_enc_output_arr[7], m_enc_output_arr[8],
                container.m_networks[100], container.m_networks[101], container.m_networks[102], container.m_networks[103],
                container.m_networks[104], container.m_networks[105], container.m_networks[106], container.m_networks[107],
                container.m_networks[108], container.m_networks[109], container.m_networks[110], container.m_networks[111],
                1, &e_encoder[7], &e_encoder[8]
                );

            v_Encoder(
                m_enc_output_arr[8], m_enc_output_arr[9],
                container.m_networks[112], container.m_networks[113], container.m_networks[114], container.m_networks[115],
                container.m_networks[116], container.m_networks[117], container.m_networks[118], container.m_networks[119],
                container.m_networks[120], container.m_networks[121], container.m_networks[122], container.m_networks[123],
                1, &e_encoder[8], &e_encoder[9]
                );

            v_Encoder(
                m_enc_output_arr[9], m_enc_output_arr[10],
                container.m_networks[124], container.m_networks[125], container.m_networks[126], container.m_networks[127],
                container.m_networks[128], container.m_networks[129], container.m_networks[130], container.m_networks[131],
                container.m_networks[132], container.m_networks[133], container.m_networks[134], container.m_networks[135],
                1, &e_encoder[9], &e_encoder[10]
                );

            v_Encoder(
                m_enc_output_arr[10], m_enc_output_arr[11],
                container.m_networks[136], container.m_networks[137], container.m_networks[138], container.m_networks[139],
                container.m_networks[140], container.m_networks[141], container.m_networks[142], container.m_networks[143],
                container.m_networks[144], container.m_networks[145], container.m_networks[146], container.m_networks[147],
                1, &e_encoder[10], &e_encoder[11]
            );
        }

        // layer_norm
        cl_event e_normalized;
        LOG("normalize", v_layer_norm(m_enc_output_arr[11], m_normalized, container.m_networks[148], container.m_networks[149], 1, &e_encoder[11], &e_normalized));

        // load class token
        cl_event e_load_class_token;
        err = clEnqueueCopyBuffer(container.queue, m_normalized, m_class_token, 0, 0, class_token_size, 1, &e_normalized, &e_load_class_token);
        CHECK_CL_ERROR(err);

        // convert class token into output (probabilites)
        cl_event e_class_output;
        LOG(
            "cls_output",
            v_linear_layer(
                m_class_token, container.m_networks[150], container.m_networks[151], m_class_output,
                1, EMBED_DIM, NUM_CLASSES,
                1, &e_load_class_token, &e_class_output
            );
        );

        // sofemax
        cl_event e_softmax;
        LOG("sofemax", v_Softmax(m_class_output, m_probabilities, NUM_CLASSES, 1, &e_class_output, &e_softmax));


        // read result
        err = clEnqueueReadBuffer(container.queue, m_probabilities, CL_FALSE, 0, probabilities_size, probabilities[i], 1, &e_softmax, &e_process_end[i]);
        CHECK_CL_ERROR(err);
    }

    // clWaitForEvents(IMAGE_COUNT, e_process_end);
    clFinish(container.queue);



    // release mem objs
    err = clReleaseMemObject(m_img);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_patch_embedded);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_flatten_transposed);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_class_token_prepended);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_position_embeded);
    CHECK_CL_ERROR(err);
    for (int i = 0; i < 12; ++i) {
        err = clReleaseMemObject(m_enc_output_arr[i]);
        CHECK_CL_ERROR(err);
    }
    err = clReleaseMemObject(m_normalized);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_class_token);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_class_output);
    CHECK_CL_ERROR(err);
    CHECK_CL_ERROR(err);
    err = clReleaseMemObject(m_probabilities);
    CHECK_CL_ERROR(err);


    cleanup();

    printf(">> [ViT_cl] : ended\n");
}
