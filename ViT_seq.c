#include "ViT_seq.h"


#define UNUSED(var) \
    (void)(var);

////////////////////////////////////////////////////////////////////////////////////
// constants

static const int size[] = {
    EMBED_DIM * (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE), // conv2D
    EMBED_DIM * (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE), // flatten and transpose
    EMBED_DIM * ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE) + 1), // class token
    EMBED_DIM * ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE) + 1) // position embedding
};

static const int enc_size = EMBED_DIM * ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE) + 1);


////////////////////////////////////////////////////////////////////////////////////
// static funtion declaration


static void v_Conv2d (
    float* input, float* output, 
    Network weight, Network bias
);

static void v_flatten_transpose (float* input, float* output);

static void class_token (
    float* patch_tokens, float* final_tokens, 
    Network cls_tk
);

static void v_pos_emb (
    float* input, float* output, 
    Network pos_emb
);

static void Encoder(
    float* input, float* output,
    Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
    Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b
);

static void multihead_attn(
    float* input, float* output,
    Network in_weight, Network in_bias, 
    Network out_weight, Network out_bias
);

static void mlp_block (
    float* input, float* output, 
    Network fc1_weight, Network fc1_bias, 
    Network fc2_weight, Network fc2_bias
);

static float v_gelu(float x);

static void Softmax(float* logits, float* probabilities, int length);

static void layer_norm (
    float* input, float* output, 
    Network weight, Network bias
);

static void linear_layer (
    float* input, float* output, 
    int tokens, int in_features, int out_features, 
    Network weight, Network bias
);



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
    150, 151: softmax

*/

void ViT_seq (
    ImageData* image, 
    Network* networks, 
    float** probabilities
) {
    const int token_size = ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE) + 1);

    UNUSED(token_size);
    
    float* layer[4];
    for (int i = 0; i < 4; i++) {
        layer[i] = (float*)malloc(sizeof(float) * size[i]);
    }
    
    // encoding layer
    float* enc_layer[12];
    for (int i = 0; i < 12; i++) {
        enc_layer[i] = (float*)malloc(sizeof(float) * enc_size);
    }
    
    // encoding output
    float* enc_output;
    enc_output = (float*)malloc(sizeof(float) * enc_size);




    for (int i = 0; i < image->n; i++) {
        /*patch embedding*/
        float* patch_embedded = layer[0];
        v_Conv2d(image[i].data, patch_embedded, networks[1], networks[2]);

        /*flatten and transpose*/
        float* flatten_transposed = layer[1];
        v_flatten_transpose(patch_embedded, flatten_transposed);

        /*prepend class token*/
        float* clas_token_prepended = layer[2];
        class_token(flatten_transposed, clas_token_prepended, networks[0]);

        /*position embedding*/
        float* position_embeded = layer[3];
        v_pos_emb(clas_token_prepended, position_embeded, networks[3]);

        /*Encoder - 12 Layers*/
        {
            Encoder(position_embeded, enc_layer[0],
                networks[4], networks[5], networks[6], networks[7],
                networks[8], networks[9], networks[10], networks[11],
                networks[12], networks[13], networks[14], networks[15]);

            Encoder(enc_layer[0], enc_layer[1],
                networks[16], networks[17], networks[18], networks[19],
                networks[20], networks[21], networks[22], networks[23],
                networks[24], networks[25], networks[26], networks[27]);

            Encoder(enc_layer[1], enc_layer[2],
                networks[28], networks[29], networks[30], networks[31],
                networks[32], networks[33], networks[34], networks[35],
                networks[36], networks[37], networks[38], networks[39]);

            Encoder(enc_layer[2], enc_layer[3],
                networks[40], networks[41], networks[42], networks[43],
                networks[44], networks[45], networks[46], networks[47],
                networks[48], networks[49], networks[50], networks[51]);

            Encoder(enc_layer[3], enc_layer[4],
                networks[52], networks[53], networks[54], networks[55],
                networks[56], networks[57], networks[58], networks[59],
                networks[60], networks[61], networks[62], networks[63]);

            Encoder(enc_layer[4], enc_layer[5],
                networks[64], networks[65], networks[66], networks[67],
                networks[68], networks[69], networks[70], networks[71],
                networks[72], networks[73], networks[74], networks[75]);

            Encoder(enc_layer[5], enc_layer[6],
                networks[76], networks[77], networks[78], networks[79],
                networks[80], networks[81], networks[82], networks[83],
                networks[84], networks[85], networks[86], networks[87]);

            Encoder(enc_layer[6], enc_layer[7],
                networks[88], networks[89], networks[90], networks[91],
                networks[92], networks[93], networks[94], networks[95],
                networks[96], networks[97], networks[98], networks[99]);

            Encoder(enc_layer[7], enc_layer[8],
                networks[100], networks[101], networks[102], networks[103],
                networks[104], networks[105], networks[106], networks[107],
                networks[108], networks[109], networks[110], networks[111]);

            Encoder(enc_layer[8], enc_layer[9],
                networks[112], networks[113], networks[114], networks[115],
                networks[116], networks[117], networks[118], networks[119],
                networks[120], networks[121], networks[122], networks[123]);

            Encoder(enc_layer[9], enc_layer[10],
                networks[124], networks[125], networks[126], networks[127],
                networks[128], networks[129], networks[130], networks[131],
                networks[132], networks[133], networks[134], networks[135]);

            Encoder(enc_layer[10], enc_layer[11],
                networks[136], networks[137], networks[138], networks[139],
                networks[140], networks[141], networks[142], networks[143],
                networks[144], networks[145], networks[146], networks[147]);
        }

        // normalize
        layer_norm(enc_layer[11], enc_output, networks[148], networks[149]);

        // load class token
        float* cls_token = (float*)malloc(sizeof(float) * EMBED_DIM);
        memcpy(cls_token, enc_output, sizeof(float) * EMBED_DIM);
        
        // convert class token into output
        float* cls_output = (float*)malloc(sizeof(float) * NUM_CLASSES);
        linear_layer(
            cls_token, cls_output, 
            1, EMBED_DIM, NUM_CLASSES, 
            networks[150], networks[151]
        );

        // sofemax
        Softmax(cls_output, probabilities[i], NUM_CLASSES);
    }
}



////////////////////////////////////////////////////////////////////////////////////
// sub funcitons

// image patch embedding (convolution)
// input[IN_CAHNS][PATCH_SIZE][PATCH_SIZE] => output[EMBED_DIM][n_patch_per_image][n_patch_per_image]
static void v_Conv2d (
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
static void v_flatten_transpose (float* input, float* output) {
    int output_size = IMG_SIZE / PATCH_SIZE;
    int num_patches = output_size * output_size;

    UNUSED(num_patches);

    for (int oh = 0; oh < output_size; oh++) {
        for (int ow = 0; ow < output_size; ow++) {
            int patch_idx = oh * output_size + ow;
            for (int oc = 0; oc < EMBED_DIM; oc++) {
                int idx_input = (oc * output_size + oh) * output_size + ow;
                int idx_output = patch_idx * EMBED_DIM + oc;

                // output[patch_idx][EMBED_DIM] == output[oh][ow][EMBED_DIM]
                // input[oc][oh][ow]
                output[idx_output] = input[idx_input];
                //printf("%f ",output[idx_output]);
            }
        }
    }
}



/* ------------------------------------------------------------------------------ */
// prepend class tokens in front
// input[total_num_patches][EMBED_DIM] => output[total_num_patches + 1][EMBED_DIM]
static void class_token (
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

    int total_tokens = num_patches + 1; // class token + patch tokens
    for (int i = 0; i < total_tokens * EMBED_DIM; i++) {
        //("%f ", final_tokens[i]);
    }
    //printf("\n");
}



/* ------------------------------------------------------------------------------ */
// add position embedding data
// input[total_num_patches + 1][EMBED_DIM] => output[total_num_patches + 1][EMBED_DIM]
static void v_pos_emb (
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
static void Encoder(
    float* input, float* output,
    Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
    Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b
) {
    int n_tokens = ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE)) + 1;
    size_t buffer_size = sizeof(float) * n_tokens * EMBED_DIM;
    
    // normalize input
    float* input_normalized = (float*)malloc(buffer_size);
    layer_norm(input, input_normalized, ln1_w, ln1_b);
    
    // multi-head self attention
    float* attn_out = (float*)malloc(buffer_size);
    multihead_attn(input_normalized, attn_out, attn_w, attn_b, attn_out_w, attn_out_b);

    /*Residual1*/
    // skip-connection
    float* residual = (float*)malloc(buffer_size);
    for (int i = 0; i < n_tokens * EMBED_DIM; i++) {
        residual[i] = input[i] + attn_out[i];
    }

    // normalize again
    float* residual_normalized = (float*)malloc(buffer_size);
    layer_norm(residual, residual_normalized, ln2_w, ln2_b);

    /* MLP */
    float* mlp_out = (float*)malloc(buffer_size);
    mlp_block(residual_normalized, mlp_out, mlp1_w, mlp1_b, mlp2_w, mlp2_b);

    /*Residual2*/
    // skip connection again
    for (int i = 0; i < n_tokens * EMBED_DIM; i++) {
        output[i] = residual[i] + mlp_out[i];
    }

    // wrap up
    free(input_normalized);
    free(attn_out);
    free(residual);
    free(residual_normalized);
    free(mlp_out);
}


// multi-head self attention
// input[total_num_patches + 1][EMBED_DIM]
static void multihead_attn(
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

    // calculate Q, K, V per token
    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < EMBED_DIM; i++) {
            // set bias as initial value
            float sum_q = in_bias.data[Q_dim + i];
            float sum_k = in_bias.data[K_dim + i];
            float sum_v = in_bias.data[V_dim + i];

            // do calculation
            for (int j = 0; j < EMBED_DIM; j++) {
                // sum_q += input[t][j] * in_weight.data[Q_dim + i][j] 이런 식
                sum_q += input[t * EMBED_DIM + j] * in_weight.data[(Q_dim + i) * EMBED_DIM + j];
                sum_k += input[t * EMBED_DIM + j] * in_weight.data[(K_dim + i) * EMBED_DIM + j];
                sum_v += input[t * EMBED_DIM + j] * in_weight.data[(V_dim + i) * EMBED_DIM + j];
            }

            // store results
            Q[t * EMBED_DIM + i] = sum_q;
            K[t * EMBED_DIM + i] = sum_k;
            V[t * EMBED_DIM + i] = sum_v;
        }
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

        // softmax scores
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
static void mlp_block (
    float* input, float* output, 
    Network fc1_weight, Network fc1_bias, 
    Network fc2_weight, Network fc2_bias
) {
    int tokens = ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE)) + 1; //197
    int Embed_dim = EMBED_DIM; //768
    int hidden_dim = ((int)(EMBED_DIM * MLP_RATIO)); //3072

    UNUSED(Embed_dim);

    float* fc1_out = (float*)malloc(sizeof(float) * tokens * hidden_dim);
    linear_layer(
        input, fc1_out, 
        tokens, EMBED_DIM, hidden_dim, 
        fc1_weight, fc1_bias
    );

    // apply GELU
    for (int i = 0; i < tokens * hidden_dim; i++) {
        fc1_out[i] = v_gelu(fc1_out[i]);
    }

    // fc2: (tokens, in_dim)
    linear_layer(
        fc1_out, output, 
        tokens, hidden_dim, EMBED_DIM, 
        fc2_weight, fc2_bias
    );

    free(fc1_out);
}


// GELU function
static float v_gelu(float x) {
    return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
}




/* ------------------------------------------------------------------------------ */
//
static void Softmax(float* logits, float* probabilities, int length) {
    // ��ġ �������� ���� �ִ밪 ���
    float max_val = logits[0];
    for (int i = 1; i < length; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    // �� ���ҿ� ���� exp(logit - max_val)�� ����ϰ� �ջ�
    float sum_exp = 0.0f;
    for (int i = 0; i < length; i++) {
        probabilities[i] = expf(logits[i] - max_val);
        sum_exp += probabilities[i];
    }

    // Ȯ�������� ����ȭ
    for (int i = 0; i < length; i++) {
        probabilities[i] /= sum_exp;
    }
}



////////////////////////////////////////////////////////////////////////////////////
// rather common functions

// normalize
// input[total_num_patches + 1][EMBED_DIM] => output[total_num_patches + 1][EMBED_DIM]
static void layer_norm (
    float* input, float* output, 
    Network weight, Network bias
) {
    int token = ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE)) + 1;
    
    for (int t = 0; t < token; t++) {
        // cal sum, sum_of_square per token
        float sum = 0.0, sum_of_square = 0.0;
        for (int i = 0; i < EMBED_DIM; i++) {
            float val = input[t * EMBED_DIM + i];
            sum += val;
            sum_of_square += val * val;
        }
        
        // 순차적으로 평균, 분산, 표준편차의 역수 계산
        float mean = sum / EMBED_DIM;
        float var = sum_of_square / EMBED_DIM - mean * mean;
        float inv_std = 1.0f / sqrtf(var + EPSILON);

        // normalize values
        for (int i = 0; i < EMBED_DIM; i++) {
            int idx = t * EMBED_DIM + i;
            output[idx] = ((input[idx] - mean) * inv_std) * weight.data[i] + bias.data[i];
        }
    }
}


// do linear transform with input matrix(network)
static void linear_layer (
    float* input, float* output, 
    int tokens, int in_features, int out_features, 
    Network weight, Network bias
) {
    for (int t = 0; t < tokens; t++) {
        for (int o = 0; o < out_features; o++) {
            float sum = bias.data[o];

            for (int i = 0; i < in_features; i++) {
                sum += input[t * in_features + i] * weight.data[o * in_features + i];
            }

            output[t * out_features + o] = sum;
        }
    }
}
