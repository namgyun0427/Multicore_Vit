#include "Network.h"

////////////////////////////////////////////////////////////////////////////////
// functions

ImageData* load_image_data(const char* filename) {
    FILE* f = NULL;
    errno_t err = fopen_s(&f, filename, "rb");
    if (err != 0) {
        // do nithin'
    }
    if (f == NULL) {
        perror("���� ���� ����");
        return NULL;
    }


    int header[4];
    if (fread(header, sizeof(int), 4, f) != 4) {
        perror("��� �б� ����");
        fclose(f);
        return NULL;
    }

    int n = header[0];
    int c = header[1];
    int h = header[2];
    int w = header[3];
    int image_size = c * h * w;
    //printf("%d * %d * %d = %d!!\n", c, h, w, image_size);
    size_t total_elements = n * image_size;


    float* all_data = (float*)malloc(total_elements * sizeof(float));
    if (all_data == NULL) {
        perror("��ü ������ ���� �޸� �Ҵ� ����");
        fclose(f);
        return NULL;
    }
    if (fread(all_data, sizeof(float), total_elements, f) != total_elements) {
        perror("�̹��� ������ �б� ����");
        free(all_data);
        fclose(f);
        return NULL;
    }
    fclose(f);

    ImageData* images = (ImageData*)malloc(n * sizeof(ImageData));
    if (images == NULL) {
        perror("ImageData �迭 �޸� �Ҵ� ����");
        free(all_data);
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        images[i].n = n;
        images[i].c = c;
        images[i].h = h;
        images[i].w = w;

        images[i].data = (float*)malloc(image_size * sizeof(float));
        if (images[i].data == NULL) {
            perror("���� �̹��� ������ �޸� �Ҵ� ����");

            for (int j = 0; j < i; j++) {
                free(images[j].data);
            }
            free(images);
            free(all_data);
            return NULL;
        }
        

        memcpy(images[i].data, all_data + i * image_size, image_size * sizeof(float));
    }

    free(all_data);
    return images;
}


// extract index from file name
static int parse_index_from_filename(const char* filename) {
    // if file doesn't start with "Weight_", fail
    if (strncmp(filename, "Weight_", 7) != 0) {
        return -1;
    }

    const char* start = filename + 7;
    const char* end = strchr(start, '_');
    if (!end) {
        return -1;
    }
    size_t len = end - start;

    char index_str[16] = { 0 };
    if (len >= sizeof(index_str))
        len = sizeof(index_str) - 1;

    strncpy_s(index_str, sizeof(index_str), start, len);
    index_str[len] = '\0';

    return atoi(index_str);
}

void load_weights(const char* directory, Network network[], int count) {
    DIR* dir = opendir(directory);
    if (!dir) {
        perror("���丮 ���� ����");
        exit(EXIT_FAILURE);
    }

    // init network values
    for (int i = 0; i < count; i++) {
        network[i].data = NULL;
        network[i].size = 0;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        // if file doesn't start with "Weight_", skip it
        if (strncmp(entry->d_name, "Weight_", 7) != 0)
            continue;

        // if extension != ".bin", skip it
        const char* ext = strrchr(entry->d_name, '.');
        if (!ext || strcmp(ext, ".bin") != 0)
            continue;

        // extract index fro filename
        int idx = parse_index_from_filename(entry->d_name);
        if (idx < 0 || idx >= count)
            continue;

        // get filepath
        char filepath[512];
        snprintf(filepath, sizeof(filepath), "%s/%s", directory, entry->d_name);

        // open file
        FILE* fp = NULL;
        errno_t err = fopen_s(&fp, filepath, "rb");
        if (err != 0) {
            // ���� ó��: ���� ���⿡ ������ ���
        }

        // get file_size
        fseek(fp, 0, SEEK_END);
        long file_size = ftell(fp);
        rewind(fp);
        if (file_size < 0) {
            fclose(fp);
            continue;
        }

        // cal num of data
        size_t num_floats = file_size / sizeof(float);

        // allocate buffer
        float* buffer = (float*)malloc(file_size);
        if (!buffer) {
            perror("�޸� �Ҵ� ����");
            fclose(fp);
            exit(EXIT_FAILURE);
        }

        // load data into buffer
        size_t read_size = fread(buffer, sizeof(float), num_floats, fp);
        if (read_size != num_floats) {
            perror("���� �б� ����");
            free(buffer);
            fclose(fp);
            continue;
        }
        fclose(fp);

        // normalize to reduce floating-point acuumulation error
        for (size_t i = 0; i < num_floats; i++) {
            buffer[i] = roundf(buffer[i] * 1000000.0f) / 1000000.0f;
        }

        // set network values
        network[idx].data = buffer;
        network[idx].size = num_floats;
    }

    closedir(dir);
}
