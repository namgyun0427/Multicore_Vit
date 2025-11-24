#include "posix_port.h"

typedef int errno_t;

errno_t fopen_s(FILE** pFile, const char* filename, const char* mode) {
    // 1. Parameter Validation:
    //    Check if pFile, filename, or mode are NULL.
    //    If any are NULL, set errno to EINVAL and return EINVAL.
    if (pFile == NULL || filename == NULL || mode == NULL) {
        if (pFile != NULL) { // If pFile is not NULL, set *pFile to NULL
            *pFile = NULL;
        }
        errno = EINVAL;
        return EINVAL;
    }

    // 2. Attempt to open the file:
    //    This part would involve low-level operating system calls
    //    to open the specified file with the given mode.
    FILE* temp_file_ptr = fopen(filename, mode);

    // 3. Check for opening success/failure:
    if (temp_file_ptr == NULL) {
        // If file opening failed, set *pFile to NULL and
        // set errno based on the reason for failure (e.g., ENOENT, EACCES).
        *pFile = NULL;
        // Example: errno = ENOENT; (File not found)
        return errno; // Return the error code
    } else {
        // If successful, assign the file pointer to *pFile and return 0.
        *pFile = temp_file_ptr;
        return 0; // Success
    }
}

errno_t strncpy_s(char *dst, size_t dstsz, const char *src, size_t count) {
    if (dst == NULL) {
        return EINVAL;
    }
    if (dstsz == 0) {
        return EINVAL;
    }

    dst[0] = '\0';

    if (src == NULL) {
        return EINVAL;
    }

    /* 실제 복사 크기 계산 */
    size_t to_copy = count;
    if (to_copy >= dstsz) {
        dst[0] = '\0';
        return ERANGE;
    }

    /* 복사 수행 */
    memcpy(dst, src, to_copy);
    dst[to_copy] = '\0';

    return 0;
}