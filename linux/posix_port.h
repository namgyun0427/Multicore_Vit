#pragma once
#ifndef _POSIX_PORT
#define _POSIX_PORT

#include <errno.h>
#include <stdio.h>
#include <memory.h>

typedef int errno_t;

errno_t fopen_s(FILE** pFile, const char* filename, const char* mode);
errno_t strncpy_s(char *dst, size_t dstsz, const char *src, size_t count);

#endif // _POSIX_PORT