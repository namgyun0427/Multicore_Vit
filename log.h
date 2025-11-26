#pragma once
#ifndef _LOG_H
#define _LOG_H


#define LOG(msg, code) \
    do {    \
        clock_t start = clock(); \
        code;   \
        clock_t end = clock();  \
        printf("%s, %s:%d, %.2f\n", msg, __FILE__, __LINE__, (double)(end - start) * 1000 / CLOCKS_PER_SEC); \
    } while (0);


#endif // _LOG_H
