#pragma once

#ifndef COMPARATOR_H
#define COMPARATOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int parse_line(const char* line, int* label, float* prob);
static void trim_newline(char* str);

int comparator(void);

#endif // COMPARATOR_H
