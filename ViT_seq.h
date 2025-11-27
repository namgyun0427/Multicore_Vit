#pragma once

#ifndef _ViT_seq_H
#define _ViT_seq_H

#ifdef _WIN32
    #pragma warning(disable : 4996)
#endif


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "log.h"
#include "constants.h"
#include "Network.h"

void ViT_seq(ImageData* image, Network* networks, float** prb);

#endif // _ViT_seq_H
