#ifndef CUDA_ALGORITHMS_H
#define CUDA_ALGORITHMS_H


#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"


#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif


template<typename DataPtr, typename BinaryOperation>
__device__ DataPtr inclusive_scan_block(DataPtr datain, DataPtr dataout, int N, BinaryOperation op);

#endif