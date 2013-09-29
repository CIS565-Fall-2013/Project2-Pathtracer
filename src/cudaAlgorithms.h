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



//For easy avoidance of bank conflicts
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5 

#define NO_BANK_CONFLICTS


#ifdef NO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n)    \
	(((n) >> NUM_BANKS) + ((n) >> (2 * LOG_NUM_BANKS)))  
#else
	#define CONFLICT_FREE_OFFSET(n)    ((n) >> NUM_BANKS)  
#endif
#define MAX_BLOCK_DIM_X 1024
#define MAX_GRID_DIM_X 65535


struct Add : std::binary_function<float,float,float> {
__host__ __device__   float operator() (float a, float b) {return (a+b);}
};


struct Multiply : std::binary_function<float,float,float> {
__host__ __device__ float operator() (float a, float b) {return (a*b);}
};


template<typename DataType>
__host__ DataType exclusive_scan_sum(DataType* datain, DataType* dataout, int N);
template<typename DataType, typename BinaryOperation>
__device__ DataType exclusive_scan_block(DataType* datain, DataType* dataout, int N, BinaryOperation op);
template<typename DataType, typename BinaryOperation>
__device__ DataType inclusive_scan_block(DataType* datain, DataType* dataout, int N, BinaryOperation op);



template<typename DataType>
__host__ DataType inclusive_scan_sum(DataType* datain, DataType* dataout, int N);

#endif