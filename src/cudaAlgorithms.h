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

template <class Arg1, class Arg2, class Result>
struct binary_function {
	typedef Arg1 first_argument_type;
	typedef Arg2 second_argument_type;
	typedef Result result_type;
};


struct Add : binary_function<float,float,float> {
  float operator() (float a, float b) {return (a+b);}
};


struct Multiply : binary_function<float,float,float> {
  float operator() (float a, float b) {return (a*b);}
};

template<typename DataPtr, typename BinaryOperation>
__device__ DataPtr inclusive_scan_block(DataPtr datain, DataPtr dataout, int N, BinaryOperation op);


#endif