#pragma once
#define GLM_SWIZZLE
#include "util.h"
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"


//void pathTracerKernelWrapper( //unsigned char* const outputImage, 
//                             float* const outputImage,
//                             float* const directIllum,
//                             float* const indirectIllum, 
//                             int width, int height, _CameraData cameraData,
//                             const _Primitive* const primitives, int primitiveNum,
//                              const _Light* const lights, int lightNum, _Material* mtl, int mtlNum,
//                              int DOPsampleCount, curandState *state, curandStateSobol32_t* sobolState,unsigned short iteration );

void pathTracerKernelWrapper( _Param* param );
void pathTracerEyeRayKernelWrapper( _Param* param );

//void BDPTLightPathWrapper();
void setupRandSeedWrapper( int dimX, int dimY, curandState* states ) ;
void setupSobolRandSeedWrapper( int width, int height, curandStateSobol32_t* states, unsigned int*vector ) ;

void initMarkerWrapper(  int width, int height, int* d_marker );