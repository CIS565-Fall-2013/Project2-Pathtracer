#pragma once
#define GLM_SWIZZLE
#include "util.h"
#include "glm/glm.hpp"

void rayTracerKernelWrapper( unsigned char* const outputImage, int width, int height, _CameraData cameraData,
                             const _Primitive* const primitives, int primitiveNum,
                              const _Light* const lights, int lightNum, _Material* mtl, int mtlNum, int DOPsampleCount, curandState *state );


void setupRandSeedWrapper( int dimX, int dimY, curandState* states ) ;