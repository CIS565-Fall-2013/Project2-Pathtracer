// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef RAYTRACEKERNEL_H
#define PATHTRACEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"


#define LIGHT_NUM 1
#define ANTI_NUM 1
//#define STREAMCOMPACTION 1
#define TITLESIZESC 64//this title size is for stream compaction
#define TITLESIZE 8
//#define MOTIONBLUR 1
//#define DOF 16
#define MAXDEPTH 7
//#define TEXTUREMAP 6 //Right now only support sphere, the number mean which object you want to texture it

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void cudaRaytraceCore(uchar4* pos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, glm::vec3* preColors, int *BMPRed, int *BMPGreen, int *BMPBlue, int BMPSize[2]);

#endif
