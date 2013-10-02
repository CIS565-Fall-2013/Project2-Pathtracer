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

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void cudaRaytraceCore(uchar4* pos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, mytexture* textures, int numberOfTextures);
void setupProjection (projectionInfo &ProjectionParams, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov);
void onDeviceErrorExit (cudaError_t errorCode, glm::vec3 *cudaimage, staticGeom *cudageom, material * materialColours, int numberOfMaterials);

__host__ __device__ glm::vec3 reflectRay (glm::vec3 incidentRay, glm::vec3 normal);  
__device__ bool isShadowRayBlocked (ray r, glm::vec3 lightPos, staticGeom *geomsList, sceneInfo objectCountInfo);
__host__ __device__ bool isApproximate (float valToBeCompared, float valToBeCheckedAgainst) ;
//{ if ((valToBeCompared >= valToBeCheckedAgainst-0.001) && (valToBeCompared <= valToBeCheckedAgainst+0.001)) return true;	return false; }
__device__ unsigned long getIndex (int x, int y, int MaxWidth);
//{	return (unsigned long) y*MaxWidth + x ;	}
__device__ glm::vec3 getColour (material Material, glm::vec2 UVcoords);

#endif
