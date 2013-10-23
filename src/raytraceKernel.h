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

void cudaRaytraceCore(uchar4* pos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials,
	geom* geoms, int numberOfGeoms, int numberOfCubes, int numberOfSpheres, bool cameraMoved);

#endif

  /*///////////////////////////////////////////////////////// SCAN TEST

  int numchars = 8;

  char* test = new char[numchars];
  test[0] = 'a';
  test[1] = 'b';
  test[2] = 'c';
  test[3] = 'd';
  test[4] = 'e';
  test[5] = 'f';
  test[6] = 'g';
  test[7] = 'h';

  char* cudatest = NULL;
  cudaMalloc((void**)&cudatest, numchars*sizeof(char));
  cudaMemcpy( cudatest, test, numchars*sizeof(char), cudaMemcpyHostToDevice);

  int* condition = new int[numchars];
  condition[0] = 1;
  condition[1] = 0;
  condition[2] = 1;
  condition[3] = 1;
  condition[4] = 0;
  condition[5] = 0;
  condition[6] = 1;
  condition[7] = 0;

  int* cudacondition = NULL;
  cudaMalloc((void**)&cudacondition, numchars*sizeof(int));
  cudaMemcpy( cudacondition, condition, numchars*sizeof(int), cudaMemcpyHostToDevice);

  int* cudatemp = NULL;
  cudaMalloc((void**)&cudatemp, numchars*sizeof(int));
  
  int log2n = (int)ceil(log(float(numchars)) / log(2.0f));
  for (int d = 1; d <= log2n; d++){
	 streamCompact<<<2, 2>>>( cudatest, cudacondition, cudatemp, d);
	 cudaMemcpy(cudacondition, cudatemp, numchars*sizeof(int), cudaMemcpyDeviceToDevice); //memcpy
  }
  
	 int* indexes = new int[numchars];
	 cudaMemcpy( indexes, cudacondition, numchars*sizeof(int), cudaMemcpyDeviceToHost);
	  for (int i = 0; i < numchars; i++){
		  if(condition[i]){
			  test[indexes[i]-1] = test[i];
		  }
	  }

	  for (int i = 0; i < 4; i ++){
		  std::cout<<test[i]<<", ";
	  }
	  std::cout<<std::endl;
	  getchar();


  /////////////////////////////////////////////////////////// SCAN TEST*/