// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>

#define POSITION(g) multiplyMV(g.transform, glm::vec4(0,0,0,1))
#define SPECULAR(materials, g) materials[g.materialid].specularExponent
#define REFLECTIVE(materials, g) materials[g.materialid].hasReflective
#define IS_LIGHT(materials, geoms, idx) !epsilonCheck(materials[geoms[idx].materialid].emittance, 0.0f)
#define COLOR(materials, g) materials[g.materialid].color
#define SPEC_COLOR(materials, g) materials[g.materialid].specularColor
#define EMITTANCE(materials, g) materials[g.materialid].emittance

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

glm::vec3* cudaimage;
cameraData cam;

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng1(hash(index));
  thrust::default_random_engine rng2(hash(index));
  thrust::uniform_real_distribution<float> u01(-1,1);

  return glm::vec3((float) u01(rng1), (float) u01(rng1),(float) u01(rng1));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, float x, float y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
    glm::vec3 A = glm::cross(view, up);
	glm::vec3 B = glm::cross(A, view);
	glm::vec3 M = eye + view;

	float phi = glm::radians(fov.y);
	float theta = glm::radians(fov.x);
	float C = glm::length(view);

	glm::vec3 V = glm::normalize(B) * (C * tan(phi));
	glm::vec3 H = glm::normalize(A) * (C * tan(theta));

	float sx = (float) x / (resolution.x - 1.0f);
	float sy = (float) y / (resolution.y - 1.0f);

	glm::vec3 P = M + (2 * sx - 1.0f)*H	 + (1.0f - 2 * sy)*V;

	ray r;
	r.origin = eye;
	r.direction = glm::normalize(P - eye);
	return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, float iterations){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x / iterations *255.0;
      color.y = image[index].y / iterations *255.0;
      color.z = image[index].z / iterations *255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void pathtraceRay(ray* rays, glm::vec2 resolution, float time, int traceDepth, cameraData cam, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* mats, int* lightIds, int numberOfLights/*, mesh* meshes, int numberOfMeshes*/){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if((x<=resolution.x && y<=resolution.y)){
		ray r = rays[index];

		if(!r.isContinue) return;
	
		/*if(traceDepth == 0){
			float focal_length = 10.0f;
			glm::vec3 focal_point = cam.position + focal_length * cam.view;
		
			float t = 1.0f / glm::dot(r.direction, cam.view) * glm::dot(focal_point - r.origin, cam.view);
			glm::vec3 aimed = r.origin + t * r.direction;
			r.origin = r.origin + 1.0f * generateRandomNumberFromThread(resolution, time, x, y);
			r.direction = glm::normalize(aimed - r.origin);
		}*/
	
		float intersect;
		int geomId;

		glm::vec3 intersectionPoint, normal;
		glm::vec3 light_pos, ray_incident, reflectedRay;

		// Intersection Checking
		intersect = isIntersect(r, intersectionPoint, normal, geoms, numberOfGeoms, geomId);
		
		if (epsilonCheck(intersect, -1.0f)){
			r.color = glm::vec3(0.0f);
			r.isContinue = false;
		}else{
			if(IS_LIGHT(mats, geoms, geomId)){
				r.color = r.color * COLOR(mats, geoms[geomId]) * EMITTANCE(mats, geoms[geomId]);
				r.isContinue = false;
			}else if(REFLECTIVE(mats, geoms[geomId]) > .001f){
				r.color *= COLOR(mats, geoms[geomId]);
				if(epsilonCheck(glm::length(glm::cross(r.direction, normal)), 0.0f)) reflectedRay = -1.0f * normal;
				else if(epsilonCheck(glm::dot(-1.0f * r.direction, normal), 0.0f)) reflectedRay = r.direction;
				else reflectedRay = r.direction - 2.0f * normal * glm::dot(r.direction, normal);
				r.direction = reflectedRay;
				r.origin = intersectionPoint + .001f * r.direction;
			}else{
				// DIFFUSE CASE
				thrust::default_random_engine rng(hash(time * index * (traceDepth + 1)));
				thrust::uniform_real_distribution<float> u01(0,1);
				r.color = r.color * COLOR(mats, geoms[geomId]), 0.0f, 1.0f;
				
				r.direction = calculateRandomDirectionInHemisphere(normal,(float)u01(rng),(float)u01(rng));

				r.origin = intersectionPoint + .001f * r.direction;
			}
		}
		rays[index] = r;
	}
}

void __global__ setUpRays(glm::vec2 resolution, float time, cameraData cam, ray* rays){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x <= resolution.x && y <= resolution.y){
		thrust::default_random_engine rng(hash(index*time));
		thrust::uniform_real_distribution<float> u01(-.5, .5);

		ray r = raycastFromCameraKernel(resolution, time, x + (float)u01(rng), y + (float)u01(rng), cam.position, cam.view, cam.up, cam.fov);
		r.isContinue = true;
		r.px = index;
		r.color = glm::vec3(1.0);

		rays[index] = r;
	}
}

void __global__ accumulateColor(glm::vec2 resolution, glm::vec3* image, ray* rays){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x <= resolution.x && y <= resolution.y){
		ray r = rays[index];
		int img_index = r.px;
		image[index] += r.color;
	}
}

void __global__ setColor(glm::vec2 resolution, ray* rays){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x <= resolution.x && y <= resolution.y){
		rays[index].color = glm::vec3(0,1,0);
	}
}

void __global__ createPredicate(ray* rays, int* pred, glm::vec2 resolution){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x <= resolution.x && y <= resolution.y){
		pred[index] = rays[index].isContinue ? 1 : 0;
	}
}

void __global__ scanByBlock(int* pred, int* outArr, int* interSums, int n, int numLiveRays){
	int index = threadIdx.x;
	int global_index = blockIdx.x * n + threadIdx.x;
	__shared__ int temp[4];

	int pout = 1, pin = 0;
	temp[pout * n + index] = index > 0 && global_index < numLiveRays ? pred[global_index - 1] : 0;
	temp[pin * n + index] = index > 0 && global_index < numLiveRays ? pred[global_index - 1] : 0;
	__syncthreads();
	for(int offset = 1; offset < n; offset *= 2){
		if (index >= offset){
			temp[pout * n + index] += temp[pin * n + index - offset];
		}
		__syncthreads();

		// copy to buffer
		temp[pin * n + index] = temp[pout * n + index];
		__syncthreads();
	}
	if(index == n - 1) interSums[blockIdx.x] = temp[pout * n + index] + pred[global_index];
	pred[global_index] = temp[pout * n + index];
}

void __global__ applySums(int* partialSums, int* scanArr, int n, int numLiveRays){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index <= numLiveRays){
		scanArr[index] += partialSums[index / n];
	}
}

void __global__ scatter(ray* prev, int* scanArr, ray* buffer, int numRays){
	int k = threadIdx.x;
	if(k <= numRays){
		if(prev[k].isContinue){
			buffer[scanArr[k]] = prev[k];
		}
	}
}

void scan(int* pred, int* outArr, int numLiveRays){
	int threadsPerBlock = 2;
	int numBlocks = (int)ceil((double)numLiveRays / (double)threadsPerBlock);
	
	// Alloc space for summing array
	int* interSums = NULL;
	cudaMalloc((void**)&interSums, sizeof(int) * numBlocks);

	int* interSumsBuffer = NULL;
	cudaMalloc((void**)&interSumsBuffer, sizeof(int) * numBlocks);

	// Scan predicate by block
	scanByBlock<<<numBlocks, threadsPerBlock>>>(pred, outArr, interSums, threadsPerBlock, numLiveRays);

	int numInterSumBlocks;
	int numInterSums = numBlocks;
	int offset = threadsPerBlock;

	// Store each of the last index of blocks into summing array
	do{
		// Perform prefix sum on intermittent partial sums of arrays
		numInterSumBlocks = (int)ceil((double)numInterSums / threadsPerBlock);
		scanByBlock<<<numInterSumBlocks, threadsPerBlock>>>(interSums, outArr, interSumsBuffer, threadsPerBlock, numInterSums);
		applySums<<<numBlocks, threadsPerBlock>>>(interSums, pred, offset, numLiveRays);
		offset = offset * 2;

		// Set variables to sum on a smaller partial sum array if needed
		int* temp = interSums;
		interSums = interSumsBuffer;
		interSumsBuffer = interSums;
		numInterSums = (int)ceil((double)numInterSums/2);
	}while(numInterSumBlocks > 1);
	cudaFree(interSums);
	cudaFree(interSumsBuffer);
}

void testScan(){
    // test scanByBlock
	int* test_eight = new int[8];
    test_eight[0] = 1, test_eight[1] = 0, test_eight[2] = 1, test_eight[3] = 1, test_eight[4] = 0, test_eight[5] = 1, test_eight[6] = 0, test_eight[7] = 1;

	int* test_outscan = new int[8];

  int* cuda_test_eight = NULL;
  cudaMalloc((void**)&cuda_test_eight, 8 * sizeof(int));
  cudaMemcpy(cuda_test_eight, test_eight, 8 * sizeof(int), cudaMemcpyHostToDevice);
  
  
  int* cuda_test_outscan = NULL;
  cudaMalloc((void**)&cuda_test_outscan, 8 * sizeof(int));

  scan(cuda_test_eight, cuda_test_outscan, 7);

  cudaMemcpy(test_eight, cuda_test_eight, 8 * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(test_outscan, cuda_test_outscan, 8 * sizeof(int), cudaMemcpyDeviceToHost);

  	std::cout << "INSCAN" << std::endl;
	for(int idx = 0; idx < 8; idx++){
	  std::cout << test_eight[idx] << std::endl;
	}

	std::cout << "OUTSCAN" << std::endl;
	for(int idx = 0; idx < 8; idx++){
		std::cout << test_outscan[idx] << std::endl;
	}

  cudaFree(cuda_test_eight);
  cudaFree(cuda_test_outscan);
}

void testScanByBlock(int* test_eight, int* test_intersum, int* test_outscan){
	  // test scanByBlock
  int* cuda_test_eight = NULL;
  cudaMalloc((void**)&cuda_test_eight, 7 * sizeof(int));
  cudaMemcpy(cuda_test_eight, test_eight, 7 * sizeof(int), cudaMemcpyHostToDevice);
  
  int* cuda_test_intersum = NULL;
  cudaMalloc((void**)&cuda_test_intersum, 7 * sizeof(int));
  cudaMemcpy(cuda_test_intersum, test_intersum, 7 * sizeof(int), cudaMemcpyHostToDevice);
  
  int* cuda_test_outscan = NULL;
  cudaMalloc((void**)&cuda_test_outscan, 8 * sizeof(int));

  dim3 test_blocksPerGrid (1,0,0);
  dim3 test_threadsPerBlock (8,0,0);
  scanByBlock<<<1,8>>>(cuda_test_eight, cuda_test_outscan, cuda_test_intersum, 8, 7);

  cudaMemcpy(test_eight, cuda_test_eight, 7 * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(test_outscan, cuda_test_outscan, 7 * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(test_intersum, cuda_test_intersum, 7 * sizeof(int), cudaMemcpyDeviceToHost);
  
  cudaFree(cuda_test_eight);
  cudaFree(cuda_test_intersum);
  cudaFree(cuda_test_outscan);
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, 
	int* lightIds, int numberOfLights){
  
  int maxTraceDepth = 8;
  int traceDepth = 0; 

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

  material* cudamats = NULL;
  cudaMalloc((void**)&cudamats, numberOfMaterials * sizeof(material));
  cudaMemcpy(cudamats, materials, numberOfMaterials * sizeof(material), cudaMemcpyHostToDevice);

  int* cudalightids = NULL;
  cudaMalloc((void**)&cudalightids, numberOfLights * sizeof(int));
  cudaMemcpy(cudalightids, lightIds, numberOfLights * sizeof(int), cudaMemcpyHostToDevice);

  //package camera
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  // allocate ray array
  ray* cudarays_a = NULL;
  ray* cudarays_b = NULL;
  cudaMalloc((void**)&cudarays_a,(int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(ray));
  cudaMalloc((void**)&cudarays_b,(int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(ray));

  ray* cudarays;
  ray* buffer;
  int numRays = (int)renderCam->resolution.x * (int)renderCam->resolution.y;
  glm::vec2 resolution = renderCam->resolution;
  dim3 blockDimension(32, 1);
  dim3 gridDimension(numRays/32, 1);
  // set up rays
  setUpRays<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, cudarays_a);

  while(traceDepth < maxTraceDepth){
	  cudarays = traceDepth % 2 ? cudarays_a : cudarays_a;
	  buffer = traceDepth % 2 ? cudarays_b : cudarays_a;
	  pathtraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(cudarays, resolution, (float)iterations, traceDepth, cam, cudaimage, cudageoms, numberOfGeoms, cudamats, cudalightids, numberOfLights);

	  //scatter<<<1, numRays>>>(cudarays, buffer, numRays);

	  //// Reset kernel sizes
	  //fullBlocksPerGrid = ((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize));

	  // Free memory
	  //cudaFree( pred );
	  //cudaFree( scanArr );

	  traceDepth++;
  }

  accumulateColor<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaimage, cudarays);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, (float)iterations);
  
  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamats );
  cudaFree( cudalightids );
  cudaFree( cudarays_a );
  cudaFree( cudarays_b );
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();
  checkCUDAError("Kernel failed!");
}

void clearImageBuffer(camera* renderCam){
	int tileSize = 8;
    dim3 threadsPerBlock(tileSize, tileSize);
    dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

	// Copy image from CPU to GPU
	cudaimage = NULL;
	cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

	// Clear Image
	clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaimage);

	// Retrieve image
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	// Sync Threads
	cudaThreadSynchronize();
}

void testScanBlock(int* inArr, int* outArr, int numElem){
	  int threads = 8;
	  int* pred = NULL;

	  cudaMalloc((void**)&pred, numElem * sizeof(int));
	  cudaMemcpy(inArr, pred, numElem * sizeof(int), cudaMemcpyHostToDevice);

	  int* scanArr = NULL;
	  cudaMalloc((void**)&scanArr, numElem * sizeof(int));

	  int* testInterSum = NULL;
	  cudaMalloc((void**)&testInterSum, numElem / threads * sizeof(int));
	  
	  dim3 t1(threads,1);
	  dim3 t2(numElem / threads,1);
	  
	  scanByBlock<<<t2, t1>>>(pred, scanArr, testInterSum, threads, numElem);

	  int* testOutArray = new int[8];
	  cudaMemcpy(scanArr, testOutArray, sizeof(int) * numElem, cudaMemcpyDeviceToHost);

	  cudaThreadSynchronize();
	  checkCUDAError("scan failed");
}