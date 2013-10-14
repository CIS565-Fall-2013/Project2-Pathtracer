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
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int index){
  thrust::default_random_engine rng1(hash(time));
  thrust::default_random_engine rng2(hash(index * time));
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
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index <= resolution.x * resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, float iterations){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x <= resolution.x && y <= resolution.y){

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
__global__ void pathtraceRay(ray* rays, glm::vec2 resolution, float time, int traceDepth, int maxDepth, cameraData cam, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* mats, int* lightIds, int numberOfLights, mesh* meshes, int numberOfMeshes, face* faces, int numberOfFaces){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	return;
	
	if(index <= resolution.x * resolution.y){
		ray r = rays[index];

		if(!r.isContinue) return;
	
		if(traceDepth == 0){
			float focal_length = 12.0f;
			glm::vec3 focal_point = cam.position + focal_length * cam.view;
		
			float t = 1.0f / glm::dot(r.direction, cam.view) * glm::dot(focal_point - r.origin, cam.view);
			glm::vec3 aimed = r.origin + t * r.direction;
			r.origin = r.origin + 1.0f * generateRandomNumberFromThread(resolution, time, index);
			r.direction = glm::normalize(aimed - r.origin);
		}
	
		float intersect;
		int geomId;

		glm::vec3 intersectionPoint, normal;
		glm::vec3 light_pos, ray_incident, reflectedRay;

		// Intersection Checking
		intersect = isIntersect(r, intersectionPoint, normal, geoms, numberOfGeoms, meshes, faces, geomId);

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
		if(traceDepth == maxDepth - 1) r.isContinue = false; // force accumulator to pickup color on 
		rays[index] = r;
	}
}

void __global__ setUpRays(glm::vec2 resolution, float time, cameraData cam, ray* rays){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x <= resolution.x && y <= resolution.y){
		thrust::default_random_engine rng(hash(index*time));
		thrust::uniform_real_distribution<float> u01(-1.0, 1.0);

		ray r = raycastFromCameraKernel(resolution, time, x + (float)u01(rng), y + (float)u01(rng), cam.position, cam.view, cam.up, cam.fov);
		r.isContinue = true;
		r.px = index;
		r.color = glm::vec3(1.0);

		rays[index] = r;
	}
}

void __global__ accumulateColor(int numRays, glm::vec3* image, ray* rays){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(index < numRays){
		ray r = rays[index];
		if(!r.isContinue){
			int img_index = r.px;
			image[img_index] += r.color;
		}
	}
}

void __global__ createPredicate(ray* rays, int* pred, int numLiveRays){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index < numLiveRays){
		pred[index] = rays[index].isContinue ? 1 : 0;
	}
}

void __global__ scanByBlock(int* pred, int* interSums, int n, int numLiveRays){
	int index = threadIdx.x;
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ int temp[];

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

void __global__ applySums(int* partialSums, int* scanArr, int n, int numLiveRays, int* remainRays){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index <= numLiveRays){
		scanArr[index] += partialSums[index / n];
	}
	if(index == numLiveRays - 1) remainRays[0] = scanArr[index];
}

void __global__ scatter(ray* prev, int* scanArr, ray* buffer, int numRays){
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if(k < numRays){
		if(prev[k].isContinue){
			int idx = scanArr[k];
			buffer[idx] = prev[k];
		}
	}
}

void scan(int* pred, int numLiveRays, int* remainRays, int threadsPerBlock, int* interSums_a, int* interSums_b){
	int numBlocks = (int)ceil((double)numLiveRays / (double)threadsPerBlock);
	
	// Scan predicate by block, allocate shared memory dynamically
	scanByBlock<<<numBlocks, threadsPerBlock,  2 * threadsPerBlock * sizeof(int)>>>(pred, interSums_a, threadsPerBlock, numLiveRays);

	int numInterSumBlocks = numBlocks;
	int numInterSums = numBlocks;
	int offset = threadsPerBlock;
	int* interSums; int* interSumsBuffer;
	int idx = 0;

	do{
		interSums = idx % 2 == 0 ? interSums_a : interSums_b;
		interSumsBuffer = idx % 2 == 0 ? interSums_b : interSums_a;

		numInterSumBlocks = (int)ceil((double)numInterSumBlocks / threadsPerBlock);

		// Perform prefix sum on intermittent partial sums of arrays
		scanByBlock<<<numInterSumBlocks, threadsPerBlock,  2 * threadsPerBlock * sizeof(int)>>>(interSums, interSumsBuffer, threadsPerBlock, numInterSums);
		applySums<<<numBlocks, threadsPerBlock>>>(interSums, pred, offset, numLiveRays, remainRays);
		offset = offset * threadsPerBlock;

		// Set variables to sum on a smaller partial sum array if needed
		numInterSums = numInterSumBlocks;
		idx++;
	}while(numInterSumBlocks > 1);
}

void streamCompact(ray* src, ray* dest, int& numLiveRays, int& fullBlocksPerGrid, int threadsPerBlock, glm::vec2 resolution, int* remainRays, int* pred, int* interSums_a, int* interSums_b){
	 // Create Predicate Array
	 createPredicate<<<fullBlocksPerGrid, threadsPerBlock>>>(src, pred, numLiveRays);
	 
	 // Scan
	 scan(pred, numLiveRays, remainRays, threadsPerBlock, interSums_a, interSums_b);

	 // Scatter
	 scatter<<<fullBlocksPerGrid, threadsPerBlock>>>(src, pred, dest, numLiveRays);

	 // Recalculate kernel dimensions
	 fullBlocksPerGrid = (int) ceil((float)numLiveRays/threadsPerBlock);
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, 
	int* lightIds, int numberOfLights, mesh* meshes, int numberOfMeshes, face* faces, int numberOfFaces){
  
  int maxTraceDepth = 16;
  int traceDepth = 0; 

  // set up crucial magic
  int threadsPerBlock = 128;
  int fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x) * (renderCam->resolution.y)/threadsPerBlock));
  
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
	newStaticGeom.meshId = geoms[i].meshId;
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

  mesh* cudaMeshes = NULL;
  cudaMalloc((void**)&cudaMeshes, numberOfMeshes * sizeof(mesh));
  cudaMemcpy(cudaMeshes, meshes, numberOfMeshes * sizeof(mesh), cudaMemcpyHostToDevice);

  face* cudaFaces = NULL;
  cudaMalloc((void**)&cudaFaces, numberOfFaces * sizeof(face));
  cudaMemcpy(cudaFaces, faces, numberOfFaces * sizeof(face), cudaMemcpyHostToDevice);

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

  // array for updating number of rays
  int* remainRays = new int[1];
  int* cuda_remainRays = NULL;
  cudaMalloc((void**)&cuda_remainRays, sizeof(int));

  // predicat, partial sum array and buffers
  int* cuda_pred = NULL;
  int* cuda_interSum_a = NULL;
  int* cuda_interSum_b = NULL;
  cudaMalloc((void**)&cuda_pred, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(int));
  cudaMalloc((void**)&cuda_interSum_a, (int)ceil(renderCam->resolution.x * renderCam->resolution.y / threadsPerBlock) * sizeof(int));
  cudaMalloc((void**)&cuda_interSum_b, (int)ceil(renderCam->resolution.x * renderCam->resolution.y / threadsPerBlock) * sizeof(int));

  int tileSize = 8;
  dim3 setupThreads(tileSize, tileSize);
  dim3 setupBlocks((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

  std::cout << "setup" << std::endl;
  // set up rays
  setUpRays<<<setupBlocks, setupThreads>>>(renderCam->resolution, (float)iterations, cam, cudarays_a);

  while(traceDepth < maxTraceDepth && numRays > 1){
	  cudarays = traceDepth % 2 == 0 ? cudarays_a : cudarays_b;
	  buffer = traceDepth % 2 == 0? cudarays_b : cudarays_a;
	  
	  pathtraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(cudarays, resolution, (float)iterations, traceDepth, maxTraceDepth, cam, cudaimage, cudageoms, numberOfGeoms, cudamats, cudalightids, numberOfLights, cudaMeshes, numberOfMeshes, cudaFaces, numberOfFaces);
	  
	  accumulateColor<<<fullBlocksPerGrid, threadsPerBlock>>>(numRays, cudaimage, cudarays);
	  
	  if(traceDepth < maxTraceDepth - 1) 
		  streamCompact(cudarays, buffer, numRays, fullBlocksPerGrid, threadsPerBlock, resolution, cuda_remainRays, cuda_pred, cuda_interSum_a, cuda_interSum_b);

	  cudaMemcpy(remainRays, cuda_remainRays, sizeof(int), cudaMemcpyDeviceToHost);
	  numRays = remainRays[0] + 1;


	  traceDepth++;
  }
  sendImageToPBO<<<setupBlocks, setupThreads>>>(PBOpos, renderCam->resolution, cudaimage, (float)iterations);


  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamats );
  cudaFree( cudalightids );
  cudaFree( cudarays_a );
  cudaFree( cudarays_b );
  cudaFree( cuda_remainRays );
  cudaFree( cuda_pred );
  cudaFree( cuda_interSum_a );
  cudaFree( cuda_interSum_b );
  cudaFree( cudaMeshes );
  cudaFree( cudaFaces );
  delete geomList;
  delete remainRays;

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

