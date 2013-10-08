// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <math.h> // log2
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define MAXDEPTH 50 // max raytrace depth

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
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
			color.x = image[index].x*255.0/iterations;
      color.y = image[index].y*255.0/iterations;
      color.z = image[index].z*255.0/iterations;

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

//---------------------------------------------
//--------------Helper functions---------------
//---------------------------------------------

// Returns true if every component of a is greater than the corresponding component of b
__host__ __device__ bool componentCompare(glm::vec3 a, glm::vec3 b) {
	return (a[0] > b[0] && a[1] > b[1] && a[2] > b[2]);
}

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y,glm::vec3 eye,
																								glm::vec3 view, glm::vec3 up, glm::vec2 fov, float focal, float aperture){
  ray r;
  r.origin = eye;
	r.active = true;

	// values for computing ray direction
  float phi = glm::radians(fov.y);
	float theta = glm::radians(fov.x);
	glm::vec3 A = glm::normalize(glm::cross(view, up));
	glm::vec3 B = glm::normalize(glm::cross(A, view));
	glm::vec3 M = eye + view;
	glm::vec3 V = B * glm::length(view) * tan(phi);
	glm::vec3 H = A * glm::length(view) * tan(theta);

	// super sampling for anti-aliasing
	thrust::default_random_engine rng(hash(time));
	thrust::uniform_real_distribution<float> u01(0, 1);
	float fx = x + (float)u01(rng);
	float fy = y + (float)u01(rng);

	glm::vec3 P = M + (2*fx/(resolution.x-1)-1) * H + (2*(1-fy/(resolution.y-1))-1) * V;
	r.direction = glm::normalize(P-eye);

	if (abs(focal+1) > THRESHOLD) {
		// for depth of field
		// get the intersection with the focal plane
		glm::vec3 pointOnFocalPlane = eye + r.direction * focal;

		// jitter sample position
		thrust::uniform_real_distribution<float> u02(-aperture/2, aperture/2);
		r.origin += A * u02(rng) + B * u02(rng);
		r.direction = glm::normalize(pointOnFocalPlane - r.origin);
	}

  return r;
}

// Kernel that populates the pool of rays with rays from camera to pixels
__global__ void generateRay(cameraData cam, float time, ray* raypool) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * cam.resolution.x);

	if((x<=cam.resolution.x && y<=cam.resolution.y)){
		raypool[index] = raycastFromCameraKernel(cam.resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov, cam.focal, cam.aperture);
		raypool[index].index = index;
		raypool[index].baseColor = glm::vec3(1, 1, 1);
	}
}

__global__ void pathtraceRay(cameraData cam, float time, int rayDepth, ray* rays, int numberOfRays, glm::vec3* colors,
														 staticGeom* geoms, int numberOfGeoms, material* materials){
	int rayIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (rayIdx < numberOfRays) {
		int pixelIdx = rays[rayIdx].index;
		float seed = time * rayIdx * (rayDepth+1);

		if (rays[rayIdx].active) {
			glm::vec3 minIntersection, minNormal; // closest intersection point and the normal at that point
			int minIdx = getClosestIntersection(geoms, numberOfGeoms, rays[rayIdx], minIntersection, minNormal);

			if (minIdx != -1) {
				material mtl = materials[geoms[minIdx].materialid]; // does caching make it faster?
				if (mtl.emittance > THRESHOLD) { // light
					glm::vec3 color = rays[rayIdx].baseColor * mtl.color * mtl.emittance;
					colors[pixelIdx] += color;
					rays[rayIdx].active = false;
				}
				else {
					if (mtl.hasReflective > THRESHOLD || mtl.hasRefractive > THRESHOLD) {
						if (!isDiffuseRay(seed, mtl.hasDiffuse)) {
							float IOR = mtl.indexOfRefraction;
							if (glm::dot(rays[rayIdx].direction, minNormal) > 0) { // reverse normal and index of refraction if we are inside the object
								minNormal *= -1;
								IOR = 1/(IOR + THRESHOLD);
							}
							if (mtl.hasRefractive > THRESHOLD) { // if the surface has refraction
								glm::vec3 VT = getRefractedRay(rays[rayIdx].direction, minNormal, IOR);
								if (glm::length(VT) > THRESHOLD && (mtl.hasReflective < THRESHOLD || isRefractedRay(seed, IOR, rays[rayIdx].direction, minNormal, VT))) {
									rays[rayIdx].direction = VT;
									rays[rayIdx].origin = minIntersection + VT * (float)THRESHOLD;
									rays[rayIdx].baseColor *= mtl.color;
									return;
								}
							}
							// if the surface only has reflection
							glm::vec3 VR = getReflectedRay(rays[rayIdx].direction, minNormal);
							rays[rayIdx].origin = minIntersection + VR * (float)THRESHOLD;
							rays[rayIdx].direction = VR;
							rays[rayIdx].baseColor *= mtl.color;
							return;
						}
					}
					
					// use diffuse shading model
					if (glm::dot(rays[rayIdx].direction, minNormal) > 0) { // reverse normal if we are inside the object
						minNormal *= -1;
					}
					
					// shoot diffuse reflected ray
					thrust::default_random_engine rng(hash(seed)); // include rayDepth in this since we don't want
																																									// the direction in every bounce to be the same
					thrust::uniform_real_distribution<float> u01(0, 1);
					rays[rayIdx].direction = calculateRandomDirectionInHemisphere(minNormal, (float)u01(rng), (float)u01(rng));
					rays[rayIdx].origin = minIntersection + rays[rayIdx].direction * (float)THRESHOLD;
					rays[rayIdx].baseColor = rays[rayIdx].baseColor * mtl.color;
				}
			}
			else {
				rays[rayIdx].active = false;
			}
		}
	}
}

// Copy ray active data to scanArray
__global__ void copy(ray* raypool, int* scanArray) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	scanArray[index] = (int)(raypool[index].active);
}

// Scan using shared memory
__global__ void sharedMemoryScan(int* scanArray, int* sumArray) {
	__shared__ int subArray1[64];
	__shared__ int subArray2[64];

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	subArray1[threadIdx.x] = scanArray[index];
	subArray2[threadIdx.x] = scanArray[index];
	__syncthreads();

	int d = 1;
#pragma unroll
	for (; d<=ceil(log((float)blockDim.x)/log(2.0f)); ++d) {
		if (threadIdx.x >= ceil(pow((float)2, (float)(d-1)))) {
			int prevIdx = threadIdx.x - ceil(pow((float)2, (float)(d-1)));
			if (d % 2 == 1) {
				subArray2[threadIdx.x] = subArray1[threadIdx.x] + subArray1[prevIdx];
			}
			else {
				subArray1[threadIdx.x] = subArray2[threadIdx.x] + subArray2[prevIdx];
			}
		}
		else {
			if (d % 2 == 1) {
				subArray2[threadIdx.x] = subArray1[threadIdx.x];
			}
			else {
				subArray1[threadIdx.x] = subArray2[threadIdx.x];
			}
		}
		__syncthreads();
	}

	if (d % 2 == 1) {
		scanArray[index] = subArray1[threadIdx.x];
		if (threadIdx.x == 63) {
			sumArray[blockIdx.x] = subArray1[threadIdx.x];
		}
	}
	else {
		scanArray[index] = subArray2[threadIdx.x];
		if (threadIdx.x == 63) {
			sumArray[blockIdx.x] = subArray2[threadIdx.x];
		}
	}
}

// Naive scan, for scanning the sum array
__global__ void naiveScan(int* scanArray1, int* scanArray2, int d) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= ceil(pow((float)2, (float)(d-1)))) {
		int prevIndex = index - (int)ceil(pow((float)2, (float)(d-1)));
		scanArray2[index] = scanArray1[index] + scanArray1[prevIndex];
	}
	else {
		scanArray2[index] = scanArray1[index];
	}
}

// Add the elements in the sum array to the scan array
__global__ void addToScanArray(int* scanArray, int* sumArray) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (blockIdx.x > 0) {
		scanArray[index] += sumArray[blockIdx.x-1];
	}
}

// Scatter kernel for stream compaction
__global__ void scatter(ray* raypool, int* scanArray, ray* newraypool) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (raypool[index].active) {
		newraypool[scanArray[index]-1] = raypool[index];
	}
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, cameraData cam, int iterations,
											staticGeom* cudageoms, int numberOfGeoms, material* cudamtls,
											glm::vec3* cudaimage, ray* raypool1, ray* raypool2, int numberOfRays,
											int* scanArray, int* sumArray1, int* sumArray2) {
  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));  //send image to GPU

	// populate the two ray pools
	generateRay<<<fullBlocksPerGrid, threadsPerBlock>>>(cam, (float)iterations, raypool1);
	cudaMemcpy(raypool2, raypool1, numberOfRays * sizeof(ray), cudaMemcpyDeviceToDevice);

	for (int i=0; i<MAXDEPTH; ++i) {
		int rayThreadsPerBlock = 64;
		int rayBlocksPerGrid = ceil((float)numberOfRays/(float)rayThreadsPerBlock);

		if (i % 2 == 0) {
			pathtraceRay<<<rayBlocksPerGrid, rayThreadsPerBlock>>>(cam, (float)iterations, i, raypool1,
				numberOfRays, cudaimage, cudageoms, numberOfGeoms, cudamtls);

			copy<<<rayBlocksPerGrid, rayThreadsPerBlock>>>(raypool1, scanArray);
		}
		else {
			pathtraceRay<<<rayBlocksPerGrid, rayThreadsPerBlock>>>(cam, (float)iterations, i, raypool2,
				numberOfRays, cudaimage, cudageoms, numberOfGeoms, cudamtls);

			copy<<<rayBlocksPerGrid, rayThreadsPerBlock>>>(raypool2, scanArray);
		}	

		sharedMemoryScan<<<rayBlocksPerGrid, rayThreadsPerBlock>>>(scanArray, sumArray1);
		int sumArrayLength = rayBlocksPerGrid;
		int naiveScanBlocksPerGrid = ((int)ceil((float)sumArrayLength/(float)64), 24); // 24 is the number of SMs on my GPU
		int naiveScanThreadsPerBlock = ceil((float)sumArrayLength/(float)naiveScanBlocksPerGrid);
		int d = 1;
		for (; d<=ceil(log(sumArrayLength)/log(2)); ++d) {
			// use double buffer
			if (d % 2 == 1) {
				naiveScan<<<naiveScanBlocksPerGrid, naiveScanThreadsPerBlock>>>(sumArray1, sumArray2, d);
			}
			else {
				naiveScan<<<naiveScanBlocksPerGrid, naiveScanThreadsPerBlock>>>(sumArray2, sumArray1, d);
			}
		}
		if (d % 2 == 1) {
			addToScanArray<<<rayBlocksPerGrid, rayThreadsPerBlock>>>(scanArray, sumArray1);
		}
		else {
			addToScanArray<<<rayBlocksPerGrid, rayThreadsPerBlock>>>(scanArray, sumArray2);
		}

		// get number of active rays
		int* numberOfActiveRays = new int[1];
		cudaMemcpy(numberOfActiveRays, scanArray + numberOfRays -1, sizeof(int), cudaMemcpyDeviceToHost);

		if (numberOfActiveRays[0] <= 0) {
			break;
		}

		// scatter
		if (i % 2 == 0) {
			scatter<<<rayBlocksPerGrid, rayThreadsPerBlock>>>(raypool1, scanArray, raypool2);
		}
		else {
			scatter<<<rayBlocksPerGrid, rayThreadsPerBlock>>>(raypool2, scanArray, raypool1);
		}

		numberOfRays = numberOfActiveRays[0];
		
		delete [] numberOfActiveRays;
	}

	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, (float)iterations);

  // make certain the kernel has completed
  cudaThreadSynchronize();
}
