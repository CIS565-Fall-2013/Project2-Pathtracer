// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>
/*

#include "utils/cuPrintf.cuh"
#include "utils/cuPrintf.cu"*/

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

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
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

__host__ __device__ glm::vec2 generateRandomNumberFromThreadForSSAA(glm::vec2 resolution, float time, float x, float y, float distanceToBoundary){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec2(x - distanceToBoundary + (float)u01(rng) * 2 * distanceToBoundary, y - distanceToBoundary + (float)u01(rng) * 2 * distanceToBoundary);
}


//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, float x, float y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
    glm::vec3 M, H, V, A, B, P;
	ray r;
    M = eye + view;
	A = glm::cross(view, up);
	B = glm::cross(A, view);
	V = B * (float)(tan(fov[1] / 180.0 * PI) * (glm::length(view) / glm::length(B)));
	H = A * (float)(tan(fov[0] / 180.0 * PI) * (glm::length(view) / glm::length(A)));

	P = M + H * (2.0f * x/(resolution[0]-1.0f) - 1.0f) + V * (2.0f * (1.0f-y/(resolution[1]-1.0f))-1.0f);
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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, int iterations){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0/float(iterations/2);
      color.y = image[index].y*255.0/float(iterations/2);
      color.z = image[index].z*255.0/float(iterations/2);

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

__host__ __device__ int rayIntersect(const ray& r, staticGeom* geoms, int numberOfGeoms, glm::vec3& intersectionPoint, glm::vec3& intersectionNormal, material* materials/*, int numberOfMaterials*/){
	float distance = 8000.0f;
	float tempDistance = -1.0f;
	glm::vec3 tempIntersctionPoint(0.0f), tempIntersectionNormal(0.0f);
	int intersIndex = -1;

	for(int i = 0; i < numberOfGeoms; i++)
	{
		tempDistance = -1.0f;
		if(geoms[i].type == GEOMTYPE::SPHERE){
			tempDistance = sphereIntersectionTest(geoms[i], r, tempIntersctionPoint, tempIntersectionNormal);
		}
		else if(geoms[i].type == GEOMTYPE::CUBE){
			tempDistance = boxIntersectionTest(geoms[i], r, tempIntersctionPoint, tempIntersectionNormal);
		}
		else 
			continue;
		if(!(abs(tempDistance + 1.0f) < EPSILON))
		{
			if(tempDistance < distance) {
				intersIndex = i;
				distance = tempDistance;
				intersectionPoint = tempIntersctionPoint;
				intersectionNormal = tempIntersectionNormal;
			}
		}
	}
//	printf("%d", intersIndex);
	return intersIndex;
}
__host__ __device__ bool ShadowRayUnblocked(glm::vec3 surfacePoint,glm::vec3 lightPosition, staticGeom* geoms, int numberOfGeoms, material* materials) // return true if unblocked
{
//	printf("in shadow test\n");
	glm::vec3 rayDir = glm::normalize(lightPosition - surfacePoint);
	ray shadowRay;
	shadowRay.origin = surfacePoint + 0.01f * rayDir;
	shadowRay.direction = rayDir;
	glm::vec3 intersPoint, intersNormal;
	int intersIndex = rayIntersect(shadowRay, geoms, numberOfGeoms, intersPoint, intersNormal, materials); 
	if(intersIndex == -1) return true;
	else if(materials[geoms[intersIndex].materialid].emittance > 0.0f) return true;
	else return false;

}
__host__ __device__ glm::vec3 raytraceRecursive(int index, const ray &r, int iteration, float currentIndexOfRefraction, int depth, int maximumDepth, staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, int* lightSources, int numberOfLights){

	glm::vec3 bgColor(0.0f);
	glm::vec3 phongColor(0.0f), reflColor(0.0f), refraColor(0.0f);;
	glm::vec3 returnColor(0.0f);


	if(depth > maximumDepth)
		return bgColor;

	// intersection test	
	glm::vec3 intersectionPoint, intersectionNormal;
	int intersIndex = rayIntersect(r, geoms, numberOfGeoms, intersectionPoint, intersectionNormal, materials);
//	if(depth == 0)
//	printf("%f %f %f\n", r.direction.x, r.direction.y, r.direction.z);
//	printf("%f %f %f\n", intersectionPoint.x, intersectionPoint.y, intersectionPoint.z);
//	printf("%d\n", intersIndex);
	/*glm::vec3 intersectionPoint, intersectionNormal;
	float distance = 8000.0f;
	float tempDistance = -1.0f;
	glm::vec3 tempIntersctionPoint(0.0f), tempIntersectionNormal(0.0f);
	int intersIndex = -1;
	printf("%f %f %f\n", tempIntersectionNormal.x, tempIntersectionNormal.y, tempIntersectionNormal.z);
	for(int i = 0; i < numberOfGeoms; i++)
	{
		tempDistance = -1.0f;
//		if(materials[geoms[i].materialid].emittance > 0.0f) continue; // do not test intesection with light source
		if(geoms[i].type == GEOMTYPE::SPHERE){
			tempDistance = sphereIntersectionTest(geoms[i], r, tempIntersctionPoint, tempIntersectionNormal);
		}
		else if(geoms[i].type == GEOMTYPE::CUBE){
			tempDistance = boxIntersectionTest(geoms[i], r, tempIntersctionPoint, tempIntersectionNormal);
		}
		else 
			continue;
		if(!(abs(tempDistance + 1.0f) < EPSILON))
		{
			if(tempDistance < distance) {
				intersIndex = i;
				distance = tempDistance;
				intersectionPoint = tempIntersctionPoint;
				intersectionNormal = tempIntersectionNormal;
			}
		}
	}*/
	
//	printf("depth = %d:\n\n", depth);
	if(intersIndex == -1) return bgColor;
	else if(materials[geoms[intersIndex].materialid].emittance > 0.0f) // intersected with light source geometry
		return materials[geoms[intersIndex].materialid].color;

	else // intersected with actual geometry
	{
		glm::vec3 reflDir, refraDir;
		float nextIndexOfRefraction;
		Fresnel fresnel; fresnel.reflectionCoefficient = 0.1; fresnel.transmissionCoefficient = 0;
//		returnColor = ka * ambientColor * materials[geoms[intersIndex].materialid].color;
//		calculateFresnel(intersectionNormal, r.direction, currentIndexOfRefraction, materials[geoms[intersIndex].materialid].indexOfRefraction, 
		if(/* material transparent */materials[geoms[intersIndex].materialid].hasRefractive == 1) // for fresnel, if refractive, it will be reflective
		{
			if(abs(currentIndexOfRefraction - 1) < 0.00001f)  // current ray is in air
			{
				refraDir = calculateTransmissionDirection(r.direction, intersectionNormal, currentIndexOfRefraction, materials[geoms[intersIndex].materialid].indexOfRefraction, nextIndexOfRefraction);
//				printf("in air %f\n", nextIndexOfRefraction);
			}
			else                                              // current ray is in glass
			{
				intersectionNormal = -intersectionNormal;
				refraDir = calculateTransmissionDirection(r.direction, intersectionNormal, currentIndexOfRefraction, 1.0f, nextIndexOfRefraction);
//				printf("in glass, depth: %d\n", depth);
				
			}
/*
			if(depth == 1)
			{*/
				/*printf("intersection index: %d\n", intersIndex);
				printf("incident direction: %f %f %f\n", r.direction.x, r.direction.y, r.direction.z);
				printf("refration direction: %f %f %f\n", refraDir.x, refraDir.y, refraDir.z);
				printf("intersectionPoint: %f %f %f\n", intersectionPoint.x, intersectionPoint.y, intersectionPoint.z);
				printf("intersectionNormal: %f %f %f\n", intersectionNormal.x, intersectionNormal.y, intersectionNormal.z);			
				printf("currentIndexOfRefraction: %f, nextIndexOfRefraction: %f\n\n", currentIndexOfRefraction, nextIndexOfRefraction);*/
//			}
			ray refraRay;
			refraRay.origin = intersectionPoint + 0.01f * refraDir;
			refraRay.direction = refraDir;
			refraColor = raytraceRecursive(index, refraRay, iteration, nextIndexOfRefraction, depth + 1, maximumDepth, geoms, numberOfGeoms, materials, numberOfMaterials, lightSources, numberOfLights);

			reflDir = calculateReflectionDirection(r.direction, intersectionNormal);
			ray reflRay;
			reflRay.origin = intersectionPoint + 0.01f * reflDir;
			reflRay.direction = reflDir;
			reflColor = raytraceRecursive(index, reflRay, iteration, currentIndexOfRefraction, depth + 1, maximumDepth, geoms, numberOfGeoms, materials, numberOfMaterials, lightSources, numberOfLights);

			fresnel = calculateFresnel(intersectionNormal, r.direction, currentIndexOfRefraction, nextIndexOfRefraction, reflDir, refraDir);
//			printf("reflectionCoefficient: %f;  transmissionCoefficient: %f\n", fresnel.reflectionCoefficient, fresnel.transmissionCoefficient);
//			return refraColor;
			returnColor = fresnel.reflectionCoefficient * reflColor +  fresnel.transmissionCoefficient * refraColor;
		}
		else if(/* material not transparent */materials[geoms[intersIndex].materialid].hasReflective == 1)
		{
			reflDir = calculateReflectionDirection(r.direction, intersectionNormal);
			ray reflRay;
			reflRay.origin = intersectionPoint + 0.01f * reflDir;
			reflRay.direction = reflDir;
			reflColor = raytraceRecursive(index, reflRay, iteration, currentIndexOfRefraction, depth + 1, maximumDepth, geoms, numberOfGeoms, materials, numberOfMaterials, lightSources, numberOfLights);

			returnColor = reflColor;
		}

#if 0
		for(int i = 0; i < numberOfLights; i++)
		{
//		if(iteration < numberOfLights)
//			printf("%d\n", numberOfLights);
//			if(materials[geoms[i].materialid].emittance > 0)
//			{
				glm::vec3 lightPos;
			    if(geoms[lightSources[i]].type == GEOMTYPE::CUBE)
				    lightPos = getRandomPointOnCube(geoms[lightSources[i]], iteration);
			    else if(geoms[lightSources[i]].type == GEOMTYPE::SPHERE)
				    lightPos = getRandomPointOnSphere(geoms[lightSources[i]], iteration);
				if(ShadowRayUnblocked(intersectionPoint, lightPos, geoms, numberOfGeoms, materials))
				{
		//					printf("shadow ray not blocked\n");
					glm::vec3 L = glm::normalize(lightPos - intersectionPoint);
		//				printf("%f %f %f\n", intersectionNormal.x, intersectionNormal.y, intersectionNormal.z);
					float dot1 = glm::clamp(glm::dot(intersectionNormal, L), 0.0f, 1.0f);
		//				printf("%f\n", dot1);
					float dot2 = glm::dot(calculateReflectionDirection(-L, intersectionNormal) ,-r.direction);
					glm::vec3 diffuse = materials[geoms[lightSources[i]].materialid].color /*lightSources[iteration].color*/ * 0.5f * materials[geoms[intersIndex].materialid].color * dot1;
					glm::vec3 specular;
					if(abs(materials[geoms[intersIndex].materialid].specularExponent) > 1e-6)
						specular = materials[geoms[lightSources[i]].materialid].color/*lightSources[iteration].color*/ * 0.1f * pow(max(dot2, 0.0f), materials[geoms[intersIndex].materialid].specularExponent);
					phongColor += /*(1.0f / numberOfLights) * */(diffuse + specular);
				
				}
//		}
		}

//		returnColor += (1.0f / numberOfLights) * (phongColor) + fresnel.reflectionCoefficient * reflColor +  fresnel.transmissionCoefficient * refraColor;
		returnColor = phongColor + fresnel.reflectionCoefficient * reflColor +  fresnel.transmissionCoefficient * refraColor;
#endif	
		else
		{
			thrust::default_random_engine rng(hash(iteration + depth));
			thrust::uniform_real_distribution<float> u01(0,1), v01(0, 1);
			float russianRoulette1 = (float)u01(rng);
			float russianRoulette2 = (float)v01(rng);

//			printf("iteration: %d;  russianRoulette1: %f;   russianRoulette2: %f;\n", iteration, russianRoulette1, russianRoulette2);

			glm::vec3 newDir = calculateRandomDirectionInHemisphere(intersectionNormal, russianRoulette1, russianRoulette2);
			ray newRay;
			newRay.origin =  intersectionPoint + 0.01f * newDir;
			newRay.direction = newDir;

			returnColor = materials[geoms[intersIndex].materialid].color * glm::clamp(glm::dot(intersectionNormal, newDir), 0.0f, 1.0f) * raytraceRecursive(index, newRay, iteration, currentIndexOfRefraction, depth + 1,  maximumDepth, geoms, numberOfGeoms, materials, numberOfMaterials, lightSources, numberOfLights);
		}

		
	}



	return returnColor;
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytracePrimary(glm::vec2 resolution, int time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, int* lightSources, int numberOfLights){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  int nSSAA = 1;
  float nStep = 1.0f / nSSAA;
  glm::vec3 antiAliasedColor(0.0f);

 
  if((x<=resolution.x && y<=resolution.y)){//for resolution undivisible by tileSize, some threads do not have a ray to test (out of screen)
//	 if((x == 400 && y == 650)){
//	  printf("**********************\n");
	  glm::vec2 jitteredScreenCoords;
	  for(float i = x - 0.5f + nStep * 0.5f; i < x +0.5f; i += nStep)
	  {
		  for(float j = y - 0.5f + nStep * 0.5f; j < y +0.5f; j += nStep)
		  {
			  jitteredScreenCoords = generateRandomNumberFromThreadForSSAA(resolution, time, i, j, 0.5f * nStep);
			  ray r = raycastFromCameraKernel(resolution, time, jitteredScreenCoords.x, jitteredScreenCoords.y, cam.position, cam.view, cam.up, cam.fov);
//			  ray r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
		//    colors[index] = generateRandomNumberFromThread(resolution, time, x, y);
			  antiAliasedColor += nStep * nStep * raytraceRecursive(index, r, time, 1.0f, 0, rayDepth, geoms, numberOfGeoms, materials, numberOfMaterials, lightSources, numberOfLights);
		//	  printf("%f %f %f\n", colors[index].x, colors[index].y, colors[index].z);
		  }
	  }
//	  colors[index] += 1.0f / 800.0f * antiAliasedColor;
	  colors[index] += pow(1.0f/((float)time + 1.0f), 2.0f) * antiAliasedColor ;


   }
}

__global__ void rayTracerIterative(int iteration, int depth, glm::vec2 resolution, ray *rayPool, int rayCount, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials){

	glm::vec3 bgColor(0.0f);
	glm::vec3 phongColor(0.0f), reflColor(0.0f), refraColor(0.0f);;
	glm::vec3 returnColor(0.0f);

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);

	// intersection test	
	if(rayPool[index].index != -1){
		glm::vec3 intersectionPoint, intersectionNormal;
		int intersIndex = rayIntersect(rayPool[index], geoms, numberOfGeoms, intersectionPoint, intersectionNormal, materials);

		if(intersIndex == -1) 
		{
			colors[rayPool[index].index] += glm::vec3(0.0f);
			rayPool[index].index = -1;
			return;
		}
		else if(materials[geoms[intersIndex].materialid].emittance > 0.0f) // intersected with light source geometry
		{
			rayPool[index].accumulatedColor *= materials[geoms[intersIndex].materialid].color;		
			colors[rayPool[index].index] += rayPool[index].accumulatedColor;
//			colors[rayPool[index].index] = (float)(iteration + 1); 
			rayPool[index].index = -1;
			return;
		}
		else // intersected with actual geometry
		{
			glm::vec3 reflDir, refraDir;
			float nextIndexOfRefraction;
			Fresnel fresnel;
			if(depth == 9) 
			{
				colors[rayPool[index].index] += glm::vec3(0.0f);
				rayPool[index].index = -1;
				return;
			}

			else if(materials[geoms[intersIndex].materialid].hasRefractive == 1) // for fresnel, if refractive, it will be reflective
			{
				if(abs(rayPool[index].mediaIOR - 1) < 0.00001f)  // current ray is in air
				{
					refraDir = calculateTransmissionDirection(rayPool[index].direction, intersectionNormal, rayPool[index].mediaIOR, materials[geoms[intersIndex].materialid].indexOfRefraction, nextIndexOfRefraction);
	//				printf("in air %f\n", nextIndexOfRefraction);
				}
				else                                              // current ray is in glass
				{
					intersectionNormal = -intersectionNormal;
					refraDir = calculateTransmissionDirection(rayPool[index].direction, intersectionNormal, rayPool[index].mediaIOR, 1.0f, nextIndexOfRefraction);
	//				printf("in glass, depth: %d\n", depth);
				
				}
				reflDir = calculateReflectionDirection(rayPool[index].direction, intersectionNormal);
				fresnel = calculateFresnel(intersectionNormal, rayPool[index].direction, rayPool[index].mediaIOR, nextIndexOfRefraction, reflDir, refraDir);

				thrust::default_random_engine rng(hash(iteration *(depth + 1)* index));
				thrust::uniform_real_distribution<float> u01(0,1);
//				float russianroulette = (float)u01(rng);

				if((float)u01(rng) < fresnel.reflectionCoefficient)
				{
					ray reflRay;
					reflRay.origin = intersectionPoint + 0.01f * reflDir;
					reflRay.direction = reflDir;
					reflRay.mediaIOR = rayPool[index].mediaIOR;
					reflRay.accumulatedColor = rayPool[index].accumulatedColor;
					rayPool[index] = reflRay;
				}
				
				else
				{
					ray refraRay;
					refraRay.origin = intersectionPoint + 0.01f * refraDir;
					refraRay.direction = refraDir;
					refraRay.mediaIOR = nextIndexOfRefraction;
					refraRay.accumulatedColor = rayPool[index].accumulatedColor;
					rayPool[index] = refraRay;
				}

			}
			else if(materials[geoms[intersIndex].materialid].hasReflective == 1)
			{
				reflDir = calculateReflectionDirection(rayPool[index].direction, intersectionNormal);
				ray reflRay;
				reflRay.origin = intersectionPoint + 0.01f * reflDir;
				reflRay.direction = reflDir;
				reflRay.index = rayPool[index].index;
				reflRay.accumulatedColor = rayPool[index].accumulatedColor;

				rayPool[index] = reflRay;
			}

			else
			{
				thrust::default_random_engine rng(hash(iteration *(depth + 1)* index));
				thrust::uniform_real_distribution<float> u01(0,1), v01(0, 1);
				float russianroulette1 = (float)u01(rng);
				float russianroulette2 = (float)v01(rng);

				glm::vec3 newdir = calculateRandomDirectionInHemisphere(intersectionNormal, russianroulette1, russianroulette2);
				ray newray;
				newray.origin =  intersectionPoint + 0.01f * newdir;
				newray.direction = newdir;
				newray.index = rayPool[index].index;
				newray.accumulatedColor = rayPool[index].accumulatedColor * glm::clamp(glm::dot(intersectionNormal, newdir), 0.0f, 1.0f) * materials[geoms[intersIndex].materialid].color;

				rayPool[index] = newray;
			}
		}
	}
}

__global__ void rayTracerIterativePrimary(glm::vec2 resolution, int time, cameraData cam, ray *rayPool){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y))
  {
     ray r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
     r.index = index;
	 r.accumulatedColor = glm::vec3(1.0f);
	 r.mediaIOR = 1.0f;
	 rayPool[index] = r;
  }
}



//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  // frame: current frame, objects and cam move between frames in multi-frame mode
  // iterations: curent iteration, objects and cam do not move between iterations
  int traceDepth = 10; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
    
  // construct initial ray pool
  int rayPoolCount = (int)(renderCam->resolution.x * renderCam->resolution.y);
  ray* rayPool = new ray[rayPoolCount]; 

    //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  ray* cudaray = NULL;
  cudaMalloc((void**)&cudaray, rayPoolCount*sizeof(ray));
  cudaMemcpy( cudaray, rayPool, rayPoolCount*sizeof(ray), cudaMemcpyHostToDevice);

  // first kernel launch to populate ray pool
  rayTracerIterativePrimary<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, iterations, cam, cudaray);

//  dim3 blocksPerGrid((int)ceil((float)rayPoolCount/(float)(tileSize*tileSize)));
  //send image to GPU
  glm::vec3* cudaimage = NULL;
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

  material* cudamaterials = NULL;
  cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  size_t size;
  cudaDeviceSetLimit(cudaLimitStackSize, 10000 * sizeof(float));
  cudaDeviceGetLimit(&size, cudaLimitStackSize);

//  cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
//  printf("MallocHeapSize found to be %d\n", (int)size);
//  cudaDeviceSetLimit(cudaLimitMallocHeapSize, size_t size)
  
//  printf("Stack size found to be %d\n",(int)size);


  //kernel launches
  for(int i = 0; i < traceDepth; ++i)
  {
     rayTracerIterative<<<fullBlocksPerGrid, threadsPerBlock>>>(iterations, i, renderCam->resolution, cudaray, rayPoolCount, cudaimage, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials);

  }

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, iterations);
  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  
  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaray);
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamaterials);
  delete geomList;
  delete rayPool;
  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
