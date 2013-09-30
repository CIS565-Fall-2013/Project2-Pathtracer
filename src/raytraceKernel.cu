// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <ctime>
#include <random>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

const glm::vec3 bgColour = glm::vec3 (0.55, 0.25, 0);

void checkCUDAError(const char *msg) {
	cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
	std::cin.get ();
    exit(EXIT_FAILURE); 
  }
} 

//Sets up the projection half vectors.
void	setupProjection (projectionInfo &ProjectionParams, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov)
{
	//Set up the projection variables:
	float	degToRad = 3.1415926 / 180.0;
	float	radToDeg = 1.0 / degToRad;

	ProjectionParams.centreProj = eye+view;
	glm::vec3	eyeToProjCentre = ProjectionParams.centreProj - eye;
	glm::vec3	A = glm::cross (ProjectionParams.centreProj, up);
	glm::vec3	B = glm::cross (A, ProjectionParams.centreProj);
	float		lenEyeToProjCentre = glm::length (eyeToProjCentre);
	
	ProjectionParams.halfVecH = glm::normalize (A) * lenEyeToProjCentre * (float)tan ((fov.x*degToRad));
	ProjectionParams.halfVecV = glm::normalize (B) * lenEyeToProjCentre * (float)tan ((fov.y*degToRad));
}

// Reflects the incidentRay around the normal.
__host__ __device__ glm::vec3 reflectRay (glm::vec3 incidentRay, glm::vec3 normal)
{
	glm::vec3 reflectedRay = incidentRay - (2.0f*glm::dot (incidentRay, normal))*normal;
	return reflectedRay;
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
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov, glm::vec3 centreProj,
													glm::vec3	halfVecH, glm::vec3 halfVecV)
{
  ray r;
  r.origin = eye;
  r.direction = glm::vec3(0,0,-1);

  float normDeviceX = (float)x / (resolution.x-1);
  float normDeviceY = 1 - ((float)y / (resolution.y-1));

  glm::vec3 P = centreProj + (2*normDeviceX - 1)*halfVecH + (2*normDeviceY - 1)*halfVecV;
  r.direction = glm::normalize (P - r.origin);

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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, int nLights){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){
	  image [index] /= nLights;
      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

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

__device__ interceptInfo getIntercept (staticGeom * geoms, sceneInfo objectCountInfo, ray castRay, material* textureArray)
{
	glm::vec3 intrPoint = glm::vec3 (0, 0, 0);
	glm::vec3 intrNormal = glm::vec3 (0, 0, 0);
	glm::vec2 UVcoords = glm::vec2 (0, 0);

	float interceptValue = -32767;

	material newMaterial;
	newMaterial.color = glm::vec3 (0,0,0);
	newMaterial.specularExponent = 1.0;
	newMaterial.hasReflective = 0.0;
	newMaterial.hasRefractive = 0.0;

	interceptInfo theRightIntercept;					// Stores the lowest intercept.
	theRightIntercept.interceptVal = interceptValue;	// Initially, it is empty/invalid
	theRightIntercept.intrNormal = intrNormal;			// Intially, Normal - 0,0,0
	theRightIntercept.intrMaterial = newMaterial;

	float min = 1e6;
	// Two different loops to intersect ray with cubes and spheres.
	for (int i = 0; i < objectCountInfo.nCubes; ++i)
	{
		staticGeom currentGeom = geoms [i];

		interceptValue = boxIntersectionTest(currentGeom, castRay, intrPoint, intrNormal, UVcoords);
		if (interceptValue > 0)
		{
			if (interceptValue < min)
			{
				min = interceptValue;

				theRightIntercept.interceptVal = min;
				theRightIntercept.intrNormal = intrNormal;
				theRightIntercept.intrMaterial = textureArray [currentGeom.materialid];
				theRightIntercept.UV = UVcoords;
			}
		}
	}

	for (int i = objectCountInfo.nCubes; i <= (objectCountInfo.nCubes+objectCountInfo.nSpheres); ++i)
	{
		staticGeom currentGeom = geoms [i];

		interceptValue = sphereIntersectionTest(currentGeom, castRay, intrPoint, intrNormal);
		if (interceptValue > 0)
		{
			if (interceptValue < min)
			{
				min = interceptValue;

				theRightIntercept.interceptVal = min;
				theRightIntercept.intrNormal = intrNormal;
				theRightIntercept.intrMaterial = textureArray [currentGeom.materialid];
			}
		}
	}

	return theRightIntercept;
}

__device__ unsigned long getIndex (int x, int y, int MaxWidth)
{	return (unsigned long) y*MaxWidth + x ;	}

__host__ __device__ bool isApproximate (float valToBeCompared, float valToBeCheckedAgainst) 
{ if ((valToBeCompared >= valToBeCheckedAgainst-0.001) && (valToBeCompared <= valToBeCheckedAgainst+0.001)) return true;	return false; }

__device__ glm::vec3 getColour (material Material, glm::vec2 UVcoords)
{
	if (Material.hasTexture)
	{	
		unsigned long texelXY, texelXPlusOneY, texelXYPlusOne, texelXPlusOneYPlusOne;
		float xInterp = (Material.Texture.texelWidth * UVcoords.x) - floor (Material.Texture.texelWidth * UVcoords.x);
		float yInterp = (Material.Texture.texelHeight * UVcoords.y) - floor (Material.Texture.texelHeight * UVcoords.y);

		texelXY = getIndex ((int)floor (Material.Texture.texelWidth * UVcoords.x), (int)floor (Material.Texture.texelHeight * UVcoords.y), Material.Texture.texelWidth);
		texelXPlusOneY = getIndex ((int)ceil (Material.Texture.texelWidth * UVcoords.x), (int)floor (Material.Texture.texelHeight * UVcoords.y), Material.Texture.texelWidth);
		texelXYPlusOne = getIndex ((int)floor (Material.Texture.texelWidth * UVcoords.x), (int)ceil (Material.Texture.texelHeight * UVcoords.y), Material.Texture.texelWidth);
		texelXPlusOneYPlusOne = getIndex ((int)ceil (Material.Texture.texelWidth * UVcoords.x), (int)ceil (Material.Texture.texelHeight * UVcoords.y), Material.Texture.texelWidth);

		glm::vec3 xInterpedColour1, xInterpedColour2, finalColour;
		xInterpedColour1 = xInterp * Material.Texture.texels [texelXPlusOneY] + (1-xInterp)* Material.Texture.texels [texelXY];
		xInterpedColour2 = xInterp * Material.Texture.texels [texelXPlusOneYPlusOne] + (1-xInterp)* Material.Texture.texels [texelXYPlusOne];
		finalColour = yInterp * xInterpedColour2 + (1-yInterp) * xInterpedColour1;

		return finalColour;
	}
	else
		return Material.color;
}

// Calclates the direct lighting at a given point, which is calculated from castRay and interceptVal of theRightIntercept. 
__device__ glm::vec3 calcShade (interceptInfo theRightIntercept, glm::vec3 lightVec, glm::vec3 eye, ray castRay, 
								material* textureArray, float ks, float kd, glm::vec3 lightCol)
{
	glm::vec3 shadedColour = glm::vec3 (0,0,0);
	if ((theRightIntercept.interceptVal > 0))
	{
		if (theRightIntercept.intrMaterial.hasReflective)
			shadedColour = theRightIntercept.intrMaterial.specularColor;
		else
			shadedColour = theRightIntercept.intrMaterial.color;
		//// Diffuse shading
		//glm::vec3 intrPoint = castRay.origin + theRightIntercept.interceptVal*castRay.direction;	// The intersection point.
		//glm::vec3 intrNormal = glm::normalize (eye - intrPoint); // intrNormal is the view vector.
		//float interceptValue = max (glm::dot (theRightIntercept.intrNormal, lightVec), (float)0); // Diffuse Lighting is given by (N.L); N being normal at intersection pt and L being light vector.
		//intrPoint = (getColour (theRightIntercept.intrMaterial, theRightIntercept.UV) * kd * interceptValue);			// Reuse intrPoint to store partial product (kdId) of the diffuse shading computation.
		//shadedColour += multiplyVV (lightCol*emittance, intrPoint);		// shadedColour will have diffuse shaded colour. 
		//
		//// Specular shading
		//lightVec = glm::normalize (reflectRay (-lightVec, theRightIntercept.intrNormal)); // Reuse lightVec for storing the reflection of light vector around the normal.
		//interceptValue = max (glm::dot (lightVec, intrNormal), (float)0);				// Reuse interceptValue for computing dot pdt of specular.
		//shadedColour += (lightCol * ks * pow (interceptValue, theRightIntercept.intrMaterial.specularExponent));

		//// Quick and Dirty fix for lights.
		//if ((theRightIntercept.intrMaterial.emittance > 0) && (interceptValue > 0))
		//	shadedColour = glm::vec3 (1,1,1);
		//else 
//		if ((rayDepth >= 5) && (theRightIntercept.intrMaterial.emittance < 1))
//			shadedColour = glm::vec3 (0);
	}

	return	shadedColour;
}

//TODO: Done!
//Core raytracer kernel
__global__ void raytraceRay (float time, cameraData cam, int rayDepth, glm::vec3* colors, staticGeom* geoms, 
							 material* textureArray, renderInfo * RenderParams, sceneInfo objectCountInfo, 
							 bool *primaryArrayOnDevice, ray *rayPoolOnDevice, int rayPoolLength, glm::vec3 lightPosition)
{
  __shared__ staticGeom light;
  __shared__ renderInfo RenderParamsOnBlock;
  __shared__ float ks;
  __shared__ float kd;
  __shared__ glm::vec3 lightPos;
  __shared__ glm::vec3 lightCol;
  __shared__ float nLights;
  __shared__ int sqrtLights;
  __shared__ float stepSize;
//  __shared__ float lightEmittance;

  extern __shared__ glm::vec3 arrayPool [];
  __shared__ glm::vec3 *colourBlock; 
  __shared__ bool *primArrayBlock;
  __shared__ ray *rayPoolBlock;

  if ((threadIdx.x == 0) && (threadIdx.y == 0))
  {
	  RenderParamsOnBlock = *RenderParams;
	  ks = RenderParams->ks;
	  kd = RenderParams->kd;
	  nLights = RenderParams->nLights;
	  sqrtLights = RenderParams->sqrtLights;
	  stepSize = RenderParams->lightStepSize;
	  light = geoms [0];
	  lightPos = lightPosition;
	  lightCol = RenderParams->lightCol;
//	  lightEmittance = textureArray [light.materialid].emittance;

	  colourBlock = arrayPool;
	  primArrayBlock = (bool *) &colourBlock [blockDim.x * blockDim.y];
	  rayPoolBlock = (ray *) &primArrayBlock [blockDim.x * blockDim.y];
	  for (int i = 0; i < blockDim.y; i ++)
		  for (int j = 0; j < blockDim.x; j ++)
		  {
			  // We have a 1-D array of blocks in the grid. From a thread's perspective, it is a 2-D array.
			  // Ray pool is a massive 1-D array, so we need to compute the index of the element of ray pool
			  // that each thread will handle.
			  int index = (blockIdx.x * blockDim.x) + j +			// X-part: straightforward
						  (i * (int)(blockDim.x * ceil ((float)rayPoolLength / (float)(blockDim.x*blockDim.y))));  // Y-part: as below:
			  // No. of blocks in the grid = ceil (rayPoolLength / (blockDim.x*blockDim.y))
			  // Multiplying that with the no. threads in a block gives the no. of threads in a single row of grid.
			  // Multiplying that with row number (threadIdx.y) and adding the x offset (X-part) gives the index.
			  int index2 = i*blockDim.x + j;
			  primArrayBlock [index2] = primaryArrayOnDevice [index];
			  rayPoolBlock [index2] = rayPoolOnDevice [index];

			  // We recompute the index for the colour array since it represents a frame
			  // and each index represents a pixel. If we don't, stream compaction would 
			  // mess things up.
			  index = rayPoolBlock [index2].y*cam.resolution.x + rayPoolBlock [index2].x;
 			  colourBlock [index2] = colors [index];
			  // colourBlock [index2] therefore represents colour computed by ray through the pixel (x,y)
		  }
  }
  __syncthreads ();


  int threadID = (blockIdx.x * blockDim.x) + threadIdx.x +			
				 (threadIdx.y * (int)(blockDim.x * ceil ((float)rayPoolLength / (float)(blockDim.x*blockDim.y))));
  int randomSeed = threadID*time;
  glm::vec3 shadedColour = glm::vec3 (0);
  if (threadID < rayPoolLength)
  {
	threadID = threadIdx.y*blockDim.x + threadIdx.x;
	ray currentRay = rayPoolBlock [threadID];
	interceptInfo theRightIntercept = getIntercept (geoms, objectCountInfo, currentRay, textureArray);
	glm::vec3 lightVec; 
		
	lightVec = glm::normalize (lightPosition - (currentRay.origin + (currentRay.direction*theRightIntercept.interceptVal)));
	shadedColour += (calcShade (theRightIntercept, lightVec, cam.position, currentRay, textureArray, ks, kd, lightCol));

	if ((theRightIntercept.intrMaterial.emittance > 0) || (theRightIntercept.interceptVal < 0))
		primArrayBlock [threadID] = false;	// Ray did not hit anything or it hit light, so kill it.
	else
		calculateBSDF  (currentRay, 
						currentRay.origin + currentRay.direction * theRightIntercept.interceptVal, 
						theRightIntercept.intrNormal, glm::vec3 (0), AbsorptionAndScatteringProperties (), 
						randomSeed, theRightIntercept.intrMaterial.color, glm::vec3 (0), theRightIntercept.intrMaterial);

	rayPoolBlock [threadID] = currentRay;
	
	if (glm::length (colourBlock [threadID]) > 0)
		colourBlock [threadID] *= shadedColour;			// Add computed shade to shadedColour.
	else
		colourBlock [threadID] = shadedColour;
  }

  __syncthreads ();

  if ((threadIdx.x ==0) && threadIdx.y == 0)
  {
	  for (int i = 0; i < blockDim.y; i ++)
		  for (int j = 0; j < blockDim.x; j ++)
		  {
			  // Calculate the index.
			  int index = (blockIdx.x * blockDim.x) + j +												// X-part
						  (i * (int)(blockDim.x * ceil ((float)rayPoolLength / (float)(blockDim.x*blockDim.y))));		// Y-part

			  threadID = i*blockDim.x + j;
			  primaryArrayOnDevice [index] = primArrayBlock [threadID];
			  rayPoolOnDevice [index] = rayPoolBlock [threadID];

			  // Recalculate the index for colour array.
			  index = rayPoolBlock [threadID].y*cam.resolution.x + rayPoolBlock [threadID].x;
			  colors [index] = colourBlock [threadID];
		  }
  }
}

// Kernel to create the initial pool of rays.
__global__ void createRayPool (ray *rayPool, bool *primaryArray, int *secondaryArray, 
								cameraData cam, projectionInfo ProjectionParams)
{
	int x = (blockDim.x * blockIdx.x) + threadIdx.x;
	int y = (blockDim.y * blockIdx.y) + threadIdx.y;
	int threadID = x  +
				   y * cam.resolution.y;
	if (threadID < cam.resolution.x*cam.resolution.y)
	{
		rayPool [threadID] = raycastFromCameraKernel (cam.resolution, 0, x, y, cam.position, 
														cam.view, cam.up, cam.fov, ProjectionParams.centreProj, 
														ProjectionParams.halfVecH, ProjectionParams.halfVecV);
		rayPool [threadID].x = (blockDim.x * blockIdx.x) + threadIdx.x;
		rayPool [threadID].y = (blockDim.y * blockIdx.y) + threadIdx.y;

		primaryArray [threadID] = true;
		secondaryArray [threadID] = 0;
	}
}

__global__ void copyArray (bool *from, int *to, int fromLength)
{
	/*__shared__ int blockArrays []
	__shared__ bool *pArrayBlock;
	__shared__ int *sArrayBlock;
	
	if (threadIdx.x == 0)
	{
		pArrayBlock = (bool *)blockArrays;
		sArrayBlock = (int *)&pArrayBlock [blockDim.x];
	}
	__syncthreads ();*/

//	int blockIndex = threadIdx.x;
	int globalIndex = blockDim.x*blockIdx.x + threadIdx.x;

	if (globalIndex < fromLength)
		to [globalIndex] = from [globalIndex];
}

__global__ void copyArray (ray *from, ray *to, int fromLength)
{
	/*__shared__ int blockArrays []
	__shared__ bool *pArrayBlock;
	__shared__ int *sArrayBlock;
	
	if (threadIdx.x == 0)
	{
		pArrayBlock = (bool *)blockArrays;
		sArrayBlock = (int *)&pArrayBlock [blockDim.x];
	}
	__syncthreads ();*/

//	int blockIndex = threadIdx.x;
	int globalIndex = blockDim.x*blockIdx.x + threadIdx.x;

	if (globalIndex < fromLength)
		to [globalIndex] = from [globalIndex];
}

__global__ void copyArray (int *from, int *to, int fromLength)
{
	/*__shared__ int blockArrays []
	__shared__ bool *pArrayBlock;
	__shared__ int *sArrayBlock;
	
	if (threadIdx.x == 0)
	{
		pArrayBlock = (bool *)blockArrays;
		sArrayBlock = (int *)&pArrayBlock [blockDim.x];
	}
	__syncthreads ();*/

//	int blockIndex = threadIdx.x;
	int globalIndex = blockDim.x*blockIdx.x + threadIdx.x;

	if (globalIndex < fromLength)
		to [globalIndex] = from [globalIndex];
}

// Kernel to do inclusive scan.
__global__ void inclusiveScan (int *secondaryArray, int *tmpArray, int primArrayLength, int iteration)
{
//	__shared__ int blockArrays []
//	__shared__ bool *pArrayBlock;
//	__shared__ int *sArrayBlock;

//	if (threadIdx.x == 0)
//	{
//		pArrayBlock = (bool *)blockArrays;
//		sArrayBlock = (int *)&pArrayBlock [blockDim.x];
//	}

	int		curIndex = blockDim.x*blockIdx.x + threadIdx.x;
	int		prevIndex = curIndex - (int)pow ((float)2, (float)(iteration-1));

	if (curIndex < primArrayLength)
		if (curIndex >= pow ((float)2, (float)iteration-1))
			tmpArray [curIndex] = secondaryArray [curIndex] + secondaryArray [prevIndex];
}

// Kernel to shift all elements of Array to the right. 
// The last element is thrown out in the process and the first element becomes 0.
// Can convert an inclusive scan result to an exclusive scan.
__global__ void	shiftRight (int *Array, int *secondArray, int arrayLength)
{
	int		curIndex = blockDim.x*blockIdx.x + threadIdx.x;
	if (curIndex < arrayLength)
	{
		if (curIndex > 0)
			secondArray [curIndex] = Array [curIndex - 1];
		else
			secondArray [curIndex] = 0;
	}

//	__syncthreads ();
	// Make sure we make the first element 0 only after all the shifting has taken place.
//	Array [curIndex] = secondArray [curIndex];
}


// Kernel to do stream compaction.
__global__ void	compactStream (ray *rayPoolOnDevice, ray *tempRayPool, bool *primaryArrayOnDevice, int *secondaryArray, int rayPoolLengthOnDevice)
{
	int		curIndex = blockDim.x*blockIdx.x + threadIdx.x;
	if (curIndex < rayPoolLengthOnDevice)
	{
		int secondArrayIndex = secondaryArray [curIndex];
		if (primaryArrayOnDevice [curIndex])
			tempRayPool [secondArrayIndex] = rayPoolOnDevice [curIndex];
	}
}

// Device function to check if a point is in shadow.
// Given a ray r, it intersects it with all the primitives in the scene to check whether it intersects
// with any other primitive than light. If it does, then the point (ray's origin) is in shadow, and it returns true. 
__device__ bool isShadowRayBlocked (ray r, glm::vec3 lightPos, staticGeom *geomsList, sceneInfo objectCountInfo)
{
	float min = 1e6, interceptValue;
	glm::vec3 intrPoint, intrNormal;
	glm::vec2 UVcoords = glm::vec2 (0, 0);
	for (int i = 0; i < objectCountInfo.nCubes; ++i)
	{
		staticGeom currentGeom = geomsList [i];
		interceptValue = boxIntersectionTest(currentGeom, r, intrPoint, intrNormal, UVcoords);
		if (interceptValue > 0)
		{
			if (interceptValue < min)
				min = interceptValue;
		}
	}

	for (int i = objectCountInfo.nCubes; i <= (objectCountInfo.nCubes+objectCountInfo.nSpheres); ++i)
	{
		staticGeom currentGeom = geomsList [i];
		interceptValue = sphereIntersectionTest(currentGeom, r, intrPoint, intrNormal);
		if (interceptValue > 0)
		{
			if (interceptValue < min)
				min = interceptValue;
		}
	}

	if (glm::length (lightPos - r.origin) > (min+0.1))
		return true;
	return false;
}

// This kernel will accumulate all the colours calculated in an iteration into the actual colour array.
__global__ void		accumulateIterationColour (glm::vec3* accumulator, glm::vec3* iterationColour, glm::vec2 resolution)
{
	int index = (blockDim.y*blockIdx.y + threadIdx.y) * resolution.x + 
				(blockDim.x*blockIdx.x + threadIdx.x);
	if (index < resolution.x*resolution.y)
		accumulator [index] += iterationColour [index];
}

// This kernel surveys the rays that are still alive and replaces
// the colours of their respective pixels with noise (0,0,0)
__global__ void		addNoise (glm::vec3 *localColours, ray *rayPoolOnDevice, int rayPoolLength, glm::vec2 resolution)
{
	// Index calculation, as in raytraceRay
	int index = (blockIdx.x * blockDim.x) + threadIdx.x +															// X-part
				(threadIdx.y * (int)(blockDim.x * ceil ((float)rayPoolLength / (float)(blockDim.x*blockDim.y))));	// Y-part
	if (index < rayPoolLength)
	{
		// Index re-calculation for colour array, as in raytraceRay
		ray currentRay = rayPoolOnDevice [index];
		int colourIndex = currentRay.y * resolution.x + currentRay.x;
		localColours [colourIndex] = glm::vec3 (0);
	}
}

// If errorCode is not cudaSuccess, kills the program.
void onDeviceErrorExit (cudaError_t errorCode, glm::vec3 *cudaimage, staticGeom *cudageoms, material * materialColours, int numberOfMaterials)
{
  if (errorCode != cudaSuccess)
  {
	  std::cout << "\nError while trying to send texture data to the GPU!";
	  std::cin.get ();

	  if (cudaimage)
		cudaFree( cudaimage );
	  if (cudageoms)
		cudaFree( cudageoms );
	  if (materialColours)
	  {
		   /*for (int i = 0; i < numberOfMaterials; i ++)
		   {
			   if (materialColours [i].hasTexture)
				cudaFree (materialColours[i].Texture.texels);

			   if (materialColours [i].hasNormalMap)
				cudaFree (materialColours[i].NormalMap.texels);
		   }*/
		  cudaFree (materialColours);
	  }

	  cudaimage = NULL;
	  cudageoms = NULL;
	  materialColours = NULL;

	  exit (EXIT_FAILURE);
  }
}

//TODO: Done!
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces
  projectionInfo	ProjectionParams;
  float degToRad = 3.1415926 / 180.0;
	
  // Set up projection.
	ProjectionParams.centreProj = renderCam->positions [frame]+renderCam->views [frame];
	glm::vec3	eyeToProjCentre = ProjectionParams.centreProj - renderCam->positions [frame];
	glm::vec3	A = glm::cross (eyeToProjCentre, renderCam->ups [frame]);
	glm::vec3	B = glm::cross (A, eyeToProjCentre);
	float		lenEyeToProjCentre = glm::length (eyeToProjCentre);
	
	ProjectionParams.halfVecH = glm::normalize (A) * lenEyeToProjCentre * (float)tan ((renderCam->fov.x*degToRad) / 2.0);
	ProjectionParams.halfVecV = glm::normalize (B) * lenEyeToProjCentre * (float)tan ((renderCam->fov.y*degToRad) / 2.0);

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaFinalImage = NULL;
  cudaMalloc((void**)&cudaFinalImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaFinalImage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  // package geometry to be sent to GPU global memory
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  sceneInfo		primCounts;
  
  // Reorder geometry so that light is the first item in geomList,
  // followed by cubes and then spheres. Doing so reduces divergence.
  int count = 1;	int lightIndex = 0;
  bool lightSet = false;
  for(int i=0; i<numberOfGeoms; i++)
  {
	  if ((geoms [i].materialid == 8) && !lightSet)
	  {
		staticGeom newStaticGeom;
		newStaticGeom.type = geoms[i].type;
		newStaticGeom.materialid = geoms[i].materialid;
		newStaticGeom.translation = geoms[i].translations[frame];
		newStaticGeom.rotation = geoms[i].rotations[frame];
		newStaticGeom.scale = geoms[i].scales[frame];
		newStaticGeom.transform = geoms[i].transforms[frame];
		newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
		geomList[0] = newStaticGeom;
		
		lightSet = true;
		lightIndex = i;
	  }

	  else if (geoms [i].type == CUBE)
	  {
		staticGeom newStaticGeom;
		newStaticGeom.type = geoms[i].type;
		newStaticGeom.materialid = geoms[i].materialid;
		newStaticGeom.translation = geoms[i].translations[frame];
		newStaticGeom.rotation = geoms[i].rotations[frame];
		newStaticGeom.scale = geoms[i].scales[frame];
		newStaticGeom.transform = geoms[i].transforms[frame];
		newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
		geomList[count] = newStaticGeom;
		count ++;
	  }
  }

  if (!lightSet)
  {
	  geomList [0] = geomList [count-1];
	  count --;
  }
  // Lights may only be cubes.
  primCounts.nCubes = count;
  
  for(int i=0; i<numberOfGeoms; i++)
  {
	  if (geoms [i].type == SPHERE)
	  {
		staticGeom newStaticGeom;
		newStaticGeom.type = geoms[i].type;
		newStaticGeom.materialid = geoms[i].materialid;
		newStaticGeom.translation = geoms[i].translations[frame];
		newStaticGeom.rotation = geoms[i].rotations[frame];
		newStaticGeom.scale = geoms[i].scales[frame];
		newStaticGeom.transform = geoms[i].transforms[frame];
		newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
		geomList[count] = newStaticGeom;
		count ++;
	  }
  }

  primCounts.nSpheres = count - primCounts.nCubes;
  primCounts.nMeshes = 0;
  materials [geoms [lightIndex].materialid].color *= materials [geoms [lightIndex].materialid].emittance;

  // Allocate memory. We'll copy it later (because we're moving objects around for Motion blur).
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom)); 
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);


  // Copy materials to GPU global memory:
  material		*materialColours = NULL;
  glm::vec3		*colourArray = NULL;

  // Guard against shallow copying here.. Materials has a pointer pointing to Texture data.
  int sizeOfMaterialsArr = numberOfMaterials * (sizeof (material));
  cudaError_t returnCode1 = cudaMalloc((void**)&materialColours, numberOfMaterials*sizeof(material));
  onDeviceErrorExit (returnCode1, cudaFinalImage, cudageoms, materialColours, numberOfMaterials);
  cudaMemcpy (materialColours, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  // TODO: Texture mapping: Use index to a texture array.
  // Deep copying textures and normal maps:
  //glm::vec3 *texture = NULL;
  //glm::vec3 *norMap = NULL;
  //material *copyMaterial = new material [numberOfMaterials];	// SUCKS!
  //for (int i = 0; i < numberOfMaterials; i ++)
  //{
	 // copyMaterial [i] = materials [i];
	 // copyMaterial [i].Texture.texels = NULL;
	 // copyMaterial [i].NormalMap.texels = NULL;
	 // int noOfTexels = 0, noOfNMapTexels = 0;
	 // if (copyMaterial [i].hasTexture)
	 // {
		//  noOfTexels = materials [i].Texture.texelHeight * materials [i].Texture.texelWidth;
		//  cudaError_t returnCode2 = cudaMalloc ((void **)&texture, noOfTexels * sizeof (glm::vec3));
		//  onDeviceErrorExit (returnCode2, cudaimage, cudageoms, materialColours, numberOfMaterials);
		//  copyMaterial [i].Texture.texels = texture;
	 // }

	 // if (copyMaterial [i].hasNormalMap)
	 // {
		//  noOfNMapTexels = materials [i].NormalMap.texelHeight * materials [i].NormalMap.texelWidth;
		//  cudaError_t returnCode2 = cudaMalloc ((void **)&norMap, noOfNMapTexels * sizeof (glm::vec3));
		//  onDeviceErrorExit (returnCode2, cudaimage, cudageoms, materialColours, numberOfMaterials);
		//  copyMaterial [i].NormalMap.texels = norMap;
	 // }
  //}

  //cudaMemcpy (materialColours, copyMaterial, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  //for (int i = 0; i < numberOfMaterials; i ++)
  //{

	 // if (noOfTexels)
		//  cudaMemcpy( curMaterialDevice->Texture.texels, materials [i].Texture.texels, noOfTexels*sizeof(glm::vec3), cudaMemcpyHostToDevice);
	 // if (noOfNMapTexels)
		//  cudaMemcpy (curMaterialDevice->NormalMap.texels, materials [i].NormalMap.texels, noOfNMapTexels*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  //}

  // Need to check whether the above method is correct.

  // Copy the render parameters like ks, kd values, the no. of times the area light is sampled, 
  // the position of the light samples w/r to the light's geometry and so on.
  renderInfo	RenderParams, *RenderParamsOnDevice = NULL;
  RenderParams.ks = 0.4;
  RenderParams.kd = 1 - RenderParams.ks;
  RenderParams.nLights = renderCam->iterations;
  RenderParams.sqrtLights = sqrt ((float)RenderParams.nLights);
  RenderParams.lightStepSize = 1.0/(RenderParams.sqrtLights-1);
  RenderParams.lightPos = glm::vec3 (0, -0.6, 0);
  RenderParams.lightCol = materials [geoms [lightIndex].materialid].color * materials [geoms [lightIndex].materialid].emittance;
  cudaMalloc ((void **)&RenderParamsOnDevice, sizeof (renderInfo));
  cudaMemcpy (RenderParamsOnDevice, &RenderParams, sizeof (renderInfo), cudaMemcpyHostToDevice);

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  int nIterations = renderCam->iterations;

  time_t startTime = time (NULL);
  std::default_random_engine randomNumGen (hash (startTime));
  std::uniform_real_distribution<float> jitter ((float)0, (float)0.142);

  float movement = 3.0/nIterations;
  int nBounces = 6;
  int oneEighthDivisor = nIterations / 8;

  // For each point sampled in the area light, launch the raytraceRay Kernel which will compute the diffuse, specular, ambient
  // and shadow colours. It will also compute reflected colours for reflective surfaces.
  for (int i = 0; i < nIterations; i ++)
  {
	  glm::vec3* cudaimage = NULL;
	  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	  cudaMemset (cudaimage, 0, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));


	  float zAdd = jitter (randomNumGen);
	  float xAdd = jitter (randomNumGen); 
	  glm::vec3 curLightSamplePos = glm::vec3 (RenderParams.lightPos.x /*+ ((i%RenderParams.sqrtLights)*RenderParams.lightStepSize)*/, 
												RenderParams.lightPos.y, 
												RenderParams.lightPos.z/* + ((i/RenderParams.sqrtLights)*RenderParams.lightStepSize)*/);
	  
	  // Area light sampled in a jittered grid to reduce banding.
	  curLightSamplePos.z += zAdd;
	  curLightSamplePos.x += xAdd;
	  
	  if (!(i%oneEighthDivisor))	// Supersampling at 8x!
	  {
		cam.position.y += zAdd*0.002;
		cam.position.x += xAdd*0.002;
	  }

	  if (!((i*4)/(3*nIterations)))	
	  {
		  // Motion blur!
		  geomList [primCounts.nCubes].translation += glm::vec3 (movement, 0, 0);
		  glm::mat4 transform = utilityCore::buildTransformationMatrix(geomList [primCounts.nCubes].translation, 
																	   geomList [primCounts.nCubes].rotation, 
																	   geomList [primCounts.nCubes].scale);
		  geomList [primCounts.nCubes].transform = utilityCore::glmMat4ToCudaMat4(transform);
		  geomList [primCounts.nCubes].inverseTransform = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));
	  }

	  //  Now copy the geometry list to the GPU global memory.
	  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

	  glm::vec3 lightPos = multiplyMV (geomList [0].transform, glm::vec4 (curLightSamplePos, 1.0));
	  
	  // Create Ray Pool. 
	  int rayPoolLength = cam.resolution.x * cam.resolution.y;
//	  ray *rayPool = new ray [rayPoolLength];
	  ray *rayPoolOnDevice = NULL;
	  cudaMalloc ((void **)&rayPoolOnDevice, rayPoolLength * sizeof (ray));

	  // Primary Array		-> Array holding the true/false value specifying whether the ray is alive (true) or dead (false).
	  // Secondary Array	-> Array that will hold the indices of rays that are alive. Used in stream compaction.
	  bool *primaryArray = new bool [rayPoolLength];
	  memset (primaryArray, true, rayPoolLength * sizeof(bool));
	  bool *primaryArrayOnDevice = NULL;
	  cudaMalloc ((void **)&primaryArrayOnDevice, rayPoolLength * sizeof (bool));

	  int *secondaryArray = NULL;
	  cudaMalloc ((void **)&secondaryArray, rayPoolLength * sizeof (int));
	  int *secondaryArrayOnHost = new int [rayPoolLength];

	  // Launch createRayPool kernel to create the ray pool and populate the primary and secondary arrays.
	  fullBlocksPerGrid = dim3 ((int)ceil(float(cam.resolution.x)/threadsPerBlock.x), (int)ceil(float(cam.resolution.y)/threadsPerBlock.y));
	  createRayPool<<<fullBlocksPerGrid, threadsPerBlock>>> (rayPoolOnDevice, primaryArrayOnDevice, secondaryArray, cam, ProjectionParams);
//	  cudaDeviceSynchronize ();
//	  checkCUDAError("createRayPool Kernel failed!");
	  
	  dim3 threadsPerBlock1D (threadsPerBlock.x*threadsPerBlock.y);
	  // Iterate until nBounces: launch kernel to trace each ray bounce.
	  for (int j = 0; j < nBounces; ++j)
	  {
		// The backup secondary array. We use this for many things because we can't do stuff in-place in parallel.
	    int *secondaryArray2 = NULL;
		cudaMalloc ((void **)&secondaryArray2, rayPoolLength * sizeof (int));

		// The core raytraceRay kernel launch
		fullBlocksPerGrid = dim3 ((int)ceil(float(rayPoolLength)/(threadsPerBlock.x*threadsPerBlock.y))); 
		raytraceRay<<<fullBlocksPerGrid, threadsPerBlock, threadsPerBlock.x*threadsPerBlock.y*(sizeof(glm::vec3) + sizeof (bool) + sizeof(ray))>>>
			((float)j+(i*nBounces), cam, j, cudaimage, cudageoms, materialColours, RenderParamsOnDevice, 
			 primCounts, primaryArrayOnDevice, rayPoolOnDevice, rayPoolLength, lightPos);
//		checkCUDAError("raytraceRay Kernel failed!");

/////		---- CPU Stream Compaction ----		///
//		// Inefficient. Grossly inefficient. Need to look over and change as required.
//		cudaMemcpy (primaryArray, primaryArrayOnDevice, rayPoolLength * sizeof (bool), cudaMemcpyDeviceToHost);
//		cudaMemcpy (rayPool, rayPoolOnDevice, rayPoolLength * sizeof (ray), cudaMemcpyDeviceToHost);
//
//		// Stream compaction:
//		secondaryArrayOnHost [0] = 0;
//		for (int k = 1; k < rayPoolLength; ++ k)
//			secondaryArrayOnHost [k] = secondaryArrayOnHost [k-1] + primaryArray [k-1];
//
//		int count = 0;
//		for (int k = 0; k < rayPoolLength; ++ k)
//		{
//			if (primaryArray [k])
//			{
//				rayPool [secondaryArrayOnHost [k]] = rayPool [k];
//				++ count;
//			}
//		}
//
//		rayPoolLength = count;
//		cudaMemcpy (rayPoolOnDevice, rayPool, rayPoolLength * sizeof (ray), cudaMemcpyHostToDevice);
//		cudaMemset (primaryArrayOnDevice, true, rayPoolLength * sizeof (bool));

		/// ----- Stream Compaction ----- ///
		// Copy the primary array into secondaryArray and secondaryArray2.
		copyArray<<<fullBlocksPerGrid, threadsPerBlock1D>>> (primaryArrayOnDevice, secondaryArray, rayPoolLength);
//		checkCUDAError("copyArray Kernel failed!");
		copyArray<<<fullBlocksPerGrid, threadsPerBlock1D>>> (secondaryArray, secondaryArray2, rayPoolLength);
		
		// Parallel scan. We can't do it in place, so we accumulate in secondaryArray2, and copy back to secondaryArray
		// at the end of each iteration.
		for (int k = 1; k <= ceil (log2f (rayPoolLength)); ++ k) 
		{
			inclusiveScan<<<fullBlocksPerGrid, threadsPerBlock1D>>>(secondaryArray, secondaryArray2, rayPoolLength, k);
			copyArray<<<fullBlocksPerGrid, threadsPerBlock1D>>> (secondaryArray2, secondaryArray, rayPoolLength);
		}
//		checkCUDAError("inclusiveScan Kernel failed!");
	
		// Next, we convert the result of the parallel scan (inclusive) into exclusive scan.
		// Again, we use secondaryArray2 to store the shifted values, and then copy back to secondaryArray.
		shiftRight<<<fullBlocksPerGrid, threadsPerBlock1D>>>(secondaryArray, secondaryArray2, rayPoolLength);
//		checkCUDAError("shiftRight Kernel failed!");
		copyArray<<<fullBlocksPerGrid, threadsPerBlock1D>>> (secondaryArray2, secondaryArray, rayPoolLength);
//		checkCUDAError("copyArray-shiftRight Kernel failed!");

		// We're done with the backup secondary array, so let's free the memory.
		cudaFree (secondaryArray2);

		// Stream compaction. Compact the ray pool into tmpRayPool and copy back.
		ray *tmpRayPool = NULL;
		cudaMalloc ((void **)&tmpRayPool, rayPoolLength * sizeof (ray));
		compactStream<<<fullBlocksPerGrid, threadsPerBlock1D>>> (rayPoolOnDevice, tmpRayPool, primaryArrayOnDevice, secondaryArray, rayPoolLength);
//		checkCUDAError("compactStream Kernel failed!");

		// But we only need to copy back the compacted array, so first find the length of the reduced array 
		cudaMemcpy (primaryArray, primaryArrayOnDevice, rayPoolLength * sizeof (bool), cudaMemcpyDeviceToHost);
		int count = 0;
		for (int k = 0; k < rayPoolLength; ++k)
			if (primaryArray [k])
				count ++;
		rayPoolLength = count;

		// Now copy it back.
		copyArray<<<fullBlocksPerGrid, threadsPerBlock1D>>> (tmpRayPool, rayPoolOnDevice, rayPoolLength);
		cudaFree (tmpRayPool);

		// Set the primary array to all trues because all rays in the ray pool are alive, 
		// now that stream compaction has already happened.
		cudaMemset (primaryArrayOnDevice, true, rayPoolLength * sizeof (bool));
//		checkCUDAError("copyArray-compactStream Kernel failed!");
	  }
	  checkCUDAError ("One or more of the raytrace/stream compaction kernels failed. ");

	  fullBlocksPerGrid = dim3 ((int)ceil(float(rayPoolLength)/(threadsPerBlock.x*threadsPerBlock.y))); 
	  // At this point, since stream compaction has already taken place,
	  // it means that rayPoolOnDevice contains only rays that are still alive.
	  addNoise<<<fullBlocksPerGrid,threadsPerBlock>>>(cudaimage, rayPoolOnDevice, rayPoolLength, cam.resolution);

	  fullBlocksPerGrid = dim3 ((int)ceil(float(cam.resolution.x)/threadsPerBlock.x), (int)ceil(float(cam.resolution.y)/threadsPerBlock.y));
	  accumulateIterationColour<<<fullBlocksPerGrid, threadsPerBlock>>>(cudaFinalImage, cudaimage, cam.resolution);
	  checkCUDAError("accumulateIterationColour Kernel failed!");

	  cudaFree (rayPoolOnDevice);
	  cudaFree (primaryArrayOnDevice);
	  cudaFree (secondaryArray);
	  cudaFree (cudaimage);

	  rayPoolOnDevice = NULL;
	  primaryArrayOnDevice = NULL;
	  cudaimage = NULL;

	  delete [] primaryArray;
	  delete [] secondaryArrayOnHost;
//	  delete [] rayPool;

	  std::cout << "\rRendering.. " <<  ceil ((float)i/(RenderParams.nLights-1) * 100) << " percent complete.";
  }

  // Accumulate all the colours in the cudaFinalImage memory block on the GPU, and divide 
  // by the no. of light samples to get the final colour.
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaFinalImage, nIterations);
  std::cout.precision (4);
  std::cout << "\nRendered in " << difftime (time (NULL), startTime) << " seconds. \n\n";
  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaFinalImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
   if (cudaFinalImage)
		cudaFree( cudaFinalImage );
   if (cudageoms)
		cudaFree( cudageoms );
   if (materialColours)
   {
	   /*for (int i = 0; i < numberOfMaterials; i ++)
	   {
		   if (materialColours [i].hasTexture)
			cudaFree (materialColours[i].Texture.texels);

		   if (materialColours [i].hasNormalMap)
			cudaFree (materialColours[i].NormalMap.texels);
	   }*/
	   cudaFree (materialColours);
   }
   cudaFree (RenderParamsOnDevice);

   cudaFinalImage = NULL;
   cudageoms = NULL;
   materialColours = NULL;

 // make certain the kernel has completed
  cudaThreadSynchronize();
  
  delete [] geomList;

  checkCUDAError("Kernel failed!");
  
//  std::cin.get ();
}
