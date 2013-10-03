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

// Intersects the castRay with all the geometry in the scene (geoms) and returns the intercept information.
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

// Given MaxWidth of a 2D array, and the x and y co-ordinates or indices of an element, returns the equivalent 1D array index.
__device__ unsigned long getIndex (int x, int y, int MaxWidth)
{	return (unsigned long) y*MaxWidth + x ;	}

// Check for approximate equality.
__host__ __device__ bool isApproximate (float valToBeCompared, float valToBeCheckedAgainst) 
{ if ((valToBeCompared >= valToBeCheckedAgainst-0.001) && (valToBeCompared <= valToBeCheckedAgainst+0.001)) return true;	return false; }

// Given the UV coordinates (UVcoords) and a Texture, this returns the bilinearly interpolated colour at that point.
__device__ glm::vec3 getColour (mytexture &Texture, glm::vec2 UVcoords)
{	
		unsigned long texelXY, texelXPlusOneY, texelXYPlusOne, texelXPlusOneYPlusOne;
		float xInterp = (Texture.texelWidth * UVcoords.x) - floor (Texture.texelWidth * UVcoords.x);
		float yInterp = (Texture.texelHeight * UVcoords.y) - floor (Texture.texelHeight * UVcoords.y);

		texelXY = getIndex ((int)floor (Texture.texelWidth * UVcoords.x), (int)floor (Texture.texelHeight * UVcoords.y), Texture.texelWidth);
		texelXPlusOneY = getIndex ((int)ceil (Texture.texelWidth * UVcoords.x), (int)floor (Texture.texelHeight * UVcoords.y), Texture.texelWidth);
		texelXYPlusOne = getIndex ((int)floor (Texture.texelWidth * UVcoords.x), (int)ceil (Texture.texelHeight * UVcoords.y), Texture.texelWidth);
		texelXPlusOneYPlusOne = getIndex ((int)ceil (Texture.texelWidth * UVcoords.x), (int)ceil (Texture.texelHeight * UVcoords.y), Texture.texelWidth);

		glm::vec3 xInterpedColour1, xInterpedColour2, finalColour;
		xInterpedColour1 = xInterp * Texture.texels [texelXPlusOneY] + (1-xInterp)* Texture.texels [texelXY];
		xInterpedColour2 = xInterp * Texture.texels [texelXPlusOneYPlusOne] + (1-xInterp)* Texture.texels [texelXYPlusOne];
		finalColour = yInterp * xInterpedColour2 + (1-yInterp) * xInterpedColour1;

		return finalColour;
}

// Calclates the direct lighting at a given point, which is calculated from castRay and interceptVal of theRightIntercept. 
__device__ glm::vec3 calcShade (interceptInfo theRightIntercept, mytexture* textureArray)
{
	glm::vec3 shadedColour = glm::vec3 (0,0,0);
	if ((theRightIntercept.interceptVal > 0))
	{
		if ((theRightIntercept.intrMaterial.hasReflective >= 1.0) || 
			(theRightIntercept.intrMaterial.hasRefractive >= 1.0))
			shadedColour = theRightIntercept.intrMaterial.specularColor;
//		else if (theRightIntercept.intrMaterial.hasTexture)
//			shadedColour = getColour (textureArray [theRightIntercept.intrMaterial.textureid], theRightIntercept.UV);
		else
			shadedColour = theRightIntercept.intrMaterial.color;
	}

	return	shadedColour;
}

//TODO: Done!
//Core raytracer kernel
__global__ void raytraceRay (float time, cameraData cam, int rayDepth, glm::vec3* colors, staticGeom* geoms, 
							 material* textureArray, mytexture * Textures, sceneInfo objectCountInfo, 
							 bool *primaryArrayOnDevice, ray *rayPoolOnDevice, int rayPoolLength)
{
  extern __shared__ glm::vec3 arrayPool [];
  __shared__ glm::vec3 *colourBlock; 
  __shared__ bool *primArrayBlock;
  __shared__ ray *rayPoolBlock;

  if ((threadIdx.x == 0) && (threadIdx.y == 0))
  {
	  colourBlock = arrayPool;
	  primArrayBlock = (bool *) &colourBlock [blockDim.x * blockDim.y];
	  rayPoolBlock = (ray *) &primArrayBlock [blockDim.x * blockDim.y];
  }

  __syncthreads ();		// Block all threads until the colourBlock, rayPoolBlock
						// and primArrayBlock pointers have been bound properly.

  // We have a 1-D array of blocks in the grid. From a thread's perspective, it is a 2-D array.
  // Ray pool is a massive 1-D array, so we need to compute the index of the element of ray pool
  // that each thread will handle.

  int index = (blockIdx.x * blockDim.x) + threadIdx.x +			// X-part: straightforward
			(threadIdx.y * (int)(blockDim.x * ceil ((float)rayPoolLength / (float)(blockDim.x*blockDim.y))));  // Y-part: as below:
  // No. of blocks in the grid = ceil (rayPoolLength / (blockDim.x*blockDim.y))
  // Multiplying that with the no. threads in a block gives the no. of threads in a single row of grid.
  // Multiplying that with row number (threadIdx.y) and adding the x offset (X-part) gives the index.

  // threadID gives the index of the thread when the block of threads is flattened out into a 1D array.
  // We need this because we're using shared memory.
  int threadID = threadIdx.y*blockDim.x + threadIdx.x;
  int colourIndex;

  glm::vec3 shadedColour = glm::vec3 (0);

  if (index < rayPoolLength)
  {
    primArrayBlock [threadID] = primaryArrayOnDevice [index];
    rayPoolBlock [threadID] = rayPoolOnDevice [index];
	// We compute the index for the colour array separately since it represents a frame
    // and each index represents a pixel. If we don't, stream compaction would mess things up.
	colourIndex = rayPoolBlock [threadID].y*cam.resolution.x + rayPoolBlock [threadID].x;
    colourBlock [threadID] = colors [colourIndex];
    // colourBlock [threadID] therefore represents colour computed by ray through the pixel (x,y)

	interceptInfo theRightIntercept = getIntercept (geoms, objectCountInfo, rayPoolBlock [threadID], textureArray);		
	shadedColour += calcShade (theRightIntercept, Textures);

	if ((theRightIntercept.intrMaterial.emittance > 0) || (theRightIntercept.interceptVal < 0))
		primArrayBlock [threadID] = false;	// Ray did not hit anything or it hit light, so kill it.
	else
		calculateBSDF  (rayPoolBlock [threadID], 
						rayPoolBlock [threadID].origin + rayPoolBlock [threadID].direction * theRightIntercept.interceptVal, 
						theRightIntercept.intrNormal, glm::vec3 (0), AbsorptionAndScatteringProperties (), 
						index*time, theRightIntercept.intrMaterial.color, glm::vec3 (0), theRightIntercept.intrMaterial);
	
	if (glm::length (colourBlock [threadID]) > 0)
		colourBlock [threadID] *= shadedColour;			// Add computed shade to shadedColour.
	else
		colourBlock [threadID] = shadedColour;
  }

  __syncthreads ();
  
  // Copy the rayPool, Colour and Primary arrays back to global memory.
  if (index < rayPoolLength)
  {
	  primaryArrayOnDevice [index] = primArrayBlock [threadID];
	  rayPoolOnDevice [index] = rayPoolBlock [threadID];
	  colors [colourIndex] = colourBlock [threadID];
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
	int globalIndex = blockDim.x*blockIdx.x + threadIdx.x;

	if (globalIndex < fromLength)
		to [globalIndex] = (int)from [globalIndex];
}

__global__ void copyArray (ray *from, ray *to, int fromLength)
{
	int globalIndex = blockDim.x*blockIdx.x + threadIdx.x;

	if (globalIndex < fromLength)
		to [globalIndex] = from [globalIndex];
}

__global__ void copyArray (int *from, int *to, int fromLength)
{
	int globalIndex = blockDim.x*blockIdx.x + threadIdx.x;

	if (globalIndex < fromLength)
		to [globalIndex] = from [globalIndex];
}

// Kernel to do inclusive scan.
// Do NOT copy the results back in the same kernel as threads in other blocks might be still accessing the same location in 
// global memory, causing a read/write conflict. Use copyArray or cudaMemcpy.
__global__ void inclusiveScan (int *secondaryArray, int *tmpArray, int primArrayLength, int iteration)
{
	unsigned long	curIndex = blockDim.x*blockIdx.x + threadIdx.x;
	long	prevIndex = curIndex - floor (pow ((float)2.0, (float)(iteration-1)));

	if (curIndex < primArrayLength)
	{
		if (/*curIndex >= floor (pow ((float)2.0, (float)(iteration-1)))*/prevIndex >= 0)
			tmpArray [curIndex] = secondaryArray [curIndex] + secondaryArray [prevIndex];
	}
}

// Kernel to shift all elements of Array to the right. 
// The last element is thrown out in the process and the first element becomes 0.
// Can convert an inclusive scan result to an exclusive scan.
// Do NOT copy the results back in the same kernel as threads in other blocks might be still accessing the same location in 
// global memory, causing a read/write conflict and erroneous values. Use copyArray or cudaMemcpy.
__global__ void	shiftRight (int *Array, bool *primaryArray, int arrayLength)
{
	unsigned long	curIndex = blockDim.x*blockIdx.x + threadIdx.x;
	if (curIndex < arrayLength)
	{
		if (primaryArray [curIndex])
			Array [curIndex] = Array [curIndex] - 1;
	}
}


// Kernel to do stream compaction.
__global__ void	compactStream (ray *rayPoolOnDevice, ray *tempRayPool, bool *primaryArrayOnDevice, int *secondaryArray, int rayPoolLengthOnDevice)
{
	unsigned long	curIndex = blockDim.x*blockIdx.x + threadIdx.x;
	if (curIndex < rayPoolLengthOnDevice)
	{
		int secondArrayIndex = secondaryArray [curIndex];
		if (primaryArrayOnDevice [curIndex])
			tempRayPool [secondArrayIndex] = rayPoolOnDevice [curIndex];
	}
}

// This kernel will accumulate all the colours calculated in an iteration into the actual colour array.
__global__ void		accumulateIterationColour (glm::vec3* accumulator, glm::vec3* iterationColour, glm::vec2 resolution)
{
	int index = (blockDim.y*blockIdx.y + threadIdx.y) * resolution.x + 
				(blockDim.x*blockIdx.x + threadIdx.x);
	if (index < resolution.x*resolution.y)
		accumulator [index] += iterationColour [index];
}

// This kernel replaces the colours of the respective pixels of all the rays in the ray pool with noise (0,0,0)
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

//TODO: Done!
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, 
						material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, 
						mytexture* textures, int numberOfTextures){
  
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

  int sizeOfMaterialsArr = numberOfMaterials * (sizeof (material));
  cudaMalloc((void**)&materialColours, numberOfMaterials*sizeof(material));
  checkCUDAError ("Could not create Materials Array!: ");
  cudaMemcpy (materialColours, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  // Package all the texture data into an array of structures.
  mytexture *textureList = new mytexture [numberOfTextures];
  for (int i=0; i < numberOfTextures; i++)
  {
	  textureList [i].texelWidth = textures [i].texelWidth;
	  textureList [i].texelHeight = textures [i].texelHeight;

	  // Malloc for texture data (RGB values) and store the pointer to device memory in texels.
	  // So that when this structure is accessed from the device, the pointer reference is valid.
	  int nTexelElements = textureList [i].texelWidth*textureList [i].texelHeight;
	  cudaMalloc((void**)&textureList [i].texels, nTexelElements*sizeof(glm::vec3)); 
	  checkCUDAError ("Error allocing memory for texture data! ");
	  cudaMemcpy (textureList [i].texels, textures [i].texels, nTexelElements*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  }
  // Send the array of textures to the GPU.
  mytexture * textureArray = NULL;
  cudaMalloc((void**)&textureArray, numberOfTextures*sizeof(mytexture)); 
  checkCUDAError ("Error allocing memory for texture array! ");
  cudaMemcpy (textureArray, textureList, numberOfTextures*sizeof(mytexture), cudaMemcpyHostToDevice);
  delete [] textureList;

  glm::vec3 lightPosInBodySpace = glm::vec3 (0, -0.6, 0);

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  unsigned int nIterations = renderCam->iterations;

  time_t startTime = time (NULL);
  std::default_random_engine randomNumGen (hash (startTime));
  std::uniform_real_distribution<float> jitter ((float)0, (float)0.142);

  float movement = 3.0/nIterations;			// For motion blur.
  int nBounces = 6;
  int oneEighthDivisor = nIterations / 8;	// For antialiasing.
  int errCount = 0;
  // For each point sampled in the area light, launch the raytraceRay Kernel which will compute the diffuse, specular, ambient
  // and shadow colours. It will also compute reflected colours for reflective surfaces.
  for (int i = 0; i < nIterations; i ++)
  {
	  glm::vec3* cudaimage = NULL;
	  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	  cudaMemset (cudaimage, 0, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));


	  float zAdd = jitter (randomNumGen);
	  float xAdd = jitter (randomNumGen); 
	  glm::vec3 curLightSamplePos = lightPosInBodySpace;
	  
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
	  
	  // Create Ray Pool. 
	  int rayPoolLength = cam.resolution.x * cam.resolution.y;
	  ray *rayPoolOnDevice = NULL;
	  cudaMalloc ((void **)&rayPoolOnDevice, rayPoolLength * sizeof (ray));

	  // Primary Array		-> Array holding the true/false value specifying whether the ray is alive (true) or dead (false).
	  bool *primaryArrayOnHost = new bool [rayPoolLength];
	  memset (primaryArrayOnHost, true, rayPoolLength * sizeof(bool));
	  bool *primaryArrayOnDevice = NULL;
	  cudaMalloc ((void **)&primaryArrayOnDevice, rayPoolLength * sizeof (bool));

	  // Secondary Array	-> Array that will hold the indices of rays that are alive. Used in stream compaction.
	  int *secondaryArrayOnDevice = NULL;
	  cudaMalloc ((void **)&secondaryArrayOnDevice, rayPoolLength * sizeof (int));
	  int *secondaryArrayOnHost = new int [rayPoolLength];

	  // Launch createRayPool kernel to create the ray pool and populate the primary and secondary arrays.
	  fullBlocksPerGrid = dim3 ((int)ceil(float(cam.resolution.x)/threadsPerBlock.x), (int)ceil(float(cam.resolution.y)/threadsPerBlock.y));
	  createRayPool<<<fullBlocksPerGrid, threadsPerBlock>>> (rayPoolOnDevice, primaryArrayOnDevice, secondaryArrayOnDevice, cam, ProjectionParams);

	  dim3 threadsPerBlock1D (threadsPerBlock.x*threadsPerBlock.y);
	  // Iterate until nBounces: launch kernel to trace each ray bounce.
	  for (int j = 0; j < nBounces; ++j)
	  {
		// The core raytraceRay kernel launch
		fullBlocksPerGrid = dim3 ((int)ceil(float(rayPoolLength)/(threadsPerBlock.x*threadsPerBlock.y))); 
		raytraceRay<<<fullBlocksPerGrid, threadsPerBlock, threadsPerBlock.x*threadsPerBlock.y*(sizeof(glm::vec3) + sizeof (bool) + sizeof(ray))>>>
			((float)j+(i*nBounces), cam, j, cudaimage, cudageoms, materialColours, textureArray, primCounts, primaryArrayOnDevice, 
			rayPoolOnDevice, rayPoolLength);

		/// ----- CPU/GPU Hybrid Stream Compaction ----- ///
		// Scan is done on the CPU, the actual compaction happens on the GPU.
		// ------------------------------------------------------------------
		// Copy the primary array from device to host.
		cudaMemcpy (primaryArrayOnHost, primaryArrayOnDevice, rayPoolLength * sizeof (bool), cudaMemcpyDeviceToHost);

		// Exclusive scan.
		secondaryArrayOnHost [0] = 0;
		for (int k = 1; k < rayPoolLength; ++ k)
			secondaryArrayOnHost [k] = secondaryArrayOnHost [k-1] + primaryArrayOnHost [k-1];
		// This is because the compactStream kernel should run on the whole, uncompacted array.
		// We'll set this to rayPoolLength once compactStream has done its job.
		int compactedRayPoolLength = secondaryArrayOnHost [rayPoolLength-1] + primaryArrayOnHost [rayPoolLength-1];

		// Stream compaction. Compact the ray pool into tmpRayPool.
		ray *tmpRayPool = NULL;
		cudaMalloc ((void **)&tmpRayPool, rayPoolLength * sizeof (ray));
		cudaMemcpy (secondaryArrayOnDevice, secondaryArrayOnHost, rayPoolLength * sizeof (int), cudaMemcpyHostToDevice);
		compactStream<<<fullBlocksPerGrid, threadsPerBlock1D>>> (rayPoolOnDevice, tmpRayPool, primaryArrayOnDevice, secondaryArrayOnDevice, rayPoolLength);

		// Now set rayPoolLength to the compacted array size, compactedRayPoolLength.
		rayPoolLength = compactedRayPoolLength;

		// Copy the ray pool from tmpRayPool back into rayPoolOnDevice.
		copyArray<<<fullBlocksPerGrid, threadsPerBlock1D>>> (tmpRayPool, rayPoolOnDevice, rayPoolLength);
		cudaFree (tmpRayPool);

		// Set the primary array to all trues because all rays in the ray pool are alive, 
		// now that stream compaction has already happened.
		cudaMemset (primaryArrayOnDevice, true, rayPoolLength * sizeof (bool));
	  }
	  checkCUDAError ("One or more of the raytrace/stream compaction kernels failed. ");

	  // At this point, since stream compaction has already taken place,
	  // it means that rayPoolOnDevice contains only rays that are still alive.
	  fullBlocksPerGrid = dim3 ((int)ceil(float(rayPoolLength)/(threadsPerBlock.x*threadsPerBlock.y))); 
	  addNoise<<<fullBlocksPerGrid,threadsPerBlock>>>(cudaimage, rayPoolOnDevice, rayPoolLength, cam.resolution);

	  fullBlocksPerGrid = dim3 ((int)ceil(float(cam.resolution.x)/threadsPerBlock.x), (int)ceil(float(cam.resolution.y)/threadsPerBlock.y));
	  accumulateIterationColour<<<fullBlocksPerGrid, threadsPerBlock>>>(cudaFinalImage, cudaimage, cam.resolution);
	  checkCUDAError("accumulateIterationColour Kernel failed!");

	  cudaFree (rayPoolOnDevice);
	  cudaFree (primaryArrayOnDevice);
	  cudaFree (secondaryArrayOnDevice);
	  cudaFree (cudaimage);

	  rayPoolOnDevice = NULL;
	  primaryArrayOnDevice = NULL;
	  cudaimage = NULL;

	  delete [] primaryArrayOnHost;
	  delete [] secondaryArrayOnHost;

	  std::cout << "\rRendering.. " <<  ceil ((float)i/(nIterations-1) * 100) << " percent complete.";
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
	   cudaFree (materialColours);
   }
   if (textureArray)
   {
	   cudaFree (textureArray);
   }

   cudaFinalImage = NULL;
   cudageoms = NULL;
   materialColours = NULL;

 // make certain the kernel has completed
  cudaThreadSynchronize();
  
  delete [] geomList;

  checkCUDAError("Kernel failed!");
}
