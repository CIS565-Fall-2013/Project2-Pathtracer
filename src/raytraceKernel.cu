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
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>
#include "glm/glm.hpp"

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

//Function that does the initial raycast from the camera given a float defined pixel. Allows pixels to be defined with subpixel resolution easily.
//20% faster than provided code.
__host__ __device__ ray raycastFromCamera(glm::vec2 resolution, float x, float y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){

	ray r;
	r.origin = eye;
	glm::vec3 right = glm::cross(view, up);

	//float d = 1.0f; use a viewing plane of 1 distance 
	glm::vec3 pixel_location = /* d* */(view + (2*x/resolution.x-1)*right*glm::tan(glm::radians(fov.x)) 
		- (2*y/resolution.y-1)*up*glm::tan(glm::radians(fov.y)));

	r.direction = glm::normalize(pixel_location);

	return r;

}


//Scales the entire image by a float scale factor. Makes averaging trivial
__global__ void scaleImageIntensity(glm::vec2 resolution, glm::vec3* image, float sf)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x<resolution.x && y<resolution.y){
		image[index] = sf*image[index];
	}
}

//Takes each ray's pixel assignment and casts a randomized ray through the pixel
__global__ void raycastFromCameraKernel(int seed, int frame, cameraData cam, renderOptions rconfig, rayState* cudaraypool, int rayPoolSize)
{	

	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int rIndex = blockId * blockDim.x + threadIdx.x; 
	if(rIndex < rayPoolSize){
		thrust::default_random_engine rng(hash(seed));
		thrust::uniform_real_distribution<float> u01(0,1);

		int pixelIndex = cudaraypool[rIndex].index;
		if(pixelIndex >= 0){
			int x = pixelIndex % int(cam.resolution.x);
			int y = (pixelIndex - x)/int(cam.resolution.x);

			//Reset other fields
			cudaraypool[rIndex].T = glm::vec3(1,1,1);
			cudaraypool[rIndex].matIndex = -1;
			cudaraypool[rIndex].r =raycastFromCamera(cam.resolution, x+u01(rng)-0.5, y+u01(rng)-0.5, cam.position, cam.view, cam.up, cam.fov);
		}
	}

}

//Takes the number of rays requested by each pixel from the pool and allocates them stocastically from a single random number
//xi1 is a uniformly distributed random number from 0 to 1
__global__ void allocateRayPool(float xi1, renderOptions rconfig, cameraData cam, glm::vec3* cudaimage, rayState* cudaraypool, int numRays)
{
	//1D blocks and 2D grid

	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int rIndex = blockId * blockDim.x + threadIdx.x; 

	if(rIndex < numRays){//Thread index range check

		int numPixels = cam.resolution.x*cam.resolution.y;

		//Allocate all rays stochastically
		if(rconfig.stocasticRayAssignment){
			float P = float(numPixels)/numRays;//compute stochastic interval
			int start =  floor(xi1*numPixels);
			cudaraypool[rIndex].index = ((int)(start + P*rIndex) % numPixels);
		}else{
			if(rIndex < numPixels)
				cudaraypool[rIndex].index = rIndex;
			else
				cudaraypool[rIndex].index = -1;
		}
	}
}

__global__ void displayRayCounts(cameraData cam, renderOptions rconfig, glm::vec3* cudaimage, rayState* cudaraypool, int numRays, float maxScale)
{
	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int rIndex = blockId * blockDim.x + threadIdx.x; 

	if(rIndex < numRays){//Thread index range check
		int pixelIndex = cudaraypool[rIndex].index;
		if(pixelIndex >= 0)
		{
			float scale = clamp(1.0f/maxScale, 0.0f, 1.0f);
			cudaimage[pixelIndex] += scale*glm::vec3(1,1,1);
		}
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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, float scaleFactor){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x<=resolution.x && y<=resolution.y){

		glm::vec3 color;      
		color.x = image[index].x*255.0*scaleFactor;
		color.y = image[index].y*255.0*scaleFactor;
		color.z = image[index].z*255.0*scaleFactor;

		//Clamp
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

__global__ void traceRayFirstHit(cameraData cam, renderOptions rconfig, float time, int bounce, glm::vec3* colors, 
						 rayState* raypool, int numRays, staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials)
{
	//Compute ray index
	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int rIndex = blockId * blockDim.x + threadIdx.x; 


	//Pixel index of -1 indicates the ray's contribution has been recorded and is no longer in flight
	if(rIndex < numRays)
	{
		//Thread has a ray, check if ray has a pixel
		int pixelIndex = raypool[rIndex].index;
		if(pixelIndex >= 0 && pixelIndex < (int)cam.resolution.x*(int)cam.resolution.y)
		{
			//valid pixel index
			ray r = raypool[rIndex].r;

			float MAX_DEPTH = 100000000000000000;
			float depth = MAX_DEPTH;
			glm::vec3 color = glm::vec3(0,0,0);
			for(int i=0; i<numberOfGeoms; i++){
				glm::vec3 intersectionPoint;
				glm::vec3 intersectionNormal;
				if(geoms[i].type==SPHERE){
					depth = sphereIntersectionTest(geoms[i], r, intersectionPoint, intersectionNormal);
				}else if(geoms[i].type==CUBE){
					depth = boxIntersectionTest(geoms[i], r, intersectionPoint, intersectionNormal);
				}else if(geoms[i].type==MESH){
					//triangle tests go here
				}else{
					//lol?
				}
				if(depth<MAX_DEPTH && depth>-EPSILON){
					MAX_DEPTH = depth;
					color = materials[geoms[i].materialid].color;
				}

			}
			colors[pixelIndex] += color;
		}	
	}
}


//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void traceRay(cameraData cam, renderOptions rconfig, float time, int bounce, glm::vec3* colors, 
						 rayState* raypool, int numRays, staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials)
{
	//Compute ray index
	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int rIndex = blockId * blockDim.x + threadIdx.x; 


	//Pixel index of -1 indicates the ray's contribution has been recorded and is no longer in flight
	if(rIndex < numRays)
	{
		//Thread has a ray, check if ray has a pixel
		int pixelIndex = raypool[rIndex].index;
		if(pixelIndex >= 0 && pixelIndex < (int)cam.resolution.x*(int)cam.resolution.y)
		{
			//valid pixel index
			ray r = raypool[rIndex].r;

			float MAX_DEPTH = 100000000000000000;
			float depth = MAX_DEPTH;
			glm::vec3 color = glm::vec3(0,0,0);
			for(int i=0; i<numberOfGeoms; i++){
				glm::vec3 intersectionPoint;
				glm::vec3 intersectionNormal;
				if(geoms[i].type==SPHERE){
					depth = sphereIntersectionTest(geoms[i], r, intersectionPoint, intersectionNormal);
				}else if(geoms[i].type==CUBE){
					depth = boxIntersectionTest(geoms[i], r, intersectionPoint, intersectionNormal);
				}else if(geoms[i].type==MESH){
					//triangle tests go here
				}else{
					//lol?
				}
				if(depth<MAX_DEPTH && depth>-EPSILON){
					MAX_DEPTH = depth;
					color = materials[geoms[i].materialid].color;
				}

			}
			colors[pixelIndex] += color;
		}	
	}
}


//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam,  renderOptions* rconfig, int frame, int iterations, int frameFilterCounter, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){

	int traceDepth = rconfig->traceDepth; //determines how many bounces the raytracer traces
	int numPixels = renderCam->resolution.x*renderCam->resolution.y;
	int rayPoolSize = (int) ceil(float(numPixels)*rconfig->rayPoolSize);

	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlockByPixel(tileSize, tileSize);
	dim3 fullBlocksPerGridByPixel((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

	// Set up a 2D grid
	// Fill up rows before adding more
	//TODO: Improve resource allocation. Slipping over once will create a lot of wasted blocks
	int blockSize = 64;
	dim3 threadsPerBlockByRay(blockSize);
	int blockCount = (int)ceil(float(rayPoolSize)/float(blockSize));

	dim3 fullBlocksPerGridByRay;
	int maxGridX = 65535;//TODO: get this dynamically
	if(blockCount > maxGridX){
		fullBlocksPerGridByRay = dim3(maxGridX, (int)ceil( blockCount / float(maxGridX)));
	}else{
		fullBlocksPerGridByRay = dim3(blockCount);
	}

	//send image to GPU
	glm::vec3* cudaimage = NULL;
	cudaMalloc((void**)&cudaimage, numPixels*sizeof(glm::vec3));
	cudaMemcpy( cudaimage, renderCam->image, numPixels*sizeof(glm::vec3), cudaMemcpyHostToDevice);

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

	///Allocations
	staticGeom* cudageoms = NULL;
	cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
	cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

	material* cudamaterials = NULL;
	cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
	cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

	rayState* cudaraypool = NULL;
	cudaMalloc((void**)&cudaraypool, rayPoolSize*sizeof(rayState));


	//package camera
	cameraData cam;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->positions[frame];
	cam.view = renderCam->views[frame];
	cam.up = renderCam->ups[frame];
	cam.fov = renderCam->fov;

	///Prep image
	if(!rconfig->frameFiltering || frameFilterCounter <= 1)
	{
		clearImage<<<fullBlocksPerGridByPixel, threadsPerBlockByPixel>>>(renderCam->resolution, cudaimage);
		frameFilterCounter = 1;
	}
	//else{
	//	scaleImageIntensity<<<fullBlocksPerGridByPixel, threadsPerBlockByPixel>>>(renderCam->resolution, cudaimage, (float)(frameFilterCounter-1));
	//}


	//Figure out which rays should go to which pixels.
	thrust::default_random_engine rng(hash(iterations*frameFilterCounter+iterations));
	thrust::uniform_real_distribution<float> u01(0,1);
	allocateRayPool<<<fullBlocksPerGridByRay, threadsPerBlockByRay>>>(u01(rng), *rconfig, cam, cudaimage, cudaraypool, rayPoolSize);

	switch(rconfig->mode)
	{
	case PATHTRACE:
		raycastFromCameraKernel<<<fullBlocksPerGridByRay, threadsPerBlockByRay>>>(iterations, frame, cam, *rconfig, cudaraypool, rayPoolSize);

		for(int bounce = 0; bounce < traceDepth; bounce++)
		{
			traceRay<<<fullBlocksPerGridByRay, threadsPerBlockByRay>>>(cam, *rconfig, iterations, bounce, cudaimage, 
				cudaraypool, rayPoolSize, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials);

		}

		break;
	case RAYCOUNT_DEBUG:
		displayRayCounts<<<fullBlocksPerGridByRay, threadsPerBlockByRay>>>(cam, *rconfig, cudaimage, cudaraypool, rayPoolSize,ceil(float(rayPoolSize)/numPixels));
		break;
	case FIRST_HIT_DEBUG:
		raycastFromCameraKernel<<<fullBlocksPerGridByRay, threadsPerBlockByRay>>>(iterations, frame, cam, *rconfig, cudaraypool, rayPoolSize);

		traceRayFirstHit<<<fullBlocksPerGridByRay, threadsPerBlockByRay>>>(cam, *rconfig, iterations, 0, cudaimage, 
				cudaraypool, rayPoolSize, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials);

		break;
	}


	//if(rconfig->frameFiltering)
	//{
	//	scaleImageIntensity<<<fullBlocksPerGridByPixel, threadsPerBlockByPixel>>>(renderCam->resolution, cudaimage, 1.0f/(frameFilterCounter));
	//}


	//retrieve image from GPU before drawing overlays and writing to screen
	cudaMemcpy( renderCam->image, cudaimage, numPixels*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	//TODO: Draw any debug overlays here



	//Draw to screen
	sendImageToPBO<<<fullBlocksPerGridByPixel, threadsPerBlockByPixel>>>(PBOpos, renderCam->resolution, cudaimage, 1.0f/float(frameFilterCounter));


	//free up stuff, or else we'll leak memory like a madman
	cudaFree( cudaimage );
	cudaFree( cudageoms );
	cudaFree( cudamaterials );
	cudaFree( cudaraypool );
	delete [] geomList;

	// make certain the kernel has completed 
	cudaThreadSynchronize();
	checkCUDAError("Kernel failed!");
}
