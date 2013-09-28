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



__host__ __device__ int estimateNumSamples(int x, int y, glm::vec2 resolution, glm::vec3* colors, renderOptions rconfig)
{
	//TODO implement more flexible options

	//Compute RMSD in local window 3x3
	int n = 0;
	glm::vec3 accumulator = glm::vec3(0,0,0);
	for(int yi = MAX(0,y - 1); yi <= MIN(y + 1, resolution.y-1); ++yi)
	{
		for(int xi = MAX(0,x - 1); xi <= MIN(x + 1, resolution.x-1); ++xi)
		{
			++n;
			int index = xi + (yi * resolution.x);
			accumulator += colors[index];
		}
	}

	glm::vec3 mean = accumulator/(float)n;
	accumulator = glm::vec3(0,0,0);


	for(int yi = MAX(0,y - 1); yi <= MIN(y + 1, resolution.y-1); ++yi)
	{
		for(int xi = MAX(0,x - 1); xi <= MIN(x + 1, resolution.x-1); ++xi)
		{
			int index = xi + (yi * resolution.x);
			accumulator += (colors[index]-mean)*(colors[index]-mean);
		}
	}

	glm::vec3 RMSD = glm::sqrt(accumulator/(float)n);

	if(RMSD.x > rconfig.aargbThresholds.x || RMSD.y > rconfig.aargbThresholds.y || RMSD.z > rconfig.aargbThresholds.z)
	{
		return rconfig.maxSamplesPerPixel;
	}
	else
	{
		return 1;
	}

}

__global__ void requestRays(glm::vec2 resolution, renderOptions rconfig, glm::vec3* colors, int* numRays)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	numRays[index] = estimateNumSamples(x,y,resolution,colors, rconfig);
}



__global__ void scaleImageIntensity(glm::vec2 resolution, glm::vec3* image, int* scaleFactor, bool scaleUp)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	float sf = scaleFactor[index];

	if(!scaleUp)
		sf = 1.0f/scaleFactor[index];

	if(x<resolution.x && y<resolution.y){
		image[index] = sf*image[index];
	}
}

//Takes the number of rays requested by each pixel from the pool and allocates them stocastically from a single random number
//scannedRayRequests is an array of ints containing the results of an inclusive scan
//xi1 is a uniformly distributed random number from 0 to 1
__global__ void allocateRayPool(float xi1, renderOptions rconfig, cameraData cam, glm::vec3* cudaimage, rayState* cudaraypool, int* scannedRayRequests, int*  raytotals, int numRays)
{
	//1D blocks and 2D grid

	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int rIndex = blockId * blockDim.x + threadIdx.x; 

	if(rIndex < numRays){//Thread index range check
		int numPixels = cam.resolution.x*cam.resolution.y;

		if(rconfig.forceOnePerPixel){
			if(rIndex < numPixels)
			{
				//Ensure that each pixel gets at least one ray
				cudaraypool[rIndex].index = rIndex;
			}else{
				//Use stochastic universal sampling on remaining rays

				int spares = numRays-numPixels;
				float P = float(scannedRayRequests[numPixels-1])/spares;//compute stochastic interval
				float start = xi1*P;
				cudaraypool[rIndex].index = ((int)(floor(start+P*(rIndex-numPixels))) % numPixels);
			}

		}else{
			//Allocate all rays stochastically
			float P = float(scannedRayRequests[numPixels-1])/numRays;//compute stochastic interval
			float start = xi1*P;
			cudaraypool[rIndex].index = ((int)(floor(start+P*(rIndex))) % numPixels);
			
		}
		atomicAdd(&raytotals[cudaraypool[rIndex].index], 1); 

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
			//TODO improve parallelism by removing atomic adds?
			atomicAdd(&cudaimage[pixelIndex].x, scale);
			atomicAdd(&cudaimage[pixelIndex].y, scale);
			atomicAdd(&cudaimage[pixelIndex].z, scale);
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


//Kernel that clears an separate buffer of ints the same size of the image
__global__ void clearIntBuffer(glm::vec2 resolution, int* buf){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y){
		buf[index] = 0;
	}
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x<=resolution.x && y<=resolution.y){

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

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, float bounce, cameraData cam, int rayDepth, glm::vec3* colors, 
							staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials)
{
	//Compute pixel and index
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	ray r = raycastFromCamera(resolution, x, y, cam.position, cam.view, cam.up, cam.fov);

	if((x<=resolution.x && y<=resolution.y)){

		float MAX_DEPTH = 100000000000000000;
		float depth = MAX_DEPTH;

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
				colors[index] = materials[geoms[i].materialid].color;
			}
		}


		//colors[index] = generateRandomNumberFromThread(resolution, time, x, y);
	}	

}


//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam,  renderOptions* rconfig, int frame, int iterations, int frameFilterCounter, int* raytotals, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){

	int traceDepth = rconfig->traceDepth; //determines how many bounces the raytracer traces
	int numPixels = renderCam->resolution.x*renderCam->resolution.y;
	int rayPoolSize = (int) ceil(float(numPixels)*rconfig->rayPoolSize);

	if(rconfig->forceOnePerPixel && rayPoolSize < numPixels)
	{
		//Error
		printf("Error Not Enough Rays in Pool\n");
		return;
	}

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

	staticGeom* cudageoms = NULL;
	cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
	cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

	material* cudamaterials = NULL;
	cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
	cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

	rayState* cudaraypool = NULL;
	cudaMalloc((void**)&cudaraypool, rayPoolSize*sizeof(rayState));

	//Array to hold samples per pixel (for adaptive anti-aliasing)
	int* cudarequestedrays = NULL;
	cudaMalloc((void**)&cudarequestedrays, numPixels*sizeof(int));

	//Array to hold accumulated ray totals
	int* cudaraytotals = NULL;
	cudaMalloc((void**)&cudaraytotals, numPixels*sizeof(int));
	cudaMemcpy( cudaraytotals, raytotals, numPixels*sizeof(int), cudaMemcpyHostToDevice);

	//package camera
	cameraData cam;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->positions[frame];
	cam.view = renderCam->views[frame];
	cam.up = renderCam->ups[frame];
	cam.fov = renderCam->fov;

	if(!rconfig->frameFiltering || frameFilterCounter <= 1)
	{
		clearIntBuffer<<<fullBlocksPerGridByPixel, threadsPerBlockByPixel>>>(renderCam->resolution, cudaraytotals);
		clearImage<<<fullBlocksPerGridByPixel, threadsPerBlockByPixel>>>(renderCam->resolution, cudaimage);
		
	}else{
		scaleImageIntensity<<<fullBlocksPerGridByPixel, threadsPerBlockByPixel>>>(renderCam->resolution, cudaimage, cudaraytotals, true);
	}
	
	//Allocate rays
	requestRays<<<fullBlocksPerGridByPixel, threadsPerBlockByPixel>>>(renderCam->resolution, *rconfig, cudaimage, cudarequestedrays);
	
	//Perform scan
	inclusive_scan_sum(cudarequestedrays, cudarequestedrays, numPixels);
	
	//Figure out which rays should go to which pixels.

	thrust::default_random_engine rng(hash(iterations));
	thrust::uniform_real_distribution<float> u01(0,1);
	allocateRayPool<<<fullBlocksPerGridByRay, threadsPerBlockByRay>>>(u01(rng), *rconfig, cam, cudaimage, cudaraypool, cudarequestedrays, cudaraytotals, rayPoolSize);
	
	if(rconfig->mode == RAYCOUNT_DEBUG)
	{
		displayRayCounts<<<fullBlocksPerGridByRay, threadsPerBlockByRay>>>(cam, *rconfig, cudaimage, cudaraypool, rayPoolSize,2.0f);
	}else{


	}
	

	if(rconfig->frameFiltering)
	{
		scaleImageIntensity<<<fullBlocksPerGridByPixel, threadsPerBlockByPixel>>>(renderCam->resolution, cudaimage, cudaraytotals, false);
	}

	
	//retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, numPixels*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	cudaMemcpy( raytotals, cudaraytotals, numPixels*sizeof(int), cudaMemcpyDeviceToHost);

	sendImageToPBO<<<fullBlocksPerGridByPixel, threadsPerBlockByPixel>>>(PBOpos, renderCam->resolution, cudaimage);


	//free up stuff, or else we'll leak memory like a madman
	cudaFree( cudaimage );
	cudaFree( cudageoms );
	cudaFree( cudamaterials );
	cudaFree( cudaraypool );
	cudaFree( cudaraytotals);
	cudaFree( cudarequestedrays );
	delete [] geomList;

	// make certain the kernel has completed 
	cudaThreadSynchronize();
	checkCUDAError("Kernel failed!");
}
