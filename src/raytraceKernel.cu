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

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__global__ void raycastFromCameraKernel(glm::vec2 resolution, float time, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov, ray* rays){

  // pixel index for ray
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  //establish "right" camera direction
  glm::normalize(eye); glm::normalize(view);
  glm::vec3 right = glm::normalize(glm::cross(up, view));
  
  // calculate P1 and P2 in both x and y directions
  glm::vec3 image_center = eye + view;
  glm::vec3 P1_X = image_center - tan((float)4.0*fov.x)*right;
  glm::vec3 P2_X = image_center + tan((float)4.0*fov.x)*right;
  glm::vec3 P1_Y = image_center - tan((float)4.0*fov.y)*up;
  glm::vec3 P2_Y = image_center + tan((float)4.0*fov.y)*up;
  
  glm::vec3 bottom_left  = P1_X + (P1_Y - image_center);
  glm::vec3 bottom_right = P2_X + (P1_Y - image_center);
  glm::vec3 top_left     = P1_X + (P2_Y - image_center);

  glm::vec3 imgRight = bottom_right - bottom_left;
  glm::vec3 imgUp    = top_left - bottom_left;

  // supersample the pixels by taking a randomly offset ray in each iteration
  glm::vec3 random_offset = generateRandomNumberFromThread(resolution, time, x, y);
  float x_offset = random_offset.x;
  float y_offset = random_offset.y;
  glm::vec3 img_point = bottom_left + ((float)x + x_offset)/(float)resolution.x*imgRight + ((float)y + y_offset)/(float)resolution.y*imgUp;
  glm::vec3 direction = glm::normalize(img_point - eye); 

  // return value
  ray r;
  r.x = x;
  r.y = y;
  r.alive = true;
  r.origin = eye; 
  r.direction = direction;
  r.coeff = glm::vec3(1.0f, 1.0f, 1.0f);
  r.currentIOR = 1.0;

  rays[index] = r;
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

// perform exclusive scan
__global__ void createTempArray(ray* R_in, int* R_temp) {
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (R_in[index].alive == true)	{ R_temp[index] = 1; }
	else							{ R_temp[index] = 0; }

	__syncthreads();
}

// perform exclusive scan
__global__ void inclusiveScan(int* R_scan_in, int* R_scan_out, int depth) {
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
    if (index >= pow(2.0, depth-1))
		R_scan_out[index] = R_scan_in[index - (int)pow(2.0, depth-1)] + R_scan_in[index];
	else
		R_scan_out[index] = R_scan_in[index];
		
	__syncthreads();
}

// shift from inclusive to exclusive scan and store the numberOfRays
__global__ void inclusive2exclusive(int* R_inclusive, int* R_exclusive, int *numberOfRays) {
	
	//*numberOfRays /= 2.0;

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (index == 0)	{ R_exclusive = 0; }
	else			{ R_exclusive[index] = R_inclusive[index-1]; }
}

// shift from inclusive to exclusive scan and store the numberOfRays
__global__ void scatter(int* R_temp, int* R_scan, ray* R_in, ray* R_out) {
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (R_temp[index] == 1)
		R_out[R_scan[index]] = R_in[index];
}

// Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, int rayDepth, glm::vec3* colors, staticGeom* geoms, 
							int numberOfGeoms, material* materials, int iterations, ray* rays) {

	// Find index of pixel and create empty color vector
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Get initial ray from camera through this position
	ray inRay = rays[index];
	inRay.origin = inRay.origin + (float)ERROR*inRay.direction;

	// Allocate secondary ray 
	ray outRay;
  
	// Luminance value to be returned for ray
	int colorIndex = inRay.x + (inRay.y * resolution.x);
	
	// Return values for the intersection test
	R3Intersection intersection;

	// Find the closest geometry intersection along the ray
	float t;
	float min_t = -1.0;
	for (int i = 0; i < numberOfGeoms; i++) {
		staticGeom geom = geoms[i];
		t = geomIntersectionTest(geom, inRay, intersection.point, intersection.normal);
		if ((t > ERROR) && (t < min_t || min_t < 0.0)) {
			min_t = t;
			intersection.material = &materials[geom.materialid];
		}
	}
	
	if (min_t == -1.0) {
		outRay.alive = false;
		outRay.coeff = glm::vec3(0,0,0);
	}
	
	else if(intersection.material->emittance > 0)
	{
		outRay.alive = false;
		outRay.coeff = (inRay.coeff * intersection.material->color * intersection.material->emittance);
		colors[colorIndex] += outRay.coeff / (float)iterations;
	}

	else {
		
		int BSDF = calculateBSDF(inRay, intersection.point, intersection.normal, intersection.material, float(time*rayDepth));
		switch (BSDF) {

			case DIFFUSE:
			{
				glm::vec3 rand = generateRandomNumberFromThread(resolution, time*rayDepth, inRay.x, inRay.y);
				outRay.direction = calculateRandomDirectionInHemisphere(intersection.normal, rand.x, rand.y);
				outRay.coeff = inRay.coeff * intersection.material->color;
				outRay.currentIOR = inRay.currentIOR;
			}
			break;

			case SPECULAR:
			{
				outRay.direction = calculateReflectionDirection(intersection.normal, inRay.direction);
				outRay.currentIOR = inRay.currentIOR;
				outRay.coeff = inRay.coeff;
			}
			break;

			case TRANSMIT:
			{
				float incidentIOR = inRay.currentIOR;
				float transmittedIOR;
				if (glm::dot(inRay.direction, intersection.normal) > 0)
					transmittedIOR = 1.0;
				else
					transmittedIOR = intersection.material->indexOfRefraction;

				outRay.direction = calculateTransmissionDirection(intersection.normal, inRay.direction, incidentIOR, transmittedIOR);
				outRay.coeff = inRay.coeff;
			}
			break;
		}

		// Constant properties of outgoing ray, regardless of switch case
		glm::vec3 ro = glm::vec3(intersection.point);
		outRay.origin = ro;
		outRay.x = inRay.x;
		outRay.y = inRay.y;
		outRay.alive = true;
	}
	rays[index] = outRay;
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
	// determines how many bounces the raytracer traces
	int traceDepth = 6;		

	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
	//send image to GPU
	glm::vec3* cudaimage = NULL;
	cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
	//package geometry and send to GPU
	staticGeom* geomList = new staticGeom[numberOfGeoms];
	for(int i=0; i<numberOfGeoms; i++){
		staticGeom newStaticGeom;
		newStaticGeom.objectid = geoms[i].objectid;
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
  
	//package materials and send to GPU
	material* materialList = new material[numberOfMaterials];
	for (int i=0; i<numberOfMaterials; i++){
		material newMaterial;
		newMaterial.color = materials[i].color;
		newMaterial.specularExponent = materials[i].specularExponent;
		newMaterial.specularColor = materials[i].specularColor;
		newMaterial.hasReflective = materials[i].hasReflective;
		newMaterial.hasRefractive = materials[i].hasRefractive;
		newMaterial.indexOfRefraction = materials[i].indexOfRefraction;
		newMaterial.hasScatter = materials[i].hasScatter;
		newMaterial.absorptionCoefficient = materials[i].absorptionCoefficient;
		newMaterial.reducedScatterCoefficient = materials[i].reducedScatterCoefficient;
		newMaterial.emittance = materials[i].emittance;
		materialList[i] = newMaterial;
	}
  
	material* cudamaterials = NULL;
	cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
	cudaMemcpy( cudamaterials, materialList, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);
  
	// package camera
	cameraData cam;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->positions[frame];
	cam.view = renderCam->views[frame];
	cam.up = renderCam->ups[frame];
	cam.fov = renderCam->fov;

	// create array of rays to feed to kernel call
	int numberOfRays = cam.resolution.x * cam.resolution.y;
	ray* rayList = new ray[numberOfRays];
	ray* cudarays = NULL;
	cudaMalloc((void**)&cudarays, numberOfRays*sizeof(ray));
	cudaMemcpy(cudarays, rayList, numberOfRays*sizeof(ray), cudaMemcpyHostToDevice);
	raycastFromCameraKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(cam.resolution, (float)iterations, cam.position, cam.view, cam.up, cam.fov, cudarays);

	// kernel launching loop
	for (int d = 0; d < traceDepth; d++) {

		// determine block sizes & threads per block
		int raytraceTileSize = 128;
		dim3 raytraceThreadsPerBlock(raytraceTileSize);
		dim3 raytraceFullBlocksPerGrid((int)ceil(float(numberOfRays)/float(raytraceTileSize)));
  
		// kernel call to trace all active rays
		raytraceRay<<<raytraceFullBlocksPerGrid, raytraceThreadsPerBlock>>>(renderCam->resolution, (float)iterations, d+1, cudaimage, cudageoms, numberOfGeoms, cudamaterials, renderCam->iterations, cudarays);
		
		// create array copy for stream compaction
		ray* cudaR_out = NULL;
		cudaMalloc((void**)&cudaR_out, numberOfRays*sizeof(ray));
		
		// create temp array for stream compaction
		int* cudaR_temp = NULL;
		cudaMalloc((void**)&cudaR_temp, numberOfRays*sizeof(int));

		// create scan array for stream compaction
		int* cudaR_scan_in  = NULL;
		int* cudaR_scan_out = NULL;
		cudaMalloc((void**)&cudaR_scan_in,  numberOfRays*sizeof(int));
		cudaMalloc((void**)&cudaR_scan_out, numberOfRays*sizeof(int));
		
		// populate the temp array and copy it to the initial state of the scan array
		createTempArray<<<raytraceFullBlocksPerGrid, raytraceThreadsPerBlock>>>(cudarays, cudaR_temp);
		cudaMemcpy(cudaR_scan_in, cudaR_temp, numberOfRays*sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(cudaR_scan_out, cudaR_temp, numberOfRays*sizeof(int), cudaMemcpyDeviceToDevice);
		
		// kernel call to perform exclusive scan
		for (int p = 1; p <= ceil(log2((double)numberOfRays)); p++) {
			inclusiveScan<<<raytraceFullBlocksPerGrid, raytraceThreadsPerBlock>>>(cudaR_scan_in, cudaR_scan_out, p);
			cudaMemcpy(cudaR_scan_in, cudaR_scan_out, numberOfRays*sizeof(int), cudaMemcpyDeviceToDevice);
		}
		
		int* numRays = new int[1];
		cudaMemcpy(numRays, cudaR_scan_in+(numberOfRays-2), sizeof(int), cudaMemcpyDeviceToHost);
		numberOfRays = numRays[0];

		inclusive2exclusive<<<raytraceFullBlocksPerGrid, raytraceThreadsPerBlock>>>(cudaR_scan_in, cudaR_scan_out, &numberOfRays);
		scatter<<<raytraceFullBlocksPerGrid, raytraceThreadsPerBlock>>>(cudaR_temp, cudaR_scan_out, cudarays, cudaR_out);
		
		// perform stream compaction on array of rays, Rout array
		cudaMemcpy(cudarays, cudaR_out, numberOfRays*sizeof(ray), cudaMemcpyDeviceToDevice);
		
		delete numRays;
		cudaFree( cudaR_out );
		cudaFree( cudaR_temp );
		cudaFree( cudaR_scan_in );
		cudaFree( cudaR_scan_out );
	}
	
	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);
  
	// retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  
	// free up stuff, or else we'll leak memory like a madman
	cudaFree( cudaimage );
	cudaFree( cudageoms );
	cudaFree( cudamaterials );
	cudaFree( cudarays );
	delete geomList;
	delete materialList;
	delete rayList;


	// make certain the kernel has completed
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}
