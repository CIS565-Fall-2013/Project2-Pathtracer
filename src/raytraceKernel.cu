// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/copy.h>

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
#include <time.h>  
#include <windows.h>



#define LIGHT_NUM 1
#define ANTI_NUM 1
#define STREAMCOMPATION 0
#define TITLESIZE 32

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
	system("pause");
    exit(EXIT_FAILURE); 
  }
} 

struct rayPixel
{
	ray r;
	int index;
	bool isDone;
	bool isFirst;
	int x;
	int y;
};

struct is_done
{
    __host__ __device__
    bool operator()(const rayPixel r)
    {
		return r.isDone;
    }
};

thrust::host_vector<rayPixel> globalRayPool;



//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

__host__ __device__ int generateRandomNumber(glm::vec2 resolution, float time, int x, int y)
{
	int index = x + (y * resolution.x);
   
	thrust::default_random_engine rng(hash(index*time));
	thrust::uniform_real_distribution<float> u01(0,1);

	return (int)(u01(rng)) * 10;
}

__host__ __device__ int generateRandomNumber(float time)
{
	thrust::default_random_engine rng(hash(time));
	thrust::uniform_real_distribution<float> u01(0,1);

	return (float)(u01(rng));
}


//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, float x, float y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  
	//printf("%f  ", x);

	glm::vec3 A = glm::cross(view, up);
	glm::vec3 B = glm::cross(A, view);
	glm::vec3 M = eye + view;

	float absC = glm::length(view);
	float absA = glm::length(A);
	float absB = glm::length(B);

	glm::vec3 H = (A * absC * (float)tan(fov.x*(PI/180))) / absA;
	glm::vec3 V = (B * absC * (float)tan(fov.y*(PI/180))) / absB;

	glm::vec3 P = M + (2 * (x / (float)(resolution.x - 1)) - 1) * H + (1 - 2 * (y / (float)(resolution.y - 1))) * V;
	glm::vec3 D = (P - eye) / glm::length(P - eye);

	ray r;
	r.origin = eye;
	r.direction = D;	
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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, float time){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;

      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

	  color /= time;//sijie
	  
      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
	  //printf(" rgb %f %f %f ", color.x, color.y, color.z);

      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}


__global__ void sendImageToPBOSC(uchar4* PBOpos, glm::vec3* image, float time, int poolSize, glm::vec3* preimage)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y; 
  
  if(x<poolSize){

	  int index = x;
      glm::vec3 color;

	  /*color.x = ((image[index].x + preimage[index].x) / time) * 255.0;
	  color.y = ((image[index].y + preimage[index].y) / time) * 255.0;
	  color.z = ((image[index].z + preimage[index].z) / time) * 255.0;*/

	  color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

	  color /= time;//sijie
	  
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


__global__ void addImageColor(glm::vec3* currImage, glm::vec3* preImage, glm::vec2 resolution)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
  
	if(x<=resolution.x && y<=resolution.y)
	{
		currImage[index] = (currImage[index] + preImage[index]);
	}
}

__host__ __device__  bool ShadowRayUnblock(staticGeom* geoms, int numberofGeoms, glm::vec3 intersectionPoint, int lightIndex, int geomIndex, glm::vec3 normal, glm::vec3 lightPos)
{	
	ray r;	
	r.direction = glm::normalize(lightPos - intersectionPoint);
	r.origin = intersectionPoint + .1f * r.direction;

	//surface parallel to the light
	if(glm::dot(r.direction, normal) < 1e-5 && glm::dot(r.direction, normal) > -1e-5)
		return false;

	glm::vec3 tempInterPoint, tempNormal;	
	float t = FLT_MAX;
	int intersectIndex = -1;
	for(int i = 0; i < numberofGeoms; ++i)
	{		
		float temp;
		if(geoms[i].type == CUBE)
			temp = boxIntersectionTest(geoms[i], r, tempInterPoint, tempNormal);
		else
			temp = sphereIntersectionTest(geoms[i], r, tempInterPoint, tempNormal);

		if(temp < t && temp != -1.0f)
		{
			t = temp;
			intersectIndex = i;
		}
	}

	if(intersectIndex == lightIndex)
		return true;
	else
		return false;
}

__host__ __device__ bool checkIntersect(staticGeom* geoms, int numberOfGeoms, ray r, glm::vec3& intersectionPoint, glm::vec3& normal, int& geomIndex)
{
	float t = FLT_MAX;
	//int geomIndex = 0;
	for(int i = numberOfGeoms - 1; i >= 0; --i)
	{	
		float temp;
		//if(geoms[i].materialid != 7 || geoms[i].materialid != 8)
		if(geoms[i].type == SPHERE)
			temp = sphereIntersectionTest(geoms[i], r, intersectionPoint, normal);
		else if(geoms[i].type == CUBE)
			temp = boxIntersectionTest(geoms[i], r, intersectionPoint, normal);

		if(temp != -1.0f && temp < t)
		{
			t = temp;
			geomIndex = i;		
		}
	}	

	if(t != FLT_MAX){
		//get the intersection point and normal
		if(geoms[geomIndex].type == SPHERE)
			sphereIntersectionTest(geoms[geomIndex], r, intersectionPoint, normal);
		else
			boxIntersectionTest(geoms[geomIndex], r, intersectionPoint, normal);
		return true;
	}
	else
	{
		return false;
	}
}

__host__ __device__ void pathTraceRecursive(ray r, int rayDepth, staticGeom* geoms, int numberOfGeoms, material* materials, glm::vec3& color, 
											cameraData cam, float time, int x, int y, glm::vec3* lightPos, int lightIndex)
{
	if(rayDepth >= 3)
	{
		return;
	}

	glm::vec3 intersectionPoint, normal;
	int geomIndex;
	if(!checkIntersect(geoms, numberOfGeoms, r, intersectionPoint, normal, geomIndex))
	{
		color = glm::vec3(0,0,0);
		return;
	}

	material currMaterial = materials[geoms[geomIndex].materialid];

	glm::vec3 ran = generateRandomNumberFromThread(cam.resolution, time, x, y);
	if(ran.z < 0.5f)
	{
		color = currMaterial.emittance * currMaterial.color;
		return;
	}
	else{
		ray newRay;
		newRay.direction = calculateRandomDirectionInHemisphere(normal, ran.x, ran.y);
		newRay.origin = intersectionPoint + 0.01f * newRay.direction;

		float cosTheta = glm::dot(newRay.direction, normal);
		float BDRF = 2.0f * 0.4f * cosTheta;
		pathTraceRecursive(newRay, rayDepth + 1, geoms, numberOfGeoms, materials, color, cam, time, x, y, lightPos, lightIndex);
		float diffuseTerm;
		diffuseTerm = glm::dot(glm::normalize(lightPos[0] - intersectionPoint), normal);
		diffuseTerm = max(diffuseTerm, 0.0f);

		color = color * currMaterial.color/* * diffuseTerm*/;	
	}
	
	return;
}

__host__ __device__ void pathTraceIterative(ray r, int rayDepth, staticGeom* geoms, int numberOfGeoms, material* materials, glm::vec3& color, 
											cameraData cam, float time, int x, int y, glm::vec3* lightPos, int lightIndex)
{
	glm::vec3 acol = glm::vec3(1,1,1);
	color = glm::vec3(1,1,1);

	for(int i = 0; i < 5; i++)
	{
		glm::vec3 intersectionPoint, normal;
		int geomIndex;
		if(!checkIntersect(geoms, numberOfGeoms, r, intersectionPoint, normal, geomIndex))
		{
			color = glm::vec3(0,0,0);
			return;
		}

		material currMaterial = materials[geoms[geomIndex].materialid];		
		glm::vec3 ran = generateRandomNumberFromThread(cam.resolution, time + (i+1), x, y);

		if(geomIndex == lightIndex){
			color *= currMaterial.emittance * currMaterial.color;
			return;
		}

		float cosTheta = glm::dot(r.direction, normal);

		if(currMaterial.hasReflective > 0.0f)//Reflective
		{
			glm::vec3 reflectionDirection = calculateReflectionDirection(normal, r.direction);			
			r.origin = intersectionPoint + 0.01f * reflectionDirection;
			r.direction = reflectionDirection;	
		}
		else if(currMaterial.hasRefractive> 0.0f)//Refractive
		{

			Fresnel fresnel;
			float inOrOut = glm::dot(r.direction, normal);
			glm::vec3 reflectedColor, refractedColor;
			if(inOrOut < 0)
				fresnel = calculateFresnel(normal, r.direction, 1.0f, currMaterial.indexOfRefraction, reflectedColor, refractedColor);
			else
				fresnel = calculateFresnel(-normal, r.direction, currMaterial.indexOfRefraction, 1.0f, reflectedColor, refractedColor);

			float refractive = currMaterial.indexOfRefraction;
			glm::vec3 refractedDirection;
			
			if(ran.z < .5f)
			{
				if(inOrOut < 0)
					refractedDirection = calculateTransmissionDirection(normal, r.direction, 1.0, refractive);
				else
					refractedDirection = calculateTransmissionDirection(-normal, r.direction, refractive, 1.0); 					
				r.origin = intersectionPoint + 0.01f * refractedDirection;
				r.direction = refractedDirection;
				color *= fresnel.transmissionCoefficient;
				continue;
			}
			else
			{
				glm::vec3 reflectionDirection = calculateReflectionDirection(normal, r.direction);			
				r.origin = intersectionPoint + 0.01f * reflectionDirection;
				r.direction = reflectionDirection;	
				color *= fresnel.reflectionCoefficient;
				continue;
			}
		}
		else//Diffuse
		{
			r.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, ran.x, ran.y));
			r.origin = intersectionPoint + 0.001f * r.direction;
		}
		
		float diffuseTerm;
		diffuseTerm = glm::dot(glm::normalize(lightPos[0] - intersectionPoint), normal);
		diffuseTerm = max(diffuseTerm, 0.0f);

		color *= currMaterial.color;
	}
	color = glm::vec3(0,0,0);
	return;
}


__global__ void pathtraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, staticGeom* geoms,
							int numberOfGeoms, material* materials, int numberOfMaterials, glm::vec3* lightPos, int lightIndex, int iteration)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if((x<=resolution.x && y<=resolution.y))
	{
		glm::vec3 ran = generateRandomNumberFromThread(cam.resolution, time, x, y);
		//glm::vec3 camPosition = glm::vec3(cam.position.x + (float)(1 + ran.x)/1.0f, cam.position.y + (float)(1 + ran.y)/1.0f, cam.position.z + (float)(1 + ran.z)/1.0f);
		glm::vec3 color;		
		ray r = raycastFromCameraKernel(resolution, time, (float)(x + (float)(1 + ran.x)/1.0f), (float)(y + (float)(1.0f + ran.y)/ 1.0f), cam.position, cam.view, cam.up, cam.fov);
		
		float focalLength = 15.f;
		glm::vec3 aimedPosition = r.origin + focalLength * r.direction;
		//pathTraceIterative(r, 0, geoms, numberOfGeoms, materials, color, cam, time, x, y, lightPos, lightIndex);
		//colors[index] += color;
		for(int i = 0; i < 1; i++){
			glm::vec3 rand = generateRandomNumberFromThread(cam.resolution, time * (i+1), x, y);

			glm::vec3 camPosition = glm::vec3(cam.position.x + (float)rand.x, cam.position.y + (float)rand.y, cam.position.z +  (float)rand.z);

			ray jitterRay;
			jitterRay.origin = camPosition;
			jitterRay.direction = glm::normalize(aimedPosition - camPosition);

			pathTraceIterative(jitterRay, 0, geoms, numberOfGeoms, materials, color, cam, time, x, y, lightPos, lightIndex);
			colors[index] += color;
		}
		color[index] /= 1.f;
	}
}


__host__ __device__ int colorCheck(glm::vec3 color)
{
	if(epsilonCheck(color.x, 0.0f) && epsilonCheck(color.y, 0.0f) && epsilonCheck(color.z, 0.0f))
		return 0;
	else
		return 1;
}

__host__ __device__ int pathTraceIterativeSC(ray& r, staticGeom* geoms, int numberOfGeoms, material* materials, glm::vec3& color, 
											cameraData cam, float time, int x, int y, glm::vec3* lightPos, int lightIndex)
{
	color = glm::vec3(1,1,1);
	
	glm::vec3 intersectionPoint, normal;
	int geomIndex;
	if(!checkIntersect(geoms, numberOfGeoms, r, intersectionPoint, normal, geomIndex))
	{		
		return 0;
	}

	material currMaterial = materials[geoms[geomIndex].materialid];		
	glm::vec3 ran = generateRandomNumberFromThread(cam.resolution, time, x, y);

	if(geomIndex == lightIndex)
	{
		color *= currMaterial.emittance * currMaterial.color;
		return 2;
	}

	float cosTheta = glm::dot(r.direction, normal);

	if(currMaterial.hasReflective > 0.0f)//Reflective
	{
		glm::vec3 reflectionDirection = calculateReflectionDirection(normal, r.direction);			
		r.origin = intersectionPoint + 0.01f * reflectionDirection;
		r.direction = reflectionDirection;	
	}
	else if(currMaterial.hasRefractive> 0.0f)//Refractive
	{

		Fresnel fresnel;
		float inOrOut = glm::dot(r.direction, normal);
		glm::vec3 reflectedColor, refractedColor;
		if(inOrOut < 0)
			fresnel = calculateFresnel(normal, r.direction, 1.0f, currMaterial.indexOfRefraction, reflectedColor, refractedColor);
		else
			fresnel = calculateFresnel(-normal, r.direction, currMaterial.indexOfRefraction, 1.0f, reflectedColor, refractedColor);

		float refractive = currMaterial.indexOfRefraction;
		glm::vec3 refractedDirection;
			
		if(ran.z < .5f)
		{
			if(inOrOut < 0)
				refractedDirection = calculateTransmissionDirection(normal, r.direction, 1.0, refractive);
			else
				refractedDirection = calculateTransmissionDirection(-normal, r.direction, refractive, 1.0); 					
			r.origin = intersectionPoint + 0.01f * refractedDirection;
			r.direction = refractedDirection;
			color *= fresnel.transmissionCoefficient;
			return colorCheck(color);
		}
		else
		{
			glm::vec3 reflectionDirection = calculateReflectionDirection(normal, r.direction);			
			r.origin = intersectionPoint + 0.01f * reflectionDirection;
			r.direction = reflectionDirection;	
			color *= fresnel.reflectionCoefficient;
			return colorCheck(color);
		}
	}
	else//Diffuse
	{
		r.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, ran.x, ran.y));
		r.origin = intersectionPoint + 0.001f * r.direction;
	}

	color *= currMaterial.color;// * diffuseTerm;
	return colorCheck(color);
}



__global__ void pathTraceSC(rayPixel* rayPool, int poolSize, cameraData cam,  float time, int rayDepth, glm::vec3* colors, staticGeom* geoms,
							int numberOfGeoms, material* materials, int numberOfMaterials, glm::vec3* lightPos, int lightIndex, int iteration)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + poolSize * y;
	int t = threadIdx.x;
	if(index < poolSize){
		//if(!rayPool[index].isDone){
		//__shared__ rayPixel sharedRayPixel[TITLESIZE + 1];
		//sharedRayPixel[threadIdx.x] = rayPool[index];
		
		
		__syncthreads();
		if(rayPool[index].isFirst)//first time for ray pool
		{			
			glm::vec3 ran = generateRandomNumberFromThread(cam.resolution, time, x, y);
			rayPool[index].r = raycastFromCameraKernel(cam.resolution, time, (float)(rayPool[index].x + (float)(1 + ran.x)/1.0f), 
				(float)(rayPool[index].y + (float)(1.0f + ran.y)/ 1.0f), cam.position, cam.view, cam.up, cam.fov);	
			rayPool[index].isFirst = false;
			float focalLength = 15.f;
			glm::vec3 rand = generateRandomNumberFromThread(cam.resolution, time * 2, x, y);
			glm::vec3 aimedPosition = rayPool[index].r.origin + focalLength * rayPool[index].r.direction;
			glm::vec3 camPosition = glm::vec3(cam.position.x + (float)rand.x, cam.position.y + (float)rand.y, cam.position.z + (float)rand.z);
			//ray jitterRay;
			rayPool[index].r.origin = camPosition;
			rayPool[index].r.direction = glm::normalize(aimedPosition - camPosition);
		}

		glm::vec3 color;		
		int isContinue = pathTraceIterativeSC(rayPool[index].r, geoms, numberOfGeoms, materials, color, cam, time, x, y, lightPos, lightIndex);
	
		if(isContinue != 0)
			colors[rayPool[index].index] *= color;
		else
			colors[rayPool[index].index] = glm::vec3(0,0,0);
		if(isContinue == 0 || isContinue == 2)
			rayPool[index].isDone = true;

		rayPool[index] = rayPool[index];
	//}
	}
}


__global__ void threadTest()
{
	__syncthreads();
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x;
	
}


//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, glm::vec3* preColors){
  
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0);

	//time(&timer1) * 1000;
	int traceDepth = 1; //determines how many bounces the raytracer traces

	thrust::host_vector<rayPixel> rayPool;
	thrust::device_vector<rayPixel> deviceRayPool;
  
#ifdef STREAMCOMPATION
  //initialize the ray pool
  if(deviceRayPool.size() == 0)
  {	 	  
	  for(int j = 0; j < renderCam->resolution.y; ++j)
	  {
		  for(int i = 0; i < renderCam->resolution.x; ++i)
		  {
			  glm::vec3 ran = generateRandomNumberFromThread(renderCam->resolution, iterations, i, j);
			  int index = i + (j * renderCam->resolution.x);			
			  rayPixel newRayPixel;
			  newRayPixel.isFirst = true;
			  newRayPixel.index = index;
			  newRayPixel.isDone = false;
			  newRayPixel.x = i;
			  newRayPixel.y = j;
			  preColors[index] = renderCam->image[index];
			  renderCam->image[index] = glm::vec3(1,1,1);			  
			  rayPool.push_back(newRayPixel);		
			  
		  }
	  }
	  //++realSample;
	  //printf("sample %f \n", realSample);
  } 
#endif 
  
  //Copy ray pool to device
  /*rayPixel* cudaRayPool = NULL;
  cudaMalloc((void**)&cudaRayPool, rayPool.size()*sizeof(rayPixel));
  cudaMemcpy( cudaRayPool, &rayPool[0], rayPool.size()*sizeof(rayPixel), cudaMemcpyHostToDevice);*/
  deviceRayPool = rayPool;
  rayPixel* cudaRayPool = thrust::raw_pointer_cast(&deviceRayPool[0]);
  
  // set up crucial magic
  int tileSize;	  
#ifdef STREAMCOMPATION
  tileSize = TITLESIZE;
  dim3 threadsPerBlock(tileSize, 2);
  dim3 fullBlocksPerGrid((int)ceil(float(deviceRayPool.size())/float(tileSize)), 1);
#else
  tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
#endif

  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  glm::vec3* cudaPreImage = NULL;
  cudaMalloc((void**)&cudaPreImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaPreImage, preColors, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
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

  //passing material
  material* cudamaterial = NULL;
  cudaMalloc((void**)&cudamaterial, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamaterial, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  int lightIndex;
  for(int i = 0; i < numberOfGeoms; i++)
  {
	  if(geomList[i].materialid == 7 || geoms[i].materialid == 8)
	  {
		  lightIndex = i;
		  break;
	  }
  }

  int lightNum = LIGHT_NUM;
  glm::vec3 *lightPos = new glm::vec3[lightNum];
 // for(int i = 0; i < lightNum; i++)
  //{
  lightPos[0] = getRandomPointOnCube(geomList[lightIndex], (float)iterations);
  //}

  glm::vec3* cudaLightPos = NULL;
  cudaMalloc((void**)&cudaLightPos, lightNum*sizeof(glm::vec3));
  cudaMemcpy(cudaLightPos, lightPos, lightNum*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  //size_t size;
  //cudaDeviceSetLimit(cudaLimitStackSize, 10000*sizeof(rayPixel));
  //cudaDeviceSetLimit(cudaLimitMallocHeapSize, 50000*sizeof(rayPixel));
  //cudaDeviceGetLimit(&size, cudaLimitStackSize);

//kernel launches
#ifdef STREAMCOMPATION
  int count = 0;
  //printf("original device size %d ", deviceRayPool.size());
while(deviceRayPool.size() > 0 && count < 10)
{
		fullBlocksPerGrid = dim3((int)ceil(float(deviceRayPool.size())/float(tileSize)), 1);

		/*pathTraceSC<<<fullBlocksPerGrid, threadsPerBlock>>>(cudaRayPool, deviceRayPool.size(), cam, (float)(iterations+(count +1)), traceDepth, cudaimage, cudageoms,
			numberOfGeoms, cudamaterial, numberOfMaterials, cudaLightPos, lightIndex, renderCam->iterations);
*/
		threadTest<<<fullBlocksPerGrid, threadsPerBlock>>>();

		count++; 		
				
		thrust::device_ptr<rayPixel> iteratorStart(cudaRayPool);
		thrust::device_ptr<rayPixel> iteratorEnd = iteratorStart + deviceRayPool.size();
		deviceRayPool.erase( thrust::remove_if(iteratorStart, iteratorEnd, is_done()), deviceRayPool.end());
		cudaRayPool = NULL;
		cudaRayPool = thrust::raw_pointer_cast(deviceRayPool.data());

		//retrieve image from GPU
		//cudaMemcpy(&rayPool[0], cudaRayPool, rayPool.size()*sizeof(rayPixel), cudaMemcpyDeviceToHost);
		//rayPool.erase(thrust::remove_if(rayPool.begin(), rayPool.end(), is_done()), rayPool.end());	
			
		//printf("device size %d ", deviceRayPool.size());
		//rayPool.resize(deviceRayPool.size());
		//= deviceRayPool;
		//thrust::copy(deviceRayPool.begin(), deviceRayPool.end(), rayPool.begin());
		//cudaFree( cudaRayPool );
		
		//cudaMalloc((void**)&cudaRayPool, rayPool.size()*sizeof(rayPixel));
		//cudaMemcpy( cudaRayPool, &rayPool[0], rayPool.size()*sizeof(rayPixel), cudaMemcpyHostToDevice);
		//cudaMalloc((void**)&cudaRayPool, deviceRayPool.size()*sizeof(rayPixel));		
}
threadsPerBlock = dim3(tileSize, tileSize);
fullBlocksPerGrid = dim3((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
addImageColor<<<fullBlocksPerGrid, threadsPerBlock>>>(cudaimage, cudaPreImage, renderCam->resolution); 
//sendImageToPBOSC<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, cudaimage, (float)iterations, renderCam->resolution.x * renderCam->resolution.y, cudaPreImage);
sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, (float)iterations);
#else
	pathtraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, 
	numberOfGeoms, cudamaterial, numberOfMaterials, cudaLightPos, lightIndex, renderCam->iterations);  

	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, (float)iterations);
#endif
	 
  
   cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamaterial);
  cudaFree( cudaLightPos );
  //cudaFree( cudaRayPool );
  cudaFree( cudaPreImage );
  delete geomList;  
  
  // make certain the kernel has completed
  cudaThreadSynchronize();

  cudaEventRecord( stop, 0);
  cudaEventSynchronize( stop );

  float seconds = 0.0f;
  cudaEventElapsedTime( &seconds, start, stop);
  
  printf("time %f \n", seconds);
  checkCUDAError("Kernel failed!");
}
