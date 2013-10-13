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
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"

//#define DEPTH_OF_FIELD
//#define MOTION_BLUR
//#define STREAM_COMPACTION

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

struct ray_is_dead{  
	__host__ __device__  bool operator()(const ray r)  
	{    
		return r.index == -1;  
	}
};
//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0, 1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

__host__ __device__ glm::vec2 generateRandomNumberFromThreadForSSAA(glm::vec2 resolution, float time, float x, float y, float distanceToBoundary){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec2(x - distanceToBoundary + (float)u01(rng) * 2 * distanceToBoundary, y - distanceToBoundary + (float)u01(rng) * 2 * distanceToBoundary);
//  return glm::vec2(x, y);   // debugging: turn off anti-aliasing
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
  
  if(x<=resolution.x && y<=resolution.y)
//  if(x == 290 && y == 200)
  {

      glm::vec3 color;
      color.x = image[index].x*255.0/float(iterations);
      color.y = image[index].y*255.0/float(iterations);
      color.z = image[index].z*255.0/float(iterations);

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

__host__ __device__ int rayIntersect(const ray& r, staticGeom* geoms, int numberOfGeoms, glm::vec3& intersectionPoint, glm::vec3& intersectionNormal, material* materials){
	float distance = 8000.0f;
	float tempDistance = -1.0f;
	glm::vec3 tempIntersctionPoint(0.0f), tempIntersectionNormal(0.0f);
	int intersIndex = -1;
//#pragma unroll numberOfGeoms
	for(int i = 0; i < numberOfGeoms; i++)
	{
		if(geoms[i].type == GEOMTYPE::SPHERE){
			tempDistance = sphereIntersectionTest(geoms[i], r, tempIntersctionPoint, tempIntersectionNormal);
		}
		else if(geoms[i].type == GEOMTYPE::CUBE){
			tempDistance = boxIntersectionTest(geoms[i], r, tempIntersctionPoint, tempIntersectionNormal);
		}
		else if(geoms[i].type == GEOMTYPE::MESH){
			tempDistance = meshIntersectionTest(geoms[i], r, tempIntersctionPoint, tempIntersectionNormal);
		}
/*
		if(r.index == 294752){
			printf("Geom Index: %d----------------------------------\n", i);
			printf("tempDistance: %f----------------------------------\n", tempDistance);
			printf("tempIntersctionPoint.x = %f, tempIntersctionPoint.y = %f, tempIntersctionPoint.z = %f\n", tempIntersctionPoint.x, tempIntersctionPoint.y, tempIntersctionPoint.z);
			printf("real distance: %f\n", glm::length(tempIntersctionPoint - r.origin));
		}*/
		if(abs(tempDistance + 1.0f) > EPSILON && tempDistance < distance)
		{
			intersIndex = i;
			distance = tempDistance;
			intersectionPoint = tempIntersctionPoint;
			intersectionNormal = tempIntersectionNormal;
		}
	}
	return intersIndex;
}
__host__ __device__ bool ShadowRayUnblocked(glm::vec3 surfacePoint,glm::vec3 lightPosition, staticGeom* geoms, int numberOfGeoms, material* materials/*, float &lightArea, glm::vec3 &lightNormal*/) // return true if unblocked
{
	glm::vec3 rayDir = glm::normalize(lightPosition - surfacePoint);
	ray shadowRay;
	shadowRay.origin = surfacePoint + 0.01f * rayDir;
	shadowRay.direction = rayDir;
	glm::vec3 intersPoint, intersNormal;
	int intersIndex = rayIntersect(shadowRay, geoms, numberOfGeoms, intersPoint, intersNormal, materials); 
	if(intersIndex == -1) {
		printf("not intersected!\n");
		return false;
	}else if(materials[geoms[intersIndex].materialid].emittance > 0.0f){
//		lightArea = getAreaAndNormalOnCube(geoms[intersIndex], intersPoint, lightNormal);
		return true;
	}else return false;

}
#if 0
__host__ __device__ glm::vec3 raytraceRecursive(int index, const ray &r, int iteration, float currentIndexOfRefraction, int depth, int maximumDepth, staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, int* lightSources, int numberOfLights){

	glm::vec3 bgColor(0.0f);
	glm::vec3 phongColor(0.0f), reflColor(0.0f), refraColor(0.0f);;
	glm::vec3 returnColor(0.0f);


	if(depth > maximumDepth)
		return bgColor;

	// intersection test	
	glm::vec3 intersectionPoint, intersectionNormal;
	int intersIndex = rayIntersect(r, geoms, numberOfGeoms, intersectionPoint, intersectionNormal, materials);
	
	if(intersIndex == -1) return bgColor;
	else if(materials[geoms[intersIndex].materialid].emittance > 0.0f) // intersected with light source geometry
		return materials[geoms[intersIndex].materialid].color;

	else // intersected with actual geometry
	{
		glm::vec3 reflDir, refraDir;
		float nextIndexOfRefraction;
		Fresnel fresnel; fresnel.reflectionCoefficient = 0.1; fresnel.transmissionCoefficient = 0;

		if(materials[geoms[intersIndex].materialid].hasRefractive == 1) // for fresnel, if refractive, it will be reflective
		{
			if(abs(currentIndexOfRefraction - 1) < 0.00001f)  // current ray is in air
			{
				refraDir = calculateTransmissionDirection(r.direction, intersectionNormal, currentIndexOfRefraction, materials[geoms[intersIndex].materialid].indexOfRefraction, nextIndexOfRefraction);
			}
			else                                              // current ray is in glass
			{
				intersectionNormal = -intersectionNormal;
				refraDir = calculateTransmissionDirection(r.direction, intersectionNormal, currentIndexOfRefraction, 1.0f, nextIndexOfRefraction);
				
			}

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
			returnColor = fresnel.reflectionCoefficient * reflColor +  fresnel.transmissionCoefficient * refraColor;
		}
		else if(materials[geoms[intersIndex].materialid].hasReflective == 1)
		{
			reflDir = calculateReflectionDirection(r.direction, intersectionNormal);
			ray reflRay;
			reflRay.origin = intersectionPoint + 0.01f * reflDir;
			reflRay.direction = reflDir;
			reflColor = raytraceRecursive(index, reflRay, iteration, currentIndexOfRefraction, depth + 1, maximumDepth, geoms, numberOfGeoms, materials, numberOfMaterials, lightSources, numberOfLights);

			returnColor = reflColor;
		}
		else
		{
			thrust::default_random_engine rng(hash(iteration + depth));
			thrust::uniform_real_distribution<float> u01(0,1), v01(0, 1);
			float russianRoulette1 = (float)u01(rng);
			float russianRoulette2 = (float)v01(rng);

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
	  glm::vec2 jitteredScreenCoords;
	  for(float i = x - 0.5f + nStep * 0.5f; i < x +0.5f; i += nStep)
	  {
		  for(float j = y - 0.5f + nStep * 0.5f; j < y +0.5f; j += nStep)
		  {
			  jitteredScreenCoords = generateRandomNumberFromThreadForSSAA(resolution, time, i, j, 0.5f * nStep);
			  ray r = raycastFromCameraKernel(resolution, time, jitteredScreenCoords.x, jitteredScreenCoords.y, cam.position, cam.view, cam.up, cam.fov);
			  antiAliasedColor += nStep * nStep * raytraceRecursive(index, r, time, 1.0f, 0, rayDepth, geoms, numberOfGeoms, materials, numberOfMaterials, lightSources, numberOfLights);
		  }
	  }
	  colors[index] += pow(1.0f/((float)time + 1.0f), 2.0f) * antiAliasedColor ;
   }
}
#endif
__global__ void rayTracerIterative(int iteration, int depth, ray *rayPool, int rayCount, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;  // blockIdx.y is always zero

	int index = blockDim.y * x + y;
    
	if(index < rayCount)
	{
		// intersection test	
		glm::vec3 intersectionPoint, intersectionNormal;
		ray r = rayPool[index];
/*
		if(r.index == 294752 && depth == 0){
			printf("\n**********************Start of iterative********************\n");
		}
		if(r.index == 294752)
		{
			printf("\n---------------depth = %d--------------\n", depth);
			if(depth == 1)
				printf("break\n");
		}*/
		int intersIndex = rayIntersect(r, geoms, numberOfGeoms, intersectionPoint, intersectionNormal, materials);
/*
		if(r.index == 294752)
		{
			printf("geoms[intersIndex].type = %d\n", geoms[intersIndex].type);
			printf("r.origin.x = %f, r.origin.y = %f, r.origin.z = %f\n", r.origin.x, r.origin.y, r.origin.z);
			printf("r.direction.x = %f, r.direction.y = %f, r.direction.z = %f\n", r.direction.x, r.direction.y, r.direction.z);
			printf("r.mediaIOR = %f\n", r.mediaIOR);
			printf("r.tempColor.x = %f, r.tempColor.y = %f, r.tempColor.z = %f\n", r.tempColor.x, r.tempColor.y, r.tempColor.z);
			printf("colors[r.index].x = %f, colors[r.index].y = %f, colors[r.index].z = %f\n", colors[r.index].x, colors[r.index].y, colors[r.index].z);
			printf("intersectionPoint.x = %f, intersectionPoint.y = %f, intersectionPoint.z = %f\n", intersectionPoint.x, intersectionPoint.y, intersectionPoint.z);
		}*/
		if(intersIndex == -1) 
		{
			r.index = -1;		
		}
		else // intersected with actual geometry
		{
			glm::vec3 reflDir, refraDir;
		
			if(materials[geoms[intersIndex].materialid].hasRefractive == 1) // for fresnel, if refractive, it will be reflective
			{
				float nextIndexOfRefraction;
				if(abs(r.mediaIOR - 1) < 0.00001f)  // current ray is in air
				{
					refraDir = calculateTransmissionDirection(r.direction, intersectionNormal, r.mediaIOR, materials[geoms[intersIndex].materialid].indexOfRefraction, nextIndexOfRefraction);
				}
				else                                              // current ray is in glass
				{
					intersectionNormal = -intersectionNormal;
					refraDir = calculateTransmissionDirection(r.direction, intersectionNormal, r.mediaIOR, 1.0f, nextIndexOfRefraction);
				
				}

				reflDir = calculateReflectionDirection(r.direction, intersectionNormal);
				Fresnel fresnel = calculateFresnel(intersectionNormal, r.direction, r.mediaIOR, nextIndexOfRefraction, reflDir, refraDir);

				thrust::default_random_engine rng(hash(iteration *(depth + 1)* index));
				thrust::uniform_real_distribution<float> u01(0,1);

				if((float)u01(rng) < fresnel.reflectionCoefficient)
				{
					r.origin = intersectionPoint + 0.01f * reflDir;
					r.direction = reflDir;
					int nextIntersIndex = rayIntersect(r, geoms, numberOfGeoms, intersectionPoint, intersectionNormal, materials);
					if(depth == 0 && nextIntersIndex != -1) // hack to let direct illumination method work with reflective surface
						colors[r.index] += r.tempColor * (materials[geoms[nextIntersIndex].materialid].emittance) * materials[geoms[nextIntersIndex].materialid].color;
				}				
				else
				{
					
					r.origin = intersectionPoint + 0.01f * refraDir;
					r.direction = refraDir;
					r.mediaIOR = nextIndexOfRefraction;
				}

			}
			else if(materials[geoms[intersIndex].materialid].hasReflective == 1)
			{
				reflDir = calculateReflectionDirection(r.direction, intersectionNormal);
				r.origin = intersectionPoint + 0.01f * reflDir;
				r.direction = reflDir;
				int nextIntersIndex = rayIntersect(r, geoms, numberOfGeoms, intersectionPoint, intersectionNormal, materials);
				if(depth == 0 && nextIntersIndex != -1)// hack to let direct illumination method work with reflective surface
					colors[r.index] += r.tempColor * (materials[geoms[nextIntersIndex].materialid].emittance) * materials[geoms[nextIntersIndex].materialid].color;
			}
			else
			{   // exlude direct illumination here
//				if(index == 35771) printf("intersIndex: %d\n",intersIndex);
				colors[r.index] += r.tempColor * (depth == 1? 0 : materials[geoms[intersIndex].materialid].emittance) * materials[geoms[intersIndex].materialid].color;
				thrust::default_random_engine rng(hash(iteration *(depth + 1)* index));
				thrust::uniform_real_distribution<float> u01(0,1);
				if((float)u01(rng) < 0.2 /*&& depth > 3|| glm::length(r.tempColor) < 0.01f*/) // russian roulette rule: ray is absorbed
				{
					r.index = -1;
				}
				else
				{
					glm::vec3 newdir = calculateRandomDirectionInHemisphere(intersectionNormal, (float)u01(rng), (float)u01(rng));
					r.origin =  intersectionPoint + 0.01f * newdir;
					r.direction = newdir;
					r.tempColor *= glm::clamp(glm::dot(intersectionNormal, newdir), 0.0f, 1.0f) * materials[geoms[intersIndex].materialid].color;				
				}
			}
		}	
		rayPool[index] = r;
	}
}


__global__ void rayTracerIterativePrimary(glm::vec2 resolution, int time, cameraData cam, ray *rayPool, int rayCount, glm::vec3 *colors,
                            staticGeom* geoms, int numberOfGeoms, material *materials, int numberOfMaterials, int *lights, int numberOfLights){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    
    if((x<=resolution.x && y<=resolution.y))
    {
		
	    glm::vec2 jitteredScreenCoords = generateRandomNumberFromThreadForSSAA(resolution, time, x, y, 0.5f);
	    ray r = raycastFromCameraKernel(resolution, time, jitteredScreenCoords.x, jitteredScreenCoords.y, cam.position, cam.view, cam.up, cam.fov);
        r.index = index;
		r.tempColor = glm::vec3(1.0f);
	    r.mediaIOR = 1.0f;	    
/*
		if(r.index == 294752){
			printf("****************Primary intersection********************\n");
		}*/
//----------------shadow ray for direct illumination part-------------------------------------
		glm::vec3 intersectionPoint, intersectionNormal;
		int intersIndex = rayIntersect(r, geoms, numberOfGeoms, intersectionPoint, intersectionNormal, materials); 

		// choose which light to sample
		thrust::default_random_engine rng(hash(index*time));
		thrust::uniform_real_distribution<float> u01(0,0.9999999); // in fact, the random number generator for [0, 1) sometimes generate 1 so we want to avoid that
		float russianRoulette = (float)u01(rng);
		int sampleLightIndex = lights[(int)(russianRoulette * numberOfLights)];

/*
		if(sampleLightIndex != 8)
		printf("sampleLightIndex = %f\n", (russianRoulette * numberOfLights));*/

		// sample selected light
		glm::vec3 lightNormal;
		float lightArea;
		glm::vec3 lightPos = getRandomPointOnCube(geoms[sampleLightIndex], index*time, lightArea, lightNormal);
/*
		if(r.index == 294752){
			printf("****************Shadow test********************\n");
		}*/
		if(ShadowRayUnblocked(intersectionPoint, lightPos, geoms, numberOfGeoms, materials/*, lightArea, lightNormal*/))
		{
			glm::vec3 distanceVector = lightPos - intersectionPoint;
			glm::vec3 L = glm::normalize(distanceVector);
			float cosineTerm = glm::clamp(glm::dot(intersectionNormal, L), 0.0f, 1.0f);
			float projectedLightArea = glm::clamp(glm::dot(lightNormal, -L), 0.0f, 1.0f) * lightArea;  // light and illuminated point have to be on the same side
			glm::vec3 directLighting = materials[geoms[sampleLightIndex].materialid].emittance * materials[geoms[intersIndex].materialid].color * cosineTerm * projectedLightArea *  float(numberOfLights) / pow(glm::length(distanceVector), 2.0f) / (float)PI;
     		colors[index] += directLighting;
		}
//----------------------------------------------------------------------------------------------
#if defined(DEPTH_OF_FIELD)
		thrust::default_random_engine rng(hash(index*time));
		thrust::uniform_real_distribution<float> u(0,1);

		float u1 = u(rng);
		float v1 = u(rng);

	//	glm::vec3 offset = aperture * glm::vec3(u1 * cos((float)TWO_PI*v1), u1 * sin((float)TWO_PI*v1), 0.0f);
		glm::vec3 offset = cam.aperture * glm::normalize(generateRandomNumberFromThread(resolution, time, x, y));
		offset.z = 0.0f;
		glm::vec3 focalPlaneIntersection = r.origin + cam.focalLength * r.direction;
		r.origin += offset;
		r.direction = glm::normalize(focalPlaneIntersection - r.origin);
#endif

		rayPool[index] = r;
    }
}



//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  // frame: current frame, objects and cam move between frames in multi-frame mode
  // iterations: curent iteration, objects and cam do not move between iterations
    int traceDepth = 10; //determines how many bounces the raytracer traces
//	printf("\n--------------------------------iteration %d-------------------------\n", iterations);
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
#if defined(MOTION_BLUR)
	   if(i == 6)
	   {
		   newStaticGeom.translation += (iterations < 4000 ? (float)iterations : (float)4000) * glm::vec3(-0.0015f, 0.0f, 0.0f);
		   glm::mat4 transform = utilityCore::buildTransformationMatrix(newStaticGeom.translation, newStaticGeom.rotation, newStaticGeom.scale);
		   newStaticGeom.transform = utilityCore::glmMat4ToCudaMat4(transform);
		   newStaticGeom.inverseTransform = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));
	   }
#endif
	   newStaticGeom.vertexCount = geoms[i].vertexCount;
	   newStaticGeom.vertexList = geoms[i].vertexList;
	   newStaticGeom.faceCount = geoms[i].faceCount;
	   newStaticGeom.faceList = geoms[i].faceList;
	   newStaticGeom.boundingBoxMax = geoms[i].boundingBoxMax;
	   newStaticGeom.boundingBoxMin = geoms[i].boundingBoxMin;
	   
       geomList[i] = newStaticGeom;
    }
  
    staticGeom* cudageoms = NULL;
    cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
    cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

    material* cudamaterials = NULL;
    cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
    cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);
  
  //package camera
    cameraData cam;
    cam.resolution = renderCam->resolution;
    cam.position = renderCam->positions[frame];
    cam.view = renderCam->views[frame];
    cam.up = renderCam->ups[frame];
    cam.fov = renderCam->fov;
	cam.focalLength = renderCam->focalLength;
	cam.aperture = renderCam->aperture;

  
  // construct initial ray pool
    int rayPoolCount = (int)(renderCam->resolution.x * renderCam->resolution.y);
    ray* rayPool = new ray[rayPoolCount]; 

  // set up crucial magic
    int tileSize = 8;
    dim3 threadsPerBlock(tileSize, tileSize);
    dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

    ray* cudaray = NULL;
    cudaMalloc((void**)&cudaray, rayPoolCount*sizeof(ray));
    cudaMemcpy( cudaray, rayPool, rayPoolCount*sizeof(ray), cudaMemcpyHostToDevice);

	// copy lights
	std::vector<int> lightIndices;	
	for(int i = 0; i < numberOfGeoms; ++i){
		if(materials[geoms[i].materialid].emittance > 0) lightIndices.push_back(i);
	}
	int numberOfLights = lightIndices.size();
	int *lights = new int[numberOfLights];
	for(int i = 0; i < numberOfLights; ++i){
		lights[i] = lightIndices[i];
	}

	int* cudalight = NULL;
    cudaMalloc((void**)&cudalight, numberOfLights*sizeof(int));
    cudaMemcpy( cudalight, lights, numberOfLights*sizeof(int), cudaMemcpyHostToDevice);

	// first kernel launch to populate ray pool
    rayTracerIterativePrimary<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, iterations, cam, cudaray, rayPoolCount, cudaimage, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials, cudalight, numberOfLights);
    
	// multiple kernel launches to evaluate color
    for(int i = 0; i < traceDepth && rayPoolCount != 0; ++i)
    {
	  // resize block and grid 
	    dim3 blocksPerGrid((int)ceil((float)rayPoolCount/(float)(tileSize*tileSize)));
	    rayTracerIterative<<<blocksPerGrid, threadsPerBlock>>>(iterations, i, cudaray, rayPoolCount, cudaimage, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials);
#if defined(STREAM_COMPACTION)
	  // do stream compaction
	    thrust::device_ptr<ray> ray_ptr = thrust::device_pointer_cast(cudaray);
	    thrust::device_ptr<ray> rayPool_last = thrust::remove_if(ray_ptr, ray_ptr + rayPoolCount, ray_is_dead());
	    rayPoolCount = thrust::distance(ray_ptr, rayPool_last);	 
#endif
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
