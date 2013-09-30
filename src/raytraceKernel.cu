// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <vector>
#include <time.h>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"

using glm::vec3;
using glm::cross;
using glm::length;
using glm::dot;
using glm::normalize;


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
//Thrust doc: https://developer.nvidia.com/thrust
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, float x, float y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov)
{
  ray r;
  float width = resolution.x;
  float height = resolution.y;
  vec3 M = eye + view;
  vec3 A = cross(view, up);
  vec3 B = cross(A, view);
  vec3 H = (A * length(view) * tanf(fov.x * ((float)PI/180.0f))) / length(A);
  vec3 V = -(B * length(view) * tanf(fov.y * ((float)PI/180.0f))) / length(B); // LOOK: Multiplied by negative to flip the image
  vec3 P = M + ((2.0f*x)/(width-1)-1)*H + ((2.0f*y)/(height-1)-1)*V;
  vec3 D = P - eye;
  vec3 DN = glm::normalize(D);

  r.origin = P;
  r.direction = DN;
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
__global__ void sendImageToPBO(uchar4* PBOpos, float iteration, glm::vec2 resolution, glm::vec3* image, glm::vec3* imageAccumd){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y)
  {
      glm::vec3 color;

	  imageAccumd[index] = (imageAccumd[index] * (iteration - 1) + image[index]) / iteration;

      color.x = imageAccumd[index].x*255.0;
      color.y = imageAccumd[index].y*255.0;
      color.z = imageAccumd[index].z*255.0;

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

// Loop through geometry and test against ray.
// Returns FLT_MAX if no object is intersected with the ray, else returns t such that isectPoint = P + Dt 
// Input:  staticGeom* geoms: array of geometry in the scene
//         int numberOfGeoms: number of geoms in the scene
//		   ray r: the ray that is to be intersected with all the geometry
// Output: vec3 isectPoint: holders the intersection point.
//		   vec3 isectNormal: holds the normal at the intersection point.
//		   int matId: the index of the material of the intersected geometry
//		   
__device__ float intersectionTest(staticGeom* geoms, int numberOfGeoms, ray r, vec3 &isectPoint, vec3 &isectNormal, int &matId)
{
	float t = FLT_MAX;

	// testing intersections
	for (int i = 0 ; i < numberOfGeoms ; ++i)
	{
		if (geoms[i].type == GEOMTYPE::SPHERE)
		{
			// do sphere intersection
			vec3 isectPointTemp = vec3(0,0,0);
			vec3 isectNormalTemp = vec3(0,0,0);

			float dist = sphereIntersectionTest(geoms[i], r, isectPointTemp, isectNormalTemp);

			if (dist < t && dist != -1)
			{
				t = dist;
				isectPoint = isectPointTemp;
				isectNormal = isectNormalTemp;
				matId = geoms[i].materialid;
			}
		}
		else if (geoms[i].type == GEOMTYPE::CUBE)
		{
			// do cube intersection
			vec3 isectPointTemp = vec3(0,0,0);
			vec3 isectNormalTemp = vec3(0,0,0);

			float dist = boxIntersectionTest(geoms[i], r, isectPointTemp, isectNormalTemp);

			if (dist < t && dist != -1)
			{
				t = dist;
				isectPoint = isectPointTemp;
				isectNormal = isectNormalTemp;
				matId = geoms[i].materialid;
			}
		}
		else if (geoms[i].type == GEOMTYPE::MESH)
		{
			// do triangle intersections
		}
	} 

	return t;
}

// send out shadow feeler rays and compute the tint color
// this will generate hard shadows if num shadows is set to 1
__device__ vec3 shadowFeeler(staticGeom* geoms, int numberOfGeoms, material* materials, vec3 isectPoint, vec3 isectNormal, staticGeom lightSource, float iter, int index)
{
	vec3 tint = vec3(1,1,1);
	vec3 shadowRayIsectPoint = vec3(0,0,0);
	vec3 shadowRayIsectNormal = vec3(0,0,0);
	int shadowRayIsectMatId = -1;
	float t = FLT_MAX;
	float eps = 1e-5;
	int numShadowRays = SHADOWRAY_NUM;  
	
	// number of times the shadowRays hit the light
	float hitLight = 0;	
	float maxT = 0;
	
	for (int i = 0 ; i < numShadowRays ; ++i)
	{
		vec3 lightPosition = lightSource.translation;

		if (lightSource.type == GEOMTYPE::SPHERE && numShadowRays != 1)
		{
			lightPosition = getRandomPointOnSphere(lightSource, index * iter);
		}
		else if (lightSource.type == GEOMTYPE::CUBE && numShadowRays != 1)
		{
			lightPosition = getRandomPointOnCube(lightSource, index * iter);
		}
		
		vec3 lightToIsect = lightPosition - isectPoint;
		maxT = max(maxT, length(lightToIsect));
		ray shadowRay;
		shadowRay.direction = normalize(lightToIsect);
		shadowRay.origin = isectPoint + shadowRay.direction * eps; // consider moving this in the shadow ray direction

		t = intersectionTest(geoms, numberOfGeoms, shadowRay, shadowRayIsectPoint, shadowRayIsectNormal, shadowRayIsectMatId);

		if (t != FLT_MAX)
			hitLight += materials[shadowRayIsectMatId].emittance / (materials[shadowRayIsectMatId].emittance + eps);
	}

	tint = tint * (hitLight / (float)numShadowRays);
	return tint;
}

//Core pathtracer kernel
__device__ void pathtraceRay(ray r, float ssratio, int index, int rayDepth, glm::vec3* colors, cameraData cam,
                            staticGeom* geoms, int numberOfGeoms, material* cudamat, int numberOfMat, int* cudalightIndex, int numberOfLights, float iter)
{

}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__device__ void raytraceRay(ray r, float ssratio, int index, int rayDepth, glm::vec3* colors, cameraData cam,
                            staticGeom* geoms, int numberOfGeoms, material* cudamat, int numberOfMat, int* cudalightIndex, int numberOfLights, float iter)
{
	vec3 color = vec3(0,0,0);
	vec3 reflectedColor = vec3(0,0,0);
	vec3 bgc = vec3(0,0,0);
	colors[index] = bgc;
	vec3 ambientColor = vec3(0.1, 0.1, 0.1);
  
	if (rayDepth > MAX_DEPTH) 
	{
		//bgc
		color = vec3(0,0,0); 
		return;
	}

	vec3 isectPoint = vec3(0,0,0);
	vec3 isectNormal = vec3(0,0,0);
	int matId = -1;
	float t = FLT_MAX;

	t = intersectionTest(geoms, numberOfGeoms, r, isectPoint, isectNormal, matId); 

	if (t != FLT_MAX)
	{
		material isectMat = cudamat[matId];
		
		// reflection
		if (isectMat.hasReflective > 0)
		{
			vec3 reflectedDirection = calculateReflectionDirection(isectNormal, r.direction);
			ray reflectedRay;
			reflectedRay.direction = normalize(reflectedDirection);
			reflectedRay.origin = isectPoint + isectNormal * (float)1e-5;

			// Temp: Shoot the reflected ray and see which object it hits. Use the object's color instead.
			vec3 reflectedIsectPoint = vec3(0,0,0);
			vec3 reflectedIsectNormal = vec3(0,0,0);
			int reflectedMatId = -1;
			float rt = intersectionTest(geoms, numberOfGeoms, reflectedRay, reflectedIsectPoint, reflectedIsectNormal, reflectedMatId);

			// recurse
			raytraceRay(reflectedRay, ssratio, index, rayDepth+1, colors, cam, geoms, numberOfGeoms, cudamat, numberOfMat, cudalightIndex, numberOfLights, iter);
			reflectedColor = colors[index];
		}


		// hit light source, so use the light source's color directly
		if (isectMat.emittance != 0)
		{
			color = color + ssratio * isectMat.color;
		}
		else
		{
			float reflectance = cudamat[matId].hasReflective;
			color = ssratio * ambientColor + reflectance * reflectedColor;

			// go through each light source and compute shading
			for (int i = 0 ; i < numberOfLights ; ++i)
			{
				staticGeom lightSource = geoms[cudalightIndex[i]];
				vec3 tint = shadowFeeler(geoms, numberOfGeoms, cudamat, isectPoint, isectNormal, lightSource, iter, index);

				vec3 IsectToLight = normalize(lightSource.translation - isectPoint);
				vec3 IsectToEye = normalize(cam.position - isectPoint);
				vec3 lightColor = cudamat[lightSource.materialid].color;
				vec3 materialColor = cudamat[matId].color;
				float lightIntensity = cudamat[lightSource.materialid].emittance;
				float diffuseTerm = clamp(dot(isectNormal, IsectToLight), 0.0f, 1.0f);

				// calculate specular highlight
				vec3 LightToIsect = -IsectToLight;
				vec3 specReflectedRay = calculateReflectionDirection(isectNormal, LightToIsect);
				float specularTerm = pow(max(0.0f, dot(specReflectedRay, IsectToEye)), isectMat.specularExponent);
				float ks = 0.2;
				float kd = 0.7;
				float lightDist = length(IsectToLight);
				float distAttenuation = 1.0f / (lightDist * lightDist);
				
				color = color + (1 - reflectance) * tint * ssratio * (lightIntensity * lightColor * distAttenuation * 
					(kd * materialColor * diffuseTerm + ks * isectMat.specularColor * specularTerm * isectMat.specularExponent));
			}
		}
	}

	colors[index] = color;

}


// Calls Core raytracer kernel starting from eye
__global__ void launchRaytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* cudamat, int numberOfMat, int* cudalightIndex, int numberOfLights)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	
	// supersampling for anti aliasing
	vec3 color = vec3(0,0,0);
	float ss = 1.0f;

	if (ANTIALIASING_SWITCH)
		ss = 3.0f;
	
	float ssratio = 1.0f / (ss * ss);

	for(float i = 1 ; i <= ss ; i++)
	{
		for(float j = 1 ; j <= ss ; j++)
		{
			float ssx = i / ss - 1 / (ss*2.0f);
			float ssy = j / ss - 1 / (ss*2.0f);

			ray r = raycastFromCameraKernel(resolution, 0, ssx + x, ssy + y, cam.position, cam.view, cam.up, cam.fov);

			if (PATHTRACING_SWITCH)
			{
				pathtraceRay(r, ssratio, index, rayDepth, colors, cam, geoms, numberOfGeoms, cudamat, numberOfMat, cudalightIndex, numberOfLights, time);
			}
			else
			{
				raytraceRay(r, ssratio, index, rayDepth, colors, cam, geoms, numberOfGeoms, cudamat, numberOfMat, cudalightIndex, numberOfLights, time);
			}

			color = color + colors[index];
		}
	}

	colors[index] = color;
}


// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms)
{
  // increase stack size so recursion can be used.
  cudaDeviceSetLimit(cudaLimitStackSize, 50000*sizeof(float)); 

  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //package geometry and materials and sent to GPU
  int numberOfLights = 0;
  std::vector<int> lightIndices;

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

	if (materials[newStaticGeom.materialid].emittance != 0)
	{
		numberOfLights++;
		lightIndices.push_back(i);
	}
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

  // check number of materials
  int numberOfMat = 0;

  for (int i = 0 ; i < numberOfMaterials ; ++i)
  {
	numberOfMat++;
  }

  // set up lights indices to pass to cuda
 int* cudalightIndex = NULL;
 cudaMalloc((void**) &cudalightIndex, numberOfLights * sizeof(int));
 cudaMemcpy(cudalightIndex, &(lightIndices[0]), numberOfLights * sizeof(int), cudaMemcpyHostToDevice);

  // Regaring (void**) ...
  // All CUDA API functions return an error code (or cudaSuccess if no error occured). 
  // All other parameters are passed by reference. However, in plain C you cannot have references, 
  // that's why you have to pass an address of the variable that you want the return information to be stored. 
  // Since you are returning a pointer, you need to pass a double-pointer.
  material* cudamat = NULL;
  cudaMalloc((void**)&cudamat, numberOfMat * sizeof(material));
  cudaMemcpy(cudamat, materials, numberOfMat * sizeof(material), cudaMemcpyHostToDevice);
  
  //package camera
  //Q: Why don't we need to use cudaMalloc and cudaMemcpy here?
  //A: During kernel execution, the value cam will be put onto GPU memory stack since this value is not being passed by reference.
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  //kernel launch
  launchRaytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamat, numberOfMat, cudalightIndex, numberOfLights);
 
  // setting up previous image accumulation
  vec3* imageAccum = NULL;
  cudaMalloc((void**)&imageAccum,(int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy(imageAccum, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //kernel launch
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, (float)iterations, renderCam->resolution, cudaimage, imageAccum);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, imageAccum, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( imageAccum );
  cudaFree( cudamat );
  cudaFree( cudalightIndex );
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}

