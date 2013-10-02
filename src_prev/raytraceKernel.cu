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

#define POSITION(g) multiplyMV(g.transform, glm::vec4(0,0,0,1))
#define SPECULAR(materials, g) materials[g.materialid].specularExponent
#define REFLECTIVE(materials, g) materials[g.materialid].hasReflective
#define IS_LIGHT(materials, geoms, idx) !epsilonCheck(materials[geoms[idx].materialid].emittance, 0.0f)
#define COLOR(materials, g) materials[g.materialid].color
#define SPEC_COLOR(materials, g) materials[g.materialid].specularColor
#define EMITTANCE(materials, g) materials[g.materialid].emittance

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

glm::vec3* cudaimage;
cameraData cam;

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
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
    glm::vec3 A = glm::cross(view, up);
	glm::vec3 B = glm::cross(A, view);
	glm::vec3 M = eye + view;

	float phi = glm::radians(fov.y);
	float theta = glm::radians(fov.x);
	float C = glm::length(view);

	glm::vec3 V = glm::normalize(B) * (C * tan(phi));
	glm::vec3 H = glm::normalize(A) * (C * tan(theta));

	float sx = (float) x / (resolution.x - 1.0f);
	float sy = (float) y / (resolution.y - 1.0f);

	glm::vec3 P = M + (2 * sx - 1.0f)*H	 + (1.0f - 2 * sy)*V;

	ray r;
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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, float iterations){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x / iterations *255.0;
      color.y = image[index].y / iterations *255.0;
      color.z = image[index].z / iterations *255.0;

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
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* mats, int* lightIds, int numberOfLights){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	float focal_length = 10.0f;
	glm::vec3 focal_point = cam.position + focal_length * cam.view;

	ray r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
	
	float t = 1.0f / glm::dot(r.direction, cam.view) * glm::dot(focal_point - r.origin, cam.view);
	glm::vec3 aimed = r.origin + t * r.direction;
	r.origin = r.origin + .5f * generateRandomNumberFromThread(resolution, time, x, y);
	r.direction = 0.01f * generateRandomNumberFromThread(resolution, time, x, y) + glm::normalize(aimed - r.origin);
	
	float intersect, Ka = 0.0f, Kd = .5f, Ks = .5f;
	int geomId, bounce = 0;

	glm::vec3 intersectionPoint, normal, color, base_color = glm::vec3(1.0), bgColor = glm::vec3(0.0), ambColor = glm::vec3(0.0);
	glm::vec3 light_pos, ray_incident, reflectedRay, specular, diffuse;

	if((x<=resolution.x && y<=resolution.y)){
		while(bounce < rayDepth){
			// Intersection Checking
			intersect = isIntersect(r, intersectionPoint, normal, geoms, numberOfGeoms, geomId);

			if (epsilonCheck(intersect, -1.0f)){
				bool parallel = false;
				int i = -1;
				while(i < numberOfGeoms && !parallel){
					i++;
					if(IS_LIGHT(mats, geoms, i))
						parallel = epsilonCheck(glm::length(glm::cross(r.direction, POSITION(geoms[i]) - cam.position)), 0.0f);
				}
				if(parallel) color = COLOR(mats, geoms[i]);
				else color = bgColor;
				break;
			}else{
				if(REFLECTIVE(mats, geoms[geomId]) > .001f){
					base_color = base_color * COLOR(mats, geoms[geomId]);
					if(epsilonCheck(glm::length(glm::cross(r.direction, normal)), 0.0f)) reflectedRay = -1.0f * normal;
					else if(epsilonCheck(glm::dot(-1.0f * r.direction, normal), 0.0f)) reflectedRay = r.direction;
					else reflectedRay = r.direction - 2.0f * normal * glm::dot(r.direction, normal);
				}else{
					if(IS_LIGHT(mats, geoms, geomId)) color = COLOR(mats,geoms[geomId]);
					else{
						for(int i = 0; i < numberOfLights; i++){
							light_pos = getRandomPoint(geoms[lightIds[i]], time);
							if(!rayBlocked(intersectionPoint, light_pos, lightIds[i], geoms, numberOfGeoms, mats)){
								ray_incident = glm::normalize(intersectionPoint - light_pos);
							  
								if(epsilonCheck(glm::length(glm::cross(ray_incident, normal)), 0.0f)) reflectedRay = normal;
								else if(epsilonCheck(glm::dot(-1.0f * ray_incident, normal), 0.0f)) reflectedRay = ray_incident;
								else reflectedRay = ray_incident - 2.0f * normal * glm::dot(ray_incident, normal);
							  
								float specTerm = glm::clamp(glm::dot(glm::normalize(reflectedRay), glm::normalize(-1.0f * r.direction)), 0.0f, 1.0f);
								specular = !epsilonCheck(SPECULAR(mats, geoms[geomId]), 0.0f) ? Ks * pow(specTerm, SPECULAR(mats, geoms[geomId])) * SPEC_COLOR(mats, geoms[geomId]) * glm::clamp(COLOR(mats, geoms[lightIds[i]]) * EMITTANCE(mats, geoms[lightIds[i]]), 0.0f, 1.0f) : glm::vec3(0.0);

								float diffuseTerm = glm::clamp(glm::dot(normal, glm::normalize(light_pos - intersectionPoint)), 0.0f, 1.0f);
								Kd = !epsilonCheck(glm::length(specular), 0.0f) ? Kd : Kd + Ks;
								diffuse = diffuseTerm * COLOR(mats, geoms[geomId]) * glm::clamp(COLOR(mats, geoms[lightIds[i]]) * EMITTANCE(mats, geoms[lightIds[i]]), 0.0f, 1.0f);

								color += Ka * ambColor + diffuse * base_color + specular;
							}else{
								color += Ka * ambColor * COLOR(mats, geoms[geomId]);
							}
						}
						break;
					}
				}
				r.direction = reflectedRay;
				r.origin = intersectionPoint + .001f * r.direction;
				bounce++;
			}
		}
	}
	colors[index] += glm::clamp(color, 0.0f, 1.0f);
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, 
	int* lightIds, int numberOfLights){
  
  int traceDepth = 4; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  cudaimage = NULL;
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

  material* cudamats = NULL;
  cudaMalloc((void**)&cudamats, numberOfMaterials * sizeof(material));
  cudaMemcpy(cudamats, materials, numberOfMaterials * sizeof(material), cudaMemcpyHostToDevice);

  int* cudalightids = NULL;
  cudaMalloc((void**)&cudalightids, numberOfLights * sizeof(int));
  cudaMemcpy(cudalightids, lightIds, numberOfLights * sizeof(int), cudaMemcpyHostToDevice);
  
  //package camera
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamats, cudalightids, numberOfLights);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, (float)iterations);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamats );
  cudaFree( cudalightids );
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}

void clearImageBuffer(camera* renderCam){
	int tileSize = 8;
    dim3 threadsPerBlock(tileSize, tileSize);
    dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

	// Copy image from CPU to GPU
	cudaimage = NULL;
	cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

	// Clear Image
	clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaimage);

	// Retrieve image
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	// Sync Threads
	cudaThreadSynchronize();
}
