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
#include <set>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define MAXDEPTH 10 // max raytrace depth

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
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

//---------------------------------------------
//--------------Helper functions---------------
//---------------------------------------------

// Returns true if every component of a is greater than the corresponding component of b
__host__ __device__ bool componentCompare(glm::vec3 a, glm::vec3 b) {
	return (a[0] > b[0] && a[1] > b[1] && a[2] > b[2]);
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
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y,glm::vec3 eye,
																								glm::vec3 view, glm::vec3 up, glm::vec2 fov, float focal, float aperture){
  ray r;
  r.origin = eye;

	// values for computing ray direction
  float phi = glm::radians(fov.y);
	float theta = glm::radians(fov.x);
	glm::vec3 A = glm::normalize(glm::cross(view, up));
	glm::vec3 B = glm::normalize(glm::cross(A, view));
	glm::vec3 M = eye + view;
	glm::vec3 V = B * glm::length(view) * tan(phi);
	glm::vec3 H = A * glm::length(view) * tan(theta);

	// super sampling for anti-aliasing
	thrust::default_random_engine rng(hash(time));
	thrust::uniform_real_distribution<float> u01(0, 1);
	float fx = x + (float)u01(rng);
	float fy = y + (float)u01(rng);

	glm::vec3 P = M + (2*fx/(resolution.x-1)-1) * H + (2*(1-fy/(resolution.y-1))-1) * V;
	r.direction = glm::normalize(P-eye);

	if (abs(focal+1) > THRESHOLD) {
		// for depth of field
		// get the intersection with the focal plane
		glm::vec3 pointOnFocalPlane = eye + r.direction * focal;

		// jitter sample position
		thrust::uniform_real_distribution<float> u02(-aperture/2, aperture/2);
		r.origin += A * u02(rng) + B * u02(rng);
		r.direction = glm::normalize(pointOnFocalPlane - r.origin);
	}

  return r;
}

// Get the reflected ray direction from ray direction and normal
__host__ __device__ glm::vec3 getReflectedRay(glm::vec3 d, glm::vec3 n) {
	glm::vec3 VR; // reflected ray direction
	if (glm::length(-d - n) < THRESHOLD) {
		VR = n;
	}
	else if (abs(glm::dot(-d, n)) < THRESHOLD) {
		VR = d;
	}
	else {
		VR = glm::normalize(d - 2.0f * glm::dot(d, n) * n);
	}
	return VR;
}

// Get the refracted ray direction from ray direction, normal and index of refraction (IOR)
__host__ __device__ glm::vec3 getRefractedRay(glm::vec3 d, glm::vec3 n, float IOR) {
	glm::vec3 VT; // refracted ray direction
	float t = 1 / IOR;
	float base = 1 - t * t * (1 - pow(glm::dot(n, d), 2));
	if (base < 0) {
		 VT = glm::vec3(0, 0, 0);
	}
	else {
		VT = (-t * glm::dot(n, d) - sqrt(base)) * n + t * d; // refracted ray
		VT = glm::normalize(VT);
	}
	return VT;
}

// Determine if the randomly generated ray is a refracted ray or a reflected ray
__host__ __device__  bool isRefractedRay(float randomSeed, float IOR, glm::vec3 d, glm::vec3 n, glm::vec3 t) {
	float rpar = (IOR * glm::dot(n, d) - glm::dot(n, t)) / (IOR * glm::dot(n, d) + glm::dot(n, t));
	float rperp = (glm::dot(n, d) - IOR * glm::dot(n, t)) / (glm::dot(n, d) + IOR * glm::dot(n, t));

	// compute proportion of the light reflected
	float fr = 0.5 * (rpar * rpar + rperp * rperp);

	// determine if ray is reflected according to the proportion
	thrust::default_random_engine rng(hash(randomSeed));
	thrust::uniform_real_distribution<float> u01(0,1);
	if (u01(rng) <= fr) {
		return false;
	}
	else {
		return true;
	}
}

// Decide if the intersection point is in the shadow of a light
__host__ __device__ bool isInShadow(staticGeom* geoms, int numberOfGeoms, int lightIdx, glm::vec3 intersection,
																		int pixelIdx, float time, glm::vec3& L /*out*/) {
	// generate shadow feeler
	glm::vec3 pointOnLight = getRandomPointOnGeom(geoms[lightIdx], pixelIdx * time); // area light
	float distToLight = glm::distance(intersection, pointOnLight);
	L = glm::normalize(intersection - pointOnLight); // direction from light to point
	ray shadowFeeler;
	shadowFeeler.origin = intersection + (-L) * (float)THRESHOLD;
	shadowFeeler.direction = -L;
	
	// find out if the shadow feeler intersects other objects
	bool shadow = false;
	for (int j=0; j<numberOfGeoms; ++j) {
		if (j != lightIdx) {
			glm::vec3 intersection, normal;
			float dist = geomIntersectionTest(geoms[j], shadowFeeler, intersection, normal);
			if (abs(dist+1) > THRESHOLD && dist < distToLight) {
				shadow = true;
				break;
			}
		}
	}
	return shadow;
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, globalAttributes globalAttr, int rayDepth, glm::vec3* colors,
														staticGeom* geoms, int numberOfGeoms, material* materials, int* lightIds, int numberOfLights){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

	glm::vec3 color;

  if((x<=resolution.x && y<=resolution.y)){
		ray r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov, cam.focal, cam.aperture);
		glm::vec3 baseColor(1, 1, 1);

		for (int iteration=0; iteration<MAXDEPTH; ++iteration) {
			glm::vec3 minIntersection, minNormal; // closest intersection point and the normal at that point
			int minIdx = getClosestIntersection(geoms, numberOfGeoms, r, minIntersection, minNormal);

			if (minIdx != -1) {
				material mtl = materials[geoms[minIdx].materialid]; // does caching make it faster?
				//TODO: MAKE THIS BRANCH MORE EFFICIENT
				if (mtl.emittance > THRESHOLD) { // light
					color = glm::clamp(mtl.color * mtl.emittance, glm::vec3(0, 0, 0), glm::vec3(1, 1, 1));
				}
				else {
					if (mtl.hasReflective < THRESHOLD && mtl.hasRefractive < THRESHOLD) { // use phong shading model
						glm::vec3 ambient = globalAttr.ambient * mtl.color;
						ambient = glm::clamp(ambient, glm::vec3(0, 0, 0), glm::vec3(1, 1, 1));
						glm::vec3 diffuse(0, 0, 0);
						glm::vec3 specular(0, 0, 0);

						if (glm::dot(r.direction, minNormal) > 0) { // reverse normal if we are inside the object
							minNormal *= -1;
						}
						
						for (int i=0; i<numberOfLights; ++i) {
							glm::vec3 L;
							if (!isInShadow(geoms, numberOfGeoms, lightIds[i], minIntersection, index, time, L)) {
								material lightMtl = materials[geoms[lightIds[i]].materialid];
								glm::vec3 lightColor = glm::clamp(lightMtl.color * lightMtl.emittance, glm::vec3(0, 0, 0), glm::vec3(1, 1, 1));
								
								// compute diffuse color
								diffuse += lightColor * mtl.color * glm::clamp(glm::dot(-L, minNormal), 0.0f, 1.0f);
								if (componentCompare(diffuse, mtl.color)) {
									break;
								}

								if (mtl.specularExponent > THRESHOLD) {
									// compute specular color
									glm::vec3 LR = getReflectedRay(L, minNormal); 
									specular += lightColor * mtl.specularColor * pow(glm::clamp(glm::dot(LR, -r.direction), 0.0f, 1.0f), mtl.specularExponent);
									if (componentCompare(specular, mtl.specularColor)) {
										break;
									}
								}
							}
						}
						diffuse = glm::clamp(diffuse, glm::vec3(0, 0, 0), mtl.color);
						specular = glm::clamp(specular, glm::vec3(0, 0, 0), mtl.specularColor);
						color = glm::clamp(globalAttr.Ka * ambient + globalAttr.Kd * diffuse + globalAttr.Ks * specular, glm::vec3(0, 0, 0), glm::vec3(1, 1, 1));
						color = color * baseColor;
						break;
					}
					else {
						float IOR = mtl.indexOfRefraction;
						if (glm::dot(r.direction, minNormal) > 0) { // reverse normal and index of refraction if we are inside the object
							minNormal *= -1;
							IOR = 1/(IOR + THRESHOLD);
						}
						if (mtl.hasRefractive > THRESHOLD) { // if the surface has refraction
							glm::vec3 VT = getRefractedRay(r.direction, minNormal, IOR);
							if (glm::length(VT) > THRESHOLD && (mtl.hasReflective < THRESHOLD || isRefractedRay(index*time, IOR, r.direction, minNormal, VT))) {
								r.direction = VT;
								r.origin = minIntersection + VT * (float)THRESHOLD;
								//baseColor *= mtl.color;
								continue;
							}
						}
						// if the surface only has reflection
						glm::vec3 VR = getReflectedRay(r.direction, minNormal);
						r.origin = minIntersection + VR * (float)THRESHOLD;
						r.direction = VR;
						baseColor *= mtl.color;
					}
				}
			}
			else {
				color = glm::vec3(0, 0, 0);
			}
		}
  }
	colors[index] = (colors[index] * (time-1) + color)/time;
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, globalAttributes globalAttr, int frame, int iterations,
											material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  // keep track of the IDs of light materials
	std::set<int> lightMtlIds;
	for (int i=0; i<numberOfMaterials; ++i) {
		if (materials[i].emittance > 0) {
			lightMtlIds.insert(i);
		}
	}
	
	//package geometry and materials and sent to GPU
	// keep track of the IDs of light geometries
  staticGeom* geomList = new staticGeom[numberOfGeoms];
	std::vector<int> lightIds;
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
		if (lightMtlIds.find(newStaticGeom.materialid) != lightMtlIds.end()) {
			lightIds.push_back(i);
		}
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }
	int* lightIdList = new int[lightIds.size()];
	for (int i=0; i<lightIds.size(); i++) {
		lightIdList[i] = lightIds[i];
	}
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

	// copy materials to CUDA
	material* cudamtls = NULL;
	cudaMalloc((void**)&cudamtls, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamtls, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

	// copy lights to CUDA
	int* cudalights = NULL;
	cudaMalloc((void**)&cudalights, lightIds.size()*sizeof(int));
	cudaMemcpy( cudalights, lightIdList, lightIds.size()*sizeof(int), cudaMemcpyHostToDevice);
  
  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;
	cam.focal = renderCam->focal;
	cam.aperture = renderCam->aperture;

  //kernel launches
	raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, globalAttr, traceDepth, cudaimage,
		cudageoms, numberOfGeoms, cudamtls, cudalights, lightIds.size());

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
	cudaFree( cudamtls );
	cudaFree( cudalights );
  delete geomList;
	delete lightIdList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
