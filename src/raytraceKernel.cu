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


//global variables
rayBounce* cudaFirstPass;
rayBounce* cudaRayPool;		//for stream compaction, pool of rays that are still alive
rayBounce* cudaTempRayPool;	//for switching and replacing rays in stream compaction
int* cudaCompactA;
int* cudaCompactB;

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
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
	
	ray r;

	//ray creation from camear stuff to be used in raycastFromCameraKernel
	glm::vec3 M = eye + view;	//center of screen

	//project screen to world space
	glm::vec3 A = glm::cross(view, up);
	glm::vec3 B = glm::cross(A, view);

	float C = glm::length(view);

	float phi = fov.y/180.0f * PI;		//convert to radians
	B = glm::normalize(B);
	glm::vec3 V = C * tan(phi) * B;

	float theta = fov.x/180.0f * PI;
	A = glm::normalize(A);
	glm::vec3 H = C * tan(theta) * A;

	//find the world space coord of the pixel
	float sx = (float)x / (resolution.x-1.0f);
	float sy = (float)y / (resolution.y-1.0f);

	glm::vec3 P = M + H * (2.0f * sx - 1.0f) + V * (1.0f - 2.0f * sy);

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
  
  //iterations = 1;

  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0 / iterations;
      color.y = image[index].y*255.0 / iterations;
      color.z = image[index].z*255.0 / iterations;

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


//Does intersection on all of the objects and returns length of closest intersection
__host__ __device__ float testGeomIntersection(staticGeom* geoms, int numberOfGeoms, ray& r, glm::vec3& intersectionPoint, glm::vec3& normal, int& objID){

	float len = FLT_MAX;	
	float tempLen = -1;
	glm::vec3 tempIntersection;
	glm::vec3 tempNormal;
	
	//check for interesction
	for(int geomInd = 0; geomInd<numberOfGeoms; ++geomInd){
			
		if(geoms[geomInd].type == CUBE){
			tempLen = boxIntersectionTest(geoms[geomInd], r, tempIntersection, tempNormal);
		}

		else if (geoms[geomInd].type == SPHERE){
			tempLen = sphereIntersectionTest(geoms[geomInd], r, tempIntersection, tempNormal);
		}
			
		else if(geoms[geomInd].type == MESH){
				
		}
							
		//if intersection occurs and object is in front of previously intersected object
		if(tempLen != -1 && tempLen < len){
			len =tempLen;
			intersectionPoint = tempIntersection;
			normal = tempNormal;
			objID = geomInd;
		}
	}

	return len;

}


__global__ void streamCompact(int numRays, int* compactIn, int* compactOut, int d){

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index < numRays){
		
		int valIn = compactIn[index];

		if( index >= d){
			compactOut[index] = compactIn[index - d] + valIn;
		}
		else{
			compactOut[index] = valIn;
		}
	}

	//__syncthreads(); 
}

__global__ void buildRayPool(int* compactIn, rayBounce* rayTempPass, rayBounce* rayPass, int numRays){

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index < numRays){
		if(!rayTempPass[index].dead)
			rayPass[compactIn[index] - 1] = rayTempPass[index];
	}

}

//creates and stores first bounce rays, always at depth 1
__global__ void createRay(glm::vec2 resolution, cameraData cam, int maxDepth, int time, staticGeom* geoms, int numberOfGeoms, material* materials, 
						 int numLights, int* lightID, rayBounce* firstPass, int* compactIn, int numRays, glm::vec3* colors){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	
	glm::vec3 intersection;
	glm::vec3 normal;

	if(index < numRays){

		ray firstRay = raycastFromCameraKernel(resolution, x, y, cam.position, cam.view, cam.up, cam.fov); 

		////DOF and antialiasing setup
		//float focalLength = cam.focalLength;
		//float aperture = cam.aperture;
		//
		//glm::vec3 focalPoint = firstRay.origin + focalLength * firstRay.direction;

		////jitter camera
		//glm::vec3 jitterVal = 2.0f * aperture * generateRandomNumberFromThread(resolution, time, x, y);
		//jitterVal -= glm::vec3(aperture);
		//firstRay.origin += jitterVal;

		////find new direction
		//firstRay.direction = glm::normalize(focalPoint - firstRay.origin);

		////antialias sample per pixel
		//jitterVal = generateRandomNumberFromThread(resolution, time, x, y);
		//jitterVal -= glm::vec3(0.5f, 0.5f, 0.5f);
		//firstRay.direction += 0.0015f* jitterVal; 

		//do intersection test
		int objID = -1;
		float len = testGeomIntersection(geoms, numberOfGeoms, firstRay, intersection, normal, objID);
		
		//start building the first pool of rays
		rayBounce firstBounce = rayBounce();
		firstBounce.pixID = index;
		firstBounce.thisRay = firstRay;
		firstBounce.objID = objID;
		firstBounce.currDepth = 1;
		firstBounce.color = glm::vec3(1,1,1);		//always start as white!!

		//if no intersection, return
		if(objID == -1){
			colors[index] += glm::vec3(0,0,0);					//set to black
			firstBounce.dead = true;
			firstPass[index] = firstBounce;
			compactIn[index] = 0;
			return;	
		}
		
		int matID = geoms[objID].materialid;
		material firstMat = materials[matID];
		
		//save the first bounce information
		firstBounce.normal = normal;
		firstBounce.matID = matID;
		firstBounce.intersectPt = intersection;

		//check if material is light
		if(materials[matID].emittance > 0){
			colors[index] += firstMat.color * firstMat.emittance; 
			firstBounce.dead = true;
			compactIn[index] = 0;
			return;
		}

		if(materials[matID].hasReflective == 1){
			firstBounce.dead = false;
			compactIn[index] = 1;
			firstBounce.thisRay.direction = glm::normalize(firstRay.direction - 2.0f*glm::dot(firstRay.direction, normal)*normal);
		}
		else{
			firstBounce.dead = false;
			compactIn[index] = 1;

			//calculate diffuse direction
			glm::vec3 seed = generateRandomNumberFromThread(resolution, time, x, y);
			firstBounce.thisRay.direction = calculateRandomDirectionInHemisphere(normal, seed.x, seed.y);
		}

		//offset a little to prevent self intersection
		firstBounce.thisRay.origin = intersection + 0.0001f * firstRay.direction;
		
		//store the color
		firstBounce.color *= firstMat.color;
		
		//store first bounce
		firstPass[index] = firstBounce;

	}

}

__global__ void rayParallelTrace(glm::vec2 resolution, float time, cameraData cam, int maxDepth, glm::vec3* colors, staticGeom* geoms, int numberOfGeoms, 
								material* materials, int numLights, int* lightID, int numRays, int* compactIn, rayBounce* rayPass){

	//find the index in 1D
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index < numRays){
		
		//get the bounce at this index
		rayBounce currBounce = rayPass[index];
		currBounce.currDepth++;

		//get the ray information at this index
		ray currRay = currBounce.thisRay;
		glm::vec3 intersection;
		glm::vec3 normal;

		//do intersection test
		int objID = -1;
		float len = testGeomIntersection(geoms, numberOfGeoms, currRay, intersection, normal, objID);
		
		//start building the first pool of rays
		currBounce.objID = objID;

		//if no intersection, return
		if(objID == -1){
			colors[currBounce.pixID] += glm::vec3(0,0,0);
			currBounce.dead = true;
			rayPass[index] = currBounce;
			compactIn[index] = 0;
			return;	
		}
		
		int matID = geoms[objID].materialid;
		material currMat = materials[matID];
		
		//check if material is light
		if(materials[matID].emittance > 0){
			colors[currBounce.pixID] += currBounce.color * currMat.color * currMat.emittance;
			currBounce.dead = true;
			compactIn[index] = 0;
			return;
		}

		//update bounce information
		currBounce.normal = normal;
		currBounce.matID = matID;
		currBounce.intersectPt = intersection;

		if(materials[matID].hasReflective == 1){
			currBounce.dead = false;
			compactIn[index] = 1;
			currBounce.thisRay.direction = glm::normalize(currRay.direction - 2.0f*glm::dot(currRay.direction, normal)*normal);
		}
		else{
			//generate new rand direction
			glm::vec3 seed = generateRandomNumberFromThread(resolution, time, index, threadIdx.x);
			currBounce.thisRay.direction = calculateRandomDirectionInHemisphere(normal, seed.x, seed.y);
			currBounce.dead = false;
			compactIn[index] = 1;
		}

		currBounce.color *= currMat.color;	
		currBounce.thisRay.origin = intersection + 0.0001f * currRay.direction;

		//if(currBounce.currDepth > maxDepth)
		//	colors[currBounce.pixID] += currBounce.color;
	}

}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, staticGeom* geoms, int numberOfGeoms, 
							material* materials, int numLights, int* lightID, rayBounce* firstPass){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	glm::vec3 intersection;
	glm::vec3 normal;
	glm::vec3 surfColor;

	int currDepth = 1;

	if((x<=resolution.x && y<=resolution.y)){

		ray firstRay = raycastFromCameraKernel(resolution, x, y, cam.position, cam.view, cam.up, cam.fov); 
		glm::vec3 finalColor(1,1,1);
		
		while(currDepth <= rayDepth){

#pragma region setup
			//DOF setup thing
			float focalLength = cam.focalLength;

			float aperture = cam.aperture;
		
			glm::vec3 focalPoint = firstRay.origin + focalLength * firstRay.direction;

			//jitter camera
			glm::vec3 jitterVal = 2.0f * aperture * generateRandomNumberFromThread(resolution, time, x, y);
			jitterVal -= glm::vec3(aperture);
			firstRay.origin += jitterVal;

			//find new direction
			firstRay.direction = glm::normalize(focalPoint - firstRay.origin);

			//antialias sample per pixel
			jitterVal = generateRandomNumberFromThread(resolution, time, x, y);
			jitterVal -= glm::vec3(0.5f, 0.5f, 0.5f);
			firstRay.direction += 0.0015f* jitterVal; 

			//firstPass[index] = firstRay;
#pragma endregion setup

			//do intersection test
			int objID = -1;

			float len = testGeomIntersection(geoms, numberOfGeoms, firstRay, intersection, normal, objID);

			//if no intersection, return
			if(objID == -1){
				finalColor *= 0.0f;
				break;	
			}
		
			int matID = geoms[objID].materialid;
			surfColor = materials[matID].color;

			//check if you intersected with light, if so, just return light color
			if(materials[matID].emittance > 0){
				finalColor *= surfColor;
				break;
			}


	#pragma region lightAndShadow
			glm::vec3 diffuse(0,0,0);
			glm::vec3 phong(0,0,0);

			//do light and shadow computation
			for(int i = 0; i < numLights; ++i){

				int lightGeomID = lightID[i];
				glm::vec3 lightPos;
				glm::vec3 lightColor = materials[geoms[lightGeomID].materialid].color;

				//find a random point on the light
				if(geoms[lightGeomID].type == CUBE){
					lightPos = getRandomPointOnCube(geoms[lightGeomID], time);		//CHANGE TO TIME!
				}
				else if(geoms[lightGeomID].type == SPHERE){
					lightPos = getRandomPointOnSphere(geoms[lightGeomID], time);	//CHANGE TO TIME!
				}

				//find vector from intersection to point on light
				glm::vec3 L = lightPos - intersection;
				float distToLight = glm::length(L);
				L = glm::normalize(L);

				//check if in shadow
				objID = -1;
				ray shadowFeeler; 
				shadowFeeler.direction = L;
				shadowFeeler.origin = intersection + 0.0001f*L;		//offset origin a little bit so it doesn't self intersect
			
				glm::vec3 shadowNormal; glm::vec3 shadowIntersection;
				len = testGeomIntersection(geoms, numberOfGeoms, shadowFeeler, shadowIntersection, shadowNormal, objID);

				//if intersection occured and intersection is in between the intersection point and the light position
				if(objID != -1 && len < distToLight){
				
					if(materials[geoms[objID].materialid].emittance == 0){		//only cast shadow if we intersected with object that is not a light
						//color is ambient color
						finalColor = glm::vec3(0,0,0);
						continue;
					}
				}

				//do diffuse calculation
				diffuse += glm::clamp(glm::dot(L, normal), 0.0f, 1.0f) * surfColor * lightColor;
			
				//clamp diffuse to surface color
				diffuse.x = clamp(diffuse.x, 0.0f, surfColor.x);
				diffuse.y = clamp(diffuse.y, 0.0f, surfColor.y);
				diffuse.z = clamp(diffuse.z, 0.0f, surfColor.z);

				//specular
				if(materials[matID].specularExponent != 0){
					glm::vec3 R = glm::normalize( -L - 2.0f*glm::dot(-L, normal) *normal);
					glm::vec3 V = -firstRay.direction;			//already normalized
			
					phong += materials[matID].specularColor * 
							pow(glm::clamp(glm::dot(R, V), 0.0f, 1.0f), materials[matID].specularExponent) * lightColor;
					//phong *= 0.5f;
					//diffuse *= 0.9f;
				}

			}
	#pragma endregion lightAndShadow

			//check for reflection
			if(materials[matID].hasReflective == 1){
				//reflect
				firstRay.direction = glm::normalize(firstRay.direction - 2.0f*glm::dot(firstRay.direction, normal)*normal);
				
				//offsect a little to prevent intersection
				firstRay.origin = intersection + 0.0001f * firstRay.direction;
				currDepth++;
				
				finalColor *= glm::clamp(surfColor + phong, 0.0f, 1.0f);
			}
			else{
				finalColor *= glm::clamp(diffuse + phong, 0.0f, 1.0f);
				break;
			}

		}
	
		//output final color
		colors[index] += finalColor;
	}

}

__global__ void resetCompactVals(int* compactA, int* compactB, int imageSize){
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < imageSize)
		compactA[index] = compactB[index] = 0;
}

//allocate memory on cuda
void cudaAllocMemory(glm::vec2 resolution){

	int size = (int)resolution.x*resolution.y;
	
	//std::cout<<"allocate "<<std::endl;
	//cache the first bounce since they're the same for each iteration
	cudaFirstPass = NULL;
	cudaMalloc((void**)&cudaFirstPass, size*sizeof(rayBounce));

	cudaRayPool = NULL;
	cudaMalloc((void**)&cudaRayPool, size*sizeof(rayBounce));

	cudaTempRayPool = NULL;
	cudaMalloc((void**)&cudaTempRayPool, size*sizeof(rayBounce));

	cudaCompactA= NULL;
	cudaMalloc((void**)&cudaCompactA, size*sizeof(int));

	cudaCompactB= NULL;
	cudaMalloc((void**)&cudaCompactB, size*sizeof(int));

}

void cudaFreeMemory(){
	//std::cout<<"free memory "<<std::endl;
	cudaFree( cudaFirstPass);
	cudaFree( cudaRayPool);
	cudaFree( cudaTempRayPool);
	cudaFree( cudaCompactA);
	cudaFree( cudaCompactB);
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, bool& clear){
  
  int traceDepth = 20; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);			//each block has 8 * 8 threads
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  std::vector<int> lightVec;

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
  
	//store which objects are lights
	if(materials[geoms[i].materialid].emittance > 0)
		lightVec.push_back(i);
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

  //copy materials to memory
  material* cudaMaterials = NULL;
  cudaMalloc((void**)&cudaMaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudaMaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  //copy light ID to memeory
  int numLights = lightVec.size();
  int* lightID = new int[numLights];
  for(int i = 0; i <numLights; ++i)
	  lightID[i] = lightVec[i];
  
  int* cudaLights = NULL;
  cudaMalloc((void**)&cudaLights, numLights*sizeof(int));
  cudaMemcpy( cudaLights, lightID, numLights*sizeof(int), cudaMemcpyHostToDevice);
  
  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;
  cam.focalLength = renderCam->focalLengths[frame];
  cam.aperture = renderCam->apertures[frame];


  //start parallel raytrace
  int imageSize = (int)renderCam->resolution.x * (int)renderCam->resolution.y;
  int numRays = imageSize;
  int tileSquare = tileSize * tileSize;
 
  dim3 threadsPerBlockRayPool(tileSquare);			//each block has 8 * 8 threads
  dim3 fullBlocksPerGridRayPool((int)ceil(float(numRays) / tileSquare));	//numRays/tilesquare blocks, decreases as there are less rays

  //clear image if camera has been moved
  if(clear){
	  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaimage); 
	  clear = false;
  }
  else{
	  //build rays for the first iteration
	  //if(iterations == 1)
		  createRay<<<fullBlocksPerGrid, threadsPerBlock>>>(cam.resolution, cam, traceDepth, iterations, cudageoms, numberOfGeoms, cudaMaterials, 
															numLights, cudaLights, cudaRayPool, cudaCompactA, numRays, cudaimage);

		  //copy the first pass of rays to the temporary ray pool
		  cudaMemcpy(cudaTempRayPool, cudaRayPool, imageSize*sizeof(rayBounce), cudaMemcpyDeviceToDevice);
		  //cudaMemcpy(cudaRayPool, cudaFirstPass, imageSize*sizeof(rayBounce), cudaMemcpyDeviceToDevice);
		  //cudaMemcpy(cudaTempRayPool, cudaFirstPass, imageSize*sizeof(rayBounce), cudaMemcpyDeviceToDevice);

		  //stream compact the temporary ray pool, then copy the result to the current ray pool
		  for(int currDepth = 1; currDepth <= traceDepth; ++currDepth){

			  //do stream compaction using double buffers compact A and B, in 1D
			  cudaStreamCompaction(fullBlocksPerGridRayPool, threadsPerBlockRayPool, imageSize, traceDepth, numRays);

			  if(numRays == 0)
				  break;

			  //update the number of blocks you need
			  fullBlocksPerGridRayPool = dim3((int)ceil(float(numRays) / tileSquare));

			  //raytrace with new number of blocks and rays
			  rayParallelTrace<<<fullBlocksPerGridRayPool, threadsPerBlockRayPool>>>(cam.resolution, iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, 
																					cudaMaterials, numLights, cudaLights, numRays, cudaCompactA, cudaRayPool);

			  //copy new raypool info into tempRaypool so that you can update raypool later
			  cudaMemcpy(cudaTempRayPool, cudaRayPool, numRays*sizeof(rayBounce), cudaMemcpyDeviceToDevice);
			
		  }

  }

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, (float)iterations);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, imageSize*sizeof(glm::vec3), cudaMemcpyDeviceToHost);


  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudaMaterials);
  cudaFree( cudaLights);
  delete geomList;
  delete lightID;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}


void cudaStreamCompaction(dim3 fullBlocksPerGridRayPool, dim3 threadsPerBlockRayPool, int imageSize, int traceDepth, int& numRays){

	int compactDepth = (int)ceil(log((float)numRays)/log(2.0f));		//total number of times you have to compact to get the final sums
	int currCompactDepth = 0;
	int finalDepth = 0;

	for( int d = 1; d <= compactDepth; ++d){
		finalDepth = d;
		currCompactDepth = pow(2.0f, d-1);

		//run compaction for all threads in parallel
		//if(d % 2 ==1){		//if depth is odd, use A as input
		//	streamCompact<<<fullBlocksPerGridRayPool, threadsPerBlockRayPool>>>(numRays, cudaCompactA, cudaCompactB, currCompactDepth);
		//}
		//else{								//if depth is even, use B as input
		//	streamCompact<<<fullBlocksPerGridRayPool, threadsPerBlockRayPool>>>(numRays, cudaCompactB, cudaCompactA, currCompactDepth);
		//}
		
		streamCompact<<<fullBlocksPerGridRayPool, threadsPerBlockRayPool>>>(numRays, cudaCompactA, cudaCompactB, currCompactDepth);
		int* tempCompact = cudaCompactA;
		cudaCompactA = cudaCompactB;
		cudaCompactB = tempCompact;

		//cudaThreadSynchronize();
	}

	checkCUDAError("compact failed!");
	
	int newNumRays = 0;

	//update the number of rays you have left
	//if(finalDepth % 2 == 1){
	//	//use B as the final sum result
	//	cudaMemcpy(&newNumRays, &cudaCompactB[numRays-1], sizeof(int), cudaMemcpyDeviceToHost);
	//	
	//	buildRayPool<<<fullBlocksPerGridRayPool, threadsPerBlockRayPool>>>(cudaCompactB, cudaTempRayPool, cudaRayPool, numRays);
	//}
	//else{
	//	//use A as the final sum result
	//	cudaMemcpy(&newNumRays, &cudaCompactA[numRays-1], sizeof(int), cudaMemcpyDeviceToHost);

	//	buildRayPool<<<fullBlocksPerGridRayPool, threadsPerBlockRayPool>>>(cudaCompactA, cudaTempRayPool, cudaRayPool, numRays);
	//}

	cudaMemcpy(&newNumRays, &cudaCompactA[numRays-1], sizeof(int), cudaMemcpyDeviceToHost);
	buildRayPool<<<fullBlocksPerGridRayPool, threadsPerBlockRayPool>>>(cudaCompactA, cudaTempRayPool, cudaRayPool, numRays);
	//cudaThreadSynchronize();

	//update the number of rays only after you have shifted and built the new pool
	numRays = newNumRays;

	checkCUDAError("building raypool failed!");

}

