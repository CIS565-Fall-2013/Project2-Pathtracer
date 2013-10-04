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
#include "glm/gtc/matrix_transform.hpp"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif


//global variables
rayBounce* cudaRayPool;		//for stream compaction, pool of rays that are still alive
rayBounce* cudaTempRayPool;	//for switching and replacing rays in stream compaction
int* cudaCompactA;
int* cudaCompactB;

glm::vec3* cudaimage;
staticGeom* cudageoms;
material* cudaMaterials;
int* cudaLights;
int numLights;

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


//reflection
__host__ __device__ void reflect(glm::vec3& incoming, glm::vec3 normal, float nDotV){
	
	incoming = glm::normalize(incoming - 2.0f*nDotV*normal);

}

//refraction
__host__ __device__ void refract(glm::vec3 & incoming, glm::vec3 normal, float n1, float n2, bool fresnel, int index, int time){

	float n = n1/n2;
	float c1 = glm::dot(-incoming, normal);			//cos of angle between normal and incident ray
	float c2 = 1.0f - n*n*(1.0f - c1*c1);

	//total internal reflection
	if(c2 < 0){
		//incoming = glm::normalize(incoming - 2.0f*glm::dot(incoming, normal)*normal);
		//why is this working???
		reflect(incoming, normal, c1);
		return;
	}

	//schlick's approximation
	float r0 = (n1-n2)*(n1-n2)/(n1+n2)/(n1+n2);
	float rTheta = r0 + (1.0f-r0)*pow((1.0f-c1), 5);

	thrust::default_random_engine rng(hash(index*time));
	thrust::uniform_real_distribution<float> u01(0,1);

	if(!fresnel || (float)u01(rng) > rTheta)
		incoming = glm::normalize( n*incoming + (n*c1-sqrt(c2))*normal);
	else
		reflect(incoming, normal, -c1);

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
		if(!rayTempPass[index].dead){
			rayPass[compactIn[index] - 1] = rayTempPass[index];
		}
	}
}

//pixel parallel pathtrace
__global__ void pathTrace(glm::vec2 resolution, cameraData cam, int maxDepth, int time, staticGeom* geoms, int numberOfGeoms, material* materials, 
						 int numLights, int* lightID, glm::vec3* colors){

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
		
		while(currDepth <= maxDepth){
	
			//do intersection test
			int objID = -1;
			float len = testGeomIntersection(geoms, numberOfGeoms, firstRay, intersection, normal, objID);

			//if no intersection, return
			if(objID == -1){
				colors[index] += glm::vec3(0);			//set to black
				return;	
			}

			int matID = geoms[objID].materialid;
			material firstMat = materials[matID];

			//check if material is light
			if(materials[matID].emittance > 0){
				colors[index] += finalColor * firstMat.color * firstMat.emittance; 
				return;
			}

			float nDotv = glm::dot(firstRay.direction, normal);
			if(materials[matID].hasReflective == 1){
				firstRay.direction = glm::normalize(firstRay.direction - 2.0f*glm::dot(firstRay.direction, normal)*normal);
			}

			//pure refraction
			else if(firstMat.hasRefractive == 1 && firstMat.hasReflective == 0){
				float n1 = 1.0f;
				float n2 = firstMat.indexOfRefraction;
			
			//check if ray is inside the object
			if(nDotv > 0){
				n1 = n2;
				n2 = 1.0f;
				normal *= -1.0f;			//flip normal
			}

			float n = n1/n2;
			float c1 = glm::dot(-firstRay.direction, normal);
			float c2 = 1.0f - n*n *(1.0f- c1*c1);
			
			if(c2 < 0)
				firstRay.direction = glm::normalize(firstRay.direction - 2.0f*c1*normal);
			else{
				firstRay.direction = glm::normalize( n*firstRay.direction + (n*c1-sqrt(c2))*normal);
				}
			}
			else if(firstMat.hasRefractive == 1 && firstMat.hasReflective == 1){
			float n1 = 1.0f;
			float n2 = firstMat.indexOfRefraction;
			
			//check if ray is inside the object
			if(nDotv > 0){
				n1 = n2;
				n2 = 1.0f;
				normal *= -1.0f;			//flip normal
			}

			float n = n1/n2;
			float c1 = glm::dot(-firstRay.direction, normal);
			float c2 = 1.0f - n*n *(1.0f- c1*c1);
			
			if(c2 < 0)
				firstRay.direction = glm::normalize(firstRay.direction + 2.0f*c1*normal);
			else{
				//schlick's approximation
				float r0 = pow((n1-n2)/(n1+n2), 2);
				float rTheta = r0 +(1-r0)*pow((1.0f-c1), 5);
			
				thrust::default_random_engine rng(hash(index*time));
				thrust::uniform_real_distribution<float> u01(0,1);
			
				if((float)u01(rng) > rTheta)
					firstRay.direction = glm::normalize( n*firstRay.direction + (n*c1-sqrt(c2))*normal);
				else
					firstRay.direction = glm::normalize(firstRay.direction - 2.0f*glm::dot(firstRay.direction, normal)*normal);
			}
		}
			else{
				//calculate diffuse direction
				glm::vec3 seed = generateRandomNumberFromThread(resolution, time+1, x, y);

				if(time % 2 ==0)
					firstRay.direction = calculateRandomDirectionInHemisphere(normal, seed.x, seed.y);
				else
					firstRay.direction = calculateRandomDirectionInHemisphere(normal, seed.y, seed.z);
			}

			currDepth++;
			
			//offset a little to prevent self intersection
			firstRay.origin = intersection + 0.001f * firstRay.direction;
		
			//store the color
			finalColor *= firstMat.color;
		}


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

		//DOF and antialiasing setup
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

		//do intersection test
		int objID = -1;
		float len = testGeomIntersection(geoms, numberOfGeoms, firstRay, intersection, normal, objID);

		//start building the first pool of rays 
		rayBounce firstBounce = rayBounce();
		firstBounce.pixID = index;
		firstBounce.thisRay = firstRay;
		firstBounce.currDepth = 1;
		firstBounce.color = glm::vec3(1);

		//set all rays to alive
		compactIn[index] = 1;

		//save first bounce
		firstPass[index] = firstBounce;
	}

}


__global__ void rayParallelTrace(glm::vec2 resolution, int time, cameraData cam, int maxDepth, glm::vec3* colors, staticGeom* geoms, int numberOfGeoms, 
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
		if(currMat.emittance > 0){
			colors[currBounce.pixID] += currBounce.color * currMat.color * currMat.emittance;
			currBounce.dead = true;
			rayPass[index] = currBounce;
			compactIn[index] = 0;
			return;
		}

		float nDotv = glm::dot(normal, currRay.direction);
		//basic reflection
		if(currMat.hasReflective == 1 && currMat.hasRefractive == 0){
			reflect(currRay.direction, normal, nDotv);
		}
		//refraction
		else if(currMat.hasRefractive == 1){
			float n1 = 1.0f;
			float n2 = currMat.indexOfRefraction;
			
			//check if ray is inside the object
			if(nDotv > 0){
				n1 = n2;
				n2 = 1.0f;
				normal *= -1.0f;			//flip normal
			}

			//refract without using fresnel
			if(currMat.hasReflective == 0)
				refract(currRay.direction, normal, n1, n2, false, index, time);
			else
				//reract with fresnel flag = true
				refract(currRay.direction, normal, n1, n2, true, index, time);
		}
		else{
			//find new direction
			glm::vec3 seed = generateRandomNumberFromThread(resolution, time, 0.2f*index*currBounce.currDepth+1, threadIdx.x);
			if(time % 2 ==0)
				currRay.direction = calculateRandomDirectionInHemisphere(normal, seed.x, seed.y);
			else
				currRay.direction = calculateRandomDirectionInHemisphere(normal, seed.y, seed.z);
		}

		currBounce.dead = false;
		compactIn[index] = 1;
		currBounce.color *= currMat.color;
		currBounce.thisRay.direction = currRay.direction;
		currBounce.thisRay.origin = intersection + 0.001f * currRay.direction;

		rayPass[index] = currBounce;
	}

}


//allocate memory on cuda
void cudaAllocMemory(camera* renderCam, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){

	int size = (int)renderCam->resolution.x*(int)renderCam->resolution.y;
	
	cudaRayPool = NULL;
	cudaMalloc((void**)&cudaRayPool, size*sizeof(rayBounce));

	cudaTempRayPool = NULL;
	cudaMalloc((void**)&cudaTempRayPool, size*sizeof(rayBounce));

	cudaCompactA= NULL;
	cudaMalloc((void**)&cudaCompactA, size*sizeof(int));

	cudaCompactB= NULL;
	cudaMalloc((void**)&cudaCompactB, size*sizeof(int));
  
	//send image to GPU
	cudaimage = NULL;
	cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
	
	//package geometry and materials and sent to GPU
	staticGeom* geomList = new staticGeom[numberOfGeoms];
	std::vector<int> lightVec;

	//get geom from frame 0
	for(int i=0; i<numberOfGeoms; i++){
		staticGeom newStaticGeom;
		newStaticGeom.type = geoms[i].type;
		newStaticGeom.materialid = geoms[i].materialid;
		newStaticGeom.translation = geoms[i].translations[0];
		newStaticGeom.rotation = geoms[i].rotations[0];
		newStaticGeom.scale = geoms[i].scales[0];
		newStaticGeom.transform = geoms[i].transforms[0];
		newStaticGeom.inverseTransform = geoms[i].inverseTransforms[0];
		geomList[i] = newStaticGeom;
  
		//store which objects are lights
		if(materials[geoms[i].materialid].emittance > 0)
			lightVec.push_back(i);
	}

	cudageoms = NULL;
	cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
	cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

	//copy materials to memory
	cudaMaterials = NULL;
	cudaMalloc((void**)&cudaMaterials, numberOfMaterials*sizeof(material));
	cudaMemcpy( cudaMaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

	//copy light ID to memeory
	numLights = lightVec.size();

	int* lightID = new int[numLights];
	for(int i = 0; i <numLights; ++i)
		lightID[i] = lightVec[i];
  
	cudaLights = NULL;
	cudaMalloc((void**)&cudaLights, numLights*sizeof(int));
	cudaMemcpy( cudaLights, lightID, numLights*sizeof(int), cudaMemcpyHostToDevice);

	delete[] geomList;
	delete[] lightID;
}

void cudaFreeMemory(){

	//ray stuff
	cudaFree( cudaRayPool);
	cudaFree( cudaTempRayPool);
	cudaFree( cudaCompactA);
	cudaFree( cudaCompactB);

	//scene stuff
	cudaFree( cudaimage);
	cudaFree( cudageoms );
	cudaFree( cudaMaterials);
	cudaFree( cudaLights);
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, bool& clear){
  
	int traceDepth = 10; //determines how many bounces the raytracer traces

	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);			//each block has 8 * 8 threads
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
 
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
		//pathTrace<<<fullBlocksPerGrid, threadsPerBlock>>>(cam.resolution, cam, traceDepth, iterations, cudageoms, numberOfGeoms, cudaMaterials,
		//													numLights, cudaLights, cudaimage);
		//build rays for the first iteration
		createRay<<<fullBlocksPerGrid, threadsPerBlock>>>(cam.resolution, cam, traceDepth, iterations, cudageoms, numberOfGeoms, cudaMaterials, 															
															numLights, cudaLights, cudaRayPool, cudaCompactA, numRays, cudaimage);

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
		}
	}

	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, (float)iterations);

	//retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, imageSize*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	// make certain the kernel has completed
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}


void cudaStreamCompaction(dim3 fullBlocksPerGridRayPool, dim3 threadsPerBlockRayPool, int imageSize, int traceDepth, int& numRays){

	int compactDepth = (int)ceil(log((float)numRays)/log(2.0f));		//total number of times you have to compact to get the final sums
	int currCompactDepth = 0;

	for( int d = 1; d <= compactDepth; ++d){
		currCompactDepth = pow(2.0f, d-1);

		streamCompact<<<fullBlocksPerGridRayPool, threadsPerBlockRayPool>>>(numRays, cudaCompactA, cudaCompactB, currCompactDepth);
		int* tempCompact = cudaCompactA;
		cudaCompactA = cudaCompactB;
		cudaCompactB = tempCompact;
	}

	checkCUDAError("compact failed!");

	int newNumRays = 0;

	//swap pointers to the ray pool so you alwyas trace with cudaRayPool
	rayBounce* tempRays = cudaRayPool;
	cudaRayPool = cudaTempRayPool;
	cudaTempRayPool = tempRays;

	cudaMemcpy(&newNumRays, &cudaCompactA[numRays-1], sizeof(int), cudaMemcpyDeviceToHost);
	buildRayPool<<<fullBlocksPerGridRayPool, threadsPerBlockRayPool>>>(cudaCompactA, cudaTempRayPool, cudaRayPool, numRays);

	//update the number of rays only after you have shifted and built the new pool
	numRays = newNumRays;
	//std::cout<<newNumRays<<std::endl;

	checkCUDAError("building raypool failed!");

}


