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
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/matrix_inverse.hpp"
#include "materials.h"


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


//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov,float focalDist,float aperture, int dof){
  
  ray r;
  float theta = fov.x*PI/180.0f;
  float phi = fov.y*PI/180.0f;

  glm::vec3 A = glm::cross(view,up);
  glm::vec3 B = glm::cross(A,view);
  glm::vec3 M = eye+view; 
  glm::vec3 H = glm::normalize(A)*glm::length(view)*tan(theta);
  glm::vec3 V = glm::normalize(B)*glm::length(view)*tan(phi);

  float sx= -1.0f; //(float)x/(resolution.x-1);
  float sy = -1.0f;//1.0 - (float)y/(resolution.y-1);
  
  thrust::default_random_engine rng(hash(43231*time));
  thrust::uniform_real_distribution<float> u01(-0.95,0.95);
  thrust::uniform_real_distribution<float> u02(-1.0,1.0);

  while( sx<=0.0f || sx>=1.0f || sx<=0.0f || sy>=1.0f)
	{
		float xrand = x + u01(rng);
		float yrand = y + u01(rng);
		sx = xrand/(resolution.x-1);
		sy = 1.0f - yrand/ (resolution.y-1);
	}

  glm::vec3 P = M + (2*sx-1)*H + (2*sy - 1)*V;
  
  r.origin = eye;
  r.direction = glm::normalize(P-r.origin);

  if(dof && focalDist!=0)
  {
	  float t = (focalDist-r.origin.z)/r.direction.z;
	  P = r.origin + t*r.direction;
	  glm::vec3 xaperture = aperture*H;
	  glm::vec3 yaperture = aperture*V;
	  r.origin= r.origin + u02(rng)*xaperture + u02(rng)*yaperture;
	  r.direction = glm::normalize(P-r.origin);
  }
  r.direction = glm::normalize(P-r.origin);
  return r;
}


//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec4* image){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index<= resolution.x*resolution.y){
      image[index] = glm::vec4(0,0,0,-1);
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec4* image, ray* rays,int iterations){
  
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int x = rays[index].pixelIndex.x;
	int y = rays[index].pixelIndex.y;
	int pixelIndex = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;    
      color.x = image[pixelIndex].x*255.0/iterations;
      color.y = image[pixelIndex].y*255.0/iterations;
      color.z = image[pixelIndex].z*255.0/iterations;

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
      PBOpos[pixelIndex].w = 0;
      PBOpos[pixelIndex].x = color.x;     
      PBOpos[pixelIndex].y = color.y;
      PBOpos[pixelIndex].z = color.z;

  }
}

__device__ bool isLight(int objId, int* lights, int numberOfLights)
{
	for (int i=0; i<numberOfLights; ++i)
		if (lights[i] == objId)
			return true;
	return false;
}


//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, float bounce, cameraData cam, int rayDepth, glm::vec4* colors, 
                            int* objidbuffer, staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, 
							map* maps, int numberOfMaps,int* lights, int numberOfLights,ray* rays,int dof, int* perlinData){

  //int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  //int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  //int index = x + (y * resolution.x);
	
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int pixelIndex;
  int x=-1;
  int y=-1;
  ray r;

  if (bounce==1)
  {
	y = (int) (index/(int)resolution.x);
	x = (int) (index%(int)resolution.x);
	pixelIndex = index;
	r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov,cam.focalDist,cam.aperture,dof);
	r.active = true;
	r.pixelIndex = glm::vec2(x,y);
	r.rayColor = glm::vec3(1,1,1);
	rays[index].rayColor = r.rayColor;
	rays[index].pixelIndex = r.pixelIndex;
  }
  else
  {
	  r = rays[index];
	  x = r.pixelIndex.x;
	  y = r.pixelIndex.y;
	  pixelIndex = x + (y*resolution.x);
  }

  if((x<=resolution.x && y<=resolution.y && r.active)){
	glm::vec3 intersectionPoint;
	glm::vec3 intersectionNormal;

	int objId = findNearestGeometricIntersection(r,intersectionPoint,intersectionNormal,geoms,numberOfGeoms);

	if (bounce==1 && time< 1.5f)
	{
		colors[pixelIndex].w = intersectionPoint.z;
		objidbuffer[pixelIndex] = objId;
	}
	if (objId == -1)
	{
		rays[index].active = false;
		rays[index].rayColor = glm::vec3(0,0,0);
		return;
	}
	material mtl = materials[geoms[objId].materialid];
	if (isLight(objId,lights,numberOfLights))
	{
		rays[index].active = false;
		rays[index].rayColor.x *= mtl.color.x*mtl.emittance;
		rays[index].rayColor.y *= mtl.color.y*mtl.emittance;
		rays[index].rayColor.z *= mtl.color.z*mtl.emittance;
		colors[pixelIndex].x += rays[index].rayColor.x;
		colors[pixelIndex].y += rays[index].rayColor.y;
		colors[pixelIndex].z += rays[index].rayColor.z;
		return;
	}

	glm::vec3 emittedColor;
	glm::vec3 unabsorbedColor;
	int bsdf = calculateBSDF(r,intersectionPoint,intersectionNormal,emittedColor,colors[index],unabsorbedColor,mtl,bounce*time*index);
	
	if (bsdf == 0)
	{
		glm::vec3 surfaceColor = getSurfaceColor(intersectionPoint,intersectionNormal,mtl,objId,geoms,maps,perlinData);
		r.rayColor.x *= surfaceColor.x;
		r.rayColor.y *= surfaceColor.y;
		r.rayColor.z *= surfaceColor.z;
	}
	else if(bsdf == 1)
	{
		r.rayColor.x *= mtl.specularColor.x;
		r.rayColor.y *= mtl.specularColor.y;
		r.rayColor.z *= mtl.specularColor.z;
	}
	else if (bsdf == 2)
	{
		glm::vec3 surfaceColor = getSurfaceColor(intersectionPoint,intersectionNormal,mtl,objId,geoms,maps,perlinData);
		r.rayColor.x *= surfaceColor.x;
		r.rayColor.y *= surfaceColor.y;
		r.rayColor.z *= surfaceColor.z;
	}

	rays[index].origin = r.origin + 0.001f*r.direction;
	rays[index].direction = r.direction;
	rays[index].active = r.active;
	rays[index].pixelIndex = r.pixelIndex;
	rays[index].rayColor = r.rayColor;
   }
	
}

__global__ void createBinaryActiveArray(glm::vec2 resolution,ray* rays,int* activeRaysArray, int* dNumActive)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < *dNumActive)
	{
		activeRaysArray[index] = rays[index].active?1:0;
	}
}

__global__ void parallelScanOnGPU(int* cudaActiveArray, int* gpuParallelScanTempArray, int* dNumActive,int d)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	int exponent = powf(2,d-1);
	if (index< (*dNumActive) )
	{
	if( index>= exponent)
		gpuParallelScanTempArray[index] = cudaActiveArray[index-exponent] + cudaActiveArray[index];
	else
		gpuParallelScanTempArray[index] = cudaActiveArray[index];
	}

}

__host__ void parallelScanActiveArray(int* cudaActiveArray,int* gpuParallelScanTempArray,int* hNumActive,int* dNumActive)
{
	int numberOfThreads = *hNumActive;
	float logN  = logf(numberOfThreads);
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize*tileSize);
	dim3 fullBlocksPerGrid ( (int) ceil( (float)numberOfThreads/(tileSize*tileSize)));
	int* cudaActiveArrayCopy = NULL;
	cudaMalloc((void**)&cudaActiveArrayCopy,numberOfThreads*sizeof(int));
	cudaMemcpy(cudaActiveArrayCopy,cudaActiveArray,numberOfThreads*sizeof(int),cudaMemcpyDeviceToDevice);
	int* activeArray = cudaActiveArrayCopy;
	int* tempArray = gpuParallelScanTempArray;
	int dmax = ceil(logN)+3; 

	for(int d=1; d<= dmax; ++d)
	{
	  parallelScanOnGPU<<<fullBlocksPerGrid, threadsPerBlock>>>(activeArray,tempArray,dNumActive,d);
	  int *swapTemp = activeArray;
	  activeArray = tempArray;
	  tempArray = swapTemp;
	}	
	
	if (tempArray != gpuParallelScanTempArray)
		cudaMemcpy(activeArray,tempArray,numberOfThreads*sizeof(int),cudaMemcpyDeviceToDevice);

	cudaMemcpy(dNumActive,&gpuParallelScanTempArray[numberOfThreads-1],sizeof(int),cudaMemcpyDeviceToDevice);

	cudaFree(cudaActiveArrayCopy);
}

__global__ void streamCompact(ray* rays, ray* tempRays, int* rayActiveArray, int* scannedArray,int* dNumActive)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < *dNumActive && rayActiveArray[index])
	{
		rays[scannedArray[index]-1] = tempRays[index];
	}
}

__global__ void duplicateRaysArray(ray* dest, ray* src, int* dNumActive)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < *dNumActive)
	{
		dest[index] = src[index];
	}
}

struct is_not_active
{
	__device__ bool operator() (const ray r)
	{
		return !r.active;
	}
};

struct is_active
{
	__device__ bool operator() (const ray r)
	{
		return r.active;
	}
};


__global__ void moveWorld( staticGeom* geoms, staticGeom* prevGeoms, float t,int numberOfGeoms)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(index < numberOfGeoms)
	{
		t = 1 - t*t;
		glm::vec3 newTranslation = t*geoms[index].translation + (1-t)*prevGeoms[index].translation;
		glm::mat4 translationMat = glm::translate(glm::mat4(), newTranslation);
		glm::mat4 rotationMat = glm::rotate(glm::mat4(), geoms[index].rotation.x, glm::vec3(1,0,0));
		rotationMat = rotationMat*glm::rotate(glm::mat4(), geoms[index].rotation.y, glm::vec3(0,1,0));
		rotationMat = rotationMat*glm::rotate(glm::mat4(), geoms[index].rotation.z, glm::vec3(0,0,1));
		glm::mat4 scaleMat = glm::scale(glm::mat4(), geoms[index].scale);
		glm::mat4 a =  translationMat*rotationMat*scaleMat;
		cudaMat4 m; 
		glm::mat4 b = glm::transpose(a);
		m.x = b[0];
		m.y = b[1];
		m.z = b[2];
		m.w = b[3];
		geoms[index].transform = m;

		a = glm::transpose(glm::inverse(a));
		m.x = a[0];
		m.y = a[1];
		m.z = a[2];
		m.w = a[3];
		geoms[index].inverseTransform = m;
	}
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials,map* maps,int numberOfMaps, geom* geoms, int numberOfGeoms,int mblur,int dof){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces
  int perlinNumbers[512] = { 151,160,137,91,90,15,
  131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
  190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
  88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,
  77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
  102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208,89,18,169,200,196,
  135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,
  5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
  23,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167,43,172,9,
  129,22,39,253,19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
  251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,
  49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127,4,150,254,
  138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
  151,160,137,91,90,15,
  131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
  190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
  88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,
  77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
  102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208,89,18,169,200,196,
  135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,
  5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
  23,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167,43,172,9,
  129,22,39,253,19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
  251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,
  49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127,4,150,254,
  138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
  };


  // set up crucial magic
  int numberOfThreads = (int)(renderCam->resolution.x)*(int)(renderCam->resolution.y);
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize*tileSize);
  dim3 fullBlocksPerGrid ( (int) ceil( (float)numberOfThreads/(tileSize*tileSize)));

  //send image to GPU
  glm::vec4* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec4));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec4), cudaMemcpyHostToDevice);

  int* cudaobjidbuffer = NULL;
  cudaMalloc((void**)&cudaobjidbuffer, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(int));
  
   //package lights
  std::vector<int> lightsVector;

  ray* cudarays = NULL;
  cudaMalloc((void**)&cudarays, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(ray));

  int* cudaperlindata = NULL;
  cudaMalloc((void**)&cudaperlindata, 512*sizeof(int));
  cudaMemcpy( cudaperlindata, perlinNumbers, 512*sizeof(int), cudaMemcpyHostToDevice);

  /*ALL STUFF FOR MANUAL STREAM COMPACTION
  ray* cudatemprays = NULL;
  cudaMalloc((void**)&cudatemprays, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(ray));

  ////TEST SCAN
  //const int testNum = 2048;
  //int test[testNum];
  //for(int i=0; i<testNum;++i)
  //{
	 // test[i] = 1;
  //}

  int* cudaActiveArray = NULL;
  cudaMalloc((void**)&cudaActiveArray, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(int));
  
  ////TEST SCAN
  //cudaMalloc((void**)&cudaActiveArray, testNum*sizeof(int));
  //cudaMemcpy( cudaActiveArray,test, testNum*sizeof(int), cudaMemcpyHostToDevice);

  int* dParallelScanTempArray = NULL;
  cudaMalloc((void**)&dParallelScanTempArray, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(int));

  ////TEST SCAN
  //cudaMalloc((void**)&dParallelScanTempArray, testNum*sizeof(int));

  ////TEST SCAN
  //numberOfThreads = testNum;

  int* dNumActiveRays = NULL;
  cudaMalloc((void**)&dNumActiveRays,sizeof(int));
  cudaMemcpy( dNumActiveRays,&numberOfThreads, sizeof(int), cudaMemcpyHostToDevice);
  */

  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  staticGeom* geomListPrevFrame = new staticGeom[numberOfGeoms];

  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
	staticGeom prevStaticGeom;
	int prevFrame = frame-1;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;

	if (frame==0)
	{
		prevFrame = frame;
	}
    prevStaticGeom.type = geoms[i].type;
    prevStaticGeom.materialid = geoms[i].materialid;
    prevStaticGeom.translation = geoms[i].translations[prevFrame];
    prevStaticGeom.rotation = geoms[i].rotations[prevFrame];
    prevStaticGeom.scale = geoms[i].scales[prevFrame];
    prevStaticGeom.transform = geoms[i].transforms[prevFrame];
    prevStaticGeom.inverseTransform = geoms[i].inverseTransforms[prevFrame];
    geomListPrevFrame[i] = prevStaticGeom;

	if (materials[geoms[i].materialid].emittance > 0.0f)
		lightsVector.push_back(i);
  }
  

  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
 
  staticGeom* cudageomsPrevFrame = NULL;
  cudaMalloc((void**)&cudageomsPrevFrame, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageomsPrevFrame, geomListPrevFrame, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

  material* cudamaterials = NULL;
  cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  map* cudamaps = NULL;
  cudaMalloc((void**)&cudamaps, numberOfMaps*sizeof(map));
  cudaMemcpy( cudamaps, maps, numberOfMaps*sizeof(map), cudaMemcpyHostToDevice);


  int numberOfLights = lightsVector.size();
  int* cudalights = NULL;
  cudaMalloc( (void**)&cudalights, numberOfLights*sizeof(int));
  cudaMemcpy(cudalights,&lightsVector[0],numberOfLights*sizeof(int),cudaMemcpyHostToDevice);

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;
  cam.aperture = renderCam->aperture;
  cam.focalDist = renderCam->focalDist;

 /*TESTING STREAM COMPACTION
  int t = numberOfThreads;
  for(int k=1; k<=t; k++)
  {
  cudaMemcpy( dNumActiveRays, &k, sizeof(int),cudaMemcpyHostToDevice);
  parallelScanActiveArray(cudaActiveArray,dParallelScanTempArray,&k,dNumActiveRays);
  cudaMemcpy(&numberOfThreads,dNumActiveRays,sizeof(int),cudaMemcpyDeviceToHost);
  std::cout<<"NUM OF ACTIVE THREADS: "<<numberOfThreads<<std::endl;
  if (k == t)
  {
  int* resultArray = new int[k];
  cudaMemcpy(resultArray,cudaActiveArray,k*sizeof(int),cudaMemcpyDeviceToHost);
  for(int i=0; i<k;++i)
	  std::cout<<resultArray[i]<<" ";
  std::cout<<std::endl;
    cudaMemcpy(resultArray,dParallelScanTempArray,k*sizeof(int),cudaMemcpyDeviceToHost);
  for(int i=0; i<k;++i)
	  std::cout<<resultArray[i]<<" ";
  std::cout<<std::endl;
  
  delete[] resultArray;
  }
  }*/
   
  //Prepare scene for motion blur
 if(mblur)
 {
	 dim3 mblurThreadsPerBlock(numberOfGeoms);
	 dim3 mblurBlocksPerGrid( (int) ceil( (float)numberOfGeoms/(tileSize*tileSize)));
	 thrust::default_random_engine rng(hash(48589.0f*iterations));
     thrust::uniform_real_distribution<float> u01(0,1);
	 float interpolant = u01(rng);

	 moveWorld<<<mblurBlocksPerGrid,mblurThreadsPerBlock>>>(cudageoms, cudageomsPrevFrame, interpolant,numberOfGeoms);
 }

  thrust::device_ptr<ray> thrustRaysArray = thrust::device_pointer_cast(cudarays);
 
  //kernel launches
  for(int bounce = 1; bounce <= 8; ++bounce)
  {
	dim3 compactedBlocksPerGrid ( (int) ceil( (float)numberOfThreads/(tileSize*tileSize)));
	raytraceRay<<<compactedBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, (float)bounce, cam, traceDepth, cudaimage,cudaobjidbuffer, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials,cudamaps,numberOfMaps, cudalights,numberOfLights,cudarays,dof,cudaperlindata);
	numberOfThreads = thrust::partition(thrustRaysArray,thrustRaysArray+numberOfThreads,is_active()) - thrustRaysArray;
	//numberOfThreads = thrust::remove_if(thrustRaysArray,thrustRaysArray+numberOfThreads,is_not_active()) - thrustRaysArray;

	//MANUAL STREAM COMPACTION ATTEMPT
	//duplicateRaysArray<<<compactedBlocksPerGrid, threadsPerBlock>>>(cudatemprays,cudarays,dNumActiveRays);
	//createBinaryActiveArray<<<compactedBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,cudarays,cudaActiveArray,dNumActiveRays);
	//thrust::inclusive_scan(thrustActiveArray,thrustActiveArray+numberOfThreads,thrustTempScanArray);
	//streamCompact<<<compactedBlocksPerGrid, threadsPerBlock>>>(cudarays,cudatemprays,cudaActiveArray,dParallelScanTempArray,dNumActiveRays);
	//cudaMemcpy(dNumActiveRays,&dParallelScanTempArray[numberOfThreads-1],sizeof(int),cudaMemcpyDeviceToDevice);
	//numberOfThreads = thrustTempScanArray[numberOfThreads-1];
	//parallelScanActiveArray(cudaActiveArray,dParallelScanTempArray,&numberOfThreads,dNumActiveRays);
	//cudaMemcpy(&numberOfThreads,dNumActiveRays,sizeof(int),cudaMemcpyDeviceToHost);
	//streamCompact<<<compactedBlocksPerGrid, threadsPerBlock>>>(cudarays,cudatemprays,cudaActiveArray,dParallelScanTempArray,dNumActiveRays);
  }

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage,cudarays,iterations);
  


  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec4), cudaMemcpyDeviceToHost);
  if(iterations == 1)
	cudaMemcpy( renderCam->objIdBuffer, cudaobjidbuffer, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(int), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamaterials );
  cudaFree(cudamaps);
  cudaFree(cudalights);
  cudaFree(cudarays);
  /*CLEANUP FOR MANUAL STREAM COMPACTION DATA
  cudaFree(cudatemprays);
  cudaFree(cudaActiveArray);
  cudaFree(dParallelScanTempArray);
  cudaFree(dNumActiveRays);*/
  cudaFree(cudaobjidbuffer);
  cudaFree(cudaperlindata);
  delete [] geomList;
  delete [] geomListPrevFrame;

  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}

/* THE CODE BELOW THIS IS FOR FUNCTIONS THAT RUN IN "TEXTURE" MODE. 
THIS CODE IS DUPLICATED TO AVOID UNNECESSARY BRANCHING IN THE GPU*/

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRayT(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec4* colors, 
                            int* objidbuffer, staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, 
							map* maps, int numberOfMaps,int* lights, int numberOfLights,ray* rays,int dof, int* perlinData){

  //int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  //int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  //int index = x + (y * resolution.x);
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	int y = (int) (index/(int)resolution.x);
	int x = (int) (index%(int)resolution.x);
	int pixelIndex = index;
	ray r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov,cam.focalDist,cam.aperture,dof);  
	rays[index].pixelIndex = glm::vec2(x,y);

  if(x<=resolution.x && y<=resolution.y){
	glm::vec3 intersectionPoint;
	glm::vec3 intersectionNormal;

	int objId = findNearestGeometricIntersection(r,intersectionPoint,intersectionNormal,geoms,numberOfGeoms);

	if (time< 1.5f)
	{
		colors[pixelIndex].w = intersectionPoint.z;
		objidbuffer[pixelIndex] = objId;
	}
	if (objId == -1)
	{
		return;
	}
	material mtl = materials[geoms[objId].materialid];
	if (isLight(objId,lights,numberOfLights))
	{
		colors[pixelIndex].x += mtl.color.x*mtl.emittance;
		colors[pixelIndex].y += mtl.color.y*mtl.emittance;
		colors[pixelIndex].z += mtl.color.z*mtl.emittance;
		return;
	}
	
	glm::vec3 surfaceColor = getSurfaceColor(intersectionPoint,intersectionNormal,mtl,objId,geoms,maps,perlinData);
	colors[pixelIndex].x += surfaceColor.x;
	colors[pixelIndex].y += surfaceColor.y;
	colors[pixelIndex].z += surfaceColor.z;
}
  }

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCoreT(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials,map* maps,int numberOfMaps, geom* geoms, int numberOfGeoms,int mblur,int dof){
  
  int traceDepth = 1;
  int perlinNumbers[512] = { 151,160,137,91,90,15,
  131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
  190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
  88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,
  77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
  102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208,89,18,169,200,196,
  135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,
  5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
  23,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167,43,172,9,
  129,22,39,253,19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
  251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,
  49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127,4,150,254,
  138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
  151,160,137,91,90,15,
  131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
  190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
  88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,
  77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
  102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208,89,18,169,200,196,
  135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,
  5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
  23,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167,43,172,9,
  129,22,39,253,19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
  251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,
  49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127,4,150,254,
  138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
  };


  // set up crucial magic
  int numberOfThreads = (int)(renderCam->resolution.x)*(int)(renderCam->resolution.y);
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize*tileSize);
  dim3 fullBlocksPerGrid ( (int) ceil( (float)numberOfThreads/(tileSize*tileSize)));

  //send image to GPU
  glm::vec4* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec4));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec4), cudaMemcpyHostToDevice);

  int* cudaobjidbuffer = NULL;
  cudaMalloc((void**)&cudaobjidbuffer, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(int));
  
   //package lights
  std::vector<int> lightsVector;

  ray* cudarays = NULL;
  cudaMalloc((void**)&cudarays, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(ray));

  int* cudaperlindata = NULL;
  cudaMalloc((void**)&cudaperlindata, 512*sizeof(int));
  cudaMemcpy( cudaperlindata, perlinNumbers, 512*sizeof(int), cudaMemcpyHostToDevice);


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


	if (materials[geoms[i].materialid].emittance > 0.0f)
		lightsVector.push_back(i);
  }
  

  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
 

  material* cudamaterials = NULL;
  cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  map* cudamaps = NULL;
  cudaMalloc((void**)&cudamaps, numberOfMaps*sizeof(map));
  cudaMemcpy( cudamaps, maps, numberOfMaps*sizeof(map), cudaMemcpyHostToDevice);


  int numberOfLights = lightsVector.size();
  int* cudalights = NULL;
  cudaMalloc( (void**)&cudalights, numberOfLights*sizeof(int));
  cudaMemcpy(cudalights,&lightsVector[0],numberOfLights*sizeof(int),cudaMemcpyHostToDevice);

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;
  cam.aperture = renderCam->aperture;
  cam.focalDist = renderCam->focalDist;

 
  //kernel launches

  dim3 compactedBlocksPerGrid ( (int) ceil( (float)numberOfThreads/(tileSize*tileSize)));
  raytraceRayT<<<compactedBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage,cudaobjidbuffer, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials,cudamaps,numberOfMaps, cudalights,numberOfLights,cudarays,dof,cudaperlindata);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage,cudarays,iterations);
  
  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec4), cudaMemcpyDeviceToHost);
  if(iterations == 1)
	cudaMemcpy( renderCam->objIdBuffer, cudaobjidbuffer, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(int), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamaterials );
  cudaFree(cudamaps);
  cudaFree(cudalights);
  cudaFree(cudarays);
  cudaFree(cudaobjidbuffer);
  cudaFree(cudaperlindata);
  delete [] geomList;

  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
