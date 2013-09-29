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
	//std::cin.get();
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

//Kernel that does the initial raycast from the camera.
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
   
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);
  
  //standard camera raycast stuff
  glm::vec3 E = eye;
  glm::vec3 C = view;
  glm::vec3 U = up;
  float fovx = fov.x;
  float fovy = fov.y;
  
  float CD = glm::length(C);
  
  glm::vec3 A = glm::cross(C, U);
  glm::vec3 B = glm::cross(A, C);
  glm::vec3 M = E+C;
  glm::vec3 H = (A*float(CD*tan(fovx*(PI/180))))/float(glm::length(A));
  glm::vec3 V = (B*float(CD*tan(-fovy*(PI/180))))/float(glm::length(B));
  
  float sx = (x)/(resolution.x-1);
  float sy = (y)/(resolution.y-1);
  
  glm::vec3 P = M + (((2*sx)-1)*H) + (((2*sy)-1)*V);
  glm::vec3 PmE = P-E;
  glm::vec3 R = E + (float(200)*(PmE))/float(glm::length(PmE));
  
  glm::vec3 direction = glm::normalize(R);
  //major performance cliff at this point, TODO: find out why!
  ray r;
  r.origin = eye;
  r.direction = direction;
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

__global__ void finalizeraycolor(glm::vec2 resolution,glm::vec3* colBounce,glm::vec3* colIters,glm::vec3* colors,float iters, ray* r)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  int xx = r[index].x;
  int yy = r[index].y;
  int newindex =  xx + (yy * resolution.x);
  if(iters < 0.05)
	  iters = 1.0f;
  if((x<=resolution.x && y<=resolution.y)){
	  colIters[index] = colIters[index] + colBounce[index];
	  colIters[index][0] = ((colIters[index][0] * (iters - 1)) + colBounce[index][0])  / iters ;
	  colIters[index][1] = ((colIters[index][1] * (iters - 1)) + colBounce[index][1])  / iters ;
	  colIters[index][2] = ((colIters[index][2] * (iters - 1)) + colBounce[index][2])  / iters ;
	  colors[index] = colIters[newindex] ; //colBounce[index] ;//
	  colBounce[index] = glm::vec3(1,1,1);
  }
}

__global__ void initializeray(glm::vec2 resolution, float time,cameraData cam, ray* r,glm::vec3* colBounce,glm::vec3* colIters){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if((x<=resolution.x && y<=resolution.y)){
  ray rnew = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
  r[index].direction = rnew.direction;
  r[index].origin = rnew.origin;
  r[index].x = x ;
  r[index].y = y ;
  r[index].life = true ;
  colBounce[index] = glm::vec3(1,1,1);
  colIters[index] = glm::vec3(0,0,0);
  }
}
//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, float bounce, cameraData cam, int rayDepth, glm::vec3* colors, 
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials,ray* newr, glm::vec3* colBounce){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  //ray r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
  ray r = newr[index];
  glm::vec3 curIps;
  glm::vec3 curNorm;
  if((x<=resolution.x && y<=resolution.y)){
	
    float MAX_DEPTH = 100000000000000000;
    float depth = MAX_DEPTH;
	int geoIndex = -1; 
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
		  geoIndex = i;
		  curIps  =  intersectionPoint;
		  curNorm =  intersectionNormal;
        }
    }

	if(geoIndex != -1 && materials[geoms[geoIndex].materialid].emittance < 0.01f && (r.life == true))
	{
	
	thrust::default_random_engine rng (hash (time * index));
	thrust::uniform_real_distribution<float> xi1(0,1);
    thrust::uniform_real_distribution<float> xi2(0,1);

	
	newr[index].direction =  glm::normalize(calculateRandomDirectionInHemisphere(glm::normalize(curNorm),  (float)xi1(rng),(float)xi2(rng)));
	newr[index].origin    =  curIps + newr[index].direction  * 0.001f ; //glm::vec3 neyep = dips + ref1 * 0.001f ;
	//colors[index] = materials[geoms[geoIndex].materialid].color;
	colBounce[index] = colBounce[index] * materials[geoms[geoIndex].materialid].color;
	}
	else if(geoIndex != -1 && materials[geoms[geoIndex].materialid].emittance > 0.01f && (r.life == true))
	{
	colBounce[index] = colBounce[index] * materials[geoms[geoIndex].materialid].emittance ;
	newr[index].life == false ;
	}
	else if(geoIndex != -1 && materials[geoms[geoIndex].materialid].emittance > 0.01f && (r.life == false))
	{
	colBounce[index] = colBounce[index];
	}
	else
	{
	colBounce[index] = colBounce[index] * glm::vec3(0,0,0);
	}
	//colBounce[index] = glm::vec3(0.01,0.01,0.01) ;//colors[index] = glm::vec3(0,0,0);


    //colors[index] = generateRandomNumberFromThread(resolution, time, x, y);
   }
}


//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)) , (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
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

   //Allocate memory for ray pool
  ray* raypool = NULL;
  cudaMalloc((void**)&raypool, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(ray));
 
  //Allocate memory to store color for bounces
  glm::vec3* colorBounce = NULL;
  cudaMalloc((void**)&colorBounce, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));

  //Allocate memory to store color for each iteration accumulation
  glm::vec3* colorIters = NULL;
  cudaMalloc((void**)&colorIters , (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));

  //Initialize the ray values
  initializeray<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,(float)iterations,cam,raypool,colorBounce,colorIters);
  cudaThreadSynchronize();
  //kernel launches
  for(int bounce = 1; bounce <=5; ++bounce)
  {
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, (float)bounce, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials,raypool,colorBounce);
  cudaThreadSynchronize();
  }

  finalizeraycolor<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,colorBounce,colorIters,cudaimage,(float)iterations,raypool);
  cudaThreadSynchronize();
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamaterials );
  cudaFree(colorBounce);
  //cudaFree(colorIters);
  cudaFree(raypool);
  delete [] geomList;

  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
