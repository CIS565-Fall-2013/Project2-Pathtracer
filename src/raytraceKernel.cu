// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
// Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

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
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>

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

// Ray cast from the camera with anti-aliasing from stochastic sampling with changes of the x, y data type from integer to float
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, float x, float y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov, bool DOFflag){
  // Focal length in pixels
  float focal = resolution.y / 2.0f / tan(fov.y * (PI / 180)); 

  view = glm::normalize(view);
  up   = glm::normalize(up);
  glm::vec3 right = glm::cross(view, up);

  // 3-D raycast vector from the camera in pixels
  glm::vec3 rayCast = focal * view + (x - resolution.x / 2.0f) * right - (y - resolution.y / 2.0f) * up;

  // Output the data
  ray r;
  r.origin = eye;
  r.direction = glm::normalize(rayCast);

  // Depth of field referring to http://http.developer.nvidia.com/GPUGems/gpugems_ch23.html
  if (!DOFflag)
    return r;

  // Generate random angles
  thrust::default_random_engine rng(hash(time));
  thrust::uniform_real_distribution<float> u01(0, PI);
  thrust::uniform_real_distribution<float> u02(-1, 1);
  
  float angle = (float)u01(rng);
  // Radius of the circle of confusion (Coc)
  float radius = 0.5 * (float)u02(rng);

  glm::vec3 focalPoint = eye + r.direction * focal;
  eye = eye + radius * cos(angle) * up + radius * sin(angle) * right;
  r.origin    = eye; 
  r.direction = glm::normalize(focalPoint - eye);
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

__host__ __device__ int findPrimitiveObject(ray cameraRay, staticGeom* geoms, int numberOfGeoms, glm::vec3& normal, glm::vec3& intersectionPoint, float& distance) {
  // Initialize the object index, distance and the intersection point and the normal
  int objectIndex = -1;
  float updateDistance;
  distance = 1e7f;
  glm::vec3 updateIntersectionPoint;
  glm::vec3 updateNormal;

  // Find the nearest object to the camera
  for(int i = 0; i < numberOfGeoms; ++ i) {
    glm::vec3 updateIntersectionPoint, updateNormal;
    if(geoms[i].type == SPHERE){
      updateDistance = sphereIntersectionTest(geoms[i], cameraRay, updateIntersectionPoint, updateNormal);
      if(updateDistance > EPSILON && updateDistance < distance){
        distance = updateDistance;
        intersectionPoint = updateIntersectionPoint;
	    normal = updateNormal;
	    objectIndex = i;
      }
    } else if(geoms[i].type == CUBE) {
      updateDistance = boxIntersectionTest(geoms[i], cameraRay, updateIntersectionPoint, updateNormal);
      if(updateDistance > EPSILON && updateDistance < distance){
	  distance = updateDistance;
	  intersectionPoint = updateIntersectionPoint;
	  normal = updateNormal;
	  objectIndex = i;
      }
    } // if type
  } // for i
  return objectIndex;
}

__host__ __device__ void rayPoolUpdate(material materials, glm::vec3 intersectionPoint, glm::vec3 normal, rayPool &rays, float time, cameraData cam, glm::vec3 &colors) {
  // Determine if the object itself can be treated as light source
  if(materials.emittance > 0) {
    // If the object is a light source, then the color in the image is it own
    colors = rays.colors * materials.color * materials.emittance;
  } else {
    ray currentRay;
    currentRay.origin = rays.ray.origin;
    currentRay.direction = rays.ray.direction;
	glm::vec3 currentColor = rays.colors;
    calculateBSDF(currentRay, rays, intersectionPoint, normal, currentColor, rays.colors, materials, time, cam);
	colors = rays.colors * materials.emittance;
	rays.isTerminated = false;
	rays.ray.origin = intersectionPoint;
  } // if material
}


//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, material* materials, int numberOfMaterials, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, rayPool* rays){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  // Find the primitive object
  glm::vec3 normal, intersectionPoint;
  float distance;
  int objectIndex = findPrimitiveObject(rays[index].ray, geoms, numberOfGeoms, normal, intersectionPoint, distance);
  // If there is no object, return
  if(objectIndex == -1)
    return;

  // Initialize the material index
  int materialIndex = geoms[objectIndex].materialid;

  // Check whether the ray is terminated, if not then update the ray pool for the next bounce
  if (!rays[index].isTerminated) {
	rays[index].isTerminated = true;
    if (distance < EPSILON)
	  return;
	rayPoolUpdate(materials[materialIndex], intersectionPoint, normal, rays[index], rayDepth * rays[index].index * time, cam, colors[rays[index].index]);
  }
} // end of function
 
// Initialize the ray pool
__global__ void rayGenerator (glm::vec2 resolution, float time, material* materials, int numberOfMaterials, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, rayPool* rays) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  // Generate random numbers so as to stochastic sampling each pixel referring http://pages.cpsc.ucalgary.ca/~mario/courses/591-691-W06/PR/3-ray-tracing/3-advanced/readings/Cook_Stochastic_Sampling_TOG86.pdf
  thrust::default_random_engine rng(hash(index * time));
  thrust::uniform_real_distribution<float> u01(-0.5,0.5);
  thrust::uniform_real_distribution<float> u02(-0.5,0.5);
  float xSample = x + u01(rng);
  float ySample = y + u02(rng);

  // Initialize the first ray from the camera
  bool DOFflag = false;
  ray cameraRay = raycastFromCameraKernel(resolution, index * time, xSample, ySample, cam.position, cam.view, cam.up, cam.fov, DOFflag);
  
  // Initialize the pixel color, ray and flags
  colors[index] = glm::vec3(1);
  rays[index].colors = glm::vec3(1);
  rays[index].isTerminated = true;
  rays[index].ray.origin = glm::vec3(0);
  rays[index].ray.direction = glm::vec3(0);

  if (x > resolution.x || y > resolution.y) 
	  return;
  
  // Record the origin, direction and index for the next bounce
  glm::vec3 normal, intersectionPoint;
  float distance;
  int objectIndex = findPrimitiveObject(cameraRay, geoms, numberOfGeoms, normal, intersectionPoint, distance);
  if (distance < EPSILON || objectIndex == -1)
	return;
  
  // Update the ray pool with the first ray from the camera
  int materialIndex = geoms[objectIndex].materialid;
  rays[index].ray.origin    = cameraRay.origin;
  rays[index].ray.direction = cameraRay.direction;
  rayPoolUpdate(materials[materialIndex], intersectionPoint, normal, rays[index], index*time, cam, colors[index]);
  rays[index].index  = index;
}

// Filter the noise between iteration and frame
__global__ void frameColorFilter (glm::vec2 resolution, float time, glm::vec3* currentColor, glm::vec3* updateColor) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if (x > resolution.x || y > resolution.y)
    return;
  updateColor[index] = currentColor[index] * (time - 1) / time + updateColor[index] / time;
  glm::clamp(updateColor[index], 0, 1);
}

// Check whether the bounce is terminated or not
struct IsRayTerminated {
  __host__ __device__ bool operator() (const rayPool& rays) {
	return rays.isTerminated;
  }
};

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth      = 1;    //determines how many bounces the raytracer traces
  float MAX_DEPTH     = 8;    // Maximum bounce number
  bool motionBlurFlag = true;

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
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

	// Motion Blur Effect (No tricks, only add a small mount of x in the translation)
	if (!motionBlurFlag)
		continue;
	// Set the seventh object move along the x axis 0.001f bit each frame
    if (i == 7) {
	  // Reset the translation
	  newStaticGeom.translation.x += 0.001f;
      geoms[i].translations[frame] = newStaticGeom.translation;
	  // Use the function in utility.cpp to get new transform, here only translation changes, rotation and scale are the same as before
      glm::mat4 transform = utilityCore::buildTransformationMatrix(newStaticGeom.translation, newStaticGeom.rotation, newStaticGeom.scale);
      geoms[i].transforms[frame] = utilityCore::glmMat4ToCudaMat4(transform);
      geoms[i].inverseTransforms[frame] = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));
    }
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  // Package materials
  material* cudamaterials = NULL;
  cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  // Package current color;
  glm::vec3* cudacolor = NULL;
  cudaMalloc((void**)&cudacolor, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudacolor, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  // Package rays
  rayPool* cudarays = NULL;
  cudaMalloc((void**)&cudarays, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(rayPool));
  
  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  // Intialize the rays
  rayGenerator<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cudamaterials, numberOfMaterials, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudarays);
  checkCUDAError("Kernel failed!");
  
  ++ traceDepth;    // the ray has been bounced once in the ray generator
  dim3 lessBlocksPerGrid;
  int rayNumber = (int)renderCam->resolution.x*(int)renderCam->resolution.y;
  while (traceDepth <= MAX_DEPTH) {
    // Stream Compaction check whether the ray is terminated or not, if it is then move it from the ray pool and decrease the block number per grid for fast computation usage of thrust http://stackoverflow.com/questions/12201446/converting-thrustiterators-to-and-from-raw-pointers/12236270#12236270
	thrust::device_ptr<rayPool> deviceRaysPointer(cudarays);
	thrust::device_ptr<rayPool> deviceRaysPointerEnd = thrust::remove_if(deviceRaysPointer, deviceRaysPointer + rayNumber, IsRayTerminated());
	rayNumber = deviceRaysPointerEnd.get() - deviceRaysPointer.get();
	if (rayNumber < EPSILON)
	  break;
	lessBlocksPerGrid = dim3((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(rayNumber)/(float)renderCam->resolution.x/float(tileSize)));
	//kernel launches for different depth of ray
	raytraceRay<<<lessBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cudamaterials, numberOfMaterials, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudarays);
	++ traceDepth;
  }

  // Filter between frames
  frameColorFilter<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cudacolor, cudaimage);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamaterials );
  cudaFree( cudacolor );
  cudaFree( cudarays );
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
};