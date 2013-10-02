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

// Include thrust components ( scan, etc ... ) needed for stream compaction
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define M_PI 3.14159265359f

struct ray_data { 
  ray Ray; 
  ray Intersection;
  int Age;
  int image_index;
  glm::vec3 Incident;
  glm::vec3 Emission;
  int collision_index;
}; 

enum { DEBUG_COLLISIONS, DEBUG_DIRECTIONS, DEBUG_BRDF, DEBUG_DIFFUSE, DEBUG_NORMALS, DEBUG_DISTANCE, DEBUG_ALL }; 
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


// Compute Light Contribution to object
__host__ __device__ glm::vec3 computeLightContribution( float shadowContribution, material mat, ray current_ray, ray light_ray, glm::vec3 intersection_normal, glm::vec3 intersection_point ) {
	glm::vec3 light_vector = light_ray.direction;
	glm::vec3 viewing_vector = glm::normalize( current_ray.origin - intersection_point ); 
	glm::vec3 reflection_vector = 2*glm::dot( light_vector, intersection_normal )*intersection_normal - light_vector;
	
	// Temporarily
	float ka = 0.5; // ambient 
	float ks = 1.0; // specular reflection constant
	float kd = 0.5; // diffuse reflection constant

	glm::vec3 ambient( 1.0, 1.0, 1.0 );

	float diffuse = max(glm::dot( light_vector, intersection_normal ), 0.0);

	// Specular Component  
	float specularExponent = mat.specularExponent; // alpha, shinyiness
	glm::vec3 specColor = mat.specularColor;
	glm::vec3 specular( 0.0, 0.0, 0.0 );
		
	if ( specularExponent > 0.0 ) {
		specular = specColor*powf( max( glm::dot( reflection_vector, viewing_vector ), 0.0 ), specularExponent );
	} 
		
	// Full illumination
	glm::vec3 illumination = ka*ambient + shadowContribution*kd*diffuse + shadowContribution*ks*specular;
	return illumination*mat.color;
}

// Find closest intersection
__host__ __device__ int closestIntersection( ray r, staticGeom* geoms, int numberOfGeoms, float& intersection_dist, glm::vec3& intersection_normal, glm::vec3& intersection_point ) {
	// Check for intersections. This has way too many branches :/
	int min_intersection_ind = -1;
	float intersection_dist_new;
	glm::vec3 intersection_point_new;
	glm::vec3 intersection_normal_new;

	for (int i=0; i < numberOfGeoms; ++i ) {
	    // Check for intersection with Sphere
		if ( geoms[i].type == SPHERE ) {
		    intersection_dist_new = sphereIntersectionTest(geoms[i], r, intersection_point_new, intersection_normal_new);		
						
		} else if ( geoms[i].type == CUBE ) {
			intersection_dist_new = boxIntersectionTest(geoms[i], r, intersection_point_new, intersection_normal_new);		
		} else if ( geoms[i].type == MESH ) {
			// TODO
		}
		if (intersection_dist_new != -1 ) {
			
			// If new distance is closer than previously seen one then use the new one
			if ( intersection_dist_new < intersection_dist || intersection_dist == -1 ) {
				intersection_dist = intersection_dist_new;
				intersection_point = intersection_point_new;
				intersection_normal = intersection_normal_new;
				min_intersection_ind = i;
			}
		}	
	}
	return min_intersection_ind;
}


// Calculate light ray
__host__ __device__ ray computeLightRay( glm::vec3 light, glm::vec3 intersection_point ) {
	ray light_ray;
	light_ray.origin = intersection_point;
	light_ray.direction = glm::normalize(light - intersection_point);
	return light_ray;
}

// Calculate reflected ray
__host__ __device__ ray computeReflectedRay( ray currentRay, glm::vec3 intersection_normal, glm::vec3 intersection_point ) {
	ray reflected_ray;
	reflected_ray.origin = intersection_point;
	reflected_ray.direction = -2*glm::dot(currentRay.direction, intersection_normal)*intersection_normal + currentRay.direction;
	return reflected_ray;
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

  float scl = 1.0f/((float)iterations);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = scl*image[index].x*255.0;
      color.y = scl*image[index].y*255.0;
      color.z = scl*image[index].z*255.0;

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
//Function that does the initial raycast from the camera
__global__ void raycastFromCameraKernel( glm::vec2 resolution, cameraData cam, ray_data* ray_pool, int numberOfRays ) {

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  // If index is out of bounds 
  if ( index > numberOfRays )
    return;

  // DEBUG initialize brdf accums to 1.0

  // Create ray using pinhole camera projection
  float px_size_x = tan( cam.fov.x * (PI/180.0) );
  float px_size_y = tan( cam.fov.y * (PI/180.0) );
	
  ray r;
  r.origin = cam.position;
  r.direction = cam.view + (-2*px_size_x*x/resolution.x + px_size_x)*glm::cross( cam.view, cam.up ) \
		     + (-2*px_size_y*y/resolution.y + px_size_y)*cam.up;

  // Set ray in ray pool
  ray_pool[index].Ray = r;
  ray_pool[index].Age = 0;
  ray_pool[index].image_index = index;
  ray_pool[index].Incident = glm::vec3(1.0, 1.0, 1.0);
  ray_pool[index].Emission = glm::vec3(0.0, 0.0, 0.0);
}

/* 
   Ray Collision Kernel

   Checks for collisions with objects 
*/

__global__ void rayCollisions( int resolution, float time, ray_data* ray_pool, int numberOfRays, staticGeom* geoms, int numberOfGeoms, int* ray_mask ) {

  /*
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution);
  */
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  int obj_index;

  float intersection_dist = -1; // don't actually use
  glm::vec3 intersection_point; 
  glm::vec3 intersection_normal;

  if ( index > numberOfRays ) 
    return;

  // Find closest intersection
  obj_index = closestIntersection( ray_pool[index].Ray, geoms, numberOfGeoms, intersection_dist, intersection_normal, intersection_point );

  ray_pool[index].Intersection.origin = intersection_point;
  ray_pool[index].Intersection.direction= intersection_normal;
  ray_pool[index].collision_index = obj_index;

  // Set mask value to 0 if there is no collision
  if ( obj_index == -1 )
    ray_mask[index] = 0;
}


/* 
   Sample from BRDF and return new set of rays 
*/

__global__ void sampleBRDF( int resolution, float time, ray_data* ray_pool, int numberOfRays, material* materials, int numberOfMaterials, staticGeom* geoms, int numberOfGeoms, glm::vec3* colors, int debugMode ) {

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  int obj_index;
  material mat;

  if ( index > numberOfRays )
    return;
  
  thrust::default_random_engine rng(hash( index*time ));
  thrust::uniform_real_distribution<float> xi1(0,1);
  thrust::uniform_real_distribution<float> xi2(0,1);

  obj_index = ray_pool[index].collision_index;
  mat = materials[geoms[obj_index].materialid];

  // Sample new direction
  //thrust::default_random_engine rng(hash( time*index*i ));
  glm::vec3 new_direction = calculateRandomDirectionInHemisphere( ray_pool[index].Intersection.direction, xi1(rng), xi2(rng) );

  // Compute diffuse BRDF
  float diffuse = glm::dot( new_direction, ray_pool[index].Intersection.direction );

  glm::vec3 brdf = 2*diffuse*mat.color;

  // Debug Modes
  /*
  if ( debugMode == DEBUG_DIRECTIONS ) { 
    colors[index] += 0.5f*new_direction + 0.5f;
    ray_pool[index].Age = -1;
    return;
  } else if ( debugMode == DEBUG_NORMALS ) {
    colors[index] += 0.5f*intersection_normal + 0.5f;
    ray_pool[index].Age = -1;
    return;
  } else if ( debugMode == DEBUG_BRDF ) {
    colors[index] += brdf;
    ray_pool[index].Age = -1;
    return;
  }*/

  // DEBUG Accumulate brdf
  //ray_pool[index].brdf_weight *= brdf;
  //ray_pool[index].Illumination += mat.emittance;

  // Accumulate Lighting 
  ray_pool[index].Emission += ray_pool[index].Incident*mat.emittance*glm::normalize(mat.color);
  ray_pool[index].Incident *= brdf;

  ray_pool[index].Ray.origin = ray_pool[index].Intersection.origin;
  ray_pool[index].Ray.direction = new_direction;

}

__global__ void updateImage( glm::vec2 image_resolution, glm::vec3* image, int resolution, ray_data* ray_pool, int numberOfRays ) {

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if ( index > numberOfRays ) 
    return;

  int image_index = ray_pool[index].image_index;
  
  // The final Emmission value contains the path 
  image[image_index] += ray_pool[index].Emission;
}

/* 
   Reset ray_mask
*/
__global__ void resetRayMask( int resolution, int* ray_mask, int numberOfRays ) { 
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if ( index > numberOfRays ) 
    return;
  ray_mask[index] = 1;
}
  

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, ray_data* ray_pool, int numberOfRays, int rayDepth, glm::vec3* colors, staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, int debugMode ){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  int obj_index = -1;
  float intersection_dist = -1;
  glm::vec3 intersection_point;
  glm::vec3 intersection_normal;
  glm::vec3 intersection_point_new;
  glm::vec3 intersection_normal_new;

  glm::vec3 color;

  material mat;
  glm::vec3 new_direction;

  thrust::default_random_engine rng(hash( time*index ));
  thrust::uniform_real_distribution<float> xi1(0,1);
  thrust::uniform_real_distribution<float> xi2(0,1);

  glm::vec3 brdf_accum(1.0, 1.0, 1.0);

  if ( index <= numberOfRays ) { 
  //if((x<=resolution.x && y<=resolution.y)){
	  
    // Calculate initial ray as projected from camera
    //ray currentRay = raycastFromCameraKernel( cam.resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov );
    ray lightRay;

    // Get ray from ray pool
    ray currentRay = ray_pool[index].Ray; 

    // Iteratively trace rays until depth is reached
    /*j
    if ( debugMode == DEBUG_ALL ) {
      depth = 4;
    } else {
      depth = 1;
    }
    */

    for (int i=0; i<rayDepth; ++i) {
      obj_index = closestIntersection( currentRay, geoms, numberOfGeoms, intersection_dist, intersection_normal, intersection_point );
      if (obj_index == -1) {
	// Black color
	//colors[index] += glm::vec3( 1.0, 0.0, 0.0);
	return;
      }
      mat = materials[geoms[obj_index].materialid];

      // If material is a light then return 
      if ( mat.emittance != 0 ) {
	colors[index] = colors[index] + brdf_accum*mat.emittance;
	//return;
      }

      // Sample new direction
      //thrust::default_random_engine rng(hash( time*index*i ));
      new_direction = calculateRandomDirectionInHemisphere( intersection_normal, xi1(rng), xi2(rng) );

      // Compute diffuse BRDF
      float diffuse = glm::dot( new_direction, intersection_normal );
      glm::vec3 brdf = 2*diffuse*mat.color;
      // Accumulate brdf
      brdf_accum *= brdf;

      currentRay.origin = intersection_point;
      currentRay.direction = new_direction;

      if ( debugMode == DEBUG_ALL ) {
	continue;
      } else if ( debugMode == DEBUG_DIRECTIONS ) {
	colors[index] +=  0.5f*new_direction + 0.5f;
	return;
      } else if ( debugMode == DEBUG_BRDF ) {
	colors[index] += brdf;
	return;
      } else if ( debugMode == DEBUG_COLLISIONS ) {
	// Collisions debug mode
	colors[index] = materials[geoms[obj_index].materialid].color;
	return;
      } else if ( debugMode == DEBUG_NORMALS ) { 
	colors[index] = 0.5f*intersection_normal + 0.5f;
	return;
	//color = 0.5f*glm::vec3(0.0, 0.0, -1.0) + 0.5f;
	//color = 0.5f*lightRay.direction + 0.5f;
      } else if ( debugMode == DEBUG_DISTANCE ) {
	if ( intersection_dist == -1 ) {
	  color = glm::vec3( 0, 0, 0 );
	} else {
	  // Interest range from 0 to 10m
	  float scl = 1/10.0;
	  color = scl*intersection_dist*glm::vec3( 1.0, 1.0, 1.0 );
	}
      }
    }
  //colors[index] += color;
  }

}

__global__ void copyRays( ray_data* new_ray_pool, int number_of_rays_new, ray_data* old_ray_pool, int number_of_rays_old, int* ray_mask, int* scan_indices, int resolution ) {
    
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
 /* 
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution);
  */

  if ( index > number_of_rays_old-1 )
    return;

  if ( !ray_mask[index]  )
    return;

  new_ray_pool[scan_indices[index]] = old_ray_pool[index];
  
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

  //printf( "threadsPerBlock: %d \n", threadsPerBlock.x );
  //printf( "fullBlocksPerGrid: %d \n", fullBlocksPerGrid.x);

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

  //int debugMode = DEBUG_DIRECTIONS;
  //int debugMode = DEBUG_BRDF;
  int debugMode = DEBUG_ALL;
  //int debugMode = DEBUG_DIFFUSE;
  //int debugMode = DEBUG_NORMALS;

  // Allocate memory for ray pool
  int numberOfRays = (int)renderCam->resolution.x*(int)renderCam->resolution.y;
  ray_data* rayPool = NULL;
  cudaMalloc( (void**)&rayPool, numberOfRays*sizeof(ray_data) );

  int numberOfRaysNew;
  ray_data* rayPoolNew = NULL;
  cudaMalloc( (void**)&rayPoolNew, numberOfRays*sizeof(ray_data) );

  ray_data* rayPoolSwap;

  // Allocate ray mask and index array for stream compaction
  int* ray_mask = NULL;
  cudaMalloc( (void**)&ray_mask, numberOfRays*sizeof(int) ); 

  int* cuda_indices = NULL;
  cudaMalloc( (void**)&cuda_indices, numberOfRays*sizeof(int) );

  // Perform initial raycasts from camera
  raycastFromCameraKernel<<<fullBlocksPerGrid, threadsPerBlock>>>( renderCam->resolution, cam, rayPool, numberOfRays );

  int res = (int)ceil(sqrt((float(numberOfRays))));
  //printf("resolution: %d \n", res);
  threadsPerBlock = dim3(tileSize*tileSize);
  //fullBlocksPerGrid = dim3((int)ceil(float(res)/float(tileSize)), (int)ceil(float(res)/float(tileSize)));
  fullBlocksPerGrid = dim3((int)ceil(float(numberOfRays)/float(tileSize*tileSize)));
  //printf( "threadsPerBlock: %d \n", threadsPerBlock.x );
  //printf( "fullBlocksPerGrid: %d \n", fullBlocksPerGrid.x);

  thrust::device_ptr<int> msk_dptr = thrust::device_pointer_cast(ray_mask);  
  thrust::device_ptr<int> idx_dptr = thrust::device_pointer_cast(cuda_indices);  

  int traceDepth = 3; //determines how many bounces the raytracer traces
  // Iterate through ray calls accumulating in brdf_accums 
  for ( int i=0; i < traceDepth; ++i ) {

    resetRayMask<<<fullBlocksPerGrid, threadsPerBlock>>>( res, ray_mask, numberOfRays );
    cudaThreadSynchronize();

    rayCollisions<<<fullBlocksPerGrid, threadsPerBlock>>>( res, (float)iterations, rayPool, numberOfRays, cudageoms, numberOfGeoms, ray_mask );
    cudaThreadSynchronize();


    //thrust::device_vector<int> indices(numberOfElements); 
    thrust::device_ptr<int> msk_dptr = thrust::device_pointer_cast(ray_mask);  
    thrust::exclusive_scan(msk_dptr, msk_dptr+numberOfRays, idx_dptr ); // in-place scan
    numberOfRaysNew = idx_dptr[numberOfRays-1]+1;

    //printf("numberOfRays: %d \n", numberOfRays);

    copyRays<<<fullBlocksPerGrid, threadsPerBlock>>>( rayPoolNew, numberOfRaysNew, rayPool, numberOfRays, ray_mask, cuda_indices, res  );

    // Update ray count
    numberOfRays = idx_dptr[numberOfRays-1]+1;
    //printf("numberOfRaysNew: %d \n", numberOfRays);
    
    // Update number of threads running
    threadsPerBlock = dim3(tileSize*tileSize);
    //fullBlocksPerGrid = dim3((int)ceil(float(res)/float(tileSize)), (int)ceil(float(res)/float(tileSize)));
    fullBlocksPerGrid = dim3((int)ceil(float(numberOfRays)/float(tileSize*tileSize)));
    //printf( "threadsPerBlock: %d \n", threadsPerBlock );
    //printf( "fullBlocksPerGrid: %d \n", fullBlocksPerGrid);

    // Swap array pointers 
    rayPoolSwap = rayPool;
    rayPool = rayPoolNew;
    rayPoolNew = rayPoolSwap;

    sampleBRDF<<<fullBlocksPerGrid, threadsPerBlock>>>( res, (float)(i+1)*iterations, rayPool, numberOfRays, cudamaterials, numberOfMaterials, cudageoms, numberOfGeoms, cudaimage, debugMode );
    cudaThreadSynchronize();
  }

  // Update image data
  updateImage<<<fullBlocksPerGrid, threadsPerBlock>>>( renderCam->resolution, cudaimage, res, rayPool, numberOfRays );
  // make certain the kernel has completed
  cudaThreadSynchronize();
  

  // Return block counts, etc ... back to image size
  threadsPerBlock = dim3(tileSize, tileSize);
  fullBlocksPerGrid = dim3((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  //printf( "send image to PBO \n\n\n" );
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, iterations);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( rayPool );
  cudaFree( rayPoolNew );
  cudaFree( ray_mask );
  cudaFree( cuda_indices );
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  cudaError_t errorNum = cudaPeekAtLastError();
  if ( errorNum != cudaSuccess ) { 
      printf ("Cuda error -- %s\n", cudaGetErrorString(errorNum));
  }
  checkCUDAError("Kernel failed!");
}
