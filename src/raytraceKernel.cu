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

#define M_PI 3.14159265359f

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
__global__ void raycastFromCameraKernel( glm::vec2 resolution, cameraData cam, ray* ray_pool, int numberOfRays, glm::vec3* brdf_accums, int* obj_indices ) {

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  // If index is out of bounds 
  if ( index > numberOfRays )
    return;

  // DEBUG initialize brdf accums to 1.0
  brdf_accums[index] = glm::vec3( 1.0, 1.0, 1.0 );
  obj_indices[index] = 0;

  // Create ray using pinhole camera projection
  float px_size_x = tan( cam.fov.x * (PI/180.0) );
  float px_size_y = tan( cam.fov.y * (PI/180.0) );
	
  ray r;
  r.origin = cam.position;
  r.direction = cam.view + (-2*px_size_x*x/resolution.x + px_size_x)*glm::cross( cam.view, cam.up ) \
		     + (-2*px_size_y*y/resolution.y + px_size_y)*cam.up;

  // Set ray in ray pool
  ray_pool[index] = r;
}

/* 
   Ray Collision Kernel

   Checks for collisions with objects 
*/

__global__ void rayCollisions( glm::vec2 resolution, float time, ray* ray_pool, int numberOfRays, staticGeom* geoms, int numberOfGeoms, int* obj_indices, ray* intersection_rays ) {

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  int obj_index = -1;
  float intersection_dist = -1; // don't actually use
  glm::vec3 intersection_point; 
  glm::vec3 intersection_normal;
  ray intersection_ray;
  ray currentRay;

  if ( index > numberOfRays ) 
    return;

  // DEBUG if obj_index == -1, then don't process
  if ( obj_indices[index] == -1 )
    return;

  // Get ray out of pool
  currentRay = ray_pool[index]; 

  // Find closest intersection
  obj_index = closestIntersection( currentRay, geoms, numberOfGeoms, intersection_dist, intersection_normal, intersection_point );

  // Update collision indices and normals 
  obj_indices[index] = obj_index;

  intersection_ray.direction = intersection_normal;
  intersection_ray.origin = intersection_point;
  intersection_rays[index] = intersection_ray;

}


/* 
   Sample from BRDF and return new set of rays 
*/

__global__ void sampleBRDF( glm::vec2 resolution, float time, ray* ray_pool, int numberOfRays, int* obj_indices, ray* intersection_rays, material* materials, int numberOfMaterials, staticGeom* geoms, int numberOfGeoms, glm::vec3* colors, glm::vec3* brdf_accums, int debugMode ) {

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  int obj_index;
  glm::vec3 intersection_normal;
  glm::vec3 intersection_point;
  material mat;

  thrust::default_random_engine rng(hash( time*index ));
  thrust::uniform_real_distribution<float> xi1(0,1);
  thrust::uniform_real_distribution<float> xi2(0,1);

  if ( index > numberOfRays )
    return;

  obj_index = obj_indices[index];
  intersection_normal = intersection_rays[index].direction;
  intersection_point = intersection_rays[index].origin;
  mat = materials[geoms[obj_index].materialid];
    
  // DEBUG if obj_index == -1, then don't process
  if ( obj_index == -1 )
    return;

  // If material is a light then return 
  if ( mat.emittance != 0  && debugMode == DEBUG_ALL) {
    colors[index] += brdf_accums[index]*mat.emittance;

    // DEBUG if obj_index == -1, then don't process
    obj_indices[index] = -1;
    return;
  }

  // Sample new direction
  //thrust::default_random_engine rng(hash( time*index*i ));
  glm::vec3 new_direction = calculateRandomDirectionInHemisphere( intersection_normal, xi1(rng), xi2(rng) );

  // Compute diffuse BRDF
  float diffuse = glm::dot( new_direction, intersection_normal );
  glm::vec3 brdf = 2*diffuse*mat.color;

  // Debug Modes
  if ( debugMode == DEBUG_DIRECTIONS ) { 
    colors[index] += 0.5f*new_direction + 0.5f;
    obj_indices[index] = -1;
    return;
  } else if ( debugMode == DEBUG_NORMALS ) {
    colors[index] += 0.5f*intersection_normal + 0.5f;
    obj_indices[index] = -1;
    return;
  } else if ( debugMode == DEBUG_BRDF ) {
    colors[index] += brdf;
    obj_indices[index] = -1;
    return;
  }

  // DEBUG Accumulate brdf
  brdf_accums[index] *= brdf;

  ray_pool[index].origin = intersection_point;
  ray_pool[index].direction = new_direction;

}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, ray* ray_pool, int numberOfRays, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, 
			    int debugMode ){

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

  if((x<=resolution.x && y<=resolution.y)){
	  
    // Calculate initial ray as projected from camera
    //ray currentRay = raycastFromCameraKernel( cam.resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov );
    ray lightRay;

    // Get ray from ray pool
    ray currentRay = ray_pool[index]; 

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
	colors[index] += glm::vec3( 0.0, 0.0, 0.0);
	return;
      }

      mat = materials[geoms[obj_index].materialid];

      // If material is a light then return 
      if ( mat.emittance != 0 ) {
	colors[index] += brdf_accum*mat.emittance;
	return;
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

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 2; //determines how many bounces the raytracer traces

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
  ray* rayPool = NULL;
  cudaMalloc( (void**)&rayPool, numberOfRays*sizeof(ray) );

  // Allocate normals and object indices
  int* obj_indices = NULL; 
  cudaMalloc( (void**)&obj_indices, numberOfRays*sizeof(int) );

  ray* intersectionRays = NULL;
  cudaMalloc( (void**)&intersectionRays, numberOfRays*sizeof(ray) );

  glm::vec3* brdf_accums = NULL;
  cudaMalloc( (void**)&brdf_accums, numberOfRays*sizeof(glm::vec3) );
  

  // Perform initial raycasts from camera
  raycastFromCameraKernel<<<fullBlocksPerGrid, threadsPerBlock>>>( renderCam->resolution, cam, rayPool, numberOfRays, brdf_accums, obj_indices );


  //for ( int i=0; i < traceDepth; ++i ) {
    // Perform collision checks for rays in rayPool

    rayCollisions<<<fullBlocksPerGrid, threadsPerBlock>>>( renderCam->resolution, (float)iterations, rayPool, numberOfRays, cudageoms, numberOfGeoms, obj_indices, intersectionRays );

    sampleBRDF<<<fullBlocksPerGrid, threadsPerBlock>>>( renderCam->resolution, (float)iterations, rayPool, numberOfRays, obj_indices, intersectionRays, cudamaterials, numberOfMaterials, cudageoms, numberOfGeoms, cudaimage, brdf_accums, debugMode );

    rayCollisions<<<fullBlocksPerGrid, threadsPerBlock>>>( renderCam->resolution, (float)iterations, rayPool, numberOfRays, cudageoms, numberOfGeoms, obj_indices, intersectionRays );

    sampleBRDF<<<fullBlocksPerGrid, threadsPerBlock>>>( renderCam->resolution, (float)iterations, rayPool, numberOfRays, obj_indices, intersectionRays, cudamaterials, numberOfMaterials, cudageoms, numberOfGeoms, cudaimage, brdf_accums, debugMode );

    rayCollisions<<<fullBlocksPerGrid, threadsPerBlock>>>( renderCam->resolution, (float)iterations, rayPool, numberOfRays, cudageoms, numberOfGeoms, obj_indices, intersectionRays );

    sampleBRDF<<<fullBlocksPerGrid, threadsPerBlock>>>( renderCam->resolution, (float)iterations, rayPool, numberOfRays, obj_indices, intersectionRays, cudamaterials, numberOfMaterials, cudageoms, numberOfGeoms, cudaimage, brdf_accums, debugMode );


  // Old ray tracing code
  //raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, rayPool, numberOfRays, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials, debugMode);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, iterations);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( rayPool );
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( intersectionRays );
  cudaFree( obj_indices );
  cudaFree( brdf_accums );
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  cudaError_t errorNum = cudaPeekAtLastError();
  if ( errorNum != cudaSuccess ) { 
      printf ("Cuda error -- %s\n", cudaGetErrorString(errorNum));
  }
  checkCUDAError("Kernel failed!");
}
