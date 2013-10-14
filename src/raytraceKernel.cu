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
#include <cuda_runtime_api.h>
#include "glm/glm.hpp"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <iostream>

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


__device__ glm::vec3 reflect(glm::vec3 const & I, glm::vec3 const & N)
{
  return I - 2.0f * glm::dot(N, I) * N;
}

__device__ bool isRayUnblocked(glm::vec3 const & point1, glm::vec3 const & point2, staticGeom* geoms, int numberOfGeoms)
{
  glm::vec3 DIRECTION(point2 - point1);
  float DISTANCE = glm::length(DIRECTION);

  // Offset start position in ray direction by small distance to prevent self collisions
  float DELTA = 0.001f;
  ray r;
  r.origin = point1 + DELTA * DIRECTION;
  r.direction = glm::normalize(DIRECTION);

  for (int i=0; i<numberOfGeoms; ++i)
  {
	glm::vec3 intersectionPoint;
    glm::vec3 normal;
    float intersectionDistance = geomIntersectionTest(geoms[i], r, intersectionPoint, normal);

	// Does not intersect so check next primitive
	if (intersectionDistance <= 0.0f) continue;

    // Take into consideration intersection only between the two points.
	  if (intersectionDistance < DISTANCE) return false;
  }

  return true;
}

/*
__global__ void raytraceRay(glm::vec2 resolution, int time, float bounce, cameraData cam, int rayDepth, glm::vec3* colors, 
							staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, ray* d_rays)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if ( x >= resolution.x || y >= resolution.y ) return;
	ray r = d_rays[index];

	// ============================================
	// Determine closest intersection with geometry
	// ============================================
	
	float distance = -1.0f;
	glm::vec3 intersection;
	glm::vec3 normal;
	int materialIdx;
	for (int i = 0; i < numberOfGeoms; ++i)
	{
		float newDistance;
		glm::vec3 newIntersection;
		glm::vec3 newNormal;
		switch (geoms[i].type)
		{
			case SPHERE:
				newDistance = sphereIntersectionTest(geoms[i], r, newIntersection, newNormal);
				break;
			case CUBE:
				newDistance = boxIntersectionTest(geoms[i], r, newIntersection, newNormal);
				break;
			case MESH:
				newDistance = -1.0f;
				break;
		}
		if ( newDistance < 0.0f ) continue;
		if ( distance < 0.0f || (distance > 0.0f && newDistance < distance) )
		{
			distance = newDistance;
			intersection = newIntersection;
			normal = newNormal;
			materialIdx = geoms[i].materialid;
		}
	}
	
	// ============================================
	// Paint pixel
	// ============================================

	// No hit
	if ( distance < 0.0f )
	{
		colors[index] = glm::vec3(0.0f, 0.0f, 0.0f);
		//colors[index] = generateRandomNumberFromThread(resolution, time, x, y);
		return;
	}

	// Simple local reflectance model (local illumination model formula)
	float reflectivity = 0.0f;
	float transmittance = 1.0f - reflectivity;
	glm::vec3 materialColor = materials[materialIdx].color;
	glm::vec3 reflectedColor(0.0f, 0.0f, 0.0f);
	glm::vec3 ambientLightColor(1.0f, 1.0f, 1.0f);
	
	float AMBIENT_WEIGHT = 0.2f;	// Ka - Ambient reflectivity factor
	float DIFFUSE_WEIGHT = 0.3f;	// Kd - Diffuse reflectivity factor
	float SPECULAR_WEIGHT = 0.5f;	// Ks - Specular reflectivity factor

	glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
	glm::vec3 color = AMBIENT_WEIGHT * ambientLightColor * materialColor;

	thrust::default_random_engine rng(hash(index*time));
	thrust::uniform_real_distribution<float> u01(-0.15f, 0.15f);
	for ( int i = 0; i < 1; ++i)
	{
		glm::vec3 lightPosition(0.25f + (float) u01(rng), 1.0f, (float) u01(rng));
		// Unit vector from intersection point to light source
		glm::vec3 LIGHT_DIRECTION = glm::normalize(lightPosition - intersection);
		// Direction of reflected light at intersection point
		glm::vec3 LIGHT_REFLECTION = glm::normalize(reflect(-1.0f*LIGHT_DIRECTION, normal));

		// Determine diffuse term
		float diffuseTerm;
		diffuseTerm = glm::dot(normal, LIGHT_DIRECTION);
		diffuseTerm = glm::clamp(diffuseTerm, 0.0f, 1.0f);

		// Determine specular term
		float specularTerm = 0.0f;
		if ( materials[materialIdx].specularExponent - 0.0f > 0.001f )
		{
			float SPECULAR_EXPONENT = materials[materialIdx].specularExponent;
			glm::vec3 EYE_DIRECTION = glm::normalize(cam.position - intersection);
			specularTerm = glm::dot(LIGHT_REFLECTION, EYE_DIRECTION);
			specularTerm = pow(fmaxf(specularTerm, 0.0f), SPECULAR_EXPONENT);
			specularTerm = glm::clamp(specularTerm, 0.0f, 1.0f);
		}

		if (isRayUnblocked(intersection, lightPosition, geoms, numberOfGeoms))
		{
			color += DIFFUSE_WEIGHT * lightColor * materialColor * diffuseTerm / 1.0f;
			color += SPECULAR_WEIGHT * lightColor * specularTerm / 1.0f;
		}
	}

	glm::vec3 new_color = reflectivity*reflectedColor + transmittance*color;

	if ( time > 1 )
	{
		colors[index] += (new_color - colors[index]) / (float)time;
		return;
	}
	colors[index] = new_color;
}
*/

// Requires:
//		x = 0 to width-1
//		y = 0 to height-1
// Jittering based only on random_seed (not x or y).
__host__ __device__ glm::vec3 GetRayDirectionFromCamera(const cameraData& cam, int x, int y, int random_seed)
{
	float random1, random2;         // Random # between 0 and 1 (from random_seed).

	// Set random numbers.
	{
		thrust::default_random_engine rng(hash(random_seed));
		thrust::uniform_real_distribution<float> u01(0,1);
		random1 = u01(rng);
		random2 = u01(rng);
	}

	float width = (float) cam.resolution.x;
	float height = (float) cam.resolution.y;
	
	glm::vec3 c(cam.view);          // View direction (unit vector) from eye.
	glm::vec3 e(cam.position);      // Camera center position.
	glm::vec3 m = e + c;            // Midpoint of screen.
	glm::vec3 u(cam.up);            // Up vector.
	glm::vec3 a = glm::cross(c, u); // c x u TODO: make sure this is well defined
	glm::vec3 b = glm::cross(a, c); // a x c TODO: make sure this is well defined
  
	glm::vec3 v;	                // Vertical vector from "m" to top of screen.
	glm::vec3 h;	                // Horizontal vector from "m" to right of screen.

	// Calculate v & h
	{
		float phi = cam.fov.y * PI / 180.0f / 2.0f;
		float screen_ratio = height / width;
		v = b * tan(phi) / (float)glm::length(b);
		float theta = atan(glm::length(v)/screen_ratio / (float)glm::length(c));
		h = a * (float)glm::length(c) * tan(theta) / (float)glm::length(a);
	}
  
	// Obtain a unit vector in the direction from the eye to a pixel point (x, y) on screen
	float sx = (x + random1) / width;				// Without jitter: x / (width - 1.0f)
	float sy = (y + random2) / height;				//				   y / (height - 1.0f)
	glm::vec3 p = m - (2*sx - 1)*h - (2*sy - 1)*v;	// World position of point (x, y) on screen 
	return glm::normalize(p-e);
}

// Initialize all rays using camera data.
// # of rays = # of pixels
__global__ void InitRay(cameraData cam, int random_seed, ray* d_rays, glm::vec3* d_lights, bool* d_is_ray_alive, int* d_ray_idx)
{
	int width = cam.resolution.x;
	int height = cam.resolution.y;
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int idx = x + (y * width);

	if ( x >= width || y >= height ) return;

	d_rays[idx].origin	  = cam.position;
	d_rays[idx].direction = GetRayDirectionFromCamera(cam, x, y, random_seed);
	d_lights[idx]		  = glm::vec3(1.0f);
	d_is_ray_alive[idx]	  = true;
	d_ray_idx[idx]		  = idx;
}



// Modifies:
//		p: Intersection point.
//		n: Normal unit vector at intersection.
//		material_id: Of intersected object.
// Return true if intersected.
__device__ bool GetClosestIntersection(ray& r, staticGeom* geoms, int num_geoms, material* materials,
									   glm::vec3& p, glm::vec3& n, int& material_id)
{
	float distance = -1.0f;

	for ( int i=0; i < num_geoms; ++i )
	{
		// Ignore emitters.
		//if ( IsEmitter(geoms[i].materialid, materials) ) continue;

		glm::vec3 new_intersection;
		glm::vec3 new_normal;
		float new_distance = geomIntersectionTest(geoms[i], r, new_intersection, new_normal);

		if ( new_distance < 0.0f ) continue;
		if ( distance < 0.0f || (distance > 0.0f && new_distance < distance) )
		{
			distance = new_distance;
			p = new_intersection;
			n = new_normal;
			material_id = geoms[i].materialid;
		}
	}

	if ( distance < 0.0f) return false;
	return true;
}

__host__ __device__ bool IsEmitter(int id, material* materials)
{
	return ( materials[id].emittance > 0.5f );
}

__device__ void SetAverageColor(glm::vec3* colors, int idx, glm::vec3& new_color, int iterations)
{
	if ( iterations > 1 )
	{
		colors[idx] += (new_color - colors[idx]) / (float)iterations;
		return;
	}
	colors[idx] = new_color;
}


__global__ void TraceRay(int iterations, int depth, int max_depth, int num_pixels, ray* d_rays, int num_rays, glm::vec3* d_lights, bool* d_is_ray_alive, int* d_ray_idx,
		glm::vec3* colors, staticGeom* geoms, int num_geoms, material* materials, int num_materials)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	/*
	int debug_i = 0;
	if (x == 641 && y == 177)
	{
		debug_i ++;
	}
	debug_i ++;
	*/

	if ( idx >= num_rays ) return;
	if ( !d_is_ray_alive[idx] ) return;
	
	// Copy global memory to register.
	ray ray_in = d_rays[idx];
	glm::vec3 light = d_lights[idx];

	bool is_intersected;
	glm::vec3 p;				// Intersection point.
	glm::vec3 n;				// Normal unit vector at intersection.
	int material_id;			// Of intersected object.
	
	is_intersected = GetClosestIntersection(ray_in, geoms, num_geoms, materials, p, n, material_id);
	
	// No hit, return (light * bg).
	if ( !is_intersected )
	{
		glm::vec3 bg_color(0.2f);
		glm::vec3 new_color = light * bg_color;

		d_is_ray_alive[idx] = false;
		SetAverageColor(colors, d_ray_idx[idx], new_color, iterations);
		return;
	}
	
	// Hit emitter, return (light * emitter).
	if ( IsEmitter(material_id, materials) )
	{
		glm::vec3 new_color = light * materials[material_id].color * materials[material_id].emittance;

		d_is_ray_alive[idx] = false;
		SetAverageColor(colors, d_ray_idx[idx], new_color, iterations);
		return;
	}

	// Make ray_out in random direction.
	ray ray_out;
	//ray_out.direction = UniformRandomHemisphereDirection(n, (float) (iterations-1) * max_depth * num_pixels + depth * num_pixels + idx);
	float xi1, xi2;
	{
		thrust::default_random_engine rng(hash((float) iterations * (depth+1) * idx));
		thrust::uniform_real_distribution<float> u01(0,1);
		xi1 = u01(rng);
		xi2 = u01(rng);
	}
	if ( materials[material_id].hasReflective )
	{
		ray_out.direction = reflect(ray_in.direction, glm::normalize(n));
	}
	else
	{
		ray_out.direction = calculateRandomDirectionInHemisphere(glm::normalize(n), xi1, xi2);
	}
	ray_out.origin = p + 0.001f * ray_out.direction;

	// Update light & ray.
	d_lights[idx] = light * materials[material_id].color;
	d_rays[idx] = ray_out;
	



	// Kill rays with negligible throughput.
	
	// Direct illumination.

	// For each light...
	/*
	int num_lights = 0;
	for ( int i=0; i < num_geoms; ++i )
	{
		// Ignore non-emitters.
		if ( materials[geoms[i].materialid].emittance < 0.5f ) continue;
		
		++ num_lights;

		// 1) Sample a point on light
		glm::vec3 point_on_light;
		point_on_light = getRandomPointOnGeom(geoms[i], iterations+depth);

		// 2) L += [throughput] * [avg of visible lights]
		glm::vec3 direct_L(0.0f);
		if ( isRayUnblocked(p, point_on_light, geoms, num_geoms) )
		{
			direct_L += throughput * materials[geoms[i].materialid].color
		}
		L += direct_L / (float) num_lights;
	}
	

	throughput = throughput * materials[material_id].color;

	//glm::vec3 new_color = ;
	SetAverageColor(colors, idx, new_color, iterations);
	*/
}


__global__ void CompactRays(int* td_v, ray* d_rays, glm::vec3* d_lights, bool* d_is_ray_alive, int* d_ray_idx, int num_rays,
	                                   ray* d_rays_copy, glm::vec3* d_lights_copy, bool* d_is_ray_alive_copy, int* d_ray_idx_copy)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if ( idx >= num_rays ) return;
	if ( !d_is_ray_alive[idx] ) return;
	
	int copy_idx = td_v[idx];
	d_rays_copy[copy_idx] = d_rays[idx];
	d_lights_copy[copy_idx] = d_lights[idx];
	d_is_ray_alive_copy[copy_idx] = true;
	d_ray_idx_copy[copy_idx] = d_ray_idx[idx];
}



// Wrapper for the __global__ call that sets up the kernel calls and does memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* cam, int frame, int iterations, material* materials, int num_materials, geom* geoms, int num_geoms)
{
	int width = cam->resolution.x;
	int height = cam->resolution.y;
	int num_pixels = width * height;

	// Device memory size.
	int tile_size = 8;
	dim3 threadsPerBlock(tile_size, tile_size);
	dim3 fullBlocksPerGrid(ceil((float)width/tile_size), ceil((float)height/tile_size));

	// Copy image to GPU.
	glm::vec3* d_image = NULL;
	cudaMalloc((void**)&d_image, num_pixels*sizeof(glm::vec3));
	cudaMemcpy(d_image, cam->image, num_pixels*sizeof(glm::vec3), cudaMemcpyHostToDevice);
	
	// Package geometry.
	staticGeom* geomList = new staticGeom[num_geoms];
	for ( int i=0; i<num_geoms; ++i )
	{
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
	
	// Copy geometry to GPU.
	staticGeom* d_geoms = NULL;
	cudaMalloc((void**)&d_geoms, num_geoms*sizeof(staticGeom));
	cudaMemcpy( d_geoms, geomList, num_geoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
	
	// Copy materials to GPU.
	material* cudamaterials = NULL;
	cudaMalloc((void**)&cudamaterials, num_materials*sizeof(material));
	cudaMemcpy( cudamaterials, materials, num_materials*sizeof(material), cudaMemcpyHostToDevice);

	// Package camera.
	cameraData cam_data;
	cam_data.resolution = cam->resolution;
	cam_data.position = cam->positions[frame];
	cam_data.view = cam->views[frame];
	cam_data.up = cam->ups[frame];
	cam_data.fov = cam->fov;

	// Allocate GPU memory for rays & initialize them.
	ray* d_rays			 = NULL;
	glm::vec3* d_lights	 = NULL;
	bool* d_is_ray_alive = NULL;
	int* d_ray_idx		 = NULL;
	cudaMalloc((void**)&d_rays,			num_pixels*sizeof(ray));
	cudaMalloc((void**)&d_lights,		num_pixels*sizeof(glm::vec3));
	cudaMalloc((void**)&d_is_ray_alive,	num_pixels*sizeof(bool));
	cudaMalloc((void**)&d_ray_idx,		num_pixels*sizeof(int));
	InitRay<<<fullBlocksPerGrid, threadsPerBlock>>>(cam_data, iterations, d_rays, d_lights, d_is_ray_alive, d_ray_idx);

	// Start raytracer kernel.
	int num_rays = num_pixels;
	int max_depth = 10; // # of bounces when raytracing.
	for ( int depth = 0; depth < max_depth; ++depth )
	{
		// Determine # of kernels to launch based on # of rays.
		int num_threads_per_block = 128;
		int num_blocks_per_grid = ceil((float)num_rays / num_threads_per_block);

		// Update d_rays & d_lights based on intersected object.
		TraceRay<<<num_blocks_per_grid, num_threads_per_block>>>(iterations, depth, max_depth, num_pixels, d_rays, num_rays, d_lights, d_is_ray_alive, d_ray_idx, d_image, d_geoms, num_geoms, cudamaterials, num_materials);
		
		// Update d_rays by removing dead rays (stream compaction).
		thrust::device_ptr<bool> td_is_ray_alive = thrust::device_pointer_cast(d_is_ray_alive);
		thrust::device_vector<int> td_v(num_rays);
		thrust::exclusive_scan(td_is_ray_alive, td_is_ray_alive + num_rays, td_v.begin());
		
		int num_copy_rays = td_v[num_rays-1] + (int) td_is_ray_alive[num_rays-1];
		ray* d_rays_copy		  = NULL;
		glm::vec3* d_lights_copy  = NULL;
		bool* d_is_ray_alive_copy = NULL;
		int* d_ray_idx_copy		  = NULL;
		cudaMalloc((void**)&d_rays_copy,		 num_copy_rays*sizeof(ray));
		cudaMalloc((void**)&d_lights_copy,		 num_copy_rays*sizeof(glm::vec3));
		cudaMalloc((void**)&d_is_ray_alive_copy, num_copy_rays*sizeof(bool));
		cudaMalloc((void**)&d_ray_idx_copy,		 num_copy_rays*sizeof(int));
		CompactRays<<<num_blocks_per_grid, num_threads_per_block>>>(thrust::raw_pointer_cast(td_v.data()), d_rays, d_lights, d_is_ray_alive, d_ray_idx, num_rays, d_rays_copy, d_lights_copy, d_is_ray_alive_copy, d_ray_idx_copy);
		cudaDeviceSynchronize();
		cudaFree(d_rays);
		cudaFree(d_lights);
		cudaFree(d_is_ray_alive);
		cudaFree(d_ray_idx);
		num_rays = num_copy_rays;
		d_rays = d_rays_copy;
		d_lights = d_lights_copy;
		d_is_ray_alive = d_is_ray_alive_copy;
		d_ray_idx = d_ray_idx_copy;
	}

	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, cam->resolution, d_image);

	// Retrieve image from GPU.
	cudaMemcpy( cam->image, d_image, num_pixels*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	// Free memory.
	cudaFree( d_image );
	cudaFree( d_geoms );
	cudaFree( cudamaterials );
	cudaFree( d_rays );
	cudaFree( d_lights );
	cudaFree( d_is_ray_alive );
	cudaFree( d_ray_idx );
	delete [] geomList;

	// Make sure the kernel has completed.
	cudaDeviceSynchronize();

	checkCUDAError("Kernel failed!");
}
