// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

#include "sceneStructs.h"
#include "cudaMat4.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include <thrust/random.h>

//Some forward declarations
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t);
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
__host__ __device__ glm::vec3 getSignOfRay(ray r);
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//Quick and dirty epsilon check
__host__ __device__ bool epsilonCheck(float a, float b){
    if(fabs(fabs(a)-fabs(b))<EPSILON){
        return true;
    }else{
        return false;
    }
}

//Self explanatory
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t){
	// Add a big epsilon amount .001 forward along the shadow ray (.0001 previously) so as to avoid floating problem
  return r.origin + float(t-.001)*glm::normalize(r.direction);
}

//LOOK: This is a custom function for multiplying cudaMat4 4x4 matrixes with vectors.
//This is a workaround for GLM matrix multiplication not working properly on pre-Fermi NVIDIA GPUs.
//Multiplies a cudaMat4 matrix and a vec4 and returns a vec3 clipped from the vec4
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}

//Gets 1/direction for a ray
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r){
  return glm::vec3(1.0/r.direction.x, 1.0/r.direction.y, 1.0/r.direction.z);
}

//Gets sign of each component of a ray's inverse direction
__host__ __device__ glm::vec3 getSignOfRay(ray r){
  glm::vec3 inv_direction = getInverseDirectionOfRay(r);
  return glm::vec3((int)(inv_direction.x < 0), (int)(inv_direction.y < 0), (int)(inv_direction.z < 0));
}

//TODO: IMPLEMENT THIS FUNCTION
//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
  
  // Find the inverse transformed ray in the origin centered coordinates
  glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));
  ray rt; rt.origin = ro; rt.direction = rd;

  // Find the inverse direction ray avoiding -/+ 0 determination
  glm::vec3 inverseDirection     = getInverseDirectionOfRay(rt);
  glm::vec3 signInverseDirection = getSignOfRay(rt);

	// Initialize the minimum and maximum distance to intersection to compare
	float tmin, tmax, minLimit, maxLimit;
	minLimit = -1e7;
	maxLimit = 1e7;
	tmin     = minLimit;
	tmax     = maxLimit;

  // Find the bounding volumn's maximum and minimum extent t1/t0
  glm::vec3 t0, t1;
  t0.x = (0.5 * (2 * signInverseDirection.x - 1) - ro.x) * inverseDirection.x;
	t1.x = (0.5 * (1 - 2 * signInverseDirection.x) - ro.x) * inverseDirection.x;
  t0.y = (0.5 * (2 * signInverseDirection.y - 1) - ro.y) * inverseDirection.y;
	t1.y = (0.5 * (1 - 2 * signInverseDirection.y) - ro.y) * inverseDirection.y;
	t0.z = (0.5 * (2 * signInverseDirection.z - 1) - ro.z) * inverseDirection.z;
	t1.z = (0.5 * (1 - 2 * signInverseDirection.z) - ro.z) * inverseDirection.z;
  
	// Surface flags to point out which is the nearest and farthest surface along the intersection direction
	int nearFlag, farFlag;  // which can be one value in pool of {1, 2, 3}, respectively indicating two surfaces along x || y || z axis
	
  // Compare maxima and minima to determine the intersections
	if(t0.x < 0 && t1.x < 0)
		return -1;            // whether there is only inverse intersection

	if(t0.x > 0) {
	  tmin     = (tmin > t0.x) ? tmin : t0.x;
		if(tmin == t0.x)
		  nearFlag = 1;         // update tmin if t0.x is positive and finite and set x to be the nearest surface axis
	}
	if(t1.x > 0) {
	  tmax    = (tmax < t1.x) ? tmax : t1.x;
		if(tmax == t1.x)
		  farFlag = 1;          // update tmax if t1.x is positive and finite and set x to be the nearest surface axis
	}

	// Then compare the current tmin and tmax with y, update
	if(t0.y < 0 && t1.y < 0)
		return -1;
	if(tmin > t1.y || t0.y > tmax)
	  return -1;            // whether the ray passes around the cube

	if(t0.y > 0) {
	  tmin     = (tmin > t0.y) ? tmin : t0.y;
		if(tmin == t0.y)
		  nearFlag = 2;         // update tmin if t0.y is positive and larger than tmin and set y to be the nearest surface axis
	}
	if(t1.y > 0) {
	  tmax    = (tmax < t1.y) ? tmax : t1.y;
		if(tmax == t1.y)
		  farFlag = 2;          // update tmax if t1.y is positive and less than tmax and set y to be the nearest surface axis
	}

  // Then compare the current tmin and tmax with z, update
  if(t0.z < 0 && t1.z < 0)
		return -1;
	if(tmin > t1.z || t0.z > tmax)
	  return -1;

	if(t0.z > 0) {
	  tmin     = (tmin > t0.z) ? tmin : t0.z;
		if(tmin == t0.z)
		  nearFlag = 3;         // update tmin if t0.z is positive and larger than tmin and set z to be the nearest surface axis
	}
	if(t1.z > 0 && t1.z < tmax) {
	  tmax    = (tmax < t1.z) ? tmax : t1.z;
		if(tmax == t1.z)
		  farFlag = 3;          // update tmax if t1.z is positive and less than tmax and set z to be the nearest surface axis
	}

	glm::vec3 intersectionPointTmp, normalTmp;
	// Update the normal and intersection point if the tmax or tmax are not all infinity
	if(epsilonCheck(tmin, minLimit) == 0) {
		intersectionPointTmp = getPointOnRay(rt, tmin);
		if(nearFlag == 1) {
		  normalTmp = glm::vec3(-1, 0, 0); 
			if(signInverseDirection.x)
				normalTmp.x = 1;
		} else if(nearFlag == 2) {
		  normalTmp = glm::vec3(0, -1, 0);
			if(signInverseDirection.y)
				normalTmp.y = 1;
		} else if(nearFlag ==3){
		  normalTmp = glm::vec3(0, 0, -1);
			if(signInverseDirection.z)
				normalTmp.z = 1;
		}
	} else if(epsilonCheck(tmax, maxLimit) == 0) {
		intersectionPointTmp = getPointOnRay(rt, tmax);
		if(farFlag == 1) {
		  normalTmp = glm::vec3(-1, 0, 0);
			if(signInverseDirection.x)
				normalTmp.x = 1;
		} else if(farFlag == 2) {
		  normalTmp = glm::vec3(0, -1, 0);
			if(signInverseDirection.y)
				normalTmp.y = 1;
		} else {
		  normalTmp = glm::vec3(0, 0, -1);
			if(signInverseDirection.z)
				normalTmp.z = 1;
		}
	} else {
		return -1;
	}

	// Find the distance between real intersection point and transform back the normal
  glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(intersectionPointTmp, 1.0f));
  intersectionPoint = realIntersectionPoint;
	normal = glm::normalize(multiplyMV(box.transform, glm::vec4(normalTmp, 0.0f)));
  return glm::length(r.origin - realIntersectionPoint);

}

//LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
//Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
  
  float radius = .5;
        
  glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin,1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction,0.0f)));

  ray rt; rt.origin = ro; rt.direction = rd;
  
  float vDotDirection = glm::dot(rt.origin, rt.direction);
  float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - pow(radius, 2));
  if (radicand < 0){
    return -1;
  }
  
  float squareRoot = sqrt(radicand);
  float firstTerm = -vDotDirection;
  float t1 = firstTerm + squareRoot;
  float t2 = firstTerm - squareRoot;
  
  float t = 0;
  if (t1 < 0 && t2 < 0) {
      return -1;
  } else if (t1 > 0 && t2 > 0) {
      t = glm::min(t1, t2); // min
  } else {
      t = glm::max(t1, t2); // max
  }

  glm::vec3 realIntersectionPoint = multiplyMV(sphere.transform, glm::vec4(getPointOnRay(rt, t), 1.0));
  glm::vec3 realOrigin = multiplyMV(sphere.transform, glm::vec4(0,0,0,1));

  intersectionPoint = realIntersectionPoint;
  normal = glm::normalize(realIntersectionPoint - realOrigin);
        
  return glm::length(r.origin - realIntersectionPoint);
}

//returns x,y,z half-dimensions of tightest bounding box
__host__ __device__ glm::vec3 getRadiuses(staticGeom geom){
    glm::vec3 origin = multiplyMV(geom.transform, glm::vec4(0,0,0,1));
    glm::vec3 xmax = multiplyMV(geom.transform, glm::vec4(.5,0,0,1));
    glm::vec3 ymax = multiplyMV(geom.transform, glm::vec4(0,.5,0,1));
    glm::vec3 zmax = multiplyMV(geom.transform, glm::vec4(0,0,.5,1));
    float xradius = glm::distance(origin, xmax);
    float yradius = glm::distance(origin, ymax);
    float zradius = glm::distance(origin, zmax);
    return glm::vec3(xradius, yradius, zradius);
}

//LOOK: Example for generating a random point on an object using thrust.
//Generates a random point on a given cube
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed){

    thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u01(0,1);
    thrust::uniform_real_distribution<float> u02(-0.5,0.5);

    //get surface areas of sides
    glm::vec3 radii = getRadiuses(cube);
    float side1 = radii.x * radii.y * 4.0f; //x-y face
    float side2 = radii.z * radii.y * 4.0f; //y-z face
    float side3 = radii.x * radii.z* 4.0f; //x-z face
    float totalarea = 2.0f * (side1+side2+side3);
    
    //pick random face, weighted by surface area
    float russianRoulette = (float)u01(rng);
    
    glm::vec3 point = glm::vec3(.5,.5,.5);
    
    if(russianRoulette<(side1/totalarea)){
        //x-y face
        point = glm::vec3((float)u02(rng), (float)u02(rng), .5);
    }else if(russianRoulette<((side1*2)/totalarea)){
        //x-y-back face
        point = glm::vec3((float)u02(rng), (float)u02(rng), -.5);
    }else if(russianRoulette<(((side1*2)+(side2))/totalarea)){
        //y-z face
        point = glm::vec3(.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2))/totalarea)){
        //y-z-back face
        point = glm::vec3(-.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2)+(side3))/totalarea)){
        //x-z face
        point = glm::vec3((float)u02(rng), .5, (float)u02(rng));
    }else{
        //x-z-back face
        point = glm::vec3((float)u02(rng), -.5, (float)u02(rng));
    }
    
    glm::vec3 randPoint = multiplyMV(cube.transform, glm::vec4(point,1.0f));

    return randPoint;
       
}

//TODO: IMPLEMENT THIS FUNCTION
//Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed){
	
	// Generate random number
	thrust::default_random_engine rng(hash(randomSeed));
	thrust::uniform_real_distribution<float> u01(0,TWO_PI);
	thrust::uniform_real_distribution<float> u02(0,PI);

	// Two independent variable expressing the point
	float theta, phi;
	theta = (float)u01(rng);
	phi   = (float)u02(rng);
	
	// Transform back to the real sphere coordinates
	glm::vec3 point(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));
	glm::vec3 randPoint = multiplyMV(sphere.transform, glm::vec4(point,1.0f));

	return randPoint;
}

#endif


