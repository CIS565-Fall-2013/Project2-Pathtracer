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
__host__ __device__ float boxIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
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
  return r.origin + t*glm::normalize(r.direction);
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

__host__ __device__ cudaMat4 transposeMat(cudaMat4 m){
	cudaMat4 transposedMat;
	transposedMat.x.x = m.x.x; transposedMat.x.y = m.y.x; transposedMat.x.z = m.z.x; transposedMat.x.w = m.w.x;
	transposedMat.y.x = m.x.y; transposedMat.y.y = m.y.y; transposedMat.y.z = m.z.y; transposedMat.y.w = m.w.y;
	transposedMat.z.x = m.x.z; transposedMat.z.y = m.y.z; transposedMat.z.z = m.z.z; transposedMat.z.w = m.w.z;
	transposedMat.w.x = m.x.w; transposedMat.w.y = m.y.w; transposedMat.w.z = m.z.w; transposedMat.w.w = m.w.w;
	return transposedMat;
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

	glm::vec3 bounds[2];
	bounds[0] = glm::vec3(-0.5f, -0.5f, -0.5f); // unit cube in OS with unit length on each edge
	bounds[1] = glm::vec3(0.5f, 0.5f, 0.5f);
	// transform the ray to object space
	ray rOS;
	rOS.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
	rOS.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));
		
	glm::vec3 invDirOfRay = getInverseDirectionOfRay(rOS);
	int signOfRay[3];
	signOfRay[0] = (invDirOfRay.x < 0);
	signOfRay[1] = (invDirOfRay.y < 0);
	signOfRay[2] = (invDirOfRay.z < 0);

	double tmin, tmax, tymin, tymax, tzmin, tzmax, t;
	tmin = (bounds[signOfRay[0]].x - rOS.origin.x) * invDirOfRay.x;
	tmax = (bounds[1-signOfRay[0]].x - rOS.origin.x) * invDirOfRay.x;
	tymin = (bounds[signOfRay[1]].y - rOS.origin.y) * invDirOfRay.y;
	tymax = (bounds[1-signOfRay[1]].y - rOS.origin.y) * invDirOfRay.y;

	if ((tmin > tymax) || (tymin > tmax))
		return -1.0f;
	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;
	tzmin = (bounds[(int)signOfRay[2]].z - rOS.origin.z) * invDirOfRay.z;
	tzmax = (bounds[1-(int)signOfRay[2]].z - rOS.origin.z) * invDirOfRay.z;

	if ((tmin > tzmax) || (tzmin > tmax))
		return -1.0f;
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;
	if (tmax < 0.0) return  -1.0f; // looking away

	if (tmin < 0.0) t = tmax; // inside
	else t = tmin; //outside
	
	glm::vec3 intersectionPointOS = getPointOnRay(rOS, t);
	intersectionPoint = multiplyMV(box.transform, glm::vec4(intersectionPointOS, 1.0f));
//	printf("%f %f %f\n", normal.x, normal.y, normal.z);
//	printf("%f %f\n", intersectionPointOS.x, bounds[0].x);
	if(fabs(intersectionPointOS.x - bounds[0].x) < EPSILON) 
		normal = glm::vec3(-1.0f, 0.0f, 0.0f);
	else if(fabs(intersectionPointOS.x - bounds[1].x) < EPSILON) 
		normal = glm::vec3(1.0f, 0.0f, 0.0f);
	else if(fabs(intersectionPointOS.y - bounds[0].y) < EPSILON) 
		normal = glm::vec3(0.0f, -1.0f, 0.0f);
	else if(fabs(intersectionPointOS.y - bounds[1].y) < EPSILON) 
		normal = glm::vec3(0.0f, 1.0f, 0.0f);
	else if(fabs(intersectionPointOS.z - bounds[0].z) < EPSILON) 
		normal = glm::vec3(0.0f, 0.0f, -1.0f);
	else if(fabs(intersectionPointOS.z - bounds[1].z) < EPSILON) 
		normal = glm::vec3(0.0f, 0.0f, 1.0f);
	else
		std::cout<<"Box intersection test error!"<<std::endl;
	glm::vec3 normalTip = intersectionPointOS + normal;
	glm::vec3 normalTipWS = multiplyMV(box.transform, glm::vec4(normalTip, 1.0f));
//		normal = glm::vec3(1.0f, 1.0f, 1.0f);
//	printf("%f %f %f\n", normalTipWS.x, normalTipWS.y, normalTipWS.z);
//	cudaMat4 invTransMat = transposeMat(box.inverseTransform);
//	cudaMat4 invTransMat = utilityCore::glmMat4ToCudaMat4(glm::inverse(utilityCore::cudaMat4ToGlmMat4(transposeMat(box.transform))));
//	normal = multiplyMV(invTransMat, glm::vec4(normal, 0.0f));
//	normal = glm::vec3(0.0f, 0.0f, 1.0f);
	normal = glm::normalize(normalTipWS - intersectionPoint);
//	float l = glm::length(normalTipWS - intersectionPoint);
//    normal = multiplyMV(box.transform, glm::vec4(normal, 0.0f));
//	printf("%f %f %f\n", normal.x, normal.y, normal.z);
	
//	printf("%f\n", l);
    return t;
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
    return -1.0f;
  }
  
  float squareRoot = sqrt(radicand);
  float firstTerm = -vDotDirection;
  float t1 = firstTerm + squareRoot;
  float t2 = firstTerm - squareRoot;
  
  float t = 0;
  if (t1 < 0 && t2 < 0) {
      return -1.0f;
  } else if (t1 > 0 && t2 > 0) {
      t = min(t1, t2);
  } else {
      t = max(t1, t2);
  }

  glm::vec3 realIntersectionPoint = multiplyMV(sphere.transform, glm::vec4(getPointOnRay(rt, t), 1.0f));
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
	// still uniformly distributed after scaling?
	thrust::default_random_engine rng(hash(randomSeed));
	thrust::uniform_real_distribution<float> u01(0,1);

	float u = (float)u01(rng);
	float v = (float)u01(rng);

	float theta = TWO_PI * u;
	float phi = acos(2*v - 1.0f);
	float radius = 0.5f;

	float x = radius * sin(phi) * cos(theta);
	float y = radius * cos(phi);
	float z = radius * sin(phi) * sin(theta);
	glm::vec3 point = glm::vec3(x, y, z);

	glm::vec3 randPoint = multiplyMV(sphere.transform, glm::vec4(point,1.0f));
    return glm::vec3(0,0,0);
}

#endif


