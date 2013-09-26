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

//Self explanatory
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t);

//LOOK: This is a custom function for multiplying cudaMat4 4x4 matrixes with vectors.
//This is a workaround for GLM matrix multiplication not working properly on pre-Fermi NVIDIA GPUs.
//Multiplies a cudaMat4 matrix and a vec4 and returns a vec3 clipped from the vec4
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
//Overload for doing the same with a cudaMat4 and a vec3.
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec3 v, bool isVector = false);

// Component-wise vector multiply function. Multiplies two vec3s.
__host__ __device__ glm::vec3 multiplyVV(glm::vec3 a, glm::vec3 b);

// Component-wise vector multiply. Multiplies two vec4s.
__host__ __device__ glm::vec4 multiplyVV(glm::vec4 a, glm::vec4 b);

//Gets sign of each component of a ray's inverse direction
__host__ __device__ glm::vec3 getSignOfRay(ray r);

//Gets 1/direction for a ray
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);

//TODO: Done!
//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);

//LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
//Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);

//LOOK: Example for generating a random point on an object using thrust.
//Generates a random point on a given cube
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a);

//Quick and dirty epsilon check
__host__ __device__ bool epsilonCheck(float a, float b);

//returns x,y,z half-dimensions of tightest bounding box
__host__ __device__ glm::vec3 getRadiuses(staticGeom geom);

//TODO: Done!
//Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed);


//------------------------------------------------------------------------------------------------
//		IMPLEMENTATIONS
//------------------------------------------------------------------------------------------------


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
  return r.origin + float(t-.0001)*glm::normalize(r.direction);
}

__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}

__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec3 v, bool isVector)
{
  float w;
  if (isVector)
	  w = 0;
  else
	  w = 1;
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*w);
  return r;
}

__host__ __device__ glm::vec3 multiplyVV(glm::vec3 a, glm::vec3 b)
{
	glm::vec3 r(0,0,0);
	r.x = a.x * b.x;
	r.y = a.y * b.y;
	r.z = a.z * b.z;
	return r;
}

__host__ __device__ glm::vec4 multiplyVV(glm::vec4 a, glm::vec4 b)
{
	glm::vec4 r(0,0,0,0);
	r.x = a.x * b.x;
	r.y = a.y * b.y;
	r.z = a.z * b.z;
	r.w = a.w * b.w;
	return r;
}

__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r){
  return glm::vec3(1.0/r.direction.x, 1.0/r.direction.y, 1.0/r.direction.z);
}

__host__ __device__ glm::vec3 getSignOfRay(ray r){
  glm::vec3 inv_direction = getInverseDirectionOfRay(r);
  return glm::vec3((int)(inv_direction.x < 0), (int)(inv_direction.y < 0), (int)(inv_direction.z < 0));
}

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
      t = min(t1, t2);
  } else {
      t = max(t1, t2);
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

__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed)
{
	thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u (0, 1);
	thrust::uniform_real_distribution<float> v (0, 1);
	float radius = getRadiuses (sphere).x;
	return glm::vec3 (radius*sin (PI*u(rng))*cos (TWO_PI*v(rng)), radius*sin (PI*u(rng))*sin (TWO_PI*v(rng)), radius*cos (PI*u(rng)));
}

__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2 &UVcoords)
{
	// Uses the slab method to check for intersection.
	// Refer http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm for details.

	// Define the constants. tnear = -INFINITY ; tfar = +INFINITY (+/- 1e6 for practical purposes)
	float tnear = -1e6, tfar = 1e6;
	float epsilon = 1e-3;

	// Body space extremities.
	float lowerLeftBack [3] = {-0.5, -0.5, -0.5};
	float upperRightFront [3] = {0.5, 0.5, 0.5};

	ray transformedRay;
	// Transform the ray from global to model space.
	transformedRay.origin = multiplyMV (box.inverseTransform, glm::vec4 (r.origin, 1.0));
	transformedRay.direction = glm::normalize (multiplyMV (box.inverseTransform, glm::vec4 (r.direction, 0.0)));

	float transRayOrigArr [3];
	transRayOrigArr [0] = transformedRay.origin.x;
	transRayOrigArr [1] = transformedRay.origin.y;
	transRayOrigArr [2] = transformedRay.origin.z;

	float transRayDirArr [3];
	transRayDirArr [0] = transformedRay.direction.x;
	transRayDirArr [1] = transformedRay.direction.y;
	transRayDirArr [2] = transformedRay.direction.z;

	// For each X, Y and Z, check for intersections using the slab method as described above.
	for (int loopVar = 0; loopVar < 3; loopVar ++)
	{
		if (fabs (transRayDirArr [loopVar]) < epsilon)
		{
			if ((transRayOrigArr [loopVar] < lowerLeftBack [loopVar]-epsilon) || (transRayOrigArr [loopVar] > upperRightFront [loopVar]+epsilon))
				return -1;
		}
		else
		{
			float t1 = (lowerLeftBack [loopVar] - transRayOrigArr [loopVar]) / transRayDirArr [loopVar];
			float t2 = (upperRightFront [loopVar] - transRayOrigArr [loopVar]) / transRayDirArr [loopVar];

			if (t1 > t2+epsilon)
			{
				t2 += t1;
				t1 = t2 - t1;
				t2 -= t1;
			}

			if (tnear < t1-epsilon)
				tnear = t1;

			if (tfar > t2-epsilon)
				tfar = t2;

			if (tnear > tfar+epsilon)
				return -1;

			if (tfar < 0-epsilon)
				return -1;
		}
	}

	// Get the intersection point in model space.
	glm::vec4 intersectionPointInBodySpace = glm::vec4 (getPointOnRay (transformedRay, tnear), 1.0);
	
	UVcoords = glm::vec2 (0, 0);
	if ( (isApproximate (intersectionPointInBodySpace.x, 0.5))  || // YZ Faces
		 (isApproximate (intersectionPointInBodySpace.x, -0.5)) )
	{
		UVcoords.x = intersectionPointInBodySpace.y;
		UVcoords.y = intersectionPointInBodySpace.z;
	}
	else if ((isApproximate (intersectionPointInBodySpace.y, 0.5)) || // XZ Faces
			 (isApproximate (intersectionPointInBodySpace.y, -0.5)))
	{
		UVcoords.x = intersectionPointInBodySpace.x;
		UVcoords.y = intersectionPointInBodySpace.z;
	}
	else if ((isApproximate (intersectionPointInBodySpace.z, 0.5)) || // XY Face
			 (isApproximate (intersectionPointInBodySpace.z, -0.5)))
	{
		UVcoords.x = intersectionPointInBodySpace.x;
		UVcoords.y = intersectionPointInBodySpace.y;
	}

	UVcoords.x += 0.5;	// u
	UVcoords.y += 0.5;	// v

	glm::vec4 bodySpaceOrigin = glm::vec4 (0,0,0,1);

	normal = glm::vec3 (0, 0, 0);

	float normalArr [3];
	normalArr [0] = normal.x;
	normalArr [1] = normal.y;
	normalArr [2] = normal.z;

	float intrPtBodySpaceArr [3];
	intrPtBodySpaceArr [0] = intersectionPointInBodySpace.x;
	intrPtBodySpaceArr [1] = intersectionPointInBodySpace.y;
	intrPtBodySpaceArr [2] = intersectionPointInBodySpace.z;

	float bodySpaceOrigArr [3];
	bodySpaceOrigArr [0] = bodySpaceOrigin.x;
	bodySpaceOrigArr [1] = bodySpaceOrigin.y;
	bodySpaceOrigArr [2] = bodySpaceOrigin.z;

	for (int loopVar = 0; loopVar < 3; loopVar ++)
	{	
		float diff = intrPtBodySpaceArr [loopVar] - bodySpaceOrigArr [loopVar];
		float diffAbs = fabs (diff);
		if ((diffAbs >= 0.5-epsilon) && (diffAbs <= 0.5+epsilon))
		{	
			normalArr [loopVar] = diff / diffAbs;
			break;
		}
	}

	glm::vec4 normalTobeTransformed = glm::vec4 (normalArr [0], normalArr [1], normalArr [2], 0);
	cudaMat4 transposeBoxInvTransform;
	transposeBoxInvTransform.x.x = box.inverseTransform.x.x;	transposeBoxInvTransform.x.y = box.inverseTransform.y.x;	transposeBoxInvTransform.x.z = box.inverseTransform.z.x;	transposeBoxInvTransform.x.w = box.inverseTransform.w.x;
	transposeBoxInvTransform.y.x = box.inverseTransform.x.y;	transposeBoxInvTransform.y.y = box.inverseTransform.y.y;	transposeBoxInvTransform.y.z = box.inverseTransform.z.y;	transposeBoxInvTransform.y.w = box.inverseTransform.w.y;
	transposeBoxInvTransform.z.x = box.inverseTransform.x.z;	transposeBoxInvTransform.z.y = box.inverseTransform.y.z;	transposeBoxInvTransform.z.z = box.inverseTransform.z.z;	transposeBoxInvTransform.z.w = box.inverseTransform.w.z;
	transposeBoxInvTransform.w.x = box.inverseTransform.x.w;	transposeBoxInvTransform.w.y = box.inverseTransform.y.w;	transposeBoxInvTransform.w.z = box.inverseTransform.z.w;	transposeBoxInvTransform.w.w = box.inverseTransform.w.w;

	// Transform the intersection point & the normal to world space.
	intersectionPoint = multiplyMV (box.transform, intersectionPointInBodySpace);
	normal = multiplyMV (transposeBoxInvTransform, normalTobeTransformed);
	normal = glm::normalize (normal);
	return glm::length (r.origin - intersectionPoint);
}

#endif


