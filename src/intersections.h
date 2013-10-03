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
__host__ __device__ float boxIntersectionTest(staticGeom cube, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
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
  return r.origin + float(t-.0001)*glm::normalize(r.direction);
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
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal)
{
	// convert ray to object space
	glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

	// check intersection with cube 
	float rayPosXYZ[3] = {ro.x, ro.y, ro.z};
	float rayDirXYZ[3] = {rd.x, rd.y, rd.z};
	float maxBoxXYZ[3] = {0.5, 0.5, 0.5};
	float minBoxXYZ[3] = {-0.5, -0.5, -0.5};

	float Tnear = -1.0f * FLT_MAX;
	float Tfar = FLT_MAX;

	ray rt;
	rt.origin = ro;
	rt.direction = rd;

	for (int i = 0 ; i < 3 ; ++i)
	{
		if (rayDirXYZ[i] == 0)
		{
			if (rayPosXYZ[i] < minBoxXYZ[i] || rayPosXYZ[i] > maxBoxXYZ[i])
			{
				Tnear = -1.0f * FLT_MAX;
				Tfar = FLT_MAX;
				return -1;
			}
		}
		else
		{

			float t1 = (minBoxXYZ[i] - rayPosXYZ[i]) / rayDirXYZ[i];
			float t2 = (maxBoxXYZ[i] - rayPosXYZ[i]) / rayDirXYZ[i];

			if (t1 > t2)
			{
				float temp = t1;
				t1 = t2;
				t2 = temp;
			}
			 
			if (t1 > Tnear)
				Tnear = t1;

			if (t2 < Tfar)
				Tfar = t2;

			if (Tnear > Tfar || Tfar < 0)
			{
				Tnear = -1.0f * FLT_MAX;
				Tfar = FLT_MAX;
				return -1;
			}
		}
	}

	float returnVal = -1;

    if (Tnear < 0)
	{

		glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(rt, Tfar), 1.0));
		intersectionPoint = realIntersectionPoint;
		glm::vec3 realOrigin = multiplyMV(box.transform, glm::vec4(0,0,0,1));
		returnVal = Tfar;
	}
	else
	{
		glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(rt, Tnear), 1.0));
		intersectionPoint = realIntersectionPoint;
		glm::vec3 realOrigin = multiplyMV(box.transform, glm::vec4(0,0,0,1));
		returnVal = Tnear;
	}

	
	// compute normal
	glm::vec3 frontNormal = glm::vec3(0,0,1);
	glm::vec3 backNormal = glm::vec3(0,0,-1);
	glm::vec3 rightNormal = glm::vec3(1,0,0);
	glm::vec3 leftNormal = glm::vec3(-1,0,0);
	glm::vec3 topNormal = glm::vec3(0,1,0);
	glm::vec3 bottomNormal = glm::vec3(0,-1,0);

	// TODO: verify these bounds. Note the z coordinates
	glm::vec3 minBoxCoordinate(minBoxXYZ[0], minBoxXYZ[1], maxBoxXYZ[2]);
	glm::vec3 maxBoxCoordinate(maxBoxXYZ[0], maxBoxXYZ[1], minBoxXYZ[2]);

	glm::vec3 localNormal = glm::vec3(0,0,0); // depending on which plane the intersection point is on, localNormal will be set accordingly

	float frontCheck = FLT_MAX;
	float backCheck = FLT_MAX;
	float leftCheck = FLT_MAX;
	float rightCheck = FLT_MAX;
	float topCheck = FLT_MAX;
	float bottomCheck = FLT_MAX;
	float eps = 0.001;
	
	glm::vec3 localIsectPoint = getPointOnRay(rt, returnVal);

	frontCheck = glm::dot(localIsectPoint - minBoxCoordinate, frontNormal);
	backCheck = glm::dot(localIsectPoint - maxBoxCoordinate, backNormal);
	leftCheck = glm::dot(localIsectPoint - minBoxCoordinate, leftNormal);
	rightCheck = glm::dot(localIsectPoint - maxBoxCoordinate, rightNormal);
	topCheck = glm::dot(localIsectPoint - maxBoxCoordinate, topNormal);
	bottomCheck = glm::dot(localIsectPoint - minBoxCoordinate, bottomNormal);

	// front	
	if ( frontCheck < eps && frontCheck > -eps )
	{
		normal = glm::normalize(multiplyMV(box.transform, glm::vec4(frontNormal,0.0f)));
	}
	// back
	else if ( backCheck < eps && backCheck > -eps )
	{
		normal = glm::normalize(multiplyMV(box.transform, glm::vec4(backNormal,0.0f)));
	}
	// left
	else if ( leftCheck < eps && leftCheck > -eps)
	{
		normal = glm::normalize(multiplyMV(box.transform, glm::vec4(leftNormal,0.0f)));
	}
	// right
	else if ( rightCheck < eps && rightCheck > -eps)
	{
		normal = glm::normalize(multiplyMV(box.transform, glm::vec4(rightNormal,0.0f)));
	}
	// top
	else if ( topCheck < eps && topCheck > -eps)
	{
		normal = glm::normalize(multiplyMV(box.transform, glm::vec4(topNormal,0.0f)));
	}
	// bottom
	else if ( bottomCheck < eps && bottomCheck > -eps)
	{
		normal = glm::normalize(multiplyMV(box.transform, glm::vec4(bottomNormal,0.0f)));
	}
	else
	{
		// error condition
		returnVal = -1;
	}

	return returnVal;
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


// triangle intersection.
__host__ __device__ float triangleIntersectionTest(const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3, 
												   staticGeom geom, ray r, glm::vec3& intersectionPoint, glm::vec3& normal)
{
	// convert ray to object space
	glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

	glm::vec3 n1 = glm::cross((v2-v1),(v3-v1)); // assume ccw order of vertices
	n1 = glm::normalize(n1);

	float nd = glm::dot (n1, rd);

	if (nd == 0) // parallel to plane, no intersection
		return -1;

	float t = (glm::dot(n1, v1) - glm::dot(n1, ro)) / nd;

	if (t <= 0)
		return -1; 

	glm::vec3 localIntersectionPoint = ro + rd * t;

	if( glm::dot(glm::cross((v2-v1),(localIntersectionPoint-v1)),n1) > 0 &&
		glm::dot(glm::cross((v3-v2),(localIntersectionPoint-v2)),n1) > 0 &&
		glm::dot(glm::cross((v1-v3),(localIntersectionPoint-v3)),n1) > 0 )
	{
		normal = glm::normalize(multiplyMV(geom.transform, glm::vec4(n1,0.0f)));
		intersectionPoint = multiplyMV(geom.transform, glm::vec4(localIntersectionPoint, 1.0));
		return t;
	}
	else
	{
		return -1;
	}

	//vec3 objSpP0 = r.OSrayPos;
	//vec3 objSpV0 = r.OSrayDir;

	//vec3 n = cross((point2-point1),(point3-point1)); // assume ccw order of vertices
	//n = normalize(n);

	//float nd = dot(n,objSpV0); // dot product between normal and ray direction

	//if(nd == 0) // parallel to plane, no intersection
	//	return -1;

	//float t = (dot(n,point1) - dot(n,objSpP0)) / nd;

	//vec3 ip = objSpP0 + objSpV0 * t;

	//if( dot(cross((point2-point1),(ip-point1)),n) >= 0 &&
	//	dot(cross((point3-point2),(ip-point2)),n) >= 0 &&
	//	dot(cross((point1-point3),(ip-point3)),n) >= 0 )
	//{
	//	if(isect.t > t || isect.t == -1)
	//	{				
	//		isect.t = t;
	//		isect.normal = n;
	//		isect.computeNormalWS = true;
	//	}

	//	return t;
	//}
	//else
	//{
	//	return -1;
	//}
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
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed)
{
	thrust::default_random_engine rng(hash(randomSeed));
	thrust::uniform_real_distribution<float> xi1(0, 1);
	thrust::uniform_real_distribution<float> xi2(0, 1);

	float theta = 2.0f * PI * xi1(rng);
	float phi = acosf(2 * xi2(rng) - 1);

	float x = sin(phi) * sin(theta);
	float y = sin(phi) * cos(theta);
	float z = cos(phi);

	glm::vec3 point = glm::vec3(x,y,z);
	glm::vec3 finalPoint = multiplyMV(sphere.transform, glm::vec4(point, 1.0f));

	return finalPoint;
}

#endif


