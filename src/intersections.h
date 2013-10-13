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
#include <math.h>
#include <thrust/host_vector.h>

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


//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection

__host__ __device__ bool CheckInBox(glm::vec3 targ)
{
	float x=targ.x;float y=targ.y;float z=targ.z;
	return (abs(x)<=0.5f+0.0001f && abs(y)<=0.5f+0.0001f && abs(z)<=0.5f+0.0001f);
}


//LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
//Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float sphereIntersection(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal, float defaultRadius,glm::vec2& textureCoord){
  
  float radius = defaultRadius;
        
  glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin,1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction,0.0f)));

  ray rt; rt.origin = ro; rt.direction = rd;
  
  float vDotDirection = glm::dot(rt.origin, rt.direction);
  float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - radius*radius);
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

__host__ __device__ float boxIntersection(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& textureCoord){


 
  float radius=0.9;
  glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin,1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction,0.0f)));
  ray rt; rt.origin = ro; rt.direction = rd;
  textureCoord=glm::vec2(0,0);
  // float vDotDirection = glm::dot(rt.origin, rt.direction);
  //float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - radius*radius)+0.00001f;
  //if (radicand < 0){
  //  return -1;
  //}

  //float squareRoot = sqrt(radicand);
  //float firstTerm = -vDotDirection;
  //float t1 = firstTerm + squareRoot;
  //float t2 = firstTerm - squareRoot;
  //
  //if (t1 < 0 && t2 < 0) {
  //    return -1;
  //}

	float t; glm::vec3 x;
	float maxt=100000000;
	float finalt=maxt;
	glm::vec3 hitnormal;
	
	if(abs(rd.x)>EPSILON)
	{//left face
		t=-(ro.x+0.5f)/rd.x; x=ro+rd*t;
		if(t>EPSILON && t<finalt && CheckInBox(x)){finalt=t;hitnormal=glm::vec3(-1,0,0);textureCoord=glm::vec2(x.y+0.5,x.z+0.5);}
		//right face
		t=-(ro.x-0.5f)/rd.x; x=ro+rd*t;
		if(t>EPSILON && t<finalt && CheckInBox(x)){finalt=t;hitnormal=glm::vec3(1,0,0);textureCoord=glm::vec2(x.y+0.5,x.z+0.5);}
	}
	
	if(abs(rd.y)>EPSILON)
	{ //bottom face
		t=-(ro.y+0.5f)/rd.y; x=ro+rd*t;
		if(t>EPSILON && t<finalt && CheckInBox(x)) {finalt=t;hitnormal=glm::vec3(0,-1,0);textureCoord=glm::vec2(x.z+0.5,x.x+0.5);}
		//top face
		t=-(ro.y-0.5f)/rd.y; x=ro+rd*t;
		if(t>EPSILON && t<finalt && CheckInBox(x)) {finalt=t;hitnormal=glm::vec3(0,1,0);textureCoord=glm::vec2(x.z+0.5,x.x+0.5);}
	}
		
	if(abs(rd.z)>EPSILON)
	{//back face
		t=-(ro.z+0.5f)/rd.z; x=ro+rd*t;
		if(t>EPSILON && t<finalt && CheckInBox(x)) {finalt=t;hitnormal=glm::vec3(0,0,-1);textureCoord=glm::vec2(x.x+0.5,x.y+0.5);}
		//front face
		t=-(ro.z-0.5f)/rd.z; x=ro+rd*t;
		if(t>EPSILON && t<finalt && CheckInBox(x)) {finalt=t;hitnormal=glm::vec3(0,0,1);textureCoord=glm::vec2(x.x+0.5,x.y+0.5);}
	}
	if(finalt>10000000.0f)  return -1;

  glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(rt, finalt), 1.0));
  normal = multiplyMV(box.transform,glm::vec4(hitnormal,0.0));
  normal=glm::vec3(normal.x,normal.y,normal.z);
  normal=glm::normalize(normal);
  intersectionPoint=realIntersectionPoint;

  return glm::length(r.origin - realIntersectionPoint);

 // return finalt;

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
	//randPoint=multiplyMV(cube.transform,glm::vec4(0,0,0,1));
    return randPoint;
       
}

//TODO: IMPLEMENT THIS FUNCTION
//Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed){
	double pi=3.14159265358979f;
	thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u01(0,pi);
    thrust::uniform_real_distribution<float> u02(-pi,pi);
	
	glm::vec3 point(sin((float)u01(rng))*cos((float)u02(rng)),sin((float)u01(rng))*sin((float)u02(rng)),cos((float)u01(rng)));
	glm::vec3 randPoint = multiplyMV(sphere.transform, glm::vec4(point,1.0f));
	return randPoint;
  
}
__host__ __device__ glm::vec3 clampColor(glm::vec3 target)
{
	return utilityCore::clampRGB(target);
}
__host__ __device__ glm::vec3 getRandomPoint(staticGeom x, float randomSeed)
{
	if(x.type==0) return getRandomPointOnSphere(x,randomSeed);
	else if (x.type==1) return getRandomPointOnCube(x,randomSeed);

}

#endif


