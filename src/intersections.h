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
__host__ __device__ float boxIntersectionTest(glm::vec3 boxMin, glm::vec3 boxMax, staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float triangleIntersectionTest(staticGeom triangle, ray r,glm::vec3 p11,glm::vec3 p12,glm::vec3 p13, glm::vec3& intersectionPoint, glm::vec3& normal);
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

//Wrapper for cube intersection test for testing against unit cubes
__host__ __device__  float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
  return boxIntersectionTest(glm::vec3(-.5,-.5,-.5), glm::vec3(.5,.5,.5), box, r, intersectionPoint, normal);
}

//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__  float boxIntersectionTest(glm::vec3 boxMin, glm::vec3 boxMax, staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
    glm::vec3 currentNormal = glm::vec3(0,0,0);

    ray ro = r;

    glm::vec3 iP0 = multiplyMV(box.inverseTransform,glm::vec4(r.origin, 1.0f));
    glm::vec3 iP1 = multiplyMV(box.inverseTransform,glm::vec4(r.origin+r.direction, 1.0f));
    glm::vec3 iV0 = iP1 - iP0;

    r.origin = iP0; 
    r.direction = glm::normalize(iV0);

    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    glm::vec3 rsign = getSignOfRay(r);
    glm::vec3 rInverseDirection = getInverseDirectionOfRay(r);

    if((int)rsign.x==0){
      tmin = (boxMin.x - r.origin.x) * rInverseDirection.x;
      tmax = (boxMax.x - r.origin.x) * rInverseDirection.x;
    }else{
      tmin = (boxMax.x - r.origin.x) * rInverseDirection.x;
      tmax = (boxMin.x - r.origin.x) * rInverseDirection.x;
    }

    if((int)rsign.y==0){
      tymin = (boxMin.y - r.origin.y) * rInverseDirection.y;
      tymax = (boxMax.y - r.origin.y) * rInverseDirection.y;
    }else{
      tymin = (boxMax.y - r.origin.y) * rInverseDirection.y;
      tymax = (boxMin.y - r.origin.y) * rInverseDirection.y;
    }

    if ( (tmin > tymax) || (tymin > tmax) ){
        return -1;
    }
    if (tymin > tmin){
        tmin = tymin;
    }
    if (tymax < tmax){
        tmax = tymax;
    }

    if((int)rsign.z==0){
      tzmin = (boxMin.z - r.origin.z) * rInverseDirection.z;
      tzmax = (boxMax.z - r.origin.z) * rInverseDirection.z;
    }else{
      tzmin = (boxMax.z - r.origin.z) * rInverseDirection.z;
      tzmax = (boxMin.z - r.origin.z) * rInverseDirection.z;
    }

    if ( (tmin > tzmax) || (tzmin > tmax) ){
        return -1;
    }
    if (tzmin > tmin){
        tmin = tzmin;
    }
    if (tzmax < tmax){
        tmax = tzmax;
    }
    if(tmin<0){
        return -1;
    }

    glm::vec3 osintersect = r.origin + tmin*r.direction;

    if(abs(osintersect.x-abs(boxMax.x))<.001){
        currentNormal = glm::vec3(1,0,0);
    }else if(abs(osintersect.y-abs(boxMax.y))<.001){
        currentNormal = glm::vec3(0,1,0);
    }else if(abs(osintersect.z-abs(boxMax.z))<.001){
        currentNormal = glm::vec3(0,0,1);
    }else if(abs(osintersect.x+abs(boxMin.x))<.001){
        currentNormal = glm::vec3(-1,0,0);
    }else if(abs(osintersect.y+abs(boxMin.y))<.001){
        currentNormal = glm::vec3(0,-1,0);
    }else if(abs(osintersect.z+abs(boxMin.z))<.001){
        currentNormal = glm::vec3(0,0,-1);
    }

    intersectionPoint = multiplyMV(box.transform, glm::vec4(osintersect, 1.0));



    normal = multiplyMV(box.transform, glm::vec4(currentNormal,0.0));
    return glm::length(intersectionPoint-ro.origin);
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

// Triangle intersection test 
__host__ __device__ float triangleIntersectionTest(staticGeom triangle, ray r,glm::vec3 p1,glm::vec3 p2,glm::vec3 p3, glm::vec3& intersectionPoint, glm::vec3& normal){
	 glm::vec3 ro = multiplyMV(triangle.inverseTransform, glm::vec4(r.origin,1.0f));
	 glm::vec3 rd = glm::normalize(multiplyMV(triangle.inverseTransform, glm::vec4(r.direction,0.0f)));

	 ray rt; rt.origin = ro; rt.direction = rd;
	 glm::vec3 n ;
	 n = glm::normalize(glm::cross((p3-p1),(p2-p1)));
	

	glm::vec3 nf = glm::normalize(glm::cross((p3-p1),(p2-p1))) ;
	glm::vec3 onor = multiplyMV(triangle.inverseTransform, glm::vec4(nf,1.0f));
	double thit ; 
	thit  =  (float)(glm::dot(p1,n) - glm::dot(ro,n))/(glm::dot(rd,n)) ;
	if(thit < 0 )
		 return -1 ;
	
	// check if the intersection with plane was inside triangle ,if not then output -1 
	float sa,ta ;
	glm::vec3 w;
	glm::vec3 rr =  getPointOnRay(rt, thit) ; 
	w = rr - p1 ;
	// Now using the parametric representation of a plane and finding the s and t values of the equation 
	// V(s,t) = V0 + s * ( V1 - V0) + t * (V2 - V0);
	// Find the s and t values using the ray equation . If s >=0 , t >=0 & s+t <= 0 ,then the point lies inside the triangle
	glm::vec3 u,v;
	u = p2-p1;
	v = p3-p1;
	float den = pow(glm::dot(u,v),2) - (glm::dot(u,u) * glm::dot(v,v)) ; 
	sa = (glm::dot(u,v)*glm::dot(w,v)  -  glm::dot(v,v) * glm::dot(w,u))/den ;
	ta = (glm::dot(u,v)*glm::dot(w,u)  -  glm::dot(u,u) * glm::dot(w,v))/den ;

	 glm::vec3 realIntersectionPoint = multiplyMV(triangle.transform, glm::vec4(getPointOnRay(rt, thit), 1.0));
     glm::vec3 realOrigin = multiplyMV(triangle.transform, glm::vec4(0,0,0,1));
	 intersectionPoint = realIntersectionPoint;

	 //normal =  glm::normalize(multiplyMV(triangle.transform, glm::vec4(n,0.0f)));

	 cudaMat4 itMat = triangle.inverseTransform ;
	 glm::vec4 r1(itMat.x.x,itMat.y.x,itMat.z.x,itMat.w.x);
	 glm::vec4 r2(itMat.x.y,itMat.y.y,itMat.z.y,itMat.w.y);
	 glm::vec4 r3(itMat.x.z,itMat.y.z,itMat.z.z,itMat.w.z);
	 glm::vec4 r4(itMat.x.w,itMat.y.w,itMat.z.w,itMat.w.w);

	 cudaMat4 ittrans;
	 ittrans.x = r1;	
	 ittrans.y = r2;
     ittrans.z = r3;
	 ittrans.w = r4;

	 normal =  glm::normalize(multiplyMV(ittrans, glm::vec4(n,0.0f)));
	 
	if ((sa >= 0) && (ta >= 0) && (sa+ta <= 1+0.001) )
	return glm::length(r.origin - realIntersectionPoint);
	else
	return -1;


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

//Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed){
  float radius=.5f;
  thrust::default_random_engine rng(hash(randomSeed));
  thrust::uniform_real_distribution<float> u01(-1,1);
  thrust::uniform_real_distribution<float> u02(0,TWO_PI);

  float theta = (float)u02(rng);
  float cosphi = (float)u01(rng);
  float sinphi = sqrt(1 - cosphi*cosphi);
  glm::vec3 point = radius*glm::vec3(sinphi*cos(theta),sinphi*sin(theta),cosphi);
  glm::vec3 randPoint = multiplyMV(sphere.transform, glm::vec4(point,1.0f));

  return randPoint;
}

#endif


