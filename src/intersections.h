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
#include <thrust/device_vector.h>
//Some forward declarations
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t);
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
__host__ __device__ glm::vec3 getSignOfRay(ray r);
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);
__host__ __device__ float boxIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float boxIntersectionTest(glm::vec3 boxMin, glm::vec3 boxMax, staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
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

//TODO: IMPLEMENT THIS FUNCTION
//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(glm::vec3 boxMin, glm::vec3 boxMax,staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
	
	ray rt;
	rt.origin = multiplyMV(box.inverseTransform,glm::vec4(r.origin,1.0f));
	rt.direction = multiplyMV(box.inverseTransform,glm::vec4(r.direction,0));	
	/*float boxMin.x = -0.5;
	float boxMax.x = 0.5;
	float boxMin.y = -0.5;
	float boxMax.y = 0.5;
	float boxMin.z = -0.5;
	float boxMax.z = 0.5;*/
	double tnear = -1000000000000000000;
	double tfar = 1000000000000000000;
	double t1, t2,tmp;
	
	//xplaner
	if(abs(rt.direction.x) < EPSILON)
	//if(rt.direction.x == 0)
	{
		if(rt.origin.x>boxMax.x || rt.origin.x<boxMin.x)
			return -1;
	}
	else
	{
		t1 = (boxMin.x - rt.origin.x)/rt.direction.x;
		t2 = (boxMax.x - rt.origin.x)/rt.direction.x;
		if(t1>t2)
		{
			tmp = t1;
			t1 = t2;
			t2 = tmp;
		}
		if(t1>tnear) tnear = t1;
		if(t2<tfar) tfar = t2;
		if(tfar<tnear) return -1;
		if(tfar<0) return -1;
		
	}

	//yplaner
	if(abs(rt.direction.y) < EPSILON)
	//if(rt.direction.y == 0)
	{
		if(rt.origin.y>boxMax.y || rt.origin.y<boxMin.y)
		{
			return -1;
		}
	}
	else
	{
		t1 = (boxMin.y - rt.origin.y)/rt.direction.y;
		t2 = (boxMax.y - rt.origin.y)/rt.direction.y;
		if(t1>t2)
		{
			tmp = t1;
			t1 = t2;
			t2 = tmp;
		}
		if(t1>tnear) tnear = t1;
		if(t2<tfar) tfar = t2;
		if(tfar<tnear) return -1;
		if(tfar <0) return -1;
	}
	//z	
	if(abs(rt.direction.z) < EPSILON)
	//if(rt.direction.z == 0)
	{
		if(rt.origin.z>boxMax.z || rt.origin.z<boxMin.z)
		{
			return -1;
		}
	}
	else
	{
		t1 = (boxMin.z - rt.origin.z)/rt.direction.z;
		t2 = (boxMax.z - rt.origin.z)/rt.direction.z;
		if(t1>t2)
		{
			tmp = t1;
			t1 = t2;
			t2 = tmp;
		}
		if(t1>tnear) tnear = t1;
		if(t2<tfar) tfar = t2;
		if(tfar<tnear) return -1;
		if(tfar <0) return -1;
	}
	//glm::vec3 interp = getPointOnRay(rt,tnear);	
	glm::vec3 interp =rt.direction; interp*=tnear;
	interp += rt.origin;
	glm::vec3 objnormal = glm::vec3(0,0,0);
	float maxv = max(abs(interp.x),abs(interp.y));
	if(abs(interp.z)>maxv)
	{
		objnormal = glm::vec3(0,0,interp.z);
	}
	else
	{
		if(abs(interp.x)>abs(interp.y))
		{
			objnormal = glm::vec3(interp.x,0,0);
		}
		else
		{

			objnormal = glm::vec3(0,interp.y,0);
		}
	}
	objnormal = glm::normalize(objnormal);	
	normal = multiplyMV(box.transform,glm::vec4(objnormal,0));	
	normal = glm::normalize(normal);
	/*if(abs(normal.x) !=1 && abs(normal.y) !=1 && abs(normal.z) !=1)
		printf("%f,%f,%f ",r.direction.x,r.direction.y,r.direction.z);*/
	intersectionPoint = multiplyMV(box.transform,glm::vec4(interp,1.0f));
	return glm::length(intersectionPoint - r.origin);

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
__host__ __device__ float Determinate(glm::vec3 c1, glm::vec3 c2)
{
	return c1[0]*c2[1]+c2[0]*c1[2]+c1[1]*c2[2]-c1[2]*c2[1]-c2[0]*c1[1]-c1[0]*c2[2];
}
__host__ __device__ float triangleIntersectionTest(staticGeom mesh,glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, ray r, glm::vec3&intersectionPoint, glm::vec3 normal)
{
	ray rt;
	rt.origin = multiplyMV(mesh.inverseTransform,glm::vec4(r.origin,1.0f));
	rt.direction = multiplyMV(mesh.inverseTransform,glm::vec4(r.direction,0));

	//plane equation : normal (dot product) x = d;
	double d = normal.x*p1.x+normal.y*p1.y+normal.z*p1.z;
	//substitude ray equation to it
	// n(p0+tv0) = d
	//np0+tnv0 = d;-> t = (d-np0)/nv0
	double temp = normal.x*rt.origin.x+normal.y*rt.origin.y+normal.z*rt.origin.z;
	double temp2 = normal.x*rt.direction.x+normal.y*rt.direction.y+normal.z*rt.direction.z;
	if (temp2 == 0)
	{
		//ray is parallel to the plane
		return -1;
	}
	double t1 = (d - temp)/temp2;
	float t = t1;
	glm::vec3 p = rt.origin+glm::vec3(t1*rt.direction.x,t1*rt.direction.y,t1*rt.direction.z);
	//tell if possible intersection point is incide the triangle
	//I use the method described on ppt 659
	//calculate s
	glm::vec3 c1(p1.y,p2.y,p3.y);
	glm::vec3 c2(p1.z,p2.z,p3.z);
	glm::vec3 c3(p1.x,p2.x,p3.x);
	double d1 = Determinate(c1,c2);
	double d2 = Determinate(c2,c3);
	double d3 = Determinate(c3,c1);
	double s = 0.5*sqrt(d1*d1+d2*d2+d3*d3);
	if (s<=EPSILON)
	{
		return -1;
	}
	c1 = glm::vec3(p.y,p2.y,p3.y);
	c2 = glm::vec3(p.z,p2.z,p3.z);
	c3 = glm::vec3(p.x,p2.x,p3.x);
	d1 = Determinate(c1,c2);
	d2 = Determinate(c2,c3);
	d3 = Determinate(c3,c1);
	double s1 = 0.5*sqrt(d1*d1+d2*d2+d3*d3)/s;

	c1 = glm::vec3(p.y,p3.y,p1.y);
	c2 = glm::vec3(p.z,p3.z,p1.z);
	c3 = glm::vec3(p.x,p3.x,p1.x);
	d1 = Determinate(c1,c2);
	d2 = Determinate(c2,c3);
	d3 = Determinate(c3,c1);
	double s2 = 0.5*sqrt(d1*d1+d2*d2+d3*d3)/s;

	c1 = glm::vec3(p.y,p1.y,p2.y);
	c2 = glm::vec3(p.z,p1.z,p2.z);
	c3 = glm::vec3(p.x,p1.x,p2.x);
	d1 = Determinate(c1,c2);
	d2 = Determinate(c2,c3);
	d3 = Determinate(c3,c1);
	double s3 = 0.5*sqrt(d1*d1+d2*d2+d3*d3)/s;
	if((s1>=0&&s1<=1)&&(s2>=0&&s2<=1)&&(s3>=0&&s3<=1)&&(s1+s2+s3-1<EPSILON))
	{
		if (t<=EPSILON)
		{
			return -1;
		}
		else
		{
			glm::vec3 realIntersectionPoint = multiplyMV(mesh.transform, glm::vec4(getPointOnRay(rt, t), 1.0));
			glm::vec3 realOrigin = multiplyMV(mesh.transform, glm::vec4(0,0,0,1));

			intersectionPoint = realIntersectionPoint;			
			return glm::length(r.origin - realIntersectionPoint);
		}
	}
	else
	{
		return -1;
	}
}
__host__ __device__ float meshIntersectionTest(staticGeom mesh, ray r, glm::vec3& intersectionPoint, glm::vec3& normal,
	glm::vec3* pbo,unsigned short* ibo, glm::vec3* nbo)
	//thrust::device_vector<glm::vec3> pbo,thrust::device_vector<unsigned short> ibo, thrust::device_vector<glm::vec3> nbo)
{
	staticGeom boundingBox;
	boundingBox.rotation = glm::vec3(0,0,0);
	boundingBox.translation = mesh.translation;
	boundingBox.transform = mesh.transform;
	float dist = boxIntersectionTest(mesh.boundingBox_min,mesh.boundingBox_max,boundingBox,r,intersectionPoint,normal);
	if(dist == -1)
		return -1;
	else
	{
		glm::vec3 v1,v2,v3;
		float currDist = -1;	
		dist = -1;
		glm::vec3 interP;
		glm::vec3 tmpNormal;
		for(int i = 0;i<mesh.numberOfTriangle;i++)
		{
			v1 = pbo[mesh.pboIndexOffset+mesh.iboIndexOffset+i*3];
			v2 = pbo[mesh.pboIndexOffset+mesh.iboIndexOffset+i*3+1];
			v3 = pbo[mesh.pboIndexOffset+mesh.iboIndexOffset+i*3+2];
			tmpNormal = nbo[mesh.nboIndexOffset + i];
			currDist = triangleIntersectionTest(mesh,v1,v2,v3,r,interP,tmpNormal);
			if(currDist!=-1&&(dist == -1 || (dist != -1 && currDist<dist)))
			{
				dist = currDist;
				intersectionPoint = interP;
				normal = tmpNormal;
			}
		}
		if(dist == -1)
			return -1;
		else
		{
			return dist;
		}
	}
	return -1;
}
__host__ __device__ float IntersectionTest(staticGeom geom, ray r,glm::vec3& intersectionPoint,glm::vec3& interNormal)
	//,glm::vec3* pbo,unsigned short* ibo, glm::vec3* nbo)
	//thrust::device_vector<glm::vec3> pbo,thrust::device_vector<unsigned short> ibo, thrust::device_vector<glm::vec3> nbo)
{ 
	float dist = -1;
	if(geom.type == SPHERE)
	{
		dist = sphereIntersectionTest(geom,r, intersectionPoint,interNormal);
	}
	else if(geom.type == CUBE)
	{
		dist = boxIntersectionTest(geom,r, intersectionPoint,interNormal);
	}
	else
	{
		//TODO MESH
		//dist = meshIntersectionTest(geom,r,intersectionPoint,interNormal,pbo,ibo,nbo);
	}
	return dist;
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
    float side3 = radii.x * radii.z * 4.0f; //x-z face
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


