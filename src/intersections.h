// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERSECTIONS_H 
#define INTERSECTIONS_H

#include <limits>
#include "sceneStructs.h"
#include "cudaMat4.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include <thrust/random.h>

//Some forward declarations
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t);
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
__host__ __device__ float determinant(cudaMat3 m);
__host__ __device__ glm::vec3 getSignOfRay(ray r);
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);
__host__ __device__ float isIntersect(ray r, glm::vec3& intersectionPoint, glm::vec3& normal, staticGeom* geoms, int numberOfGeoms, mesh* meshes, face* faces, int& geomId);
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float meshIntersectionTest(staticGeom meshGeom, ray r, mesh* meshes, face* faces, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float triangleIntersectionTest(staticGeom meshGeom, face f, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float getArea(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3);
__host__ __device__ glm::vec3 getRandomPoint(staticGeom geom, float randomSeed);
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed);

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

// Gets the determinant of a 3x3 matrix
__host__ __device__ float determinant(cudaMat3 m){
	float add = 0.0f;
	add += (m.x.x * m.y.y * m.z.z) + (m.x.y * m.y.z * m.z.x) + (m.x.z * m.y.x * m.z.y);
	add -= ((m.x.z * m.y.y * m.z.x) + (m.x.y * m.y.x * m.z.z) + (m.x.x * m.y.z * m.z.y));
	return add;
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

//Geometry agnostic wrapper around the intersection tests
__host__ __device__ float isIntersect(ray r, glm::vec3& intersectionPoint, glm::vec3& normal, staticGeom* geoms, int numberOfGeoms, mesh* meshes, face* faces, int& geomId){
	float temp = -1.0f, intersect = 1000000;
	bool changed = false;
	glm::vec3 temp_IP, temp_N;
	for(int i = 0; i < numberOfGeoms; i++){
		switch(geoms[i].type){
		case 0:
			temp = sphereIntersectionTest(geoms[i], r, temp_IP, temp_N);
			break;
		case 1:
			temp = boxIntersectionTest(geoms[i], r, temp_IP, temp_N);
			break;
		default:
			temp = meshIntersectionTest(geoms[i], r, meshes, faces, temp_IP, temp_N);
			break;
		}

		//Makes sure that we are getting the first obj that the ray hits
		if(!epsilonCheck(temp, -1.0f) && temp < intersect){
			intersect = temp;
			intersectionPoint = temp_IP;
			normal = temp_N;
			geomId = i;
			changed = true;
		}
	  }

	if(changed) return intersect;
	else return -1;
}

//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
	cudaMat4 Ti = box.inverseTransform;
	glm::vec3 R0 = multiplyMV(Ti, glm::vec4(r.origin, 1.0));
	float l = glm::length(r.direction);
	glm::vec3 Rd = multiplyMV(Ti, glm::vec4(glm::normalize(r.direction), 0.0));
	double tnear = -10000000;
	double tfar = 10000000;
	int slab = 0;
	double t, t1, t2;
	while(slab < 3){
		if(Rd[slab] == 0){
			if(R0[slab] > .5 || R0[slab] < -.5){
				return -1;
			}
		}
		t1 = (-.5 - R0[slab]) / Rd[slab];
		t2 = (.5 - R0[slab]) / Rd[slab];
		if(t1 > t2){
			double temp = t1;
			t1 = t2;
			t2 = temp;
		}
		if(t1 > tnear) tnear = t1;
		if(t2 < tfar) tfar = t2;
		if(tnear > tfar){
			return -1;
		}
		if(tfar < 0){
			return -1;
		}
		slab++;
	}

	if(tnear > -.0001) t = tnear;
	else t = tfar;

	glm::vec3 p = R0 + (float)t * Rd;
	glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(p, 1.0));
	intersectionPoint = realIntersectionPoint;

	glm::vec4 temp_normal;
	if(abs(p[0] - .5) < .001){
		temp_normal = glm::vec4(1,0,0,0);
	}else if(abs(p[0] + .5) < .001){
		temp_normal = glm::vec4(-1,0,0,0);
	}else if(abs(p[1] - .5) < .001){
		temp_normal = glm::vec4(0,1,0,0);
	}else if(abs(p[1] + .5) < .001){
		temp_normal = glm::vec4(0,-1,0,0);
	}else if(abs(p[2] - .5) < .001){
		temp_normal = glm::vec4(0,0,1,0);
	}else if(abs(p[2] + .5) < .001){
		temp_normal = glm::vec4(0,0,-1,0);
	}
	normal = glm::normalize(multiplyMV(box.transform, temp_normal));
        
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

__host__ __device__ float meshIntersectionTest(staticGeom meshGeom, ray r, mesh* meshes, face* faces, glm::vec3& intersectionPoint, glm::vec3& normal){
	mesh m = meshes[meshGeom.meshId];

	glm::vec3 temp_IP, temp_N;
	float temp = -1.0f, t_min = 100000;
	bool changed = false;

	for(int i = 0; i < m.numberOfFaces; i++){
		face f = faces[m.startFaceIdx + i];

		temp = triangleIntersectionTest(meshGeom, f, r, temp_IP, temp_N);
		if(!epsilonCheck(temp, -1.0f) && temp < t_min){
			t_min = temp;
			intersectionPoint = temp_IP;
			normal = temp_N;
			changed = true;
		}
	}
	if(changed) return t_min;
	else return -1.0f;
}

__host__ __device__ float triangleIntersectionTest(staticGeom meshGeom, face f, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
	glm::vec3 ro = multiplyMV(meshGeom.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(meshGeom.inverseTransform, glm::vec4(r.direction, 0.0f)));

	glm::vec3 p1, p2, p3;
	p1 = f.p1, p2 = f.p2, p3 = f.p3;
	
	glm::vec3 N = glm::normalize(glm::cross(p3 - p1, p2 - p1));
	float t = glm::dot(N, p1 - ro) / glm::dot(N, rd);
	glm::vec3 P = ro + t * rd;

	float s, s1, s2, s3, s_rep;
	s = getArea(p1,p2,p3);
	s_rep = 1.0f / s;
	s1 = s_rep * getArea(P, p2, p3);
	s2 = s_rep * getArea(P, p3, p1);
	s3 = s_rep * getArea(P, p1, p2);

	if(s1 < 0 || s1 > 1 || s2 < 0 || s2 > 1 || s3 < 0 || s3 > 1 || !epsilonCheck(s1 + s2 + s3 - 1, 0.0f)) return -1.0f;
	else{
		// Calculate Normal
		normal = glm::normalize(multiplyMV(meshGeom.transform, glm::vec4(N, 0.0f)));
		float sign = glm::dot(rd, normal);
		if(sign < 0.0f) normal = -1.0f * normal;
		
		// Transform intersection point to world coord
		intersectionPoint = multiplyMV(meshGeom.transform, glm::vec4(P, 1.0f));

		return glm::length(intersectionPoint - r.origin);
	}
}

__host__ __device__ float getArea(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3){
	// Get Triangle Area
	cudaMat3 m1, m2, m3;
    glm::vec3 x, y, z;
	
	m1.x = glm::vec3(p1.y, p1.z, 1.0f), m1.y = glm::vec3(p2.y, p2.z, 1.0f), m1.z = glm::vec3(p3.y, p3.z, 1.0f);
	m2.x = glm::vec3(p1.z, p1.x, 1.0f), m2.y = glm::vec3(p2.z, p2.x, 1.0f), m2.z = glm::vec3(p3.z, p3.x, 1.0f);
	m3.x = glm::vec3(p1.x, p1.y, 1.0f), m3.y = glm::vec3(p2.x, p2.y, 1.0f), m3.z = glm::vec3(p3.x, p3.y, 1.0f);

	float d1, d2, d3;

	d1 = determinant(m1), d2 = determinant(m2), d3 = determinant(m3);

	return .5f * std::sqrt(std::pow(d1,2) + std::pow(d2,2) + std::pow(d3,2)); 
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

__host__ __device__ glm::vec3 getRandomPoint(staticGeom geom, float randomSeed){
	switch(geom.type){
	case GEOMTYPE::CUBE:
		return getRandomPointOnCube(geom, randomSeed);
	case GEOMTYPE::SPHERE:
		return getRandomPointOnSphere(geom, randomSeed);
	default:
		// TODO : Generate random point on given mesh, such that there is equal probability on all surface area
		return glm::vec3(0.0);
	}
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
	thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u01(-PI,PI);
	glm::vec3 radius = getRadiuses(sphere);
	float theta = (float)u01(rng);
	float phi = (float)u01(rng);
	glm::vec3 p = glm::vec3(glm::sin(theta) * glm::cos(phi), glm::sin(theta) * glm::sin(phi), glm::cos(theta));
	glm::vec3 randPoint = multiplyMV(sphere.transform, glm::vec4(p, 1.0));
	return randPoint;
}

#endif


