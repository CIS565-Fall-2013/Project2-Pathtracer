// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef CUDASTRUCTS_H
#define CUDASTRUCTS_H

#include "glm/glm.hpp"
#include "cudaMat4.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>

enum GEOMTYPE{ SPHERE, CUBE, MESH };

struct triangle {
	int geomId;
	glm::vec3 v1;
	glm::vec3 v2;
	glm::vec3 v3;
	glm::vec3 n1;
	glm::vec3 n2;
	glm::vec3 n3;
	
	triangle(): geomId(), v1(), v2(), v3(), n1(), n2(), n3() {}
	triangle(int id, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 vn1, glm::vec3 vn2, glm::vec3 vn3)
		: geomId(id), v1(p1), v2(p2), v3(p3), n1(vn1), n2(vn2), n3(vn3) {}
};

struct transformedTriangle {
	int materialid;
	glm::vec3 v1;
	glm::vec3 v2;
	glm::vec3 v3;
	glm::vec3 n1;
	glm::vec3 n2;
	glm::vec3 n3;
};

struct ray {
	glm::vec3 origin;
	glm::vec3 direction;
	int index;
	glm::vec3 baseColor;
	bool active;
};

struct geom {
	enum GEOMTYPE type;
	int materialid;
	int frames;
	glm::vec3* translations;
	glm::vec3* rotations;
	glm::vec3* scales;
	cudaMat4* transforms;
	cudaMat4* inverseTransforms;
};

struct staticGeom {
	enum GEOMTYPE type;
	int materialid;
	glm::vec3 translation;
	glm::vec3 rotation;
	glm::vec3 scale;
	cudaMat4 transform;
	cudaMat4 inverseTransform;
};

struct cameraData {
	glm::vec2 resolution;
	glm::vec3 position;
	glm::vec3 view;
	glm::vec3 up;
	glm::vec2 fov;
	float focal;
	float aperture;
};

struct camera {
	glm::vec2 resolution;
	glm::vec3* positions;
	glm::vec3* views;
	glm::vec3* ups;
	int frames;
	glm::vec2 fov;
	unsigned int iterations;
	float focal;
	float aperture;
	glm::vec3* image;
	ray* rayList;
	std::string imageName;
};

struct material {
	glm::vec3 color;
	float specularExponent;
	glm::vec3 specularColor;
	float hasDiffuse; // diffuse contribution
	float hasReflective;
	float hasRefractive;
	float indexOfRefraction;
	float hasScatter;
	glm::vec3 absorptionCoefficient;
	float reducedScatterCoefficient;
	float emittance;
};

#endif //CUDASTRUCTS_H
