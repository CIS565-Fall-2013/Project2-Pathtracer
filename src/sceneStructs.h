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

enum GEOMTYPE{ SPHERE, CUBE, MESH };
enum MAPTYPE{ BASE,CHECKERBOARD,VSTRIPE,HSTRIPE,MARBLE};

struct ray {
	glm::vec3 origin;
	glm::vec3 direction;
	bool active;
	glm::vec2 pixelIndex;
	glm::vec3 rayColor;
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
	float focalDist;
	float aperture;
};

struct camera {
	glm::vec2 resolution;
	glm::vec3* positions;
	glm::vec3* views;
	glm::vec3* ups;
	int* objIdBuffer;
	int frames;
	glm::vec2 fov;
	unsigned int iterations;
	glm::vec4* image;
	ray* rayList;
	std::string imageName;
	float focalDist;
	float aperture;
};

struct material{
	int mapID;
	glm::vec3 color;
	float specularExponent;
	glm::vec3 specularColor;
	float hasReflective;
	float hasRefractive;
	float indexOfRefraction;
	float hasScatter;
	glm::vec3 absorptionCoefficient;
	float reducedScatterCoefficient;
	float emittance;
	float diffuseCoefficient;
};

struct map
{
	enum MAPTYPE type; 
	glm::vec3 color1;
	glm::vec3 color2;
	float width1;
	float width2;
	int smooth;
};

#endif //CUDASTRUCTS_H
