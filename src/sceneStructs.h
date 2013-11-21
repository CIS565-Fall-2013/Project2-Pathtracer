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
	int materialid;
	int v1;
	int v2;
	int v3;
	int n1;
	int n2;
	int n3;

	triangle() {};
	triangle(int vi1, int vi2, int vi3, int ni1, int ni2, int ni3) :
		v1(vi1), v2(vi2), v3(vi3), n1(ni1), n2(ni2), n3(ni3) {};
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

	// for mesh
	int vertexcount;
	int normalcount;
	int facecount;
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
	
	int textureid; // -1 means no texture
};

// texture structure on CPU
struct mtltexture {
	int width;
	int height;
	glm::vec3* colors;
};

// texture structure on CUDA
struct cudatexture {
	int width;
	int height;
	int startindex; // the index of the first pixel in the pixel array of all textures
};

#endif //CUDASTRUCTS_H