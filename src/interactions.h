// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#include "intersections.h"

struct AbsorptionAndScatteringProperties{
    glm::vec3 absorptionCoefficient;
    float reducedScatteringCoefficient;
};

//forward declaration
__host__ __device__ bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3);
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2);
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance);
__host__ __device__ glm::vec3 getReflectedRay(glm::vec3 d, glm::vec3 n);
__host__ __device__ glm::vec3 getRefractedRay(glm::vec3 d, glm::vec3 n, float IOR);
__host__ __device__ bool isDiffuseRay(float randomSeed, float hasDiffuse);
__host__ __device__  bool isRefractedRay(float randomSeed, float IOR, glm::vec3 d, glm::vec3 n, glm::vec3 t);
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2);

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance) {
  return glm::vec3(0,0,0);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering,
                                                        glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3){
  return false;
}

// Get the reflected ray direction from ray direction and normal
__host__ __device__ glm::vec3 getReflectedRay(glm::vec3 d, glm::vec3 n) {
	glm::vec3 VR; // reflected ray direction
	if (glm::length(-d - n) < THRESHOLD) {
		VR = n;
	}
	else if (abs(glm::dot(-d, n)) < THRESHOLD) {
		VR = d;
	}
	else {
		VR = glm::normalize(d - 2.0f * glm::dot(d, n) * n);
	}
	return VR;
}

// Get the refracted ray direction from ray direction, normal and index of refraction (IOR)
__host__ __device__ glm::vec3 getRefractedRay(glm::vec3 d, glm::vec3 n, float IOR) {
	glm::vec3 VT; // refracted ray direction
	float t = 1 / IOR;
	float base = 1 - t * t * (1 - pow(glm::dot(n, d), 2));
	if (base < 0) {
		 VT = glm::vec3(0, 0, 0);
	}
	else {
		VT = (-t * glm::dot(n, d) - sqrt(base)) * n + t * d; // refracted ray
		VT = glm::normalize(VT);
	}
	return VT;
}

// Determine if the reflected ray is a diffuse ray or not
__host__ __device__ bool isDiffuseRay(float randomSeed, float hasDiffuse) {
	// determine if ray is reflected according to the proportion
	thrust::default_random_engine rng(hash(randomSeed));
	thrust::uniform_real_distribution<float> u01(0,1);
	if (u01(rng) <= hasDiffuse) {
		return true;
	}
	else {
		return false;
	}
}

// Determine if the randomly generated ray is a refracted ray or a reflected ray
__host__ __device__  bool isRefractedRay(float randomSeed, float IOR, glm::vec3 d, glm::vec3 n, glm::vec3 t) {
	float rpar = (IOR * glm::dot(n, d) - glm::dot(n, t)) / (IOR * glm::dot(n, d) + glm::dot(n, t));
	float rperp = (glm::dot(n, d) - IOR * glm::dot(n, t)) / (glm::dot(n, d) + IOR * glm::dot(n, t));

	// compute proportion of the light reflected
	float fr = 0.5 * (rpar * rpar + rperp * rperp);

	// determine if ray is reflected according to the proportion
	thrust::default_random_engine rng(hash(randomSeed));
	thrust::uniform_real_distribution<float> u01(0,1);
	if (u01(rng) <= fr) {
		return false;
	}
	else {
		return true;
	}
}

//LOOK: This function demonstrates cosine weighted random direction generation in a sphere!
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2) {
    
    //crucial difference between this and calculateRandomDirectionInSphere: THIS IS COSINE WEIGHTED!
    
    float up = sqrt(xi1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = xi2 * TWO_PI;
    
    //Find a direction that is not the normal based off of whether or not the normal's components are all equal to sqrt(1/3) or whether or not at least one component is less than sqrt(1/3). Learned this trick from Peter Kutz.
    
    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
      directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
      directionNotNormal = glm::vec3(0, 1, 0);
    } else {
      directionNotNormal = glm::vec3(0, 0, 1);
    }
    
    //Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 = glm::normalize(glm::cross(normal, perpendicularDirection1));
    
    return ( up * normal ) + ( cos(around) * over * perpendicularDirection1 ) + ( sin(around) * over * perpendicularDirection2 );
    
}

//TODO: IMPLEMENT THIS FUNCTION
//Now that you know how cosine weighted direction generation works, try implementing non-cosine (uniform) weighted random direction generation.
//This should be much easier than if you had to implement calculateRandomDirectionInHemisphere.
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2) {
  return glm::vec3(0,0,0);
}

//TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
//returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__ __device__ int calculateBSDF(ray& r, glm::vec3 intersect, glm::vec3 normal, glm::vec3 emittedColor,
                                       AbsorptionAndScatteringProperties& currentAbsorptionAndScattering,
                                       glm::vec3& color, glm::vec3& unabsorbedColor, material m){

  return 1;
};

#endif
