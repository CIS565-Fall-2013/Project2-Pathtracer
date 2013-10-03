// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#include "intersections.h"

#define DIFFUSE  0
#define SPECULAR 1
#define TRANSMIT 2

struct Fresnel {
  float reflectionCoefficient;
  float transmissionCoefficient;
};

struct AbsorptionAndScatteringProperties{
    glm::vec3 absorptionCoefficient;
    float reducedScatteringCoefficient;
};

//forward declaration
__host__ __device__ bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3);
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2);
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance);
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR);
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident);
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection);
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2);
__host__ __device__ glm::vec3 computePhongTotal(ray& r, glm::vec3 intersection_point, glm::vec3 intersection_normal, material intersection_mtl, staticGeom* lights, int numberOfLights, staticGeom* geoms, int numberOfGeoms, material* materials, float time);
__host__ __device__ float computeShadowCoefficient(glm::vec3 intersection_point, staticGeom light, staticGeom* geoms, int numberOfGeoms, float time);

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance) {
  return glm::vec3(0,0,0);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering,
                                                        glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3){
  return false;
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR) {
	
	float n1 = incidentIOR;
	float n2 = transmittedIOR;
	float n = n1 / n2;

	float c1 = glm::dot(-incident, normal);
	float c2 = sqrt(1 - (n*n)*(1 - c1*c1));
	
	if (c1 > 0.0f) {
		normal = -normal;
		c1 = -c1;
	}

	glm::vec3 transmitDirection = (n*incident) + (n*c1 + c2) * normal;
	return transmitDirection;
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) {
	float IdotN = glm::dot(-incident,normal);
	glm::vec3 I;
	if (IdotN < 0.0f) { I = incident;  }
	else			  { I = -incident; }
	glm::vec3 R = glm::normalize(2*IdotN*normal - I);
	return R;
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection) {
  Fresnel fresnel;
  
	float n1 = incidentIOR;
	float n2 = transmittedIOR;
	float n = n1 / n2;

	float c1 = glm::dot(-incident, normal);
	float c2 = sqrt(1 - (n*n)*(1 - c1*c1));
	
	float R1 = glm::abs( (n1*c1 - n2*c2) / (n1*c1 + n2*c2) ) * glm::abs( (n1*c1 - n2*c2) / (n1*c1 + n2*c2) );
	float R2 = glm::abs( (n1*c2 - n2*c1) / (n1*c2 + n2*c1) ) * glm::abs( (n1*c2 - n2*c1) / (n1*c2 + n2*c1) );

	float R = (R1 + R2) / 2.0f;
	float T = 1.0 - R;

	fresnel.reflectionCoefficient   = R;
	fresnel.transmissionCoefficient = T;

	return fresnel;
}

//LOOK: This function demonstrates cosine weighted random direction generation in a sphere!
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2) {
    
    //crucial difference between this and calculateRandomDirectionInSphere: THIS IS COSINE WEIGHTED!
    
    float up = sqrt(xi1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = xi2 * TWO_PI;
    
    //Find a direction that is not the normal based off of whether or not the normal's components are all equal to sqrt(1/3) 
	//or whether or not at least one component is less than sqrt(1/3). Learned this trick from Peter Kutz.
    
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
  
	float z = xi1;
	float theta = xi2 * TWO_PI;
	
	float r = sqrt(1-z*z);
	float x = r*cos(theta);
	float y = r*sin(theta);

	return glm::vec3(x,y,z);
}

//TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
//returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__ __device__ int calculateBSDF(ray& r, glm::vec3 intersect, glm::vec3 normal, material* m, float randomSeed){
                                       //AbsorptionAndScatteringProperties& currentAbsorptionAndScattering
									   
	if (!m->hasReflective && !m->hasRefractive) { return DIFFUSE; }

	float incidentIOR    = r.currentIOR;
	float transmittedIOR = m->indexOfRefraction;

	glm::vec3 incident = r.direction;
	glm::vec3 reflectionDirection = calculateReflectionDirection(normal, incident);
	glm::vec3 transmittedDirection = calculateTransmissionDirection(normal, incident, incidentIOR, transmittedIOR);

	Fresnel fresnel = calculateFresnel(normal, r.direction, incidentIOR, transmittedIOR, reflectionDirection, transmittedDirection);
	
	double diffuse_range, specular_range;
	diffuse_range = 0.2;
	if (!m->hasRefractive) { specular_range = 1.0; }
	else				   { specular_range = diffuse_range + (1.0 - diffuse_range) * fresnel.reflectionCoefficient * m->hasReflective; }

	thrust::default_random_engine rng(hash(randomSeed));
	thrust::uniform_real_distribution<float> u01(0,1);
	float sample = (float)u01(rng);

	if (sample < diffuse_range)
		return DIFFUSE;
	else if (sample < specular_range)
		return SPECULAR;
	else
		return TRANSMIT;
};

#endif
