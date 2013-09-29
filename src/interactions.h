// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#include "intersections.h"

struct Fresnel {
	float reflectionCoefficient;
	float transmissionCoefficient;
};

struct AbsorptionAndScatteringProperties{
	glm::vec3 absorptionCoefficient;
	float reducedScatteringCoefficient;
};

//forward declaration
//__host__ __device__ bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3);
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2);
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance);
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR);
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident);
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection);
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2);

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

//Generates a random uniform direction in sphere. Note that this is a radially uniform distribution
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2) {
	float u = 2*(xi1-0.5);
	float th = 2*PI*xi2;

	glm::vec3 point;
	float root = glm::sqrt(1-u*u);

	//Find a uniform random point on a unit sphere and return it as a direction vector. Already normalized
	point.x = root*glm::cos(th);
	point.y = root*glm::sin(th);
	point.z = u;

	return point;
}


__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection) {
	Fresnel fresnel;
	if(utilityCore::epsilonCheck(glm::length(transmissionDirection), 0.0f))
	{
		//total internal reflection
		fresnel.reflectionCoefficient = 1;
		fresnel.transmissionCoefficient = 0;

	}else{

		//Assume unpolarized light
		float cos_t = glm::dot(-normal, transmissionDirection);
		float cos_i = glm::dot(normal, incident);
		float n1 = incidentIOR;
		float n2 = transmittedIOR;
		float Rdenom = (n1*cos_i+n2*cos_t);
		Rdenom *= Rdenom;//Squared

		float Rp = (n1*cos_i-n2*cos_t);
		Rp *= Rp/Rdenom;
		float Rs = (n2*cos_i-n2*cos_t);
		Rs *= Rs/Rdenom;

		fresnel.reflectionCoefficient = (Rp+Rs)/2;
		fresnel.transmissionCoefficient = 1-fresnel.reflectionCoefficient;
	}
	return fresnel;
}

//compute absorbtion through transmitted material
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance)
{
	return glm::exp(-absorptionCoefficient*distance);
}

__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR)
{
	float cos_thi = glm::dot(normal,incident);
	float eta  = incidentIOR/transmittedIOR;
	float sin2_tht = (eta*eta)*(1-cos_thi*cos_thi);
	if(sin2_tht > 0.0)
		return eta*incident + (eta*cos_thi- glm::sqrt(1-sin2_tht))*normal;
	else
		//Total internal reflection, no transmission
		return glm::vec3(0,0,0);

}

__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) {
	//nothing fancy here
	return incident-glm::dot(2.0f*normal, incident) * normal;
}


//TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
//returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__ __device__ int calculateBSDF(ray& r, glm::vec3 intersect, glm::vec3 normal, glm::vec3 emittedColor,
									  AbsorptionAndScatteringProperties& currentAbsorptionAndScattering,
									  glm::vec3& color, glm::vec3& unabsorbedColor, material m){

	

										  return 1;
};


#endif
