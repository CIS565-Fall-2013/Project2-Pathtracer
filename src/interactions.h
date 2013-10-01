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
__host__ __device__ bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3);
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2);
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance);
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR);
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident);
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection);
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

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR) {
  return glm::vec3(0,0,0);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) {
  //nothing fancy here
  return glm::vec3(0,0,0);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection) {
  Fresnel fresnel;

  fresnel.reflectionCoefficient = 1;
  fresnel.transmissionCoefficient = 0;
  return fresnel;
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

// Given the refractive indices of the materials at intersection, cosine of the incident angle and
// a random number uniformly distributed between 0 and 1, this function returns true if the Fresnel
// reflectance term is greater than or equal to the random number, signifying reflection. Otherwise, 
// it will return false, signifying refraction/transmittance.
__host__ __device__ bool calculateFresnelReflectance (float outsideRefIndex, float insideRefIndex, float cosineIncidentAngle, float uniformRandomBetween01)
{
	float RF0 = (insideRefIndex - outsideRefIndex) / (insideRefIndex + outsideRefIndex);
	RF0 = RF0 * RF0;
	
	//if (cosineIncidentAngle < 0)		// External Reflection
	//{
		float fresnelRefl = RF0 + (1-RF0)*pow ((1-abs(cosineIncidentAngle)), 5);

		if (uniformRandomBetween01 <= fresnelRefl)
			return true;	// reflectance
		return false;	// refraction
	//}
	//else								// Internal Reflection.
	//{
	//	float sinCritAngle = insideRefIndex / outsideRefIndex;
	//	float sinIncidentAngle = sqrt (1 - (cosineIncidentAngle * cosineIncidentAngle));
	//	if (sinIncidentAngle > sinCritAngle)
	//		return true;	// reflection
	//	return false;	// refraction
	//}
}

//TODO: Done!
//Now that you know how cosine weighted direction generation works, try implementing non-cosine (uniform) weighted random direction generation.
//This should be much easier than if you had to implement calculateRandomDirectionInHemisphere.
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2) {
  
	float cosTheta = 2*xi1 - 1;		// Spread out xi1 in [0,1] to [-1, 1]. 
	float sinTheta = sqrt (1 - cosTheta*cosTheta);
	float phi = TWO_PI * xi2;

	return glm::vec3(sinTheta*cos(phi), sinTheta*sin(phi), cosTheta);
}

//TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
//returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__ __device__ int calculateBSDF(ray& r, glm::vec3 intersect, glm::vec3 normal, glm::vec3 emittedColor,
                                       AbsorptionAndScatteringProperties& currentAbsorptionAndScattering,
									   float randomSeed, glm::vec3& color, glm::vec3& unabsorbedColor, 
									   material m, glm::vec3 lightDir)
{
	int retVal = 0;
	r.origin = intersect-0.01f*r.direction; //slightly perturb along normal to avoid self-intersection.
	thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u01(0, 1);
    thrust::uniform_real_distribution<float> u02(0, 1);

	if (m.hasReflective)
	{
		r.direction = glm::normalize (reflectRay (r.direction, normal));
		retVal = 1;
	}
	else if (m.hasRefractive)
	{
		float cosIncidentAngle = glm::dot (r.direction, normal);
		float insideRefIndex = m.indexOfRefraction; float outsideRefIndex = 1.0;
		if (cosIncidentAngle > 0)	// If ray going from inside to outside.
		{
			outsideRefIndex = m.indexOfRefraction;
			insideRefIndex = 1.0;
			normal = -normal;
		}

		if (calculateFresnelReflectance (outsideRefIndex, insideRefIndex, cosIncidentAngle, u01(rng)))
		{	
//			if (cosIncidentAngle > 0)	// If ray going from inside to outside.
//				normal = -normal;		// Flip the normal for reflection.
			r.direction = glm::normalize (reflectRay (r.direction, normal));
			retVal = 1;
		}
		else
		{
			// As given in Real-Time Rendering, Third Edition, pp. 396.
			/*float w = (outsideRefIndex / insideRefIndex) * glm::dot (lightDir, normal);
			float k = sqrt (1 + ((w + (outsideRefIndex / insideRefIndex)) * (w - (outsideRefIndex / insideRefIndex))));
			r.direction = (w - k)*normal - (outsideRefIndex / insideRefIndex)*lightDir;*/
			r.direction = glm::normalize (glm::refract (r.direction, normal, outsideRefIndex/insideRefIndex));
			r.origin = intersect+0.01f*r.direction;
			retVal = 2;
		}
	}
	else
	{
		float xi1, xi2;
		r.direction = glm::normalize (calculateRandomDirectionInHemisphere (normal, u01 (rng), u02 (rng)));
	}
	return retVal;
};

#endif
