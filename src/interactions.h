// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#include "intersections.h"
enum REFRSTAGE { REFR_ENTER, REFR_EXIT };

//forward declaration
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2);
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

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 incident, glm::vec3 normal) {
  //nothing fancy here
  return glm::normalize(incident-2.0f*normal*glm::dot(incident,normal));
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ float calculateFresnel(glm::vec3 incident,glm::vec3 normal, float incidentIOR, float transmittedIOR)
{
	//referred http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf

	float n = incidentIOR/transmittedIOR;
	float cosI = -glm::dot(incident, normal);
	float sinT2 = n*n*(1.0 -cosI*cosI);
	if(n > 1.0f)
		return 1.0f;

	float cosT = sqrt(1.0 -sinT2);
	float rorth = (incidentIOR*cosI - transmittedIOR*cosT)/(incidentIOR*cosI + transmittedIOR*cosT);
	float rpar = (transmittedIOR*cosI - incidentIOR*cosT)/(transmittedIOR*cosI + incidentIOR*cosT);

	return (rorth*rorth + rpar*rpar)/2.0f;
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ float calculateFresnelSchlick(glm::vec3 incident,glm::vec3 normal, float incidentIOR, float transmittedIOR)
{
	//referred http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf

	float r0 = (incidentIOR - transmittedIOR)/(incidentIOR+transmittedIOR);
	float cosX = -glm::dot(normal,incident);
	r0 *= r0;
	if( incidentIOR>transmittedIOR)
	{
		float n = incidentIOR/transmittedIOR;
		float sinT2 = n*n*(1.0f - cosX * cosX);
		if ( sinT2 > 1.0f) 
			return 1.0f;
		cosX = sqrtf(1.0f - sinT2);
	}
	float x = 1.0f - cosX;
	return r0+(1.0f - r0) *x*x*x*x*x;
}


__host__ __device__ glm::vec3 calculateRefractionDirection(glm::vec3 incident,glm::vec3 normal,material m,REFRSTAGE stage)
{
	float iorRatio = 1.0f/m.indexOfRefraction;
	if (stage == REFR_EXIT)
	{
		iorRatio = 1.0f/iorRatio;
		normal = -1.0f*normal;
	}
	return glm::normalize(glm::refract( incident, normal,iorRatio));
}

//TODO: IMPLEMENT THIS FUNCTION
//Now that you know how cosine weighted direction generation works, try implementing non-cosine (uniform) weighted random direction generation.
//This should be much easier than if you had to implement calculateRandomDirectionInHemisphere.
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2) {
	float up = xi1; 
    float over = sqrt(1 - up * up); // sin(theta)
    float around = xi2 * TWO_PI;	
	return glm::vec3( cos(around)*over,sin(around)*over,xi1);
}

//TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
//returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__ __device__ int calculateBSDF(ray& r, glm::vec3 intersect, glm::vec3 normal, glm::vec3 emittedColor,
                                       glm::vec4& color, glm::vec3& unabsorbedColor, material m, float randomSeed){

  r.origin = intersect;
  normal = glm::normalize(normal);
  thrust::default_random_engine rng(hash(randomSeed));
  thrust::uniform_real_distribution<float> u01(0,1);

 
  if (m.diffuseCoefficient>0.0f && m.hasReflective)
  {
	if (u01(rng) < m.diffuseCoefficient)
	{
		r.direction = calculateRandomDirectionInHemisphere(normal,u01(rng),u01(rng));
		return 0;
	}
	else
	{
		r.direction = calculateReflectionDirection(r.direction,normal);
		return 1;
	}
  }

  else if(m.hasReflective)
  {
	  r.direction = calculateReflectionDirection(normal,r.direction);
	  return 1;
  }

  else if (m.hasRefractive)
  {
    REFRSTAGE stage;
	float fresnelReflectance = 1.0f;
	if (glm::dot(r.direction,normal)<0)
	{
		stage = REFR_ENTER;
		fresnelReflectance = calculateFresnelSchlick(r.direction,normal,1.0f,m.indexOfRefraction);
	}
	else
	{
		stage = REFR_EXIT;
		fresnelReflectance = calculateFresnelSchlick(r.direction,normal,m.indexOfRefraction,1.0f);
	}

	if (u01(rng) < fresnelReflectance)
	{
		if(stage == REFR_EXIT)
		{
			normal = -1.0f*normal;
		}
		r.direction = calculateReflectionDirection(r.direction,normal);
		return 1;
	}
	else
	{
		r.direction = calculateRefractionDirection(r.direction,normal,m,stage);
		return 2;
	}

  }

  else
  {
	  r.direction = calculateRandomDirectionInHemisphere(normal,u01(rng),u01(rng));
	  return 0;
  }
 
};

#endif
