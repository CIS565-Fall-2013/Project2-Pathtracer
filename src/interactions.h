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

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION -- This is Done but no use all merged with calculateFresnel it's easy using to put together
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR) {
  // According to Snell's law (http://en.wikipedia.org/wiki/Snell%27s_law), calculate the cosine of incident angle and transmitted angle
  float ratio = incidentIOR / transmittedIOR;
  float cosIncident = glm::dot(normal, - incident);
  float cosTransmissionSquare = 1 - ratio * ratio * (1 - cosIncident * cosIncident);

  // Determine whether the square is negative or positive, if negative there is internal reflection
  if (cosTransmissionSquare < 0) {
	// When the transmitted angle reaches 90 degree the critical angle, there is no light to transmit and all lights reflect internally referring to http://en.wikipedia.org/wiki/Total_internal_reflection
	return calculateReflectionDirection(normal, incident);
  }

  // Calculate the transmission direction when the cosine of incident angle is negative change the formula
  float cosTransmission = glm::sqrt(cosTransmissionSquare);
  glm::vec3 transmissionDirection;
  if (cosIncident > 0.0f)
	  transmissionDirection = ratio * incident + (ratio * cosIncident - cosTransmission) * normal;
  else
	  transmissionDirection = ratio * incident + (ratio * cosIncident + cosTransmission) * normal;
  return glm::normalize(transmissionDirection);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION -- This is Done
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) {
  // nothing fancy here (simple vector addition)
  return (glm::normalize(incident - 2.0f * glm::dot(incident, normal) * normal));
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION -- This is Done
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection) {
  
  Fresnel fresnel;
  float n1    = incidentIOR;
  float n2    = transmittedIOR;
  
  // Return the default value for reflection when n1 or n2 is less than or equal zero
  if (n1 <= EPSILON && n2 <= EPSILON) {
    fresnel.reflectionCoefficient   = 1.0f;
    fresnel.transmissionCoefficient = 0.0f;
	return fresnel;
  }

  // Merged part detailed description in calculateTransmissionDirection
  float ratio = n1 / n2;
  float cosIncident = glm::dot(normal, - incident);
  float cosTransmissionSquare = 1 - ratio * ratio * (1 - cosIncident * cosIncident);

  if (cosTransmissionSquare < 0) {
	reflectionDirection = calculateReflectionDirection(normal, incident);
	fresnel.reflectionCoefficient   = 1.0f;
    fresnel.transmissionCoefficient = 0.0f;
	return fresnel;
  }

  // Calculate the transmission direction when the cosine of incident angle is negative change the formula
  float cosTransmission = glm::sqrt(cosTransmissionSquare);
  if (cosIncident > 0.0f)
	  transmissionDirection = ratio * incident + (ratio * cosIncident - cosTransmission) * normal;
  else
	  transmissionDirection = ratio * incident + (ratio * cosIncident + cosTransmission) * normal;

  // Fresnel Equation according to http://en.wikipedia.org/wiki/Fresnel_equations
  // Reflection coefficient of s-polarized light
  float Rs = 0;
  if (!epsilonCheck(n1 * cosIncident + n2 * cosTransmission, 0)) {
    Rs = glm::max(pow((n1 * cosIncident - n2 * cosTransmission) / (n1 * cosIncident + n2 * cosTransmission), 2.0f), 0.0f);
  }

  // Reflection coefficient of p-polarized light
  float Rp = 0;
  if (!epsilonCheck(n1 * cosTransmission + n2 * cosIncident, 0)) {
    Rp = glm::max(pow((n1 * cosTransmission - n2 * cosIncident) / (n1 * cosTransmission + n2 * cosIncident), 2.0f) , 0.0f);
  }

  fresnel.reflectionCoefficient   = (Rs + Rp) / 2;
  fresnel.transmissionCoefficient = 1 - fresnel.reflectionCoefficient;
  reflectionDirection = calculateReflectionDirection(normal, incident);
  return fresnel;
}

//LOOK: This function demonstrates cosine weighted random direction generation in a sphere!
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2) {
  // PERSONAL NOTES: the cosine weighted random direction is on the up direction normal to the hemisphere
  //crucial difference between this and calculateRandomDirectionInSphere: THIS IS COSINE WEIGHTED!
    
  float up = sqrt(xi1); // cos(theta)
  float over = sqrt(1 - up * up); // sin(theta)
  float around = xi2 * TWO_PI;
    
  //Find a direction that is not the normal based off of whether or not the normal's components are all equal to sqrt(1/3) or whether or not at least one component is less than sqrt(1/3). Learned this trick from Peter Kutz.
    
  glm::vec3 directionNotNormal = glm::vec3(0);
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
  // xi1, xi2 in [0, 1) the uniform weighted random direction  means each direction has equal probability of being selected 
  // Reffering to Emmanuel Agu's slides for the difference between cosine-weighted and uniform sampling
  float r = sqrt(1.0f - xi1);
  float theta = xi2 * TWO_PI;
  return glm::vec3(r * cos(theta), r * sin(theta), xi1);
}

//TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
//returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__ __device__ int calculateBSDF(ray currentRay, rayPool &nextRay, glm::vec3 intersect, glm::vec3 normal, glm::vec3 emittedColor,
                                      glm::vec3& color, material m, float time, cameraData cam){
  nextRay.ray.origin = intersect;
  glm::vec3 reflectionDirection,transmissionDirection;
  Fresnel fresnel;
  int classifier = -1;    // 0 -- diffusion, 1 -- reflection, 2 -- transmission

  // Generate the random numbers
  thrust::default_random_engine rng(hash(time));
  thrust::uniform_real_distribution<float> u01(0,1);
  float randomNumber = u01(rng);

  // Use the refractive or reflective parameter determining whether the surface is diffuse or reflected or transmitted
  if (m.hasReflective && m.hasRefractive) {
	// When the light is incidented from medium with IOR larger than 1 to the air there might exist internal reflection
	// Using Fresenel equation to find the coefficients both of refraction and reflection
	if (glm::dot(normal, currentRay.direction) < EPSILON)
		fresnel = calculateFresnel(normal, currentRay.direction, 1.0f, m.indexOfRefraction, reflectionDirection, transmissionDirection);
	else
		fresnel = calculateFresnel(-normal, currentRay.direction, m.indexOfRefraction, 1.0f, reflectionDirection, transmissionDirection);
	
	// Using Peter and Karl's method https://docs.google.com/file/d/0B72qrSEH6CGfbFV0bGxmLVJiUlU/edit, do Russian Roulette to choose whether reflection or refraction
	if (randomNumber > 0.5) {
	  classifier = 1;
	} else {
      classifier = 2;
	}
  } else if (m.hasReflective) {
    classifier = 1;
  } else if (m.hasRefractive) {
	classifier = 2;
  } else {
    // Diffision
    classifier = 0;
  } // if hasReflective && hasReflective
  
  if (classifier == 0) {
    // Diffusion
    nextRay.ray.direction = calculateRandomDirectionInHemisphere(normal, randomNumber, 100 * randomNumber);   // SO TRICKY here, increase the random angle 100 times the defects disappear!!!
	color = m.color * emittedColor;

    // Phong lighting for specular (general one not the Blinn-Phong lighting) http://en.wikipedia.org/wiki/Phong_reflection_model
    if( m.specularExponent > EPSILON ){
      float specularCoefficient = 6.0f;
      glm::vec3 V = glm::normalize(cam.position-intersect);
      glm::vec3 lightReflection=calculateReflectionDirection(normal,-nextRay.ray.direction);
      float specular = glm::pow(glm::max(glm::dot(V,lightReflection), 0.0f),m.specularExponent);
      color += specularCoefficient * specular * m.specularColor;
    }

    return 0;
  } else if (classifier == 1) {
    // Reflection
	nextRay.ray.direction = reflectionDirection;
	nextRay.coefficient = fresnel.reflectionCoefficient;
	color = emittedColor * nextRay.coefficient;

	// Phong lighting for specular (general one not the Blinn-Phong lighting) http://en.wikipedia.org/wiki/Phong_reflection_model
    if( m.specularExponent > EPSILON ){
      float specularCoefficient = 6.0f;
      glm::vec3 V = glm::normalize(cam.position-intersect);
      glm::vec3 lightReflection=calculateReflectionDirection(normal,-nextRay.ray.direction);
      float specular = glm::pow(glm::max(glm::dot(V,lightReflection), 0.0f),m.specularExponent);
      color += specularCoefficient * specular * m.specularColor;
    }
    return 1;
  } else if (classifier == 2) {
    // Transmission
    nextRay.ray.direction = transmissionDirection;
	nextRay.coefficient = fresnel.transmissionCoefficient;
	color = emittedColor * nextRay.coefficient;
    return 2;
  } // if classifier
  return -1;
};

#endif
