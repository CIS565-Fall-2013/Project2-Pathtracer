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

__host__ __device__ glm::vec3 colorMultiply(glm::vec3 c1, glm::vec3 c2)
{
	return glm::vec3(c1.x*c2.x,c1.y*c2.y,c1.z*c2.z);
}
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR) {
  
  float eta12=incidentIOR/transmittedIOR;
  float dotNI=glm::dot(normal,incident);
  float delta=1-eta12*eta12*(1-dotNI*dotNI);
  if(delta<0) return glm::normalize(incident-normal*glm::dot(incident,normal));
  return glm::normalize(normal*(-eta12*dotNI-sqrt(delta))+eta12*incident);
}

__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) {

	return glm::normalize(incident-normal*(2*glm::dot(normal,incident)));
}

__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, float specularExponent) {
  Fresnel fresnel;

  glm::vec3 transmissionDirection=calculateTransmissionDirection(normal,incident,incidentIOR, transmittedIOR);
  float theta=acos(glm::dot(normal,incident));
  float phi=acos(glm::dot(normal,transmissionDirection));
  float x1=tan(theta-phi)/tan(theta+phi);
  float x2=sin(theta-phi)/sin(theta+phi);
  x1*=x1;x2*=x2;
  
  fresnel.reflectionCoefficient = 0.5f*(x1+x2)*(1-glm::pow(0.99f,specularExponent));
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
__host__ __device__ __inline__ float rand_01()
{
	return ((float)rand()/RAND_MAX);
}
//TODO: IMPLEMENT THIS FUNCTION
//Now that you know how cosine weighted direction generation works, try implementing non-cosine (uniform) weighted random direction generation.
//This should be much easier than if you had to implement calculateRandomDirectionInHemisphere.
__host__ __device__ glm::vec3 getRandomDirectionInSphere(glm::vec3 normal,  float randomSeed) {

	glm::vec3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	} else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	} else {
		directionNotNormal = glm::vec3(0, 0, 1);
	}
	glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 = glm::normalize(glm::cross(normal, perpendicularDirection1));

	thrust::default_random_engine rng(hash((int)randomSeed));
	thrust::uniform_real_distribution<float> u01(0,1);
	//thrust::uniform_real_distribution<float> u02(0,1);

	float M_PI=3.14159265358979323846;
	float theta=u01(rng)*M_PI/2;
//	thrust::default_random_engine rng2(hash(randomSeed*2));
	float phi=u01(rng)*M_PI*2;
//	thrust::host_vector<float> h_1(2);
//	thrust::generate(h_1.begin(),h_1.end(),rand_01);

	//float theta=rand_01()*M_PI/2.0f;
	//float phi=rand_01()*M_PI*2.0f;
	float up=cos(theta);
	float s1=sin(theta)*cos(phi);
	float s2=sin(theta)*sin(phi);
  
	return glm::normalize(up*normal+perpendicularDirection1*s1+perpendicularDirection2*s2);
}

__host__ __device__ int getNextStep(int randomseed, float threshold1, float threshold2)
{
	thrust::default_random_engine rng(hash(randomseed));
	thrust::uniform_real_distribution<float> u(0,1);
	float randnum=u(rng);

	if(randnum<threshold1) return 0;
	else if (randnum<threshold2) return 1;
	else return 2;


}
//TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
//returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__ __device__ int calculateBSDF(ray& r, glm::vec3 intersect, glm::vec3 normal, glm::vec3 emittedColor,
                                       AbsorptionAndScatteringProperties& currentAbsorptionAndScattering,
                                       glm::vec3& color, glm::vec3& unabsorbedColor, material m){

  return 1;
};

__host__ __device__ glm::vec3 getTextureColor(glm::vec2 textcoord, int textureID, BMPInfo* bmps, int numberOfTexture, glm::vec3* textures)
{
	if(textureID<0) return glm::vec3(1,1,1);
	int w=bmps[textureID].width;
	int h=bmps[textureID].height;
	int x=(int)((float)textcoord.x*(float)w);
	int y=(int)((float)textcoord.y*(float)h);
	int index=x*h+y+bmps[textureID].offset;
	return textures[index];
};

#endif
