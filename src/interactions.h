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

enum BounceType{DIFFUSE, REFLECT, TRANSMIT};

//forward declaration
//__host__ __device__ bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3);
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2);
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance);
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR);
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident);
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection);
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2);
__host__ __device__ BounceType bounceRay(rayState& r, glm::vec3 intersect, glm::vec3 normal, material* mats, int mhitIndex, float xi0/*for bounce type*/, float xi1/*for importance sampling*/,float xi2/*for importance sampling*/);

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

__host__ __device__ glm::vec3 sphericalToCartesian(float phi, float th)
{
	glm::vec3 dir;
	dir.x = glm::cos(phi)*glm::cos(th);
	dir.y = glm::sin(phi)*glm::sin(th);
	dir.z = glm::cos(th);
	return dir;
}

__host__ __device__ glm::vec3 sampleSpecularReflectionDirection(glm::vec3 incident, glm::vec3 normal, float specularExp, float xi1, float xi2)
{
	float th = glm::acos(glm::pow(xi1, 1/(specularExp+1)));
	float phi = 2*PI*xi2;

	glm::vec3 randDirZ = sphericalToCartesian(phi, th);
	glm::vec3 idealMirror = calculateReflectionDirection(normal, incident);

	//TODO: Compute exponential specular
	//Rotate rand direction to specular space
	//glm::vec3 zUnit = glm::vec3(0,0,1);
	//glm::vec3 rotAxis = glm::cross(zUnit, 
	return idealMirror;
}

__host__ __device__ glm::vec3 sampleSpecularTransmissionDirection(glm::vec3 incident, glm::vec3 normal, float incidentIOR, float transmissionIOR, float specularExp, float xi1, float xi2)
{

	float th = glm::acos(glm::pow(xi1, 1/(specularExp+1)));
	float phi = 2*PI*xi2;

	glm::vec3 randDirZ = sphericalToCartesian(phi, th);
	glm::vec3 idealMirror = calculateReflectionDirection(normal, incident);

	//TODO: Compute exponential specular
	//Rotate rand direction to specular space
	//glm::vec3 zUnit = glm::vec3(0,0,1);
	//glm::vec3 rotAxis = glm::cross(zUnit, 
	return calculateTransmissionDirection(normal, incident, incidentIOR, transmissionIOR);
}


__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float n1/*incidentIOR*/, float n2/*transmittedIOR*/) {
	Fresnel fresnel;

	float cos_i = glm::dot(normal,incident);
	float eta  = n1/n2;
	float sin2_tht = (eta*eta)*(1-cos_i*cos_i);
	//Check if transmission possible
	if(sin2_tht < 1.0f){
		//Transmission possible, compute coefficients
		//Assume unpolarized light
		float cos_t = glm::sqrt(1-sin2_tht);
		float Rdenom = (n1*cos_i+n2*cos_t);
		Rdenom *= Rdenom;//Squared

		float Rp = (n1*cos_i-n2*cos_t);
		Rp *= Rp/Rdenom;
		float Rs = (n2*cos_i-n2*cos_t);
		Rs *= Rs/Rdenom;

		fresnel.reflectionCoefficient = (Rp+Rs)/2;
		fresnel.transmissionCoefficient = 1-fresnel.reflectionCoefficient;
	}else{
		//total internal reflection
		fresnel.reflectionCoefficient = 1;
		fresnel.transmissionCoefficient = 0;
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
	float radicand = 1.0-(eta*eta)*(1-cos_thi*cos_thi);
	if(radicand > 0.0)
		//TODO: Figure out why my code doesn't work.
		//return glm::refract(incident, normal, eta);
		return eta*incident - (eta*cos_thi + sqrt(radicand))*normal;
	else
		//Total internal reflection, no transmission
		return glm::vec3(0,0,0);

}

__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) {
	//nothing fancy here
	return incident-glm::dot(2.0f*normal, incident) * normal;
}


//returns type of bounce that was performed
//Takes three random numbers to use in sampling
//Do not call this function if ray hit a light source.
__host__ __device__ BounceType bounceRay(rayState& r, renderOptions rconfig, glm::vec3 intersect, glm::vec3 normal, material* mats, int mhitIndex, float xi0/*for bounce type*/, float xi1/*for importance sampling*/,float xi2/*for importance sampling*/)
{
	material m = mats[mhitIndex];//material we hit

	float mLastIOR;
	if(r.matIndex >= 0 ){
		mLastIOR = mats[r.matIndex].indexOfRefraction;//material we were traveling through
	}
	else{
		mLastIOR = rconfig.airIOR;//material we were traveling through is open space
	}

	//phong inspired light model.
	float ks = clamp(MAX(m.hasReflective, m.hasRefractive), 0.0f,1.0f);
	//float kd = 1.0f-ks; //not actually needed, but implicit in this definition

	BounceType bounceType;
	//Specular or diffuse?
	if(xi1 <= ks)
	{
		//Specular. Determine if reflective or transmited
		if(m.hasReflective && m.hasRefractive){
			//both options available, compute fresnel coeffs
			Fresnel f = calculateFresnel(normal, r.r.direction, mLastIOR, m.indexOfRefraction);

			//scale our random number by ks
			//0 <= xi1 <= ks, therefore 0 <= xi1/ks <= 1
			if(xi1/ks <= f.reflectionCoefficient){
				//reflect
				bounceType = REFLECT;
			}else{
				//transmit
				bounceType = TRANSMIT;
			}

		}else if(m.hasReflective){
			bounceType = REFLECT;

		}else{
			//Refractive only
			bounceType = TRANSMIT;
		}
	}else{
		//Diffuse 
		bounceType = DIFFUSE;
	}

	//at this point we know what type of bounce has occured. Perform bounce
	switch(bounceType)
	{
	case DIFFUSE:
		//Randomly select direction in hemisphere. Medium doesn't change. Accumulate diffuse refection color
		r.r.direction = calculateRandomDirectionInHemisphere(normal, xi1, xi2);
		r.r.origin = intersect;
		r.T *= m.color;
		break;
	case REFLECT:
		r.r.direction = sampleSpecularReflectionDirection(r.r.direction, normal, m.specularExponent, xi1, xi2);
		r.r.origin = intersect;
		r.T *= m.specularColor;
		break;
	case TRANSMIT:

		//TODO: Fix refraction
		//if(r.matIndex >= 0)
		//	normal = -normal;
		r.r.direction = sampleSpecularTransmissionDirection(r.r.direction, normal, mLastIOR, m.indexOfRefraction, m.specularExponent, xi1, xi2);
		r.r.origin = intersect;
		if(glm::dot(normal, r.r.direction) < 0.0)
		{
			//entering the material
			r.matIndex = mhitIndex;
		}else{
			//exiting
			r.matIndex = -1;
		}
		break;
	}

	return bounceType;
};


#endif
