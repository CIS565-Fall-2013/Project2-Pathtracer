#ifndef MATERIALS_H
#define MATERIALS_H

#include "sceneStructs.h"
#include "cudaMat4.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "perlin.h"


__host__ __device__ glm::vec3 bumpColor( glm::vec3 pointObjSpace, map& m, glm::vec3& shadeNormal,int* perlinPerm)
{
	float sineValue = 
		m.width1 * sinf( (pointObjSpace.x + pointObjSpace.y) * 0.05f + turbulence(10,pointObjSpace.x, pointObjSpace.y,pointObjSpace.z,m.width2, perlinPerm)) + 0.5f;
	glm::vec3 perturbation(sineValue,sineValue,sineValue);
	shadeNormal = (1-m.width2)*shadeNormal + m.width2*perturbation;
	shadeNormal = glm::normalize(shadeNormal);
	return m.color1;
}


__host__ __device__ glm::vec3 perlin(glm::vec3 pointObjSpace, map& m, int* perlinPerm)
{
	//Copied from my project,volumetric renderer in CIS 560

	float xPeriod = 0.3; //rotates defines repetition of marble lines in x direction
    float yPeriod = 0.1; //defines repetition of marble lines in y direction
	float zPeriod = 0;
    float turbPower = m.width2; //makes twists
    float turbSize = m.width1; //initial size of the turbulence
    
	float xyzValue = pointObjSpace.x * xPeriod + pointObjSpace.y * yPeriod + pointObjSpace.z*zPeriod + turbPower*turbulence(10,pointObjSpace.x, pointObjSpace.y,pointObjSpace.z,turbSize, perlinPerm);
    float sineValue = fabs(sin(xyzValue * PI));

	return sineValue*m.color1 + (1-sineValue)*m.color2;
}



__host__ __device__ glm::vec3 marble(glm::vec3 pointObjSpace, map& m, int* perlinPerm)
{ 
	//From http://www.codermind.com/articles/Raytracer-in-C++-Part-III-Textures.html
	float sineValue = 
		m.width1 * sinf( (pointObjSpace.x + pointObjSpace.y) * 0.05f + turbulence(10,pointObjSpace.x, pointObjSpace.y,pointObjSpace.z,m.width2, perlinPerm)) + 0.5f;


	return sineValue*m.color1 + (1-sineValue)*m.color2;

}


__host__ __device__ glm::vec3 checkerboardColor( glm::vec3 pointObjSpace, map& m)
{
	if(m.smooth)
	{
		float t = (2 + sinf(PI*pointObjSpace.x/m.width1)+sinf(PI*pointObjSpace.y/m.width2))/4;
		return t*m.color1 + (1-t)*m.color2;
	}

	if ( sinf(PI*pointObjSpace.x/m.width1) > 0  && sinf(PI*pointObjSpace.y/m.width2) > 0 )
		return m.color1;
	else
		return m.color2;
}

__host__ __device__ glm::vec3 hStripesColor( glm::vec3 pointObjSpace, map& m)
{
	if(m.smooth)
	{
		float t = (1 + sinf(PI*pointObjSpace.y/m.width1))/2;
		return t*m.color1 + (1-t)*m.color2;
	}

	if ( sinf(PI*pointObjSpace.y/m.width1) > 0)
		return m.color1;
	else
		return m.color2;
}

__host__ __device__ glm::vec3 vStripesColor( glm::vec3 pointObjSpace, map& m)
{
	if(m.smooth)
	{
		float t = (1 + sinf(PI*pointObjSpace.x/m.width1))/2;
		return t*m.color1 + (1-t)*m.color2;
	}

	if ( sinf(PI*pointObjSpace.x/m.width1) > 0)
		return m.color1;
	else
		return m.color2;
}


__host__ __device__ glm::vec3 getSurfaceColor(glm::vec3 shadePoint, glm::vec3& shadeNormal,material& mtl,int objId, staticGeom* geoms, map* maps,int* perlinData)
{
	glm::vec3 pointInObjSpace;

	if (geoms[objId].type == SPHERE)
	{
		glm::vec3 transformedCenter = multiplyMV( geoms[objId].transform, glm::vec4(0,0,0,1.0));
		//glm::vec3 northPoleDir = glm::normalize(multiplyMV( geoms[objId].transform, glm::vec4(0,1,0,0.0)));
		//glm::vec3 equatorDir =glm::normalize(multiplyMV( geoms[objId].transform, glm::vec4(1,0,0,0.0))); 
		glm::vec3 centerToIntersection = glm::normalize(shadePoint - transformedCenter);
		
		float u = atan2(centerToIntersection.z,centerToIntersection.x)/(2*PI);
		float v = asinf(centerToIntersection.y)/PI;

		//float r= acosf( -glm::dot( northPoleDir,centerToIntersection));
		//float v = r/PI;
		//float theta = (0.5*acosf( glm::dot(equatorDir,centerToIntersection))/(sinf(r)*PI));
		//float u = v;
		//if( glm::dot(glm::cross(northPoleDir,equatorDir),centerToIntersection)>0)
		//{
		//	u = theta;
		//}
		//else
		//	u = 1 - theta;
		pointInObjSpace = glm::vec3(u,v,0);
	}
	else
	{
	   pointInObjSpace = multiplyMV( geoms[objId].inverseTransform, glm::vec4(shadePoint,1.0));

	   float u = 0;
	   float v = 0;
	   if (fabs(pointInObjSpace.x)>=0.5f-0.001f && fabs(pointInObjSpace.x)<=0.5f+0.001f)
	   {
		   u = pointInObjSpace.y + 0.5f;
		   v = pointInObjSpace.z + 0.5f;
	   }
	   else if (fabs(pointInObjSpace.y)>=0.5f-0.001f && fabs(pointInObjSpace.y)<=0.5f+0.001f)
	   {
		   u = pointInObjSpace.x + 0.5f;
		   v = pointInObjSpace.z + 0.5f;
	   }
	   else
	   {
		   u = pointInObjSpace.x + 0.5f;
		   v = pointInObjSpace.y + 0.5f;
	   }

	   pointInObjSpace = glm::vec3(u,v,0);
	}

	map m = maps[mtl.mapID];

	if (m.type == BASE)
		return mtl.color;
	if (m.type == CHECKERBOARD)
		return checkerboardColor(pointInObjSpace,m);
	else if (m.type == VSTRIPE)
		return vStripesColor(pointInObjSpace,m);
	else if (m.type == HSTRIPE)
		return hStripesColor(pointInObjSpace,m);
	else if (m.type == MARBLE)
		return marble(shadePoint,m,perlinData);
	else if (m.type == BUMP)
		return bumpColor(shadePoint,m,shadeNormal,perlinData);
	else if (m.type == PERLIN)
		return perlin(shadePoint,m,perlinData);
	return mtl.color;
}

#endif
