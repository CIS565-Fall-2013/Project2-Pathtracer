#ifndef MATERIALS_H
#define MATERIALS_H

#include "sceneStructs.h"
#include "cudaMat4.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include <thrust/random.h>
#include "intersections.h"


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


__host__ __device__ glm::vec3 getSurfaceColor(glm::vec3 shadePoint, glm::vec3& shadeNormal,material& mtl,int objId, staticGeom* geoms, map* maps)
{
	//glm::vec3 pointInObjSpace;

	//if (geoms[objId].type == SPHERE)
	//{
	//	//glm::vec3 transformedCenter = glm::normalize(multiplyMV( geoms[objId].transform, glm::vec4(0,0,0,1.0)));
	//	//glm::vec3 northPoleDir = glm::normalize(multiplyMV( geoms[objId].transform, glm::vec4(0,1,0,0.0)));
	//	//glm::vec3 equatorDir =glm::normalize(multiplyMV( geoms[objId].transform, glm::vec4(1,0,0,0.0))); 
	//	//glm::vec3 centerToIntersection = glm::normalize(shadePoint - transformedCenter);
	//	//
	//	//float r= acosf( -glm::dot( northPoleDir,centerToIntersection));
	//	//float v = r/PI;

	//	//float theta = (0.5*acosf( glm::dot(equatorDir,centerToIntersection))/(sinf(r))*PI);

	//	//float u = v;
	//	//if( glm::dot(glm::cross(northPoleDir,equatorDir),centerToIntersection)>0)
	//	//{
	//	//	u = theta;
	//	//}
	//	//else
	//	//	u = 1 - theta;
	//	//pointInObjSpace = glm::vec3(u,v,0);

	//	pointInObjSpace = shadePoint;
	//}

	//else
	map m = maps[mtl.mapID];

	if (m.type == BASE)
		return mtl.color;
	glm::vec3 pointInObjSpace = multiplyMV( geoms[objId].inverseTransform, glm::vec4(shadePoint,1.0));	
	if (m.type == CHECKERBOARD)
		return checkerboardColor(pointInObjSpace,m);
	else if (m.type == VSTRIPE)
		return vStripesColor(pointInObjSpace,m);
	else if (m.type == HSTRIPE)
		return hStripesColor(pointInObjSpace,m);
	return mtl.color;
}

#endif
