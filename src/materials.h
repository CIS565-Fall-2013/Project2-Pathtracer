#ifndef MATERIALS_H
#define MATERIALS_H

#include "sceneStructs.h"
#include "cudaMat4.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include <thrust/random.h>


__host__ __device__ glm::vec3 gridColor( glm::vec3 pointObjSpace, material& mtl)
{
}

__host__ __device__ glm::vec3 hStripesColor( glm::vec3 pointObjSpace, material& mtl)
{

}

__host__ __device__ glm::vec3 vStripesColor( glm::vec3 pointObjSpace, material& mtl)
{

}

__host__ __device__ glm::vec3 marbleColor( glm::vec3 pointObjSpace, material& mtl)
{

}


__host__ __device__ glm::vec3 getMaterialColor(glm::vec3 shadePoint, glm::vec3& shadeNormal,material& mtl,int objId, staticGeoms* geoms)
{
	if (mtl.type == BASE)
		return mtl.color;

	else if (mtl.type == GRID)
		return mtl.color
}


#endif
