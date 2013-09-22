#pragma once

#define GLM_SWIZZLE

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

using namespace glm;

const float pi = 3.14159265f ; // For portability across platforms


class Transform  
{
public:
	Transform();
	virtual ~Transform();
	static void left(float degrees, vec3& eye, vec3& up);
	static void up(float degrees, vec3& eye, vec3& up);
	static mat4 lookAt(const vec3& eye, const vec3 &center, const vec3& up);
	static mat4 perspective(float fovy, float aspect, float zNear, float zFar);
        static mat3 rotate(const float degrees, const vec3& axis) ;
        static mat4 scale(const float &sx, const float &sy, const float &sz) ; 
        static mat4 translate(const float &tx, const float &ty, const float &tz);
        static vec3 upvector(const vec3 &up, const vec3 &zvec) ; 
};

