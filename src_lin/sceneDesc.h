#pragma once
#define GLM_SWIZZLE

#include "shape.h"
#include "sphere.h"
#include "triangle.h"
#include "light.h"
#include "material.h"
#include <vector>
#include "glm/glm.hpp"
#include "glm.h"


class SceneDesc
{
public:
    SceneDesc(void);
    SceneDesc( int w, int h ){ width = w; height = h;}
    ~SceneDesc(void);

public:
    unsigned int width;
    unsigned int height;

    glm::vec3 up;
    glm::vec3 center;
    glm::vec3 eyePos;
    glm::vec4 eyePosHomo;
    float fovy;
    int rayDepth;

    glm::vec3 ambient;

    std::vector<Shape*> primitives;
    std::vector<Light> lights;
    std::vector<Material> mtls;

    GLMmodel* model[10];
    unsigned short modelCount;
};