#pragma once
#include "glm/glm.hpp"

class Light
{
public:
    Light(void);
    ~Light(void);
    
    int type;

    glm::vec4 pos;
    glm::vec3 color;
    float attenu_const;
    float attenu_linear;
    float attenu_quadratic;

    float width;   //for area light
    float cutoff;//for spot light

    glm::vec3 normal;
};

