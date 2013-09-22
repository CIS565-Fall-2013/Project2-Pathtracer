#pragma once

#include "glm/glm.hpp"

class Material
{
public:
    glm::vec3 ambient;
    glm::vec3 emission;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
};