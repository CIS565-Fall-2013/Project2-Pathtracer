#pragma once
#include "shape.h"

class Sphere :
    public Shape
{
public:
    Sphere(void);
    ~Sphere(void);
    bool testRayIntersection( const glm::vec3 &raysource, const glm::vec3 &raydir, float &distance ) const;
    glm::vec3 getColor( const glm::vec3 &iDir, const glm::vec3 point ) const;
    glm::vec3 getNormalInPoint( const glm::vec3 &point ) const;

    std::string toString() const;

    glm::vec4 center;
    float radius;
};
