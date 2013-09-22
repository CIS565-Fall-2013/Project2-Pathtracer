#pragma once
#include "shape.h"


class Triangle :
    public Shape
{
public:
    Triangle(void);
    ~Triangle(void);
    bool testRayIntersection(const  glm::vec3 &raysource, const glm::vec3 &raydir, float &distance ) const;
    glm::vec3 getColor( const glm::vec3 &iDir, const glm::vec3 point ) const;
    glm::vec3 getNormalInPoint( const glm::vec3 &point ) const;
    std::string toString() const;

    glm::vec3 v[3];
    glm::vec4 vv[3];
    glm::vec3 n[3];

    glm::vec3 pn; //plane normal used when vertex normal not specified
};

