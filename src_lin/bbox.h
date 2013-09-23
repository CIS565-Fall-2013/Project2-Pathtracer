#pragma once

#include "shape.h"

class Bbox: public Shape
{
public:
    Bbox();
    ~Bbox();
    bool testRayIntersection( const glm::vec3 &raysource, const glm::vec3 &raydir, float &distance ) const;
    glm::vec3 getColor( const glm::vec3 &iDir, const glm::vec3 point ) const;
    glm::vec3 getNormalInPoint( const glm::vec3 &point ) const;

    std::string toString() const;

    glm::vec3 min;
    glm::vec3 max;
    unsigned short polyNum; //Number of polygons this bounding box encloses
};