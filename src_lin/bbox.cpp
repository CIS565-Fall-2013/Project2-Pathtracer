#include "bbox.h"
#include <iostream>

Bbox::Bbox()
{
}

Bbox::~Bbox(void)
{
}

std::string Bbox::toString() const
{
    return "bounding box";
}

bool Bbox::testRayIntersection( const glm::vec3 &raysource, const glm::vec3 &raydir, float &distance ) const
{
    //empty right now
    return false;
}

glm::vec3 Bbox::getColor( const glm::vec3 &iDir, const glm::vec3 point ) const
{
    //empty right now
    return glm::vec3(0);
}

glm::vec3 Bbox::getNormalInPoint( const glm::vec3 &point ) const
{
    //empty right now
    return glm::vec3(0);
}
