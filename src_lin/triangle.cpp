#include "Triangle.h"
#include <iostream>

Triangle::Triangle(void)
{
}


Triangle::~Triangle(void)
{
}

std::string Triangle::toString() const
{
    return "triangle";
}

bool Triangle::testRayIntersection( const glm::vec3 &raysource, const glm::vec3 &raydir, float &distance ) const
{

    //glm::vec3 planeNormal;
    glm::vec3 BAcrossQA;
    glm::vec3 CBcrossQB;
    glm::vec3 ACcrossQC;
    glm::vec3 point;

    float plane_delta;
    float ray_offset;

    //planeNormal = glm::normalize( glm::cross( v[1] - v[0], v[2] - v[0] ) );
    plane_delta = glm::dot( pn, v[0] );

    if( glm::dot( pn, raydir ) == 0 ) //the ray and the plane are parallel
        return false;

    ray_offset = ( plane_delta - glm::dot( pn, raysource ) ) /
                    glm::dot( pn, raydir ) ;

    point = raysource + ( ray_offset * raydir );

    BAcrossQA = glm::cross( v[1] - v[0], point - v[0] );
    CBcrossQB = glm::cross( v[2] - v[1], point - v[1] );
    ACcrossQC = glm::cross( v[0] - v[2], point - v[2] );

    if( ray_offset < 0 )
        return false;
 
    else if( glm::dot( BAcrossQA, pn ) >= 0 &&
          glm::dot( CBcrossQB, pn ) >= 0 &&
        glm::dot( ACcrossQC, pn ) >= 0 )   
    {
      
        distance = ray_offset;
        return true;
    }
    else
        return false;
}

glm::vec3 Triangle::getColor( const glm::vec3 &iDir, const glm::vec3 point ) const
{
    return glm::vec3( 0, 0, 0 );
}

glm::vec3 Triangle::getNormalInPoint( const glm::vec3 &point ) const
{
    return pn; //plane normal
}