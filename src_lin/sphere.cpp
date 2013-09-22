#include "sphere.h"
#include <iostream>

Sphere::Sphere(void)
{
}


Sphere::~Sphere(void)
{
}

std::string Sphere::toString() const
{
    return "sphere";
}

bool Sphere::testRayIntersection( const glm::vec3 &raysource, const glm::vec3 &raydir, float &distance )const
{
    float t;
    float A;
    float B; //linear term
    float C; //Scalar term

    B = 2 * glm::dot( raydir, ( raysource - glm::vec3( center.x, center.y, center.z) ) );

    C = pow( ( raysource[0] - center[0] ), 2 )
          + pow( ( raysource[1] - center[1] ), 2 )
          + pow( ( raysource[2] - center[2] ), 2 ) - ( radius*radius );

    A = glm::dot( raydir, raydir );

    if( B*B - 4*A*C  <0) //no intersection
        return false;

    t = ( -B - sqrt( B*B -4*A*C ) ) *0.5 / A;
	if( t < 0 )
        t = ( -B + sqrt( B*B -4*A*C ) ) *0.5 /A;
    
	if( t < 0 )
		return false;
    else
    {
        distance = t;

        return true;
    }
}

glm::vec3 Sphere::getColor( const glm::vec3 &iDir, const glm::vec3 point ) const
{
    return glm::vec3(0,0,0);    
}

glm::vec3 Sphere::getNormalInPoint( const glm::vec3 &point ) const
{
    return glm::normalize( point - glm::vec3( center ) );
}