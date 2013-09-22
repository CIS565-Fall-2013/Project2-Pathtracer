

#include "Transform.h"
 
mat3 Transform::rotate(const float degrees, const vec3& axis) 
{
  mat3 r_mat;
  vec3 normalizedAxis = glm::normalize( axis );
  float r_rad = degrees / 180.0f * pi;

  float cos_r = cos( r_rad );
  float sin_r = sin( r_rad );

  /////Construct the rotation matrix
  r_mat[0][0] = cos_r + normalizedAxis[0]*normalizedAxis[0]*( 1 - cos_r );
  r_mat[0][1] = axis[1]*normalizedAxis[0]*( 1 - cos_r ) + normalizedAxis[2]*sin_r;
  r_mat[0][2] = axis[2]*normalizedAxis[0]*( 1 - cos_r ) - normalizedAxis[1]*sin_r;

  r_mat[1][0] = normalizedAxis[0]*normalizedAxis[1]*( 1 - cos_r ) - normalizedAxis[2]*sin_r;
  r_mat[1][1] = cos_r + normalizedAxis[1]*normalizedAxis[1]*( 1 - cos_r );
  r_mat[1][2] = normalizedAxis[2]*normalizedAxis[1]*( 1 - cos_r ) + normalizedAxis[0]*sin_r;

  r_mat[2][0] = normalizedAxis[0]*normalizedAxis[2]*( 1 - cos_r ) + normalizedAxis[1]*sin_r;
  r_mat[2][1] = normalizedAxis[1]*normalizedAxis[2]*( 1 - cos_r ) - normalizedAxis[0]*sin_r;
  r_mat[2][2] = cos_r + normalizedAxis[2]*normalizedAxis[2]*( 1 - cos_r );
  /////

  return  r_mat;
}

void Transform::left(float degrees, vec3& eye, vec3& up) 
{
  mat3 r_mat = rotate( degrees, up );
  eye = r_mat * eye;
}

void Transform::up(float degrees, vec3& eye, vec3& up) 
{
  vec3 axis = glm::cross( glm::normalize(eye), up ); //obtain the rotating axis
  mat3 r_mat = rotate( degrees, axis );
  eye = r_mat * eye;
  
  up = glm::cross( axis, glm::normalize(eye) ); //update UP vector
}

mat4 Transform::lookAt(const vec3 &eye, const vec3 &center, const vec3 &up) 
{
  vec3 eyeToCenter = center - eye;
  vec3 normalizedEyeCnt = glm::normalize( eyeToCenter );
  vec3 normalizedUp = glm::normalize( up );
  vec3 u, v;
  mat4 transMat(0.0);
  mat4 lookAtMat(0.0);

  u = glm::cross( normalizedEyeCnt, normalizedUp );
  v = glm::cross( u,normalizedEyeCnt );

  //construct Look-At matrix
  lookAtMat[0] = vec4( u, glm::dot(-u,eye) );
  lookAtMat[1] = vec4( v, glm::dot(-v,eye) );
  lookAtMat[2] = vec4( -normalizedEyeCnt, glm::dot(normalizedEyeCnt, eye) );
  lookAtMat[3] = vec4(0,0,0, 1 );
  lookAtMat = glm::transpose( lookAtMat );
 
  return lookAtMat;
}

mat4 Transform::perspective(float fovy, float aspect, float zNear, float zFar)
{
    mat4 ret;

    float vFOV_rad = fovy * pi / 180.0f;

    //Construct the projection matrix
    ret[0] = vec4( 1.0 / ( aspect * tan( vFOV_rad /2.0 ) ), 0, 0, 0 );
    ret[1] = vec4( 0, 1.0/tan( vFOV_rad/2.0 ) ,0, 0 );
    ret[2] = vec4( 0, 0, -( zFar + zNear ) / ( zFar - zNear), -1 );
    ret[3] = vec4( 0, 0, -2 * zFar * zNear / ( zFar - zNear ), 0 );
    return ret;
}

mat4 Transform::scale(const float &sx, const float &sy, const float &sz) 
{
    mat4 ret;

    ret[0] = vec4( sx, 0, 0, 0 );
    ret[1] = vec4( 0, sy, 0, 0 );
    ret[2] = vec4( 0, 0, sz, 0 );
    ret[3] = vec4( 0, 0, 0, 1 );
    return ret;
}

mat4 Transform::translate(const float &tx, const float &ty, const float &tz) 
{
    mat4 ret;

    ret[0] = vec4( 1, 0, 0, 0 );
    ret[1] = vec4( 0, 1, 0, 0 );
    ret[2] = vec4( 0, 0, 1, 0 );
    ret[3] = vec4( tx, ty, tz, 1 );
    return ret;
}



vec3 Transform::upvector(const vec3 &up, const vec3 & zvec) 
{
    vec3 x = glm::cross(up,zvec); 
    vec3 y = glm::cross(zvec,x); 
    vec3 ret = glm::normalize(y); 
    return ret; 
}


Transform::Transform()
{

}

Transform::~Transform()
{

}