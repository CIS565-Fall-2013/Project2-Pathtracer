#pragma once
#define GLM_SWIZZLE
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include "glm\glm.hpp"

#define FLOAT_INF 0x7F800000

#define TYPE_SPHERE 0
#define TYPE_TRIANGLE 1
#define TYPE_BBOX 2
#define MAXDEPTH 8
#define DST_SCALE 2

#define cudaErrorCheck( errNo ) checkError( (errNo), __FILE__, __LINE__ )


inline void checkError( cudaError_t err, const char* const filename, const int line  )
{
    if( err != cudaSuccess )
    {
        std::cerr<<"CUDA ERROR: "<<filename<<" line "<<line<<"\n";
        std::cerr<<cudaGetErrorString( err )<<"\n";
        exit(1);
    }
}


typedef struct _Primitive
{
    int type;  

    //used for sphere type
    glm::vec3 center;
    float radius;

    //used for triangle type
    glm::vec3 vert[3];
    glm::vec3 pn; //plane normal used when vertex normal not specified

    unsigned short mtl_id;
}_Primitive;

typedef struct _Light
{
    glm::vec4 pos;
    glm::vec3 color;
    glm::vec3 normal;   //for area light
    float attenu_const;
    float attenu_linear;
    float attenu_quadratic;
    float cutoff;  
    unsigned short type; 
    float width; //for area light


}_Light;

typedef struct _CamreaData
{
    glm::vec3 eyePos;
    glm::vec3 uVec;
    glm::vec3 vVec;
    glm::vec3 wVec;
    //glm::vec2 viewportHalfDim;
    glm::vec2 offset1;
    glm::vec2 jitteredOffset1;
    glm::vec2 offset2;
    float focalDist;

}_CameraData;

typedef struct _Material
{
    glm::vec3 diffuse;
    glm::vec3 specular;
    glm::vec3 emission;
    glm::vec3 ambient;
    float shininess;
} _Material;

typedef struct _Param
{
    float** outputImage;
    float** directIllum;
    float** indirectIllum;
    glm::vec3** posBuf;
    glm::vec3** rayBuf;
    glm::vec3** normalBuf;
    int** marker;
    int* rayNum;
    int width;
    int height;
    _CameraData* cameraData;
    _Primitive** primitives;
    int primitiveNum;
    _Light** lights;
    int lightNum;
    _Material** mtl;
    int mtlNum;
    int DOPSampleCount;
    curandState** state;
    curandStateSobol32_t** sobolState;
    unsigned short* depth;
    unsigned short* iteration;
};
