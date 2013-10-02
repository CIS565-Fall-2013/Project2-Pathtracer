#pragma once

#include "raytracer.h"
#include "sceneDesc.h"
#include <FreeImage.h>
#include "util.h"
#include <curand_kernel.h>

class CudaRayTracer
{
public:
    CudaRayTracer();
    ~CudaRayTracer();
    void renderImage( cudaGraphicsResource* pboResource);
    //void renderImage( FIBITMAP* outputImage );
    void init( const SceneDesc &scene );
    void registerPBO( unsigned int pbo );
    void unregisterPBO();
    void setupDevStates(); //for random number generation
    void updateCamera( const SceneDesc &scene );
    void compactSurvivingPath();
    void resetPathDepth();
    void resetIteration();
public:
    unsigned short iteration;
    unsigned short depth;
    int numValidPath;
private:
    void cleanUp();
    void packSceneDescData( const SceneDesc &sceneDesc );

    int width;
    int height;

    //Host-side and packed data for transferring to the device
    _CameraData cameraData;
    _Primitive* h_pPrimitives;
    _Light* h_pLights;
    _Material* h_pMaterials;
    int* h_marker;

    int numPrimitive;
    int numTriangle;
    int numSphere;
    int numLight;
    int numMaterial;

    //unsigned char* d_outputImage;
    float* d_outputImage;
    glm::vec3* d_posBuffer;
    glm::vec3* d_rayBuffer;
    //float* d_specuBuffer;
    glm::vec3* d_normalBuffer;
    float* d_directIllum;
    float* d_indirectIllum;

    //for stream compaction operation
    int* d_marker;
    int* d_marker_temp;

    unsigned char* h_outputImage;
    _Primitive* d_primitives;
    _Light* d_lights;
    _Material* d_materials;

    _Param param;  

    //Cuda-OpenGL interop objects
    size_t pboSize;

    //rand state
    curandState *d_devStates;
    curandStateSobol32_t *d_sobolStates;
    unsigned int* d_vectors;
    curandDiscreteDistribution_t poisson_dist;

};