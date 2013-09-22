#include <helper_math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include "cudaRaytracerKernel.h"

__device__ glm::vec3 getSurfaceNormal( glm::vec3* point, const _Primitive* const primitive )
{
    if( primitive->type == 0 ) //sphere
        return glm::normalize( *point - primitive->center );
    else
        return primitive->pn;
}

__device__ float raySphereIntersect( const _Primitive* const sphere, 
                                     const glm::vec3* const raySource, const glm::vec3* const rayDir
                                      )
{
   glm::vec3 dst = *raySource - sphere->center;
   float B = glm::dot( dst, *rayDir );
   float C = glm::dot( dst, dst ) - sphere->radius * sphere->radius;
   float D = B*B - C;
   
   //returns the smallest positiive root that is real number, otherwise returns infiinite value
   return D > 0 ? ( -B-sqrt(D) > 0 ? -B-sqrt(D) : ( -B+sqrt(D) > 0 ? -B+sqrt(D) : FLOAT_INF) ) : FLOAT_INF;
    
}

__device__ float rayTriangleIntersect( const _Primitive* const triangle, const glm::vec3* const raySource, const glm::vec3* const rayDir )
{
    glm::vec3 BAcrossQA;
    glm::vec3 CBcrossQB;
    glm::vec3 ACcrossQC;
    glm::vec3 point;

    float plane_delta;
    float ray_offset;

    plane_delta = glm::dot( triangle->pn, triangle->vert[0] );

    if( glm::dot( triangle->pn, *rayDir ) == 0 ) //the ray and the plane are parallel
        return FLOAT_INF;

    ray_offset = ( plane_delta - glm::dot( triangle->pn, *raySource ) ) /
                    glm::dot( triangle->pn, *rayDir ) ;

    point = *raySource + ( ray_offset * (*rayDir) );

    BAcrossQA = glm::cross( triangle->vert[1] - triangle->vert[0], point - triangle->vert[0] );
    CBcrossQB = glm::cross( triangle->vert[2] - triangle->vert[1], point - triangle->vert[1] );
    ACcrossQC = glm::cross( triangle->vert[0] - triangle->vert[2], point - triangle->vert[2] );

 
    if( ray_offset > 0 && glm::dot( BAcrossQA, triangle->pn ) >= 0 &&
          glm::dot( CBcrossQB, triangle->pn ) >= 0 &&
        glm::dot( ACcrossQC, triangle->pn ) >= 0 )   
    {
      
        return ray_offset;
    }
    else
        return FLOAT_INF;
}


__device__ glm::vec3 shade( glm::vec3* point, glm::vec3* normal, glm::vec3* eyeRay, 
                           const _Material* const mtl, const glm::vec3* const lightColor,
                           const glm::vec3* const L )
{
    //glm::vec3 L;
    glm::vec3 H;
    //float lightDst;
    //float attenu;   //attenuation factor, unused right now

    glm::vec3 color(0.0f,0.0f,0.0f);

    //if( lightPos->x > .0f ) //local light
    //{
    //    L = glm::normalize( glm::vec3(*lightPos) - (*point) );
    //    //lightDst = glm::distance( (*point), glm::vec3(light->pos));
    //    //lightDst = glm::length( L );
    //    //attenu = light->attenu_const + 
    //    //            ( light->attenu_linear + light->attenu_quadratic * lightDst ) * lightDst;
    //}
    //else
    //{
    //    L = glm::normalize( glm::vec3(*lightPos) );
    //    //attenu = 1.0f;
    //}

    if( glm::dot( *L, *normal ) < 0 ) //the face is turned away from this light
        return color;

    H = glm::normalize( *L - *eyeRay );
    color = (*lightColor)   *
                ( mtl->diffuse * fmaxf( glm::dot( *normal, *L ), 0.0f ) +
                mtl->specular * powf( fmaxf( glm::dot( *normal, H ), 0.0f ), mtl->shininess ) );


    return color;
}

__device__ int raytrace( const glm::vec3* const ray, const glm::vec3* const source,
                         const _Primitive* const primitives, int primitiveNum,
                         const _Light* const lights, int lightNum, glm::vec3* point, glm::vec3* surfaceNormal )
{
    float nearest = FLOAT_INF;
    float dst;
    int   id = -1;
    int threadId = blockDim.y * threadIdx.y + threadIdx.x;

    glm::vec3 tmpP, tmpN;
    //__shared__ _Primitive s_primitive;

    for( int i = 0; i < primitiveNum; ++i )
    {
        //if( threadId == 0 )
        //{
        //    s_primitive = primitives[i];
        //}
        //__syncthreads();

        if( primitives[i].type == 0 ) //sphere
        {
            dst = raySphereIntersect( primitives+i, source ,ray );
        }
        else
        {
            dst = rayTriangleIntersect( primitives+i,  source, ray);

        }
        if( FLOAT_INF == dst )
           continue;

        tmpP = *source + ( dst * (*ray) );  
        
        tmpN = getSurfaceNormal( &tmpP, primitives+i );


        if( glm::dot( tmpN, *ray ) > 0 ) //surface turnes away from the camera
            continue;

        if( dst < nearest )
        {
            nearest = dst;
            id = i;
            *point = tmpP;
            *surfaceNormal = tmpN;
        }
        
    }

     return id;
}


__device__ glm::vec3 shadowTest( glm::vec3* point, glm::vec3* normal, glm::vec3 *eyeRay,const _Primitive* const occluders, int occluderNum, const _Material* const mtl,
                             const _Light* const light, curandState *state )
{
    glm::vec3 color( .0f, .0f, .0f );
    glm::vec3 L;
    glm::vec3 O;
    float lightDst, occluderDst;
    float shadowPct = 0;
    float delta = 1;
    float deltaX = 1;
    ushort2 LSample;
    int threadId = threadIdx.x + blockDim.x * threadIdx.y;

    if( light->type == 0 ) //point local light
    {    
        L = glm::vec3(light->pos) - *point ;
        lightDst = glm::length( L );
        LSample.x = LSample.y = 1;
    }
    else if( light->type == 1 ) //point directional light
    {    
        lightDst = FLOAT_INF;
        L = glm::vec3(light->pos);
        LSample.x = LSample.y = 1;
    }
    else if( light->type == 2 ) //area light
    {
        LSample.x = LSample.y = 4;
        deltaX = light->width * 1.0f / LSample.x;
        delta = 1.0f/ LSample.x / LSample.y; 

        //L = glm::vec3(light->pos) - *point ;
        L = ( glm::vec3(light->pos) -glm::vec3(light->width/2.0,0, light->width /2.0 )
                                     + glm::vec3( deltaX * curand_uniform(state+threadId),
                                                 0.0f, 
                                                 deltaX * curand_uniform(state+threadId) ) ) -
               *point;
        lightDst = glm::length( L );
        
    }

    if( glm::dot( *normal, L ) < 0 ) 
        return color;
   
 
    //delta = 1.0f/(LSample.x*LSample.y );

    for( int y = 1; y <= LSample.y; ++y ) 
        for( int x = 1; x <= LSample.x; ++x )
    {
        L = glm::normalize(L);
        shadowPct = 0;
        for( int i = 0; i < occluderNum; ++i )
        {


            if( occluders[i].type == 0 ) //sphere
            {
                occluderDst = raySphereIntersect( occluders+i, point ,&L );
            }
            else
            {
                occluderDst = rayTriangleIntersect( occluders+i,  point, &L );

            }
            if( FLOAT_INF == occluderDst )
               continue;


            if( occluderDst < lightDst )
            {
                shadowPct = 1.0f;
                break;
            }
        
        }
        color += (1-shadowPct) * delta * shade(point, normal, eyeRay, mtl, &light->color, &L );
        L = ( glm::vec3(light->pos) + -glm::vec3(light->width/2.0,0, light->width /2.0 ) +
                                                 glm::vec3( deltaX * ( 1.0f+x*curand_uniform(state+threadId) ),
                                                 0.0f, 
                                                 deltaX * ( 1.0f+y*curand_uniform(state+threadId) ) ) ) -
               *point;
        lightDst = glm::length( L );

    }
    return color;
}

__global__ void raycastKernel( unsigned char* const outputImage, int width, int height, _CameraData cameraData,
                         const _Primitive* const primitives, int primitiveNum,
                         const _Light* const lights, int lightNum, _Material* mtls, int mtlNum, curandState *state )
{
    
    ushort2 idx;
    float2 offset;
    glm::vec3 ray;
    glm::vec3 raysource;
    glm::vec3 incidentP;
    glm::vec3 shiftP;
    glm::vec3 surfaceNormal;
    glm::vec3 color(0.0f,0.0f,0.0f);
    glm::vec3 finalColor(0.0f,0.0f,0.0f);
    glm::vec3 cumulativeSpecular( 1.0f, 1.0f, 1.0f );
    int hitId;
    float shadowPct;
    _Material mtl;

    int outIdx;

    //generate ray based on block and thread idx
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if( idx.x > width || idx.y > height )
        return;

    outIdx = idx.y * width * 4 + 4 * idx.x; //element to shade in the output buffer

    offset.x = cameraData.viewportHalfDim.x * ( (idx.x+0.5) / (width/2.0) - 1 );
    offset.y = cameraData.viewportHalfDim.y * ( 1- (idx.y+0.5) / (height/2.0)  );

    ray = cameraData.wVec + offset.x * cameraData.vVec + offset.y * cameraData.uVec;
    ray = glm::normalize( ray );
    raysource = cameraData.eyePos;

    for( int depth = 0; depth <5; ++depth )
    {
        color.x = color.y = color.z = 0.0f; //clear color vector for use in current iteration
        hitId = raytrace( &ray,&raysource, primitives, primitiveNum, lights, lightNum, &incidentP, &surfaceNormal );
        if( hitId >= 0 )
        {
            mtl = mtls[primitives[hitId].mtl_id];
            shiftP = incidentP +  (0.001f * surfaceNormal);
            if( hitId > 1 )
            {
                for( int i = 0; i < lightNum; ++i )
                {
                    shadowPct = 0;
                    color  +=  shadowTest( &shiftP, &surfaceNormal, &ray, primitives+2, primitiveNum-2, &mtl, lights+i, state );

                    //sahding
                    //if( shadowPct ==0 )
                      //color += (1.0f-shadowPct)*shade( &incidentP, &surfaceNormal, &ray, &mtl, lights+i );
                }
            }
            color += mtl.ambient + mtl.emission;
    
        
        
            finalColor += color * cumulativeSpecular;
        
            if( glm::all(glm::equal(mtl.specular, glm::vec3(0.0f,0.0f,0.0f) ) ) )
                break;

            ray = glm::normalize( glm::reflect( ray, surfaceNormal ) );
            raysource = shiftP;
            cumulativeSpecular *= mtl.specular;
        }
        else
            break;
    }
    //write color to output buffer
    outputImage[ outIdx ] = finalColor.x > 1 ? 255 : finalColor.x * 255;
    outputImage[ outIdx + 1] = finalColor.y > 1 ? 255 : finalColor.y * 255;
    outputImage[ outIdx + 2] = finalColor.z > 1 ? 255 : finalColor.z * 255 ;

}


__global__ void rand_setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.y * blockDim.x;

    curand_init(1234, id, 0, &state[id]);
}


void rayTracerKernelWrapper( unsigned char* const outputImage, int width, int height, _CameraData cameraData,
                              const _Primitive* const primitives, int primitiveNum,
                              const _Light* const lights, int lightNum, _Material* mtl, int mtlNum, int DOPsampleCount, curandState *state )
{
    dim3 blockSize = dim3(8,8);
    dim3 gridSize = dim3( (width + blockSize.x-1)/blockSize.x, (height + blockSize.y-1)/blockSize.y );

    //The ray tracing work is done in the kernel
    raycastKernel<<< gridSize, blockSize >>>( outputImage, width, height, cameraData, primitives, primitiveNum,
                                       lights, lightNum, mtl, mtlNum, state );
    cudaErrorCheck( cudaGetLastError() );
    cudaDeviceSynchronize(); 
    cudaErrorCheck( cudaGetLastError() );
}

__global__ void setupRandSeed(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(1234, id, 0, &state[id]);
}

void setupRandSeedWrapper( int dimX, int dimY, curandState* states ) 
{
    setupRandSeed<<<dimX, dimY>>>(states);
}
