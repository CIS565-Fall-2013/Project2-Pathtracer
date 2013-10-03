#include <helper_math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include "cudaRaytracer.h"
#include "cudaRaytracerKernel.h"

__device__ glm::vec3 diffuse_direction( const glm::vec3* normal, curandState* state,curandStateSobol32_t* soboState )
{
    float theta = (float)acosf( sqrtf(1.0 - curand_uniform(state) ) );
    float phi = 2 * 3.1415926535897932384626433832795 * curand_uniform(state);

    //float theta = (float)acosf( sqrtf( 1.0 - ( (float)curand_discrete( state, poisson_dist )/UINT_MAX ) ) );
    //float phi = 2 * 3.1415926 * ( (float)curand_discrete( state, poisson_dist )/UINT_MAX);

    glm::vec3 U;
    glm::vec3 V;

    //construct the coordinate
    U = glm::cross( *normal, glm::vec3( 1.0f, 0.0f, 0.0f ) );

    if( (U.x*U.x + U.y*U.y + U.z*U.z ) < 0.01 )
    {
        U =  glm::normalize( glm::cross( *normal, glm::vec3( 0.0f, 1.0f, 0.0f ) ) );
    }
    else
        U = glm::normalize( U );
    V = glm::normalize( glm::cross( *normal, U ) );

    //convert theta & phi to direction vector
    return glm::normalize( U*(cosf(phi)*sin(theta) ) + V * ( sin(phi)*sin(theta) ) +
           (*normal)*cos(theta) );
}

__device__ glm::vec3 specular_direction( const glm:: vec3* incoming, const glm::vec3* normal, curandState* state,
                                         curandStateSobol32_t* sobolState, float* weight )
{
    //float theta = (float)acosf( sqrtf(1.0 - curand_uniform(state) ) );
    //float phi = 2 * 3.1415926 * curand_uniform(state);
    //glm::vec3 U;
    //glm::vec3 V;

    ////construct the coordinate
    //U = glm::cross( *normal, glm::vec3( 1.0f, 0.0f, 0.0f ) );
    //if( (U.x*U.x + U.y*U.y + U.z*U.z ) < 0.01f )
    //{
    //    U = glm::cross( *normal, glm::vec3( 0.0f, 1.0f, 0.0f ) );
    //}
    //V = glm::cross( *normal, U ); 

    //convert theta & phi to direction vector
   // return U*(cosf(phi)*sin(theta) ) + V * ( sin(phi)*sin(theta) ) +
    //       (*normal)*cos(theta);
    if( curand_uniform(state) < 0.2f )
    {
        //*weight = 1.0f/0.2f;
        return diffuse_direction( normal, state, sobolState );
    }
    else
    {
        *weight = 1.0f/0.8f;
        return glm::normalize( glm::reflect( *incoming, *normal) );
    }
}

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

__device__ float rayBoxIntersect( const _Primitive* const box, 
                                     const glm::vec3* const raySource, const glm::vec3* const rayDir, const glm::vec3* const invRay
                                      )
{
    float3 tmax, tmin;
    bool rayDirSign;
    if( rayDir->x >= 0 )
    {
        tmin.x = ( box->vert[0].x - raySource->x ) * invRay->x;
        tmax.x = ( box->vert[1].x - raySource->x ) * invRay->x;
    }
    else
    {
        tmax.x = ( box->vert[0].x - raySource->x ) * invRay->x;
        tmin.x = ( box->vert[1].x - raySource->x ) * invRay->x;
    }
    if( rayDir->y >= 0 )
    {
        tmin.y = ( box->vert[0].y - raySource->y ) * invRay->y;
        tmax.y = ( box->vert[1].y - raySource->y ) * invRay->y;
    }
    else
    {
        tmax.y = ( box->vert[0].y - raySource->y ) * invRay->y;
        tmin.y = ( box->vert[1].y - raySource->y ) * invRay->y;
    }
    if( tmin.x > tmax.y || tmin.y > tmax.x )
        return FLOAT_INF;
    if( tmin.y > tmin.x ) tmin.x = tmin.y;
    if( tmax.x > tmax.y ) tmax.x = tmax.y;

    if( rayDir->z >= 0 )
    {
        tmin.z = ( box->vert[0].z - raySource->z ) * invRay->z;
        tmax.z = ( box->vert[1].z - raySource->z ) * invRay->z;
    }
    else
    {
        tmax.z = ( box->vert[0].z - raySource->z ) * invRay->z;
        tmin.z = ( box->vert[1].z - raySource->z ) * invRay->z;
    }

    if( tmin.x > tmax.z || tmin.z > tmax.x )
        return FLOAT_INF;
    if( tmin.z > tmin.x ) tmin.x = tmin.z;
    if( tmax.z < tmax.x ) tmax.x = tmax.z;

    if( tmin.x <= 0 && tmax.x <= 0 ) 
        return FLOAT_INF;
    else 
        return tmin.x;
    
}

__device__ glm::vec3 shade( glm::vec3* point, glm::vec3* normal, glm::vec3* eyeRay, 
                           const _Material* const mtl, const _Light* const light,//const glm::vec3* const lightColor,
                           const glm::vec3* const L, float lightDst )
{
    //glm::vec3 L;
    glm::vec3 H;
   
    float attenu;   //attenuation factor

    glm::vec3 color(0.0f,0.0f,0.0f);

    //if( lightPos->x > .0f ) //local light
    //{
    //    L = glm::normalize( glm::vec3(*lightPos) - (*point) );
    //lightDst = glm::distance( (*point), glm::vec3(light->pos));
    //    //lightDst = glm::length( L );
    attenu = light->attenu_const + 
               ( light->attenu_linear + light->attenu_quadratic * lightDst ) * lightDst;
    //}
    //else
    //{
    //    L = glm::normalize( glm::vec3(*lightPos) );
    //    //attenu = 1.0f;
    //}

    if( glm::dot( *L, *normal ) < 0 ) //the face is turned away from this light
        return color;

    H = glm::normalize( *L - *eyeRay );
    color = (light->color)   / attenu *
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

    //int threadId = blockDim.y * threadIdx.y + threadIdx.x;

    glm::vec3 tmpP, tmpN;
    glm::vec3 invRay = glm::vec3( 1.0/ray->x, 1.0/ray->y, 1.0/ray->z );
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
            if( FLOAT_INF == dst )
                continue;
        }
        else if( primitives[i].type == 1 )
        {
            dst = rayTriangleIntersect( primitives+i,  source, ray);
            if( FLOAT_INF == dst )
                continue;
        }
        else 
        {
            dst = rayBoxIntersect( primitives+i, source, ray, &invRay );
            if( FLOAT_INF == dst )
                i += primitives[i].mtl_id-1; //skip primitives enclosed by this bounding box
            continue;
        }


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
                             const _Light* const light, curandState *state, int* shadowRayNum )
{
    glm::vec3 color( .0f, .0f, .0f );
    glm::vec3 L, invL;
    glm::vec3 O;
    float lightDst, occluderDst;
    int shadowPct = 0;
    //float delta = 1;
    //float deltaX = 1;
    ushort2 LSample;
    int threadId = threadIdx.x + blockDim.x * threadIdx.y;
    

    if( light->type == 0 ) //point local light
    {    
        L = glm::vec3(light->pos) - *point ;
        lightDst = glm::length( L );
        //LSample.x = LSample.y = 1;
        *shadowRayNum += 1;
    }
    else if( light->type == 1 ) //point directional light
    {    
        lightDst = FLOAT_INF;
        L = glm::vec3(light->pos);
        //LSample.x = LSample.y = 1;
        *shadowRayNum += 1;
    }
    else if( light->type == 2 ) //area light
    {
        //LSample.x = LSample.y = 4;
        *shadowRayNum += 1;
        //deltaX = light->width * 1.0f / LSample.x;
        //delta = 1.0f/ LSample.x / LSample.y; 

        //L = glm::vec3(light->pos) - *point ;
        //L = ( glm::vec3(light->pos) -glm::vec3(light->width/2.0,0, light->width /2.0 )
        //                             + glm::vec3( deltaX * curand_uniform(&state[threadId]),
        //                                         0.0f, 
        //                                          deltaX * curand_uniform(&state[threadId]) ) ) -
        //       *point;
        L = glm::vec3(light->pos ) + glm::vec3( (curand_uniform( state )-0.5 )*light->width, 
                                                 0.0f, 
                                                 (curand_uniform( state )-0.5 )*light->width ) - *point ;
        lightDst = glm::length( L );
        
    }
   
    if( glm::dot( *normal, L ) < 0 ) 
        return color;
   
 
    //delta = 1.0f/(LSample.x*LSample.y );

    //for( int y = 1; y <= LSample.y; ++y ) 
    //    for( int x = 1; x <= LSample.x; ++x )
    {
        L = glm::normalize(L);
        invL = glm::vec3( 1.0/L.x, 1.0/L.y, 1.0/L.z );
        shadowPct = 0;
        for( int i = 0; i < occluderNum; ++i )
        {


            if( occluders[i].type == 0 ) //sphere
            {
                occluderDst = raySphereIntersect( occluders+i, point ,&L );
                if( FLOAT_INF == occluderDst )
                   continue;
            }
            else if( occluders[i].type == 1 ) //triangle
            {
                occluderDst = rayTriangleIntersect( occluders+i,  point, &L );
                if( FLOAT_INF == occluderDst )
                    continue;

            }
            else if ( occluders[i].type == 2 )
            {
                occluderDst = rayBoxIntersect( occluders+i, point, &L, &invL );
                if( FLOAT_INF == occluderDst )
                    i += occluders[i].mtl_id-1;
                continue;
            }


            if( occluderDst < lightDst )
            {
                shadowPct = 1;
                break;
            }
        
        }
        if( shadowPct == 0 )
            color += shade(point, normal, eyeRay, mtl, light, &L, lightDst );
        //L = ( glm::vec3(light->pos) + -glm::vec3(light->width/2.0,0, light->width /2.0 ) +
        //                                         glm::vec3( deltaX * ( 1.0f+x*curand_uniform(&state[threadId]) ),
        //                                         0.0f, 
        //                                         deltaX * ( 1.0f+y*curand_uniform(&state[threadId]) ) ) ) -
        //       *point;
        //lightDst = glm::length( L );

    }
    return color;
}

__global__ void raycastKernel( //unsigned char* const outputImage,
                               float* const outputImage,
                               glm::vec3* const raySources,
                               glm::vec3* const rayDirs,
                               
                               int width, int height, _CameraData cameraData,
                               const _Primitive* const primitives, int primitiveNum,
                               const _Light* const lights, int lightNum, _Material* mtls, int mtlNum,
                               curandState *state, unsigned short iteration )
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
    int shadowSampleCount;
    _Material mtl;

    int outIdx;

    //generate ray based on block and thread idx
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if( idx.x > width || idx.y > height )
        return;
    outIdx = idx.y * width * 4 + 4 * idx.x; //element to shade in the output buffer

    //offset.x = cameraData.viewportHalfDim.x * ( (idx.x+0.5) / (width/2.0) - 1 );
    //offset.y = cameraData.viewportHalfDim.y * ( 1- (idx.y+0.5) / (height/2.0)  );
    offset.x = cameraData.offset1.x * idx.x + cameraData.offset2.x;
    offset.y = cameraData.offset1.y * idx.y + cameraData.offset2.y;

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
            shadowSampleCount = 0;
            if( hitId > 1 )
            {
                for( int i = 0; i < lightNum; ++i )
                {
                    //shadowPct = 0;
                    color  +=  shadowTest( &shiftP, &surfaceNormal, &ray, primitives+2, primitiveNum-2, &mtl, lights+i, state, &shadowSampleCount );

                    //sahding
                    //if( shadowPct ==0 )
                      //color += (1.0f-shadowPct)*shade( &incidentP, &surfaceNormal, &ray, &mtl, lights+i );
                }
                color /= (float)shadowSampleCount;
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
    outputImage[ outIdx ] = finalColor.x ;
    outputImage[ outIdx + 1] = finalColor.y ;
    outputImage[ outIdx + 2] =  finalColor.z  ;

}

__global__ void pathTracerKernel( ///unsigned char* const outputImage,
                               float* const outputImage,
                               float* const directIllum,
                               float* const indirectIllum,
                               glm::vec3* posBuf,
                               glm::vec3* rayBuf,
                               glm::vec3* normalBuf,
                               int* marker,
                               int raycount,
                               _CameraData cameraData,
                               const _Primitive* const primitives, int primitiveNum,
                               const _Light* const lights, int lightNum, _Material* mtls, int mtlNum,
                               curandState *state,curandStateSobol32_t* sobolState,
                               unsigned short depth, unsigned short iteration )
{
  
    float2 offset;
    glm::vec3 ray;
    glm::vec3 raysource;
    glm::vec3 incidentP;
    glm::vec3 shiftP;
    glm::vec3 surfaceNormal;
    glm::vec3 color(0.0f,0.0f,0.0f);
    glm::vec3 reflectRay;
    //glm::vec3 finalColor(0.0f,0.0f,0.0f);
    //glm::vec3 cumulativeSpecular( 1.0f, 1.0f, 1.0f );
    float weight;
    int hitId;
    int shadowSampleCount;
    _Material mtl;
    curandState local_state;
    int outIdx;
    int outIdx2;
    int outIdx3;
    

    //generate ray based on block and thread idx
    int idx= blockIdx.x * blockDim.x + threadIdx.x;

    if( idx >= raycount )
        return;

    //outIdx = ( idx.y * width + idx.x ) * 4; //element to shade in the output buffer
    //outIdx2 = ( idx.y * width + idx.x ) * 3;
    //outIdx3 = ( idx.y * width + idx.x ) * 3 * MAXDEPTH;
    outIdx = marker[blockIdx.x * blockDim.x + threadIdx.x];
    outIdx3 = outIdx * 3 * MAXDEPTH;
    outIdx2 =  outIdx * 4;
    local_state = state[outIdx];

    ray = rayBuf[ outIdx ];
    raysource = posBuf[ outIdx ];

    //    unsigned int test = curand_discrete( &local_state, poisson_dist );
    //for( int i = 0; i < 100; ++i )
    //    test = curand_discrete( &local_state, poisson_dist );
    //color.x += test;
    //else
    //{
    //    //ray = glm::vec3( &rayDirs[outIdx2] );
    //    ray = glm::make_vec3(  &rayDirs[outIdx2] );
    //    if( ray.x == 0 && ray.y == 0 && ray.z == 0 )
    //        return;
    //    raysource = glm::make_vec3( &raySources[outIdx2] );
    //    cumulativeSpecular = glm::make_vec3( &accuSpecu[outIdx2] );
    //}
      //use roulette to determine if the trace should stop

    //for( depth = 0; depth < MAXDEPTH; ++depth )
    //{
    if( 0.5 < curand_uniform( &local_state ) && depth < MAXDEPTH ) //Russian Roulette
    { 
      
        hitId = raytrace( &ray,&raysource, primitives, primitiveNum, lights, lightNum, &incidentP, &surfaceNormal );
        if( hitId >= 0 )
        {
            mtl = mtls[primitives[hitId].mtl_id];
            shiftP = incidentP +  (0.001f * surfaceNormal);
            
            if( mtl.emission.x <0.0001 && mtl.emission.y <0.0001 && mtl.emission.z <0.0001 ) //skip light gemoetry
            {
                shadowSampleCount = 0; //this will be accumulated in the shadowTest
                for( int i = 0; i < lightNum; ++i )
                {
                   
                    color  +=  shadowTest( &shiftP, &surfaceNormal, &ray, primitives+2, primitiveNum-2, &mtl, lights+i, &local_state, &shadowSampleCount );

                }
                color /= shadowSampleCount;
            }
            else
                color = ( mtl.ambient + mtl.emission ) * lights[0].attenu_const + 
               ( lights[0].attenu_linear + lights[0].attenu_quadratic * glm::distance( raysource, incidentP ) ) * 
               glm::distance( raysource, incidentP );
            //else
             //   color = ( light->attenu_const + 
             //             ( light->attenu_linear + light->attenu_quadratic * lightDst ) * lightDst;

            //Assume no self-emit objects currently
            //if( depth == 0 )
            //    color += mtl.ambient + mtl.emission;

            directIllum[outIdx3+depth*3  ] = color.x *2; // / 0.5
            directIllum[outIdx3+depth*3+1] = color.y *2;
            directIllum[outIdx3+depth*3+2] = color.z *2;

            //compute the solid angle-weighted term
            //weight = glm::dot( surfaceNormal, -ray ) / glm::distance( shiftP, raysource )*glm::distance( shiftP, raysource );
            //weight = glm::distance( raysource ,shiftP );
            //weight = weight * weight; 
            //weight = glm::dot( -ray, surfaceNormal ) / weight;
            //weight = 1;

            indirectIllum[outIdx3+(depth-1)*3  ] *= 2;
            indirectIllum[outIdx3+(depth-1)*3+1] *= 2;
            indirectIllum[outIdx3+(depth-1)*3+2] *=  2;


            weight = 1;
            //generate a ray for a further bounce
            if( !glm::all(glm::equal(mtl.diffuse, glm::vec3(0.0f,0.0f,0.0f) ) ) )
            {
                reflectRay = diffuse_direction( &surfaceNormal, &local_state, sobolState );
            }
            else
            {
                reflectRay = specular_direction( &ray, &surfaceNormal, &local_state, sobolState, &weight );
            }
            glm::vec3  H = glm::normalize( reflectRay - ray );
            color = ( mtl.diffuse * fmaxf( glm::dot( surfaceNormal, reflectRay ), 0.0f ) +
                     mtl.specular * powf( fmaxf( glm::dot( surfaceNormal, H ), 0.0f ), mtl.shininess ) ) ;

            //weight *= glm::dot( reflectRay, surfaceNormal );
            indirectIllum[outIdx3+depth*3  ] = color.x * weight;
            indirectIllum[outIdx3+depth*3+1] = color.y * weight;
            indirectIllum[outIdx3+depth*3+2] = color.z * weight;


            posBuf[outIdx] = shiftP;
            rayBuf[outIdx] = reflectRay;
            normalBuf[outIdx] = surfaceNormal;
            state[outIdx] = local_state;
            return;
        }

    }
    //else //No hit, mark this ray as invalid
    //{    
            marker[idx] = -1;

            //and accumuluate the color computed so far
            color.x = color.y = color.z = 0;
            for( int i = depth-1; i >= 0; --i )
            {
                color = glm::make_vec3( &directIllum[outIdx3+i*3] ) + color * glm::make_vec3(&indirectIllum[outIdx3+i*3]);
                         
            }
            
            outputImage[ outIdx2    ] = (outputImage[ outIdx2 ] * (iteration-1)+ color.x)/iteration ;
            outputImage[ outIdx2 + 1] = (outputImage[ outIdx2+1 ] * (iteration-1)+ color.y)/iteration ;
            outputImage[ outIdx2 + 2] = (outputImage[ outIdx2+2 ] * (iteration-1)+ color.z)/iteration ;

            //restore the curand state to global memory
            state[outIdx] = local_state;
            return;
    //}


    //calculate the final shading color
    //color.x = color.y = color.z = 0;
    //for( int i = depth-1; i >= 0; --i )
    //{
    //    color = glm::make_vec3( &directIllum[outIdx3+i*3] ) + color * glm::make_vec3(&indirectIllum[outIdx3+i*3]);
    //    
    //}
    //outputImage[ outIdx    ] = (outputImage[ outIdx ] * (iteration-1)+ color.x)/iteration ;
    //outputImage[ outIdx + 1] = (outputImage[ outIdx+1 ] * (iteration-1)+ color.y)/iteration ;
    //outputImage[ outIdx + 2] = (outputImage[ outIdx+2 ] * (iteration-1)+ color.z)/iteration ;
    //state[idx] = local_state;
}

void pathTracerKernelWrapper( //unsigned char* const outputImage, 
                             _Param* param )
{
    dim3 blockSize = dim3(8*8,1,1);
    dim3 gridSize = dim3( ((*param->rayNum)+ blockSize.x-1)/blockSize.x,1,1 );
    if( gridSize.x == 0 ) 
    {
        return;
    }
    //The ray tracing work is done in the kernel
    //raycastKernel<<< gridSize, blockSize >>>( outputImage, raySources, rayDirs, width, height, cameraData, primitives, primitiveNum,
    //                                   lights, lightNum, mtl, mtlNum, state, iteration );
    pathTracerKernel<<< gridSize, blockSize >>>( *param->outputImage, *param->directIllum, *param->indirectIllum, 
                                                 *param->posBuf, *param->rayBuf,*param->normalBuf, *param->marker, 
                                                 (*param->rayNum), (*param->cameraData), 
                                                 *param->primitives,
                                                 param->primitiveNum,
                                                 *param->lights, 
                                                 param->lightNum, 
                                                 *param->mtl, param->mtlNum, 
                                                 *param->state, *param->sobolState, 
                                                 *param->depth, *param->iteration );
    cudaErrorCheck( cudaGetLastError() );
    cudaDeviceSynchronize(); 
    cudaErrorCheck( cudaGetLastError() );
}

__global__ void pathTracerEyeRayKernel( ///unsigned char* const outputImage,
                               float* const outputImage,
                               float* const directIllum,
                               float* const indirectIllum, 
                               glm::vec3* posBuf,
                               glm::vec3* rayBuf,
                               glm::vec3* normalBuf,
                               int* marker,
                               int width, int height,
                               _CameraData cameraData,
                               const _Primitive* const primitives, int primitiveNum,
                               const _Light* const lights, int lightNum, _Material* mtls, int mtlNum,
                               curandState *state,curandStateSobol32_t* sobolState  )
{
    ushort2 idx;
    float2 offset;
    glm::vec3 ray;
    glm::vec3 raysource;
    glm::vec3 incidentP;
    glm::vec3 shiftP;
    glm::vec3 surfaceNormal;
    glm::vec3 color(0.0f,0.0f,0.0f);
    glm::vec3 reflectRay;
    //glm::vec3 finalColor(0.0f,0.0f,0.0f);
    //glm::vec3 cumulativeSpecular( 1.0f, 1.0f, 1.0f );
    float weight;
    int hitId;
    int shadowSampleCount;
    _Material mtl;
    curandState local_state;
    int outIdx;
    int outIdx2;
    int outIdx3;
    

    //generate ray based on block and thread idx
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if( idx.x > width || idx.y > height )
        return;
    outIdx = ( idx.y * width + idx.x ) * 4; //element to shade in the output buffer
    outIdx2 = ( idx.y * width + idx.x );
    outIdx3 = ( idx.y * width + idx.x ) * 3 * MAXDEPTH;
    local_state = state[outIdx2];

    //offset.x = cameraData.viewportHalfDim.x * ( (idx.x+0.5) / (width/2.0) - 1 );
    //offset.y = cameraData.viewportHalfDim.y * ( 1- (idx.y+0.5) / (height/2.0)  );
    offset.x = cameraData.offset1.x * (idx.x+cameraData.jitteredOffset1.x) + cameraData.offset2.x;
    offset.y = cameraData.offset1.y * (idx.y+cameraData.jitteredOffset1.y) + cameraData.offset2.y;

    ray = cameraData.wVec + offset.x * cameraData.vVec + offset.y * cameraData.uVec;
    ray = glm::normalize( ray );
    raysource = cameraData.eyePos;

        //if( 0.5 < curand_uniform( &local_state ) ) //Russian Roulette
        //{  
        //    //mark this ray invalid
        //    marker[outIdx2] = -1;
        //    return;
        //}

        hitId = raytrace( &ray,&raysource, primitives, primitiveNum, lights, lightNum, &incidentP, &surfaceNormal );
        if( hitId >= 0 )
        {
            mtl = mtls[primitives[hitId].mtl_id];
            shiftP = incidentP +  (0.001f * surfaceNormal);
            
            if( hitId > 1 ) //skip light gemoetry
            {
                shadowSampleCount = 0; //this will be accumulated in the shadowTest
                for( int i = 0; i < lightNum; ++i )
                {
                   
                    color  +=  shadowTest( &shiftP, &surfaceNormal, &ray, primitives+2, primitiveNum-2, &mtl, lights+i, &local_state, &shadowSampleCount );

                }
                color /= shadowSampleCount;
            }
            else
                color = ( mtl.ambient + mtl.emission ) * lights[0].attenu_const + 
               ( lights[0].attenu_linear + lights[0].attenu_quadratic * glm::distance( raysource, incidentP ) ) * 
               glm::distance( raysource, incidentP );

            directIllum[outIdx3 ] = color.x; // / 0.5
            directIllum[outIdx3+1] = color.y;
            directIllum[outIdx3+2] = color.z;

            //compute the solid angle-weighted term
            //weight = glm::dot( surfaceNormal, -ray ) / glm::distance( shiftP, raysource )*glm::distance( shiftP, raysource );
            //weight = glm::distance( raysource ,shiftP );
            //weight = weight * weight; 
            //weight = glm::dot( -ray, surfaceNormal ) / weight;
            //weight = 1;

            weight = 1;
            //generate a ray for a further bounce
            if(!glm::all(glm::equal(mtl.diffuse, glm::vec3(0.0f,0.0f,0.0f) ) )  )
            {
                reflectRay = diffuse_direction( &surfaceNormal, &local_state, sobolState );
            }
            else
            {
                reflectRay = specular_direction( &ray, &surfaceNormal, &local_state, sobolState, &weight );
            }
            glm::vec3  H = glm::normalize( reflectRay - ray );
            color = ( mtl.diffuse * fmaxf( glm::dot( surfaceNormal, reflectRay ), 0.0f ) +
                     mtl.specular * powf( fmaxf( glm::dot( surfaceNormal, H ), 0.0f ), mtl.shininess ) ) ;

            //weight *= glm::dot( reflectRay, surfaceNormal );
            indirectIllum[outIdx3  ] = color.x * weight;
            indirectIllum[outIdx3+1] = color.y * weight;
            indirectIllum[outIdx3+2] = color.z * weight;


            posBuf[outIdx2] = shiftP;

            rayBuf[outIdx2] = reflectRay;

            normalBuf[outIdx2] = surfaceNormal;

         
        }
        else //No hit, mark this ray as invalid
        {
            marker[outIdx2] = -1;
            state[outIdx2] = local_state;
            return;
        }


    //calculate the shading color
    //outputImage[ outIdx    ] = directIllum[outIdx3 ] ;
    //outputImage[ outIdx + 1] = directIllum[outIdx3+1 ] ;
    //outputImage[ outIdx + 2] = directIllum[outIdx3+2 ] ;
    state[outIdx2] = local_state;
}

void pathTracerEyeRayKernelWrapper( //unsigned char* const outputImage, 
                             _Param* param )
{
    dim3 blockSize = dim3(8,8);
    dim3 gridSize = dim3( (param->width + blockSize.x-1)/blockSize.x, (param->height + blockSize.y-1)/blockSize.y );

    //The ray tracing work is done in the kernel
    //raycastKernel<<< gridSize, blockSize >>>( outputImage, raySources, rayDirs, width, height, cameraData, primitives, primitiveNum,
    //                                   lights, lightNum, mtl, mtlNum, state, iteration );
    pathTracerEyeRayKernel<<< gridSize, blockSize >>>( *param->outputImage, *param->directIllum, *param->indirectIllum, 
                                                 *param->posBuf, *param->rayBuf, *param->normalBuf, *param->marker,
                                                 param->width, param->height, *param->cameraData, 
                                                 *param->primitives,
                                                 param->primitiveNum,
                                                 *param->lights, 
                                                 param->lightNum, 
                                                 *param->mtl, param->mtlNum, 
                                                 *param->state, *param->sobolState );
    cudaErrorCheck( cudaGetLastError() );
    cudaDeviceSynchronize(); 
    cudaErrorCheck( cudaGetLastError() );
}

__global__ void setupRandSeed(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init( 1234, id, 0, &state[id]);
}

__global__ void setupSobolRandSeed(curandStateSobol32_t *state, unsigned int* vector)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init( vector, id, &state[id]);
}

void setupRandSeedWrapper( int width, int height, curandState* states ) 
{
    dim3 blockSize = dim3(16*16);
    dim3 gridSize = dim3( (width*height + blockSize.x-1)/blockSize.x );
    setupRandSeed<<<gridSize, blockSize>>>(states);
    cudaErrorCheck( cudaGetLastError() );
    cudaDeviceSynchronize(); 
    cudaErrorCheck( cudaGetLastError() );
}

void setupSobolRandSeedWrapper( int width, int height, curandStateSobol32_t* states, unsigned int*vector ) 
{
    dim3 blockSize = dim3(16*16);
    dim3 gridSize = dim3( (width*height + blockSize.x-1)/blockSize.x );
    setupSobolRandSeed<<<gridSize, blockSize>>>(states, vector);
    cudaErrorCheck( cudaGetLastError() );
    cudaDeviceSynchronize(); 
    cudaErrorCheck( cudaGetLastError() );
}

__global__ void initMarkerKernel( int* marker )
{
    int id = blockIdx.x * blockDim.x +threadIdx.x ;
    marker[id] = -1;
}

void initMarkerWrapper( int width, int height, int* d_marker )
{
    dim3 blockSize = dim3(8*8);
    dim3 gridSize = dim3( (width*height + blockSize.x-1)/blockSize.x );
    initMarkerKernel<<<gridSize, blockSize>>>( d_marker );
    cudaErrorCheck( cudaGetLastError() );
    cudaDeviceSynchronize(); 
    cudaErrorCheck( cudaGetLastError() );
}