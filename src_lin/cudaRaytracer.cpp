#include <GL/glew.h>
#include <iostream>
#include <cstdlib>
#include <cuda_gl_interop.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include "cudaRaytracer.h"
#include "cudaRaytracerKernel.h"
#include "stream_compact.h"

#include "timer.h"

CudaRayTracer::CudaRayTracer()
{
    h_pPrimitives = 0;
    h_pMaterials = 0;
    h_pLights = 0;

    numPrimitive = 0;
    numTriangle = 0;
    numSphere = 0;
    numLight = 0;
    numMaterial = 0;

    d_outputImage = 0;
    h_outputImage = 0;

    d_posBuffer = 0;
    d_rayBuffer = 0;
    d_normalBuffer = 0;
    d_directIllum = 0;
    d_indirectIllum = 0;
    d_marker = 0;
    d_marker_temp = 0;
    h_marker = 0;

    d_primitives = 0;
    d_lights = 0;
    d_materials = 0;

    d_devStates = 0;
    d_sobolStates = 0;
    d_vectors = 0;

    iteration = 1;
    depth = 0;

    numValidPath = 0;

    sampleGrid[0].x = -0.375; sampleGrid[0].y = 0.125;
    sampleGrid[0].x = +0.125; sampleGrid[0].y = 0.375;
    sampleGrid[0].x = -0.125; sampleGrid[0].y = -0.375;
    sampleGrid[0].x = 0.375; sampleGrid[0].y = -0.125;
    sampleGridIdx = 0;
}

CudaRayTracer::~CudaRayTracer()
{
    cleanUp();
}

void CudaRayTracer::renderImage( cudaGraphicsResource* pboResource )
{


    cudaErrorCheck( cudaGraphicsMapResources( 1, &pboResource, 0 ) );
    cudaErrorCheck( cudaGraphicsResourceGetMappedPointer((void**) &d_outputImage, &pboSize, pboResource ) );

 
    GpuTimer timer;
    timer.Start();
    //Launch the ray tracing kernel through the wrapper
    if( depth == 0 )
    {
                
        //cudaErrorCheck( cudaMemset( (void*)d_normalBuffer, 0, sizeof(glm::vec3)  * width * height ) );
        //cudaErrorCheck( cudaMemset( (void*)d_posBuffer, 0, sizeof(glm::vec3)  * width * height ) );
        //cudaErrorCheck( cudaMemset( (void*)d_rayBuffer, 0, sizeof(glm::vec3)  * width * height ) );
        //initMarkerWrapper( width, height, d_marker_temp );
        cudaErrorCheck( cudaMemcpy( d_marker, h_marker, sizeof( int )*width*height, cudaMemcpyHostToDevice ) );
        pathTracerEyeRayKernelWrapper( &param );
    }
    else
        pathTracerKernelWrapper( &param );
    timer.Stop();
    ++depth;
    std::cout<<"Render time: "<<timer.Elapsed()<<" ms. at iteration "<<iteration<<std::endl;
    //std::cout<<"valid ray: "<<numValidPath<<"\n";
    cudaErrorCheck( cudaGraphicsUnmapResources( 1, &pboResource, 0 ) );


    compactSurvivingPath();
}

//void CudaRayTracer::renderImage( FIBITMAP* outputImage )
//{
//
//    cudaErrorCheck( cudaMemset( (void*)d_outputImage, 0, sizeof(unsigned char) * 4 * width * height ) );
//
//    GpuTimer timer;
//    //timer.Start();
//    //Launch the ray tracing kernel through the wrapper
//    //rayTracerKernelWrapper( d_outputImage, 0, 0, width, height, cameraData, 
//    //    d_primitives, numPrimitive, d_lights, numLight, d_materials, numMaterial, 1, d_devStates,iteration );
//    //timer.Stop();
//
//    std::cout<<"Render time: "<<timer.Elapsed()<<" ms."<<std::endl;
//   
//    memset( h_outputImage, 0,sizeof(unsigned char) * 4 * width * height );
//    cudaErrorCheck( cudaMemcpy( (void*)h_outputImage, d_outputImage, sizeof(unsigned char) * 4 * width * height , cudaMemcpyDeviceToHost) );
//
//    //Pixel p;
//    RGBQUAD p;
//    for( int h = 0; h < height; ++h )
//        for( int w = 0; w < width; ++w )
//        {
//            p.rgbRed = h_outputImage[ 4*h*width + 4*w];
//            p.rgbGreen = h_outputImage[ 4*h*width + 4*w+1];
//            p.rgbBlue = h_outputImage[ 4*h*width + 4*w+2];
//            p.rgbReserved = 255;
//            FreeImage_SetPixelColor( outputImage, w, height-1-h, &p );
//            //outputImage.writePixel( w, h, p );
//        }
//    
//}

void CudaRayTracer::packSceneDescData( const SceneDesc &sceneDesc )
{

    width = sceneDesc.width;
    height = sceneDesc.height;
    center = sceneDesc.center;
    up = sceneDesc.up;
    //packing the camrea setting
    eyePos = cameraData.eyePos = sceneDesc.eyePos;
    //cameraData.viewportHalfDim.y = tan( sceneDesc.fovy / 2.0 );
    //cameraData.viewportHalfDim.x = (float)width / (float) height * cameraData.viewportHalfDim.y;
    glm::vec2 viewportHalfDim;
    viewportHalfDim.y =  tan( sceneDesc.fovy / 2.0 );
    viewportHalfDim.x = (float)width / (float) height * viewportHalfDim.y;

    cameraData.offset1.x = viewportHalfDim.x * 2.0f / width;
    cameraData.offset1.y = -viewportHalfDim.y * 2.0f / height;
    cameraData.offset2.x = ( 1.0f/width - 1 )* viewportHalfDim.x;
    cameraData.offset2.y = (-1.0f/height+1) * viewportHalfDim.y;

    //apply super-sampling pattern
    cameraData.jitteredOffset1 = sampleGrid[sampleGridIdx];

    //offset.x = cameraData.viewportHalfDim.x * ( (idx.x+0.5) / (width/2.0) - 1 );
    //offset.y = cameraData.viewportHalfDim.y * ( 1- (idx.y+0.5) / (height/2.0)  );
    //construct the 3 orthoogonal vectors constitute a frame
    cameraData.uVec = glm::normalize( up );
    cameraData.wVec = glm::normalize( center - cameraData.eyePos );
    cameraData.vVec = glm::normalize( glm::cross( cameraData.wVec, cameraData.uVec ) );
    cameraData.uVec = glm::normalize( glm::cross( cameraData.vVec, cameraData.wVec ) );


    //packing the primitives
    numPrimitive = sceneDesc.primitives.size();
    h_pPrimitives = new _Primitive[ numPrimitive ]; 

    for( int i = 0; i < sceneDesc.primitives.size(); ++i )
    {
        if( sceneDesc.primitives[i]->toString().compare("sphere") == 0 )
        {
            h_pPrimitives[i].center = glm::vec3( ((Sphere*)sceneDesc.primitives[i])->center );
            h_pPrimitives[i].radius = ((Sphere*)sceneDesc.primitives[i])->radius;
            h_pPrimitives[i].type = TYPE_SPHERE; //sphere type
            h_pPrimitives[i].mtl_id = sceneDesc.primitives[i]->mtl_idx;
        }
        else if( sceneDesc.primitives[i]->toString().compare("triangle") == 0 )
        {
            for( int n = 0; n < 3; ++n )
                h_pPrimitives[i].vert[n]  =((Triangle*)sceneDesc.primitives[i])->v[n];

            //for( int n = 0; n < 3; ++n )
            //    h_pPrimitives[i].normal[n] =((Triangle*)sceneDesc.primitives[i])->n[n];

            h_pPrimitives[i].pn = ((Triangle*)sceneDesc.primitives[i])->pn;

            h_pPrimitives[i].type = TYPE_TRIANGLE; //triangle type
            h_pPrimitives[i].mtl_id = sceneDesc.primitives[i]->mtl_idx;
        }
        else if( sceneDesc.primitives[i]->toString().compare("bounding box") == 0 )
        {
            h_pPrimitives[i].vert[0] = ((Bbox*)sceneDesc.primitives[i])->min;
            h_pPrimitives[i].vert[1] = ((Bbox*)sceneDesc.primitives[i])->max;
            h_pPrimitives[i].mtl_id = ((Bbox*)sceneDesc.primitives[i])->polyNum;
            h_pPrimitives[i].type = TYPE_BBOX;
        }
        //h_pPrimitives[i].transform = sceneDesc.primitives[i]->transform;
        //h_pPrimitives[i].invTrans = sceneDesc.primitives[i]->invTrans;

    }

    //pack light sources
    numLight = sceneDesc.lights.size();
    h_pLights = new _Light[numLight];
    for( int i = 0; i < numLight; i++ )
    {
        h_pLights[i].pos = sceneDesc.lights[i].pos;
        h_pLights[i].color = sceneDesc.lights[i].color;
        h_pLights[i].attenu_linear = sceneDesc.lights[i].attenu_linear;
        h_pLights[i].attenu_const = sceneDesc.lights[i].attenu_const;
        h_pLights[i].attenu_quadratic = sceneDesc.lights[i].attenu_quadratic;

        h_pLights[i].type = sceneDesc.lights[i].type;
        h_pLights[i].normal = sceneDesc.lights[i].normal;
        h_pLights[i].width = sceneDesc.lights[i].width;
    }

    //pack materails 
    numMaterial = sceneDesc.mtls.size();
    h_pMaterials = new _Material[ numMaterial ];
    for( int i = 0; i < numMaterial; ++i )
    {
        h_pMaterials[i].ambient = sceneDesc.mtls[i].ambient;
        h_pMaterials[i].emission = sceneDesc.mtls[i].emission;
        h_pMaterials[i].diffuse = sceneDesc.mtls[i].diffuse;
        h_pMaterials[i].specular = sceneDesc.mtls[i].specular;
        h_pMaterials[i].shininess = sceneDesc.mtls[i].shininess;
    }
}

void CudaRayTracer::cleanUp()
{
    if( h_pPrimitives )
        delete [] h_pPrimitives;
    h_pPrimitives  = 0;

    if( h_pMaterials )
        delete [] h_pMaterials;
    h_pMaterials = 0;

    if( h_pLights )
        delete[] h_pLights;
    h_pLights = 0;

    if( h_outputImage )
        delete [] h_outputImage;
    h_outputImage = 0;

    if( h_marker )
        delete [] h_marker;
    h_marker = 0;

    ////if( d_outputImage )
    //    cudaErrorCheck( cudaFree( d_outputImage ) );
    //d_outputImage = 0;

    if( d_posBuffer )
       cudaErrorCheck( cudaFree( d_posBuffer ) );
    d_posBuffer = 0;
    if( d_normalBuffer )
        cudaErrorCheck( cudaFree( d_normalBuffer ) );
    d_posBuffer = 0;

    if( d_rayBuffer )
        cudaErrorCheck( cudaFree( d_rayBuffer ) );
    d_rayBuffer = 0;
    if( d_directIllum )
        cudaErrorCheck( cudaFree( d_directIllum ) );
    d_directIllum = 0;

    if( d_indirectIllum )
        cudaErrorCheck( cudaFree( d_indirectIllum ) );
    d_indirectIllum = 0;

    if( d_primitives )
        cudaErrorCheck( cudaFree( d_primitives  ) );
    d_primitives = 0;


    if( d_lights )
        cudaErrorCheck( cudaFree( d_lights ) );
    d_lights = 0;

    if( d_materials )
        cudaErrorCheck( cudaFree( d_materials ) );
    d_materials = 0;

    if( d_devStates  )
       cudaErrorCheck( cudaFree(d_devStates) );
    d_devStates = 0;
    if( d_sobolStates  )
       cudaErrorCheck( cudaFree(d_sobolStates) );
    d_sobolStates = 0;

    if( d_vectors )
        cudaErrorCheck( cudaFree(d_vectors) );
    d_vectors  = 0;

    if( d_marker )
        cudaErrorCheck( cudaFree(d_marker) );
    d_marker = 0;

    if( d_marker_temp )
        cudaErrorCheck( cudaFree(d_marker_temp) );
    d_marker = 0;

    curandDestroyDistribution(poisson_dist);
}

void  CudaRayTracer::init( const SceneDesc &scene )
{
    if( scene.width < 1 || scene.height < 1 )
        return;

    width = scene.width;
    height = scene.height;
    numValidPath = width * height;
    //Pack scene description data
    packSceneDescData( scene );

    //allocate memory in the device
    cudaErrorCheck( cudaMalloc( &d_primitives, sizeof( _Primitive ) * numPrimitive ) );
    cudaErrorCheck( cudaMalloc( &d_lights, sizeof( _Light ) * numLight ) );
    cudaErrorCheck( cudaMalloc( &d_materials, sizeof( _Material ) * numMaterial ) );
    cudaErrorCheck( cudaMalloc( &d_posBuffer, sizeof( glm::vec3 ) * width * height ) );
    cudaErrorCheck( cudaMalloc( &d_rayBuffer, sizeof( glm::vec3 ) * width * height ) );
    cudaErrorCheck( cudaMalloc( &d_normalBuffer, sizeof( glm::vec3 ) * width * height ) );
    cudaErrorCheck( cudaMalloc( &d_directIllum, sizeof( float ) * width * height * 3 * MAXDEPTH ) );
    cudaErrorCheck( cudaMalloc( &d_indirectIllum, sizeof( float ) * width * height * 3 * MAXDEPTH ) );
    //cudaErrorCheck( cudaMalloc( &d_outputImage, sizeof( unsigned char )  * width * height * 4 ) );

    //Send scene description data to the device
    cudaErrorCheck( cudaMemcpy( (void*)d_primitives, h_pPrimitives, sizeof( _Primitive ) * numPrimitive, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy( (void*)d_lights, h_pLights, sizeof( _Light ) * numLight , cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpy( (void*)d_materials, h_pMaterials, sizeof( _Material ) * numMaterial , cudaMemcpyHostToDevice ) );

    cudaErrorCheck( cudaMemset( (void*)d_directIllum, 0, sizeof(float) * 3 * width * height * MAXDEPTH ) );
    cudaErrorCheck( cudaMemset( (void*)d_indirectIllum, 0, sizeof(float) * 3 * width * height * MAXDEPTH ) ); 

    //allocate a marker array for stream compaction and populate it with index values

    h_marker = new int[ width * height ];
    for( int i = 0; i < width * height; ++i ) 
        h_marker[i] = i;
    cudaErrorCheck( cudaMalloc( (void**)&d_marker, sizeof( int ) * width * height ) );
    cudaErrorCheck( cudaMemcpy( (void*)d_marker, (void*)h_marker, sizeof( int )*width*height, cudaMemcpyHostToDevice ) );

    cudaErrorCheck( cudaMalloc( (void**)&d_marker_temp, sizeof( int ) * width * height ) );
    initMarkerWrapper( width, height, d_marker_temp );

        
    setupDevStates();


    param.outputImage = &d_outputImage;
    param.directIllum = &d_directIllum;
    param.indirectIllum = &d_indirectIllum;
    param.posBuf = &d_posBuffer;
    param.rayBuf = &d_rayBuffer;
    param.normalBuf = &d_normalBuffer;
    param.marker = &d_marker;
    param.rayNum = &numValidPath;
    param.width = width;
    param.height = height;
    param.cameraData = &cameraData;
    param.primitives = &d_primitives;
    param.primitiveNum = numPrimitive;
    param.lights = &d_lights;
    param.lightNum =numLight;
    param.mtl = &d_materials;
    param.mtlNum = numMaterial;
    param.DOPSampleCount = 0;
    param.state  = &d_devStates;
    param.sobolState = &d_sobolStates;
    param.depth = &depth;
    param.iteration = &iteration;

    //allocate host memory
    //h_outputImage = new unsigned char[ 4 * width * height ];

    std::cout<<"Path tracer initialization completed. Start rendering\n"<<std::endl;
}

void CudaRayTracer::resetPathDepth()
{
    depth = 0;
    iteration += 1;
    numValidPath = width * height;

    //rotate to next supersampling subpixel
    sampleGridIdx = sampleGridIdx+1 % 4;
    //apply super-sampling pattern
    cameraData.jitteredOffset1 = sampleGrid[sampleGridIdx];

    //jitter camera position, for DOF effect
    jitterCameraPos();
}
void CudaRayTracer::resetIteration()
{
    depth = 0;
    iteration = 1;
    numValidPath = width * height;
    cudaErrorCheck( cudaMemset( (void*)d_outputImage, 0, sizeof(float) * 4 * width * height ) );

    cudaErrorCheck( cudaMemset( (void*)d_directIllum, 0, sizeof(float) * 3 * width * height * MAXDEPTH ) );
    cudaErrorCheck( cudaMemset( (void*)d_indirectIllum, 0, sizeof(float) * 3 * width * height * MAXDEPTH ) ); 
    cudaErrorCheck( cudaMemset( (void*)d_normalBuffer, 0, sizeof(glm::vec3)  * width * height ) );
    cudaErrorCheck( cudaMemset( (void*)d_posBuffer, 0, sizeof(glm::vec3)  * width * height ) );
    cudaErrorCheck( cudaMemset( (void*)d_rayBuffer, 0, sizeof(glm::vec3)  * width * height ) );
    //initMarkerWrapper( width, height, d_marker_temp );
    //cudaErrorCheck( cudaMemcpy( d_marker, h_marker, sizeof( int )*width*height, cudaMemcpyHostToDevice ) );
}

void CudaRayTracer::updateCamera( const SceneDesc &sceneDesc )
{
    //packing the camrea setting
    eyePos = cameraData.eyePos = sceneDesc.eyePos;
    glm::vec2 viewportHalfDim;
    viewportHalfDim.y =  tan( sceneDesc.fovy / 2.0 );
    viewportHalfDim.x = (float)width / (float) height * viewportHalfDim.y;

    cameraData.offset1.x = viewportHalfDim.x * 2.0f / width;
    cameraData.offset1.y = -viewportHalfDim.y * 2.0f / height;
    cameraData.offset2.x = ( 1.0f/width - 1 )* viewportHalfDim.x;
    cameraData.offset2.y = (-1.0f/height+1) * viewportHalfDim.y;


   //apply super-sampling pattern
    //cameraData.jitteredOffset1 =  sampleGrid[sampleGridIdx];

    width = sceneDesc.width;
    height = sceneDesc.height;

    //construct the 3 orthoogonal vectors constitute a frame
    cameraData.uVec = glm::normalize( sceneDesc.up );
    cameraData.wVec = glm::normalize( sceneDesc.center - eyePos );
    cameraData.vVec = glm::normalize( glm::cross( cameraData.wVec, cameraData.uVec ) );
    cameraData.uVec = glm::normalize( glm::cross( cameraData.vVec, cameraData.wVec ) );

    jitterCameraPos();
}

void CudaRayTracer::setupDevStates()
{
    curandDirectionVectors32_t* vectors;
    if( CURAND_STATUS_SUCCESS != curandGetDirectionVectors32(  &vectors, CURAND_DIRECTION_VECTORS_32_JOEKUO6 ) )
    {
          std::cout<<"ERROR at "<<__FILE__<<":"<<__LINE__<<"\n";
          exit(1);
    }

    cudaErrorCheck( cudaMalloc( (void**)&d_devStates, width*height*sizeof(curandState) ) );

    cudaErrorCheck( cudaMalloc( (void**)&d_sobolStates, width*height*sizeof(curandStateSobol32_t) ) );
    //cudaErrorCheck( cudaMalloc( (void**)&d_vectors, sizeof(unsigned int)*32 ) );
    //cudaErrorCheck( cudaMemcpy( d_vectors, vectors[0], sizeof(unsigned int)*32, cudaMemcpyHostToDevice ) );
    //std::cout<<"state size:"<<sizeof(curandState)*width*height/1024/1024<<std::endl;
    setupRandSeedWrapper(width,height,d_devStates ) ;
    //setupSobolRandSeedWrapper( width, height, d_sobolStates, d_vectors);
    //if( CURAND_STATUS_SUCCESS != curandCreatePoissonDistribution( 100, &poisson_dist ) )
    //{
    //    std::cout<<"ERROR at "<<__FILE__<<":"<<__LINE__<<"\n";
    //    exit(1);
    //}
}

void CudaRayTracer::compactSurvivingPath()
{
    compactNaturalNum( d_marker, d_marker_temp, numValidPath );

    //copy the compact result bact to the marker-in-use array
    cudaErrorCheck( cudaMemcpy( (void*)d_marker, (void*)d_marker_temp, sizeof(int)*width*height, cudaMemcpyDeviceToDevice ) );
    initMarkerWrapper( width, height, d_marker_temp );
    //update the number of surviving valid paths
    numValidPath = countValidPath( d_marker, numValidPath );
}

void CudaRayTracer::jitterCameraPos()
{
    //cameraData.eyePos = eyePos + (( rand() % 10+1 )/10.0f )*cameraData.uVec +  (( rand() % 10+1 )/10.0f )*cameraData.vVec;
    //cameraData.wVec = glm::normalize( center - cameraData.eyePos );

    //cameraData.uVec = glm::normalize( up );
    //cameraData.wVec = glm::normalize( center - eyePos );
    //cameraData.vVec = glm::normalize( glm::cross( cameraData.wVec, cameraData.uVec ) );
    //cameraData.uVec = glm::normalize( glm::cross( cameraData.vVec, cameraData.wVec ) );

    cameraData.eyePos = eyePos + 0.1f*(rand()/(float)(RAND_MAX+1))*cameraData.uVec +   0.1f*(rand()/(float)(RAND_MAX+1))*cameraData.vVec;
    cameraData.wVec = glm::normalize( center - cameraData.eyePos );
    //cameraData.vVec = glm::normalize( glm::cross( cameraData.wVec, cameraData.uVec ) );
    //cameraData.uVec = glm::normalize( glm::cross( cameraData.vVec, cameraData.wVec ) );
}