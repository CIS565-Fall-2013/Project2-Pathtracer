#include <GL/glew.h>
#include <iostream>
#include <cuda_gl_interop.h>
#include "cudaRaytracer.h"
#include "cudaRaytracerKernel.h"


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

    //d_posBuffer = 0;
    //d_rayBuffer = 0;
    //d_specuBuffer = 0;
    d_directIllum = 0;
    d_indirectIllum = 0;

    d_primitives = 0;
    d_lights = 0;
    d_materials = 0;

    d_devStates = 0;

}

CudaRayTracer::~CudaRayTracer()
{
    cleanUp();
}

void CudaRayTracer::renderImage( cudaGraphicsResource* pboResource, int iteration )
{


    cudaErrorCheck( cudaGraphicsMapResources( 1, &pboResource, 0 ) );
    cudaErrorCheck( cudaGraphicsResourceGetMappedPointer((void**) &d_outputImage, &pboSize, pboResource ) );
    if( iteration == 1 )
    {
        cudaErrorCheck( cudaMemset( (void*)d_outputImage, 0, sizeof(float) * 4 * width * height ) );
        cudaErrorCheck( cudaMemset( (void*)d_directIllum, 0, sizeof(float) * 3 * width * height * MAXDEPTH ) );
        cudaErrorCheck( cudaMemset( (void*)d_indirectIllum, 0, sizeof(float) * 3 * width * height * MAXDEPTH ) );
        //cudaErrorCheck( cudaMemset( (void*)d_specuBuffer, 0, sizeof(float) * 3 * width * height ) );
        //cudaErrorCheck( cudaMemset( (void*)d_posBuffer, 0, sizeof(float) * 3 * width * height ) );
        //cudaErrorCheck( cudaMemset( (void*)d_rayBuffer, 0, sizeof(float) * 3 * width * height ) );
    }

    GpuTimer timer;
    timer.Start();
    //Launch the ray tracing kernel through the wrapper
    rayTracerKernelWrapper( d_outputImage, d_directIllum, d_indirectIllum, width, height, cameraData, 
        d_primitives, numPrimitive, d_lights, numLight, d_materials, numMaterial, 1, d_devStates, iteration );
    timer.Stop();

    std::cout<<"Render time: "<<timer.Elapsed()<<" ms. at iteration "<<iteration<<std::endl;
    cudaErrorCheck( cudaGraphicsUnmapResources( 1, &pboResource, 0 ) );


}

void CudaRayTracer::renderImage( FIBITMAP* outputImage )
{

    cudaErrorCheck( cudaMemset( (void*)d_outputImage, 0, sizeof(unsigned char) * 4 * width * height ) );

    GpuTimer timer;
    //timer.Start();
    //Launch the ray tracing kernel through the wrapper
    //rayTracerKernelWrapper( d_outputImage, 0, 0, width, height, cameraData, 
    //    d_primitives, numPrimitive, d_lights, numLight, d_materials, numMaterial, 1, d_devStates,iteration );
    //timer.Stop();

    std::cout<<"Render time: "<<timer.Elapsed()<<" ms."<<std::endl;
   
    memset( h_outputImage, 0,sizeof(unsigned char) * 4 * width * height );
    cudaErrorCheck( cudaMemcpy( (void*)h_outputImage, d_outputImage, sizeof(unsigned char) * 4 * width * height , cudaMemcpyDeviceToHost) );

    //Pixel p;
    RGBQUAD p;
    for( int h = 0; h < height; ++h )
        for( int w = 0; w < width; ++w )
        {
            p.rgbRed = h_outputImage[ 4*h*width + 4*w];
            p.rgbGreen = h_outputImage[ 4*h*width + 4*w+1];
            p.rgbBlue = h_outputImage[ 4*h*width + 4*w+2];
            p.rgbReserved = 255;
            FreeImage_SetPixelColor( outputImage, w, height-1-h, &p );
            //outputImage.writePixel( w, h, p );
        }
    
}

void CudaRayTracer::packSceneDescData( const SceneDesc &sceneDesc )
{

    width = sceneDesc.width;
    height = sceneDesc.height;
    //packing the camrea setting
    cameraData.eyePos = sceneDesc.eyePos;
    //cameraData.viewportHalfDim.y = tan( sceneDesc.fovy / 2.0 );
    //cameraData.viewportHalfDim.x = (float)width / (float) height * cameraData.viewportHalfDim.y;
    glm::vec2 viewportHalfDim;
    viewportHalfDim.y =  tan( sceneDesc.fovy / 2.0 );
    viewportHalfDim.x = (float)width / (float) height * viewportHalfDim.y;

    cameraData.offset1.x = viewportHalfDim.x * 2.0f / width;
    cameraData.offset1.y = -viewportHalfDim.y * 2.0f / height;
    cameraData.offset2.x = ( 1.0f/width - 1 )* viewportHalfDim.x;
    cameraData.offset2.y = (-1.0f/height+1) * viewportHalfDim.y;
        //offset.x = cameraData.viewportHalfDim.x * ( (idx.x+0.5) / (width/2.0) - 1 );
    //offset.y = cameraData.viewportHalfDim.y * ( 1- (idx.y+0.5) / (height/2.0)  );
    //construct the 3 orthoogonal vectors constitute a frame
    cameraData.uVec = glm::normalize( sceneDesc.up );
    cameraData.wVec = glm::normalize( sceneDesc.center - sceneDesc.eyePos );
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

    ////if( d_outputImage )
    //    cudaErrorCheck( cudaFree( d_outputImage ) );
    //d_outputImage = 0;

    //if( d_posBuffer )
    //    cudaErrorCheck( cudaFree( d_posBuffer ) );
    //d_posBuffer = 0;
    //if( d_specuBuffer )
    //    cudaErrorCheck( cudaFree( d_specuBuffer ) );
    //d_posBuffer = 0;

    //if( d_rayBuffer )
    //    cudaErrorCheck( cudaFree( d_rayBuffer ) );
    //d_rayBuffer = 0;
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
}

void  CudaRayTracer::init( const SceneDesc &scene )
{
    if( scene.width < 1 || scene.height < 1 )
        return;

    width = scene.width;
    height = scene.height;
    //Pack scene description data
    packSceneDescData( scene );

    //allocate memory in the device
    cudaErrorCheck( cudaMalloc( &d_primitives, sizeof( _Primitive ) * numPrimitive ) );
    cudaErrorCheck( cudaMalloc( &d_lights, sizeof( _Light ) * numLight ) );
    cudaErrorCheck( cudaMalloc( &d_materials, sizeof( _Material ) * numMaterial ) );
    //cudaErrorCheck( cudaMalloc( &d_posBuffer, sizeof( float ) * width * height * 3 ) );
    //cudaErrorCheck( cudaMalloc( &d_rayBuffer, sizeof( float ) * width * height * 3 ) );
    //cudaErrorCheck( cudaMalloc( &d_specuBuffer, sizeof( float ) * width * height * 3 ) );
    cudaErrorCheck( cudaMalloc( &d_directIllum, sizeof( float ) * width * height * 3 * MAXDEPTH ) );
    cudaErrorCheck( cudaMalloc( &d_indirectIllum, sizeof( float ) * width * height * 3 * MAXDEPTH ) );
    //cudaErrorCheck( cudaMalloc( &d_outputImage, sizeof( unsigned char )  * width * height * 4 ) );

    //Send scene description data to the device
    cudaErrorCheck( cudaMemcpy( (void*)d_primitives, h_pPrimitives, sizeof( _Primitive ) * numPrimitive, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy( (void*)d_lights, h_pLights, sizeof( _Light ) * numLight , cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpy( (void*)d_materials, h_pMaterials, sizeof( _Material ) * numMaterial , cudaMemcpyHostToDevice ) );

    //allocate host memory
    //h_outputImage = new unsigned char[ 4 * width * height ];
    
    setupDevStates();
}

void CudaRayTracer::updateCamera( const SceneDesc &sceneDesc )
{
    //packing the camrea setting
    cameraData.eyePos = sceneDesc.eyePos;
    glm::vec2 viewportHalfDim;
    viewportHalfDim.y =  tan( sceneDesc.fovy / 2.0 );
    viewportHalfDim.x = (float)width / (float) height * viewportHalfDim.y;

    cameraData.offset1.x = viewportHalfDim.x * 2.0f / width;
    cameraData.offset1.y = -viewportHalfDim.y * 2.0f / height;
    cameraData.offset2.x = ( 1.0f/width - 1 )* viewportHalfDim.x;
    cameraData.offset2.y = (-1.0f/height+1) * viewportHalfDim.y;

    width = sceneDesc.width;
    height = sceneDesc.height;

    //construct the 3 orthoogonal vectors constitute a frame
    cameraData.uVec = glm::normalize( sceneDesc.up );
    cameraData.wVec = glm::normalize( sceneDesc.center - sceneDesc.eyePos );
    cameraData.vVec = glm::normalize( glm::cross( cameraData.wVec, cameraData.uVec ) );
    cameraData.uVec = glm::normalize( glm::cross( cameraData.vVec, cameraData.wVec ) );

}

void CudaRayTracer::setupDevStates()
{
    cudaErrorCheck( cudaMalloc( (void**)&d_devStates, 8*8*sizeof(curandState) ) );
    setupRandSeedWrapper(8,8, d_devStates ) ;
}