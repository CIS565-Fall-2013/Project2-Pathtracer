#include <iostream>
#include <sstream>
#include <string>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <FreeImage.h>
#include "cudaRaytracer.h"
#include "SceneDesc.h"
#include "FileParser.h"
#include "glRoutine.h"
#include "variables.h"
#define GLM_SWIZZLE
#include "glm/glm.hpp"

using namespace std;
using namespace glm;


CudaRayTracer* cudaRayTracer = NULL;
unsigned int win_w, win_h;

SceneDesc theScene;
int win_id;

int main( int argc, char* argv[] )
{
    //FreeImage_Initialise();

    if( argc == 1 )
    {
        cout<<"Usage: CudaRaytracer.exe [scene filename]\n";
        system( "pause" );
        return 1;
    }

    if( FileParser::parse( argv[1], theScene ) != 0 )
    {
        cout<<"Can't read the scene file\n";
        system( "pause" );
        return 1;
    }
    //FileParser::parse( "testScene.scene", theScene );
    win_w = theScene.width;
    win_h = theScene.height;

    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_STENCIL );

    glutInitContextVersion( 4,0 );
    glutInitContextFlags( GLUT_FORWARD_COMPATIBLE );
    glutInitContextProfile( GLUT_COMPATIBILITY_PROFILE );

    glutInitWindowSize( theScene.width, theScene.height );
    win_id = glutCreateWindow( "GPU Path Tracer" );

    GLenum errCode = glewInit();
    if( errCode != GLEW_OK )
    {
        cerr<<"Error: "<<glewGetErrorString(errCode)<<endl;
        return 1;
    }
    if( initGL() != 0 )
        return 1;

    initGLUI( win_id );

    cudaRayTracer = new CudaRayTracer();
    cudaRayTracer->init( theScene );

    glutDisplayFunc( glut_display );
    glutReshapeFunc( glut_reshape );
    glutKeyboardFunc( glut_keyboard );
    glutIdleFunc( glut_idle );
    glutMainLoop();
    
    //float r = glm::distance( vec3( theScene.eyePos.x, 0, theScene.eyePos.z )
    //                           , vec3( theScene.center.x, 0, theScene.center.z ) );
    //float degree = 0;
    //stringstream name;

    //FIBITMAP* bitmap = FreeImage_Allocate( theScene.width, theScene.height, 32 );
    //for( int f = 0; f < 450; ++f ) //render 900 frame
    //{
    //    //compute the camera pos 
    //    theScene.eyePos.x = theScene.center.x + r * sin( degree * 3.1415926/180.0f );
    //    theScene.eyePos.z = theScene.center.z+ r*cos( degree * 3.1415926 / 180);

    //    cudaRayTracer->updateCamera(theScene);
    //    cudaRayTracer->renderImage( bitmap );

    //    name.str("");
    //    name<<"frames\\CUDA-based GPU Raytracer";
    //    name<<f<<".png";
    //    
    //    //outputImage.outputPPM( name.str().c_str() ); 
    //    FreeImage_Save( FIF_PNG, bitmap, name.str().c_str() );
    //    degree += 1.25f;
    //}
    
    
    cleanUpGL();
    delete cudaRayTracer;
    //FreeImage_DeInitialise();
    return 0;
}

