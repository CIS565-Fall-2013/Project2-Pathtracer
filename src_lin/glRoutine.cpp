#include <GL/glew.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_gl_interop.h>
#include "cudaRaytracer.h"
#include "glm/glm.hpp"
#include "glRoutine.h"
#include "variables.h"
#include "util.h"


using namespace std;

float vertexData[] = { 
    //vertex position
    -1.0f, -1.0f, 0.0f,
    1.0f, -1.0f, 0.0f,
    1.0f, 1.0f, 0.0f,
    -1.0f, -1.0f, 0.0f,
    1.0f, 1.0f, 0.0f,
    -1.0f,1.0f, 0.0f,

    //texture coordinates
    0.0f, 1.0f,
    1.0f, 1.0f,
    1.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 0.0f,
    0.0f,0.0f};


GLuint pbo;  //handle of pixel buffer object
GLuint vbo;  //handle of vertex buffer object
GLuint vao;  //handle of vertex array object
GLuint texID;

//GLSL shader related variables
GLuint fragShader;
GLuint vertShader;
GLuint shaderProg;
GLuint texLoc;

//Cuda-OpenGL interop objects
cudaGraphicsResource* pboResource;

void glut_display()
{
    //generate the image using GPU raytracer
    cudaRayTracer->renderImage( pboResource );

    //render a quad to display the image
    glClearColor( 0, 0, 0, 0 );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glActiveTexture( GL_TEXTURE0 );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo );
    glBindTexture( GL_TEXTURE_2D, texID );
    glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, win_w, win_h, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER,0 );
    
    glBindVertexArray( vao );
    glDrawArrays( GL_TRIANGLES, 0, 6 );

    glBindTexture(GL_TEXTURE_2D, 0);
    glutSwapBuffers();
}

void glut_idle()
{
    glutPostRedisplay();
}

void glut_reshape( int w, int h )
{
    win_h = h;
    win_w = w;
    //rebuild the pixel buffer object
    initPBO();

    //re-calculate the dimensions of grids
    glViewport( 0, 0, w, h );
    

}

void glut_keyboard( unsigned char key, int x, int y)
{

}

int initPBO()
{
    if( pbo ) 
    {
        //ungister from CUDA context
        cudaGraphicsUnregisterResource( pboResource);
        //destroy the existing pbo 
        glDeleteBuffers( 1, &pbo ); pbo = 0;
        glDeleteTextures( 1, &texID ); texID = 0;
    }

    //create a PBO
    glGenBuffers(1, &pbo);
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo );
    glBufferData( GL_PIXEL_UNPACK_BUFFER, sizeof( GLubyte) * win_w * win_h * 4, NULL, GL_STREAM_DRAW );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );

    //register with CUAD context
    cudaGraphicsGLRegisterBuffer( &pboResource, pbo, cudaGraphicsMapFlagsWriteDiscard );

    //create texture for displaying the rendering result
    glActiveTexture( GL_TEXTURE0);
    glGenTextures( 1, &texID );
    glBindTexture( GL_TEXTURE_2D, texID );
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, win_w, win_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glBindTexture( GL_TEXTURE_2D, 0 );

    return 0;
}

int initVertexData()
{
    glGenBuffers( 1, &vbo );
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glBufferData( GL_ARRAY_BUFFER, sizeof( float) * 36, vertexData, GL_STATIC_DRAW );

    //create and setup the vao
    glGenVertexArrays( 1, &vao );
    glBindVertexArray( vao );

    glEnableVertexAttribArray(0);
    glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, (GLubyte*)NULL );

    glEnableVertexAttribArray(1);
    glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, 0, (GLvoid*)(sizeof( float) * 18) );

    glBindBuffer( GL_ARRAY_BUFFER,0 );
    glBindVertexArray(0);

    return 0;
}

char* readFromFile( const char* filename, int* len )
{
	std::ifstream file;
	file.open( filename, std::ios::binary );
	if( !file.is_open() )
	{
        cerr<<"Read shader source failed!\n";
		return NULL;
	}

	(*len)=0;
	file.seekg( 0, std::ios::end );
	(*len) = file.tellg();
	file.seekg( 0, std::ios::beg );

	if( (*len) == 0 )
	{
		cerr<<"Shader source zero length!\n";
		return NULL;
	}

	char* buf = new char[(*len)+1];
	file.read( buf, *len );
    buf[(*len)] = '\0';
	return buf;
} 

int checkAndShowShaderStatus( const char* obj_name, GLuint obj, int check_mode )
{
	int err_code;
	int max_infolen;
	if( check_mode == 1 )
	{
		glGetShaderiv( obj, GL_COMPILE_STATUS, &err_code );
		glGetShaderiv( obj, GL_INFO_LOG_LENGTH, &max_infolen ); 
	}
	else 
	{	
		glGetProgramiv( obj, GL_LINK_STATUS, &err_code );
		glGetProgramiv( obj, GL_INFO_LOG_LENGTH, &max_infolen ); 
	}
	if( err_code != GL_TRUE )
	{
		int infolen;
		char *infobuf = new char[max_infolen+1];
		glGetShaderInfoLog( obj,max_infolen,&infolen, infobuf);
        cerr<<"ERROR("<<obj_name<<"):";
        cerr<<infobuf<<endl;
		delete [] infobuf;
		return -1;

	}
	else
		return 0;

}


GLuint initShader( GLenum shaderType, const char* shaderSourceFile )
{
    int src_len;
    GLuint shader = glCreateShader( shaderType );
 
    const char* source = readFromFile( shaderSourceFile, &src_len );
    if( source == NULL )
        return 0;

    glShaderSource( shader, 1, &source, NULL );
    glCompileShader( shader );

    delete [] source;

    if( checkAndShowShaderStatus( shaderSourceFile, shader, 1 ) != 0 )
        return 0;

    return shader;

}

GLuint initShaderProg( GLuint vertShader, GLuint fragShader )
{
    GLuint prog = glCreateProgram();

    glAttachShader( prog, vertShader );
    glAttachShader( prog, fragShader );

    glLinkProgram( prog );
    if( checkAndShowShaderStatus( "Shader Program", prog, 2 ) != 0 )
        return 0;

    //Obtain locations of
    texLoc = glGetUniformLocation( shaderProg, "tex1" );
    if( texLoc < 1 )
    {
        cerr<<"Uniform variable text1 unavailable!\n";
    }
    else
        glUniform1i( texLoc, 0 ); //set the sampler location to texture unit 0
    return prog;
}

int initGL()
{
    //init shader
    fragShader = initShader( GL_FRAGMENT_SHADER, "shaders/basic.frag" );
    vertShader = initShader( GL_VERTEX_SHADER, "shaders/basic.vert" );
    shaderProg = initShaderProg( vertShader, fragShader );

    if( fragShader < 1 || vertShader < 1  || shaderProg < 1 )
        return -1;

    glUseProgram(shaderProg);
    //init vbo
    if( initVertexData() )
        return -1;

    //init pbo
    if( initPBO() )
        return -1;

    return 0;
}

void cleanUpGL()
{
    if( pbo )
    {
        cudaGraphicsUnregisterResource( pboResource );
        glDeleteBuffers( 1, &pbo );
        pbo = 0;
    }
    if( texID )
    {
        glDeleteTextures( 1, &texID );
        texID = 0;
    }

    glDeleteBuffers( 1, &vbo );
    glDeleteVertexArrays( 1, &vao );
}