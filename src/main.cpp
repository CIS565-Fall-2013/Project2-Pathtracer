// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Takashi Furuya
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Varun Sampath and Patrick Cozzi for GLSL Loading, from CIS565 Spring 2012 HW5 at the University of Pennsylvania: http://cis565-spring-2012.github.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include "main.h"
#include "main_initializer.h"
#include "main_runtime.h"
#include <GL/glut.h>

using namespace std;

void initVAO();
GLuint initShader(const char *vertexShaderPath, const char *fragmentShaderPath);


int main(int argc, char** argv)
{
	// Set up pathtracer.
	bool is_scene_loaded	= false;
	is_render_done			= false;
	target_frame			= 0;
	is_single_frame_mode	= false;
	iterations				= 0;

	// Read command line arguments and load scene file.
	for ( int i=1; i<argc; ++i )
	{
		// header=data (e.g. scene=my_scene.txt)
		string header, data;
		istringstream liness(argv[i]);
		getline(liness, header, '=');
		getline(liness, data, '=');

		if ( strcmp(header.c_str(), "scene") == 0 )
		{
			render_scene = new scene(data);
			is_scene_loaded = true;
		}
		else if ( strcmp(header.c_str(), "frame") == 0 )
		{
			target_frame = atoi(data.c_str());
			is_single_frame_mode = true;
		}
	}

	if ( !is_scene_loaded )
	{
		cout << "Error: scene file needed!" << endl;
		return 0;
	}

	camera& cam		= render_scene->renderCam;
	int width		= cam.resolution.x;
	int height		= cam.resolution.y;

	if ( target_frame >= cam.frames )
	{
		cout << "Warning: Specified target frame is out of range, ";
		cout << "defaulting to frame 0." << endl;
		target_frame = 0;
	}

	// Initialize GLUT - Create OpenGL rendering context.
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(width, height);
	glutCreateWindow("565Raytracer");

	// Initialize GLEW
	GLenum err = glewInit();
	if ( GLEW_OK != err )
	{
		cout << "glewInit failed, aborting." << endl;
		exit(1);
	}

	// Register GL buffer with CUDA
	initCuda(&pbo, &displayImage, width, height);
	
	//runCuda();

	initVAO();
	
	GLuint passthroughProgram;
	passthroughProgram = initShader("shaders/passthroughVS.glsl", "shaders/passthroughFS.glsl");

	glUseProgram(passthroughProgram);
	glActiveTexture(GL_TEXTURE0);

	// Register window callbacks.
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(SpecialInput);

	// Start main rendering loop.
	glutMainLoop();

	return 0;
}






void initVAO(void){
    GLfloat vertices[] =
    { 
        -1.0f, -1.0f, 
         1.0f, -1.0f, 
         1.0f,  1.0f, 
        -1.0f,  1.0f, 
    };

    GLfloat texcoords[] = 
    { 
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);
    
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader(const char *vertexShaderPath, const char *fragmentShaderPath){
    GLuint program = glslUtility::createProgram(vertexShaderPath, fragmentShaderPath, attributeLocations, 2);
    GLint location;

    glUseProgram(program);
    
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}