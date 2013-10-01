
#include "main_runtime.h"
#include "sceneStructs.h"
#include "raytraceKernel.h"
#include "image.h"
#include "utilities.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include <string>
#include <iostream>

using namespace std;

void display()
{
	camera* cam = &(render_scene->renderCam);
	int width = cam->resolution.x;
	int height = cam->resolution.y;

	if ( iterations < cam->iterations )
	{
		// Compute image (single frame) in cuda and write to OpenGL buffer.
		runCuda();
	}
	else
	{
		if ( !is_render_done )
		{
			SaveFrameToImageFile();
			if( is_single_frame_mode )
			{
				cudaDeviceReset(); 
				exit(0);
			}
			is_render_done = true;
		}
		if ( target_frame < cam->frames-1 )
		{
			//clear image buffer and move onto next frame
			target_frame++;
			iterations = 0;
			for ( int i=0; i<width*height; ++i )
			{
				cam->image[i] = glm::vec3(0,0,0);
			}
			cudaDeviceReset(); 
			is_render_done = false;
		}
	}

	string title = "565Raytracer | " + utilityCore::convertIntToString(iterations) + " Iterations";
	glutSetWindowTitle(title.c_str());

	// Bind texture to OpenGL buffer.
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
			GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glClear(GL_COLOR_BUFFER_BIT);   
	
	// Draw a quad that specifies the texture coordinates at each corner.
	// VAO, shader program, and texture already bound
	glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

	glutPostRedisplay();
	glutSwapBuffers();
}


void runCuda()
{
	// Map OpenGL buffer object for writing from CUDA on a single GPU.
	// No data is moved (Win & Linux).
	// When mapped to CUDA, OpenGL should not use this buffer.
	
	uchar4 *dptr = NULL;
	iterations++;

	// 1) Map OpenGL buffer to CUDA memory.
	cudaGLMapBufferObject((void**)&dptr, pbo);
		
	// Pack geom and material arrays for passing to kernel.
	geom* geoms = new geom[render_scene->objects.size()];
	material* materials = new material[render_scene->materials.size()];
		
	for(int i=0; i<render_scene->objects.size(); i++)
	{
		geoms[i] = render_scene->objects[i];
	}
	for(int i=0; i<render_scene->materials.size(); i++)
	{
		materials[i] = render_scene->materials[i];
	}

	// 2) Compute (execute kernel) & write image from CUDA to
	//    OpenGL buffer.
	cudaRaytraceCore(dptr, &(render_scene->renderCam), target_frame, iterations, materials, render_scene->materials.size(), geoms, render_scene->objects.size() );
		
	// 3) Unmap OpenGL buffer.
	cudaGLUnmapBufferObject(pbo);

	delete [] geoms;
	delete [] materials;
}

void SaveFrameToImageFile()
{
	camera* cam = &(render_scene->renderCam);

	image outputImage(cam->resolution.x, cam->resolution.y);

	for(int x=0; x<cam->resolution.x; x++)
	{
		for(int y=0; y<cam->resolution.y; y++)
		{
			int index = x + (y * cam->resolution.x);
			outputImage.writePixelRGB(cam->resolution.x-1-x,y,cam->image[index]);
		}
	}
      
	gammaSettings gamma;
	gamma.applyGamma = true;
	gamma.gamma = 1.0/2.2;
	gamma.divisor = 1.0f;//cam->iterations;
	outputImage.setGammaSettings(gamma);
	string filename = cam->imageName;
	string s;
	stringstream out;
	out << target_frame;
	s = out.str();
	utilityCore::replaceString(filename, ".bmp", "."+s+".bmp");
	utilityCore::replaceString(filename, ".png", "."+s+".png");
	outputImage.saveImageRGB(filename);
	cout << "Saved frame " << s << " to " << filename << endl;
}

const unsigned char KEY_ESC = 27;
const unsigned char KEY_W = 'w';

void keyboard(unsigned char key, int x, int y)
{
	std::cout << key << std::endl;
	switch (key) 
	{
		case KEY_ESC:
			exit(1);
			break;
		case KEY_W:
			exit(1);
			break;
	}
}

void SpecialInput(int key, int x, int y)
{
	glm::vec3 increment(0.0f);

	switch(key)
	{
		case GLUT_KEY_UP:
			increment.z = -0.1f;
			break;
		case GLUT_KEY_DOWN:
			increment.z = 0.1f;
			break;
		case GLUT_KEY_LEFT:
			increment.x = -0.1f;
			break;
		case GLUT_KEY_RIGHT:
			increment.x = 0.1f;
			break;
	}
	*(render_scene->renderCam.positions) = *(render_scene->renderCam.positions) + increment;
}
