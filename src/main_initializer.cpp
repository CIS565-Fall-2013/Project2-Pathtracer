
#include "main_initializer.h"
#include <GL/glew.h>
#if CUDA_VERSION >= 5000
	#include <helper_cuda.h>
	#include <helper_cuda_gl.h>
	#define compat_getMaxGflopsDeviceId() gpuGetMaxGflopsDeviceId() 
#else
	#include <cutil_inline.h>
	#include <cutil_gl_inline.h>
	#define compat_getMaxGflopsDeviceId() cutGetMaxGflopsDeviceId()
#endif

// =============================================================================
// ===  OpenGL x Cuda Initialization  ==========================================
// =============================================================================

void initCuda(GLuint* pbo, GLuint* texture_id, int width, int height)
{
	// Use device with highest Gflops/s.
	cudaGLSetGLDevice( compat_getMaxGflopsDeviceId() );

	initPBO(pbo, width, height);
	initTexture(texture_id, width, height);

	// Clean up on program exit.
	//atexit(cleanupCuda);
}

void initPBO(GLuint* pbo, int width, int height)
{
	if ( !pbo ) return;

	// Set up vertex data parameter.
	int num_texels = width*height;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;
    
	// Generate GL buffer (4-channel 8-bit image)
	// and register it to be shared with CUDA.
	glGenBuffers(1, pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject(*pbo);
}


void initTexture(GLuint* texture_id, int width, int height)
{
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, texture_id);
	glBindTexture(GL_TEXTURE_2D, *texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
		GL_UNSIGNED_BYTE, NULL);
}



// =============================================================================
// ===  Cleanup  ===============================================================
// =============================================================================

void cleanupCuda()
{
	if ( pbo ) deletePBO(&pbo);
	if ( displayImage ) deleteTexture(&displayImage);
}

void deletePBO(GLuint* pbo)
{
	if ( !pbo ) return;

	// Unregister this buffer object with CUDA.
	cudaGLUnregisterBufferObject(*pbo);
    
	glBindBuffer(GL_ARRAY_BUFFER, *pbo);
	glDeleteBuffers(1, pbo);
    
	*pbo = (GLuint)NULL;
}

void deleteTexture(GLuint* tex)
{
	glDeleteTextures(1, tex);
	*tex = (GLuint)NULL;
}

