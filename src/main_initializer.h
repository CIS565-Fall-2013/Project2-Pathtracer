
#ifndef MAIN_INITIALIZER_H
#define MAIN_INITIALIZER_H

#include <GL/glew.h>

extern GLuint pbo;
extern GLuint displayImage;

// Initialization
void initCuda(GLuint* pbo, GLuint* texture_id, int width, int height);
void initPBO(GLuint* pbo, int width, int height);
void initTexture(GLuint* texture_id, int width, int height);

// Cleanup
void cleanupCuda();
void deletePBO(GLuint* pbo);
void deleteTexture(GLuint* tex);


#endif /* MAIN_INITIALIZER_H */
