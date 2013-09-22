#pragma once

#include <GL/glew.h>
#include <GL/glut.h>

void glut_display();

void glut_idle();

void glut_reshape( int w, int h );

void glut_keyboard( unsigned char key, int x, int y);

int initPBO();

int intVertexData();

GLuint initShader();
GLuint initShaderProg();

int initGL();
void cleanUpGL();
