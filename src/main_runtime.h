// Runtime calls (e.g. main loop & interaction)

#ifndef MAIN_RUNTIME_H
#define MAIN_RUNTIME_H

#include <GL/glew.h>
#include "sceneStructs.h"
#include "scene.h"

extern scene* render_scene;
extern int target_frame;
extern int iterations;
extern bool is_render_done;
extern bool is_single_frame_mode;
extern GLuint pbo;
extern GLuint displayImage;

void display();
void runCuda();
void SaveFrameToImageFile();
void keyboard(unsigned char key, int x, int y);
void SpecialInput(int key, int x, int y);

#endif /* MAIN_RUNTIME_H */
