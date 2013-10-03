-------------------------------------------------------------------------------
CIS565 Fall 2013: Project 2: CUDA Pathtracer
-------------------------------------------------------------------------------
NOTE:
-------------------------------------------------------------------------------
This project requires an NVIDIA graphics card with CUDA compute 2.0 capability! 


-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
This is a basic path tracing engine written in CUDA. 

It is capable of simulating a diverse set of materials including matte diffuse surfaces, 
metalics, glossy smooth mirrors, translucent and transparent refractive materials, 
glass and speckled tinted glass.

![This scene contains elements of every feature implemented](/renders/fuzzyglass_fresnel.bmp "Speckled Tinted Glass")

-------------------------------------------------------------------------------
Features:
-------------------------------------------------------------------------------

* Full global illumination (including soft shadows, color bleeding, etc.) by pathtracing rays through the scene. 
* Supersampled antialiasing
* Parallelization by ray instead of by pixel
* Perfect specular reflection
* Scattered specular reflection and transmission (fuzzy glass/mirrors)
* Fresnel-based reflection/refraction (i.e. glass)
* Stream compaction for optimizing high bounce counts in open scenes
* Optional Global Lighting sources for faster convergence

-------------------------------------------------------------------------------
CONTENTS:
-------------------------------------------------------------------------------
The Project2 root directory contains the following subdirectories:
	
* src/ contains the source code for the project. Both the Windows Visual Studio solution and the OSX makefile reference this folder for all source; the base source code compiles on OSX and Windows without modification.
* scenes/ contains an example scene description file.
* renders/ contains two example renders: the raytraced render from Project 1 (GI_no.bmp), and the same scene rendered with global illumination (GI_yes.bmp). 
* PROJ1_WIN/ contains a Windows Visual Studio 2010 project and all dependencies needed for building and running on Windows 7.
* PROJ1_OSX/ contains a OSX makefile, run script, and all dependencies needed for building and running on Mac OSX 10.8. 
* PROJ1_NIX/ contains a Linux makefile for building and running on Ubuntu 
  12.04 LTS. Note that you will need to set the following environment
  variables: 
    
  - PATH=$PATH:/usr/local/cuda-5.5/bin
  - LD_LIBRARY_PATH=/usr/local/cuda-5.5/lib64:/lib

The projects build and run exactly the same way as in Project0 and Project1.

-------------------------------------------------------------------------------
Interactive Controls
-------------------------------------------------------------------------------
The engine was designed so that many features could modified at runtime to allow easy exploration of the effects of various parameters. In addition, several debug modes were implemented that graphically display additional information about the scene. These options to result in more complex kernels that have a negative impact on performance. I preferred the flexibility to quickly experiment for this project, but in the path tracer I will be redesigning the kernel structure from the ground up with performance in mind.

Here is a complete list of the keypress commands you can use at runtime.

Keypress | Function
--- | ---
A | Toggles Anti-Aliasing
S | Toggles Stream Compaction
F | Toggles Frame Filtering
G | Toggles Harsh Global Shadows
f | Clears frame filter
= | Increase trace depth
- | Decrease trace dept
ESC | Exit
1 | Pathtracing Render Mode
2 | Ray Coverage Debug Mode
3 | Trace Depth Debug Mode
4 | First Hit Debug Mode
5 | Normals Debug Mode





-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
...


-------------------------------------------------------------------------------
NOTES ON GLM:
-------------------------------------------------------------------------------
This project uses GLM, the GL Math library, for linear algebra. You need to know two important points on how GLM is used in this project:

* In this project, indices in GLM vectors (such as vec3, vec4), are accessed via swizzling. So, instead of v[0], v.x is used, and instead of v[1], v.y is used, and so on and so forth.
* GLM Matrix operations work fine on NVIDIA Fermi cards and later, but pre-Fermi cards do not play nice with GLM matrices. As such, in this project, GLM matrices are replaced with a custom matrix struct, called a cudaMat4, found in cudaMat4.h. A custom function for multiplying glm::vec4s and cudaMat4s is provided as multiplyMV() in intersections.h.


-------------------------------------------------------------------------------
THIRD PARTY CODE
-------------------------------------------------------------------------------
My implementation of parallel exclusive scan in CUDA was greatly influenced by this GPUGems article: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html 
I had to rewrite it all myself and implement the arbitrary length array code, but the bulk of the code is very similar.
