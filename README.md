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
Debug Modes
-------------------------------------------------------------------------------
Ray Coverage Debug Mode should be all white if every ray is mapped correctly to a pixel.

Trace Depth Debug Mode shows how many bounces contributed to each pixel. White is the maximum number, black is 1 bounce.
![Debug Mode](/screenshots/tracedepth_debug.bmp "Trace Debug Mode")

First Hit Debug Mode just shows the raw diffuse color of the first object hit for collision verification and previsualization.

Normal debug mode colors the normals of the first impacted surface for each ray. Pure RGB colors are axis aligned.
(i.e. Red pixels have normals along the x-axis)
![Debug Mode](/screenshots/normal_debug.bmp "Normal Debug Mode")

-------------------------------------------------------------------------------
Global Lighting
-------------------------------------------------------------------------------
Early experiments with the pathtracer in closed environments showed that the final renderings could be quite dark.
![Rendering](/renders/hallofmirrors5000.0.bmp "Dark Rendering")

My first attempt to correct this was increasing the lighting intensity, which caused saturation and speckling around light emitters.
![Rendering](/renders/hallofmirrors.refractionbug.bmp "Saturation")

What do you do when you don't have enough light? Add a sun!
Here is a similar scene with a fresnel glass roof and direct overhead lighting. You can make out the blue tint of the sky through the ceiling but still see some lights reflected in it.

![Rendering](/renders/greentintedglass.0.bmp "Skylight!")

Not only did this dramatically improve the appearance and brightness of the scene, but because more rays are now hitting a "light source" the image converges much faster, especially in outdoor environments like the sundial images below.
The implementation makes changing from a bright midday sun to a peaceful moonlit night as easy as changing the material id of the global light.
The following scenes are all rendered with ONLY global lighting. Note how the shadow on the sundial tracks the light. All this works implicitly by moving the sun in the sky.
Also note the fresnel reflections off the simulated water around the island.

![Rendering](/renders/sundial1.bmp "Sundial") | ![Rendering](/renders/sundial1_moonlight.0.bmp "Sundial")
![Rendering](/renders/sundial2.0.bmp "Sundial") | ![Rendering](/renders/sundial2_moonlight.0.bmp "Sundial")
![Rendering](/renders/sundial3.0.bmp "Sundial") | ![Rendering](/renders/sundial3_moonlight.0.bmp "Sundial")

Note how the global light in this scene behaves just as any other pathtraced emitter, creating interesting effects like caustics below the lenses.

![Rendering](/renders/test.0.bmp "Caustics") 

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
I will have some figures here tomorrow morning on the effectiveness of my stream compaction versus trace depth, but I don't have access to my desktop tonight and my laptop is too slow to get decent performance metrics from.
The short form of it (minus data) is stream compaction is extremely effective for high trace depths and very open scenes. Depending on the scene I've seen 2x to 10x speedups.
However, for very low trace depths in closed environments, the additional overhead can actually degrade performance slightly.


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



-------------------------------------------------------------------------------
TAKUAscene FORMAT:
-------------------------------------------------------------------------------
This project uses a custom scene description format, called TAKUAscene.
TAKUAscene files are flat text files that describe all geometry, materials,
lights, cameras, render settings, and animation frames inside of the scene.
Items in the format are delimited by new lines, and comments can be added at
the end of each line preceded with a double-slash.

Materials are defined in the following fashion:

* MATERIAL (material ID)								//material header
* RGB (float r) (float g) (float b)					//diffuse color (light source color for global light)
* SPECX (float specx)									//specular exponent > 0. Higher values will result in more clear reflectiong/refractions. If set to -1, this will turn the material into a global light source
* SPECRGB (float r) (float g) (float b)				//specular color (background/sky color for global light)
* REFL (bool refl)									//reflectivity component. number between 0 and 1 (0==purely diffuse, 1==purely reflective)
* REFR (bool refr)									//refractivity component. number between 0 and 1 (0==purely diffuse, 1==purely refractive)
* REFRIOR (float ior)									//index of refraction
  for Fresnel effects
* SCATTER (float scatter)								//scatter flag, 0 for
  no, 1 for yes
* ABSCOEFF (float r) (float b) (float g)				//absorption
  coefficients (direct light shading for global light source) 
* RSCTCOEFF (float rsctcoeff)							//reduced scattering
  coefficient
* EMITTANCE (float emittance)							//the emittance of the
  material. Anything >0 makes the material a light source. Going above 1 will make image saturate some pixels

Cameras are defined in the following fashion:

* CAMERA 												//camera header
* RES (float x) (float y)								//resolution
* FOVY (float fovy)										//vertical field of
  view half-angle. the horizonal angle is calculated from this and the
  reslution
* ITERATIONS (float interations)							//how many
  iterations to refine the image, only relevant for supersampled antialiasing,
  depth of field, area lights, and other distributed raytracing applications
* FILE (string filename)									//file to output
  render to upon completion
* frame (frame number)									//start of a frame
* EYE (float x) (float y) (float z)						//camera's position in
  worldspace
* VIEW (float x) (float y) (float z)						//camera's view
  direction
* UP (float x) (float y) (float z)						//camera's up vector

Objects are defined in the following fashion:
* OBJECT (object ID)										//object header
* (cube OR sphere OR mesh)								//type of object, can
  be either "cube", "sphere", or "mesh". Note that cubes and spheres are unit
  sized and centered at the origin.
* material (material ID)									//material to
  assign this object
* frame (frame number)									//start of a frame
* TRANS (float transx) (float transy) (float transz)		//translation
* ROTAT (float rotationx) (float rotationy) (float rotationz)		//rotation
* SCALE (float scalex) (float scaley) (float scalez)		//scale

An example TAKUAscene file setting up two frames inside of a Cornell Box can be
found in the scenes/ directory.
