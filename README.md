-------------------------------------------------------------------------------
CIS565: Project 1: CUDA Raytracer
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
PROJECT DESCRIPTION
-------------------------------------------------------------------------------
This is a GPU path tracing program. Features implemented including:
* Basic features
	- Full global illumination (including soft shadows, color bleeding, etc.) by pathtracing rays through the scene. 
    - Properly accumulating emittance and colors to generate a final image
    - Supersampled antialiasing
    - Parallelization by ray instead of by pixel via stream compaction using Thrust library.
    - Perfect specular reflection (mirror)


* Additional features
	- Fresnel refraction and reflection for transparent objects
	- Motion blur
	- Depth of Field


To activate motion blur effects, define MOTION_BLUR in raytraceKernel.cu file.
To activate depth of field effects, define DEPTH_OF_FIELD in raytraceKernel.cu file.

Both effect are hardcoded while motion blur applies to object 6 and depth of field applies to both green spheres, which are in focus.

-------------------------------------------------------------------------------
IMPLEMENTATION DETAILS
-------------------------------------------------------------------------------
* Fresnel Transparency

Starting from my ray tracer from last project, the easiest addition would be changing my transparent objects from ray tracer 
to be based on Fresnel equations. This would first calculate the Fresnel cooefficients based on the index of refraction from each side
of the interface, normal and incident angle, then shoot both reflection ray and refraction ray and finally add the two components together,
weighted by reflective cooefficient and refractive cooefficient. The Fresnel-enhanced ray-tracer looks like this:

 ![Alt text](renders/first working fresnel.jpg?raw=true)

Throw in some mirror for awesomeness:

 ![Alt text](renders/fresnel with mirror.jpg?raw=true)


* A Naive Path Tracer

After Fresnel worked, my next objective is get a basic path tracer up and running. Although the rendering equation and all those
BRDFs and PDF seem difficult, if we just do the basic diffuse surface with cosine weighted hemisphere sampling, a naive path tracer
is in fact deceptively simple: the BRDF equals to 1 in all direction and the cosine term multiplied by BRDF gets eliminated by the 
cosine PDF.

Without messing around with ray-parallelization in case I break something in the process, I gave my path tracer a first run, which was 
disastrous.

 ![Alt text](renders/PathTracer First Run.jpg?raw=true)

Apparently, the result was not at all random and it seemed that the light followed a few very limited paths, almost like 
a reflection. Besides, the FPS was horrendously low, around 1~2. Taking Liam's advice and making random number generator 
take index of pixels, iterations and bounces, the image start to look plausible, but still with glaring artifacts.

 ![Alt text](renders/First image seemingly right.jpg?raw=true)

But I will leave it there for now and work on ray-parallelization instead.

* Ray-parallelization

The ray-parallel path tracer is better than a pixel-parallel path tracer in that it does not have as many idle threads with concluded pixel
color computation. For a path tracer, the actual bounces between rays could potentially vary a lot, which is a waste of computational resource
if a thread is allowed to wait threads which may have more bounces to calculate. For a ray-parallel implementation, a ray pool is created to 
contain all rays that are currently being traced. It is initialized with primary rays and upon contact with surface, the ray is either replaced by
another ray going other directions (reflection, refraction, diffuse-sampling), or terminated if no intersection was detected. 

With a loop of kernel calls and the help of stream compaction, we can work with fewer rays after each bounce and can allocate hardware resource among
all alive rays in a much more efficient way. 

In terms of numbers, ray parallelization allowed the FPS to increase from around 1 to 4, a dramatic advance!

Working increamentally to transfer all functionalities from pixel-parallel to ray-parallel implementation, the following images were generated:

 ![Alt text](renders/Ray parallel working (sort of).jpg?raw=true)

 ![Alt text](renders/Reflective added.jpg?raw=true)

 ![Alt text](renders/Refractive added.jpg?raw=true)


* Stream compaction




-------------------------------------------------------------------------------
SCREEN SHOTS AND VIDEOS
-------------------------------------------------------------------------------
* Project running
  ![ScreenShot](https://raw.github.com/wuhao1117/Project1-RayTracer/master/renders/project running.jpg)
	
* Final renders
  ![ScreenShot](https://raw.github.com/wuhao1117/Project1-RayTracer/master/renders/MyRender.jpg)

* Video
  https://raw.github.com/wuhao1117/Project1-RayTracer/master/renders/GPU_raytracer.mp4
-------------------------------------------------------------------------------
HOW TO BUILD
-------------------------------------------------------------------------------
* Project tested in Visual Studio 2012 in Release(5.5) configuration with 
  compute_30,sm_30

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
Tested sample scene with trace depth 2, all features enabled.


* FPS under different block sizes
   - block dimension = 8*8        : 43
   - block dimension = 9*9        : 37
   - block dimension = 10*10    : 37
   - block dimension = 11*11    : 41
   - block dimension = 12*12    : 38
   - block dimension = 13*13    : 34
   - block dimension = 14*14    : 36
   - block dimension = 15*15    : 39
   - block dimension = 16*16    : 43
   - block dimension = 17*17    : 30
   - block dimension = 18*18    : 32
   - block dimension = 19*19    : 34
   - block dimension = 20*20    : 36
   - block dimension = 21*21    : 37
   - block dimension = 22*22    : 40
   - block dimension >= 23*23    : kernel failed! too many resources requested for launch

* FPS for block dimension = 8*8 with 2X jittered SSAA: 16
   - Switch code generation from compute_20,sm_20 to compute_30,sm_30: 20
   - Use fast math: 27

It seemes to me that program executes fastest when image size (800*800) is divisible by block size. 

* Test Bench:
	* CPU: Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz
	* GPU: GeForce GTX TITAN (6GB)
	* Memory: 16GB

-------------------------------------------------------------------------------
THIRD PARTY CODE USED
-------------------------------------------------------------------------------
* An Efficient and Robust Ray¨CBox Intersection Algorithm, A. Williams, et al.  
  http://people.csail.mit.edu/amy/papers/box-jgt.pdf
* Thrust library for stream compaction

-------------------------------------------------------------------------------
TO DO LIST (listed by importance)
-------------------------------------------------------------------------------
* More BRDF models:
	* Oren-Nayar model
	* Microfacet model for refraction
	* Torrance-Sparrow model
	* Ward model
* HEAVY optimization
* Faster convergence
* Interactive camera (must do optimization first)
* OBJ file loading 
* k-d tree or BVH
