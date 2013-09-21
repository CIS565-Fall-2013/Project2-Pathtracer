-------------------------------------------------------------------------------
CIS565: Project 1: CUDA Raytracer
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
PROJECT DESCRIPTION
-------------------------------------------------------------------------------
This is a GPU ray tracing program. Features implemented including:
* Basic features
	- Raycasting from a camera into a scene through a pixel grid
	- Phong lighting for one point light source
	- Diffuse lambertian surfaces
	- Raytraced shadows
	- Cube intersection testing
	- Sphere surface point sampling

* Additional features
	- Specular reflection 
	- Soft shadows and area lights 
	- Refraction

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
  compute_20,sm_21

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


