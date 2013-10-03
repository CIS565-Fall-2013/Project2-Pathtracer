-------------------------------------------------------------------------------
#CIS565: Project 1: CUDA Raytracer
-------------------------------------------------------------------------------
#Fall 2013
-------------------------------------------------------------------------------

![Alt text](renders/title image.jpg?raw=true)
-------------------------------------------------------------------------------
PROJECT DESCRIPTION
-------------------------------------------------------------------------------
This is a GPU path tracing program. Features implemented including
#### Basic features
    - Full global illumination (including soft shadows, color bleeding, etc.) by pathtracing rays through the scene. 
    - Properly accumulating emittance and colors to generate a final image
    - Supersampled antialiasing
    - Parallelization by ray instead of by pixel via stream compaction using Thrust library.
    - Perfect specular reflection (mirror)


#### Additional features
    - Fresnel refraction and reflection for transparent objects
    - Motion blur
    - Depth of Field


To activate motion blur effects, define MOTION_BLUR in raytraceKernel.cu file.
To activate depth of field effects, define DEPTH_OF_FIELD in raytraceKernel.cu file.

Both effect are hardcoded while motion blur applies to object 6 and depth of field applies to both green spheres, which are in focus.

-------------------------------------------------------------------------------
IMPLEMENTATION DETAILS
-------------------------------------------------------------------------------
### Fresnel Transparency

Starting from my ray tracer from last project, the easiest addition would be changing my transparent objects from ray tracer 
to be based on Fresnel equations. This would first calculate the Fresnel cooefficients based on the index of refraction from each side
of the interface, normal and incident angle, then shoot both reflection ray and refraction ray and finally add the two components together,
weighted by reflective cooefficient and refractive cooefficient. The Fresnel-enhanced ray-tracer looks like this:

 ![Alt text](renders/first working fresnel.jpg?raw=true)

Throw in some mirror for awesomeness:

 ![Alt text](renders/fresnel with mirror.jpg?raw=true)


### A Naive Path Tracer

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

### Ray-parallelization

The ray-parallel path tracer is better than a pixel-parallel path tracer in that it does not have as many idle threads with concluded pixel
color computation. For a path tracer, the actual bounces between rays could potentially vary a lot, which is a waste of computational resource
if a thread is allowed to wait threads which may have more bounces to calculate. For a ray-parallel implementation, a ray pool is created to 
contain all rays that are currently being traced. It is initialized with primary rays and upon contact with surface, the ray is either replaced by
another ray going other directions (reflection, refraction, diffuse-sampling), or terminated if no intersection was detected. 

With a loop of kernel calls and the help of stream compaction, we can work with fewer rays after each bounce and can allocate hardware resource among
all alive rays in a much more efficient way. 

In terms of numbers, ray parallelization allowed the FPS to increase from around 1 to 4, a dramatic advance!

Working increamentally to transfer all functionalities from pixel-parallel to ray-parallel implementation, the following images were generated:

 ![Alt text](renders/Ray parallel working.jpg?raw=true)

 ![Alt text](renders/Reflective added.jpg?raw=true)

 ![Alt text](renders/Refractive added.jpg?raw=true)


### Stream compaction

Because we are using ray-parallelization here, the rays that are terminated should no longer take a thread and do nothing. So instead of calculating 
each bounce for resolution.x * resolution.y rays, we only calculate the "alive" rays to better utilize hardware resources. For each bounce, scan the
ray pool and remove dead rays with thrust::remove_if, then recalculate the number of blocks for the grid, launch kernel for next bounce. The performance
improvement was well visible. I will talk more about this in Performance Analysis section.

### Color accumulation

While doing this, I encountered the color accumulation issue. Before then, I just use a naive approach to adding each iteration's contribution
multiplied by 1/numberOfIterations. However, I have to wait a long time before the scene is bright enough for me to see anything, and I cannot 
let the image get better indefinitely because the number of iteration is fixed.

Another way I tried is to make the contribution of each iteration decrease exponentially, i.e., first image has contribution of 1/2, second has 1/4, 
and so on. Mathematically, the contribution of color from iteration 0 to infinity equals to 1 so it is theoretically viable. But then the color of 
the image almost stuck with the color of first iteration.

Yet another way is to add the color of current iteration with previous iteration, then divide the result by 2. But then the result color will
be unstable and very noisy because the contribution of later iterations is more significant than previous iterations.

Collin Boots and Liam both provided me with some extremely helpful insights. Basicly Collin would multiply the color with iterations before adding 
contribution of current iteration, and after the addition, divide the result color by (iteration+1), very clever method to avoid knowing the total
number of iterations beforehand while making the contribution of each iteration equivalent. Below shows color accumulation using Collin's method:

 ![Alt text](renders/Collin's accumulation.jpg?raw=true "This was only 800 iteration, so noise was expected")

With Liam's method, nothing has to be done inside the kernel, but rather, add full colors up each iteration, and before displaying, divide by iteration.
I think this is cleaner and does less computation so this is how I do in my implementation:

 ![Alt text](renders/Liam's accumulation.jpg?raw=true)

The light too dark, I consulted Liam again and multiplied emittance of the light source this time. The image looks much nicer now.

 ![Alt text](renders/Multiplying with emittance.jpg?raw=true)

Tune up iteration to 5000, it finally looks like an path-traced image like I often saw!

 ![Alt text](renders/5000 iterations.jpg?raw=true)

### Anti-aliasing

With path-tracing, Anti-aliasing is infinitely simple. Just perturbe the direction of the ray within the pixel for every iteration, and you get very nice AA!

 ![Alt text](renders/5000 iterations with AA.jpg?raw=true)

### Motion blur

Motion blur is easy too, simply translate the object a bit every iteration. This is the effect I was able to achieve in an modified Cornell Box

 ![Alt text](renders/Motion Blur.jpg?raw=true)

### Depth of Field

Browsing Internet for knowlegde of Depth of Field, I found some great sites:

	- For theory: http://http.developer.nvidia.com/GPUGems/gpugems_ch23.html

	- For implementation: 
	http://www.codermind.com/articles/Raytracer-in-C++-Depth-of-field-Fresnel-blobs.html
	http://www.keithlantz.net/2013/03/path-tracer-depth-of-field/
	http://www.colorseffectscode.com/Projects/GPUPathTracer.html (Shehzan's personel website)

Having understood the theory, I followed the algorithm described by Shehzan and wrote my own code. Here is what it looks like (focus on the 2 green spheres):

 ![Alt text](renders/Depth of Field.jpg?raw=true)


-------------------------------------------------------------------------------
HOW TO BUILD
-------------------------------------------------------------------------------
* Project tested in Visual Studio 2010 in Release(5.5) configuration with 
  compute_30,sm_30

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------

* Executin time with different block sizes (motion blur enabled, DoF disabled, 10 bounces with stream compaction and tileSize = 8)

 ![Alt text](Performance evaluation/block size vs execution time.jpg?raw=true)

* Evaluate the effect of stream compaction under different bounces (motion blur enabled, DoF disabled, and tileSize = 8)
 
 ![Alt text](Performance evaluation/Stream compaction vs no stream compaction.jpg?raw=true)

 It turns out that stream compaction is only effective when doing many bounces (>15), possibly because of the additional kernel launch overhead.
 However, making number of bounces 10 or 100 does not really make a big difference in terms of convergence rate because the majority of the rays
 will be terminated after 10 bounce. 


* Test Bench:
    * Windows 7 x64 Professional
	* CPU: Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz
	* GPU: GeForce GTX TITAN (6GB)
	* Memory: 16GB

-------------------------------------------------------------------------------
THIRD PARTY CODE USED
-------------------------------------------------------------------------------
* An Efficient and Robust Ray�CBox Intersection Algorithm, A. Williams, et al.  
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
