-------------------------------------------------------------------------------
CIS565 Project 2: CUDA Pathtracer
-------------------------------------------------------------------------------
Ricky Arietta Fall 2013
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

This project is a highly parallel version of a Monte Carlo simulated pathtracer
implemented on the GPU, augmented from a template provided by Karl Yi and Liam
Boone. It works by casting out virtual rays through an image plane 
from a user-defined camera and assigning pixel values based on intersections 
with the defined geometry and lights. The final implementation, when built, 
is capable of rendering high-quality images including full global illumination,
physically realistic area light(s) and soft shadows, geometry primitives (spheres, 
cubes), supersampled anti-aliasing, perfect recursive specular reflection, and Fresnel 
refraction. An example of a final render can be seen immediately below this 
description. The implementation of each feature is described briefly below with
a rendered image demonstrating the development of the code. Finally, there is 
some performance analysis included regarding the size and number of blocks requested 
on the GPU during runtime.

(A brief tour of the base code and a description of the scene file format
used during implementation are also included at the end of this file, adapted
from the project description provided by Patrick Cozzi and Liam Boone.)


-------------------------------------------------------------------------------
Initial Ray Casting From Camera & Geometry Primitive Intersection
-------------------------------------------------------------------------------

In order to render images using ray tracing, we needed to define the set of rays
being sent out from the camera. The camera, defined by the user input file,
includes a resolution, field of view angle, viewing direction, up direction, etc.
Using this information, I was able to define an arbitrary image plane some distance
from the camera, orthogonal to the viewing direction vector. Then, using the 
up direction and a computed third orthogonal "right" vector, I was able to
define a grid on the image plane with the same resolution as the desired image.
Then, from the camera, and given an x-index and y-index for the pixel in question,
I could compute a single ray from the camera position to the center of the corresponding
pixel in the image plane (adapted for supersampled antialiasing; see below). These
rays served as the initial rays for tracing paths through the scene. But unlike a traditional
ray tracer, these initial rays were not traced until termination -- they were inserted
into the pool that would be sampled with each wave of path traces (see Parallelization below).

Additionally, when following these rays, I needed to be able to determine any
geometry intersections along the ray direction vector, since these intersections
define the luminance value returned to the image pixel. These functions were also taken
from my Assignment 1 ray tracing implementation.  

-------------------------------------------------------------------------------
Parallelization by Ray (Instead of by Pixel) Using Stream Compaction (Self-Implemented)
-------------------------------------------------------------------------------

To maximize utilization of the hardware, this path tracer is parallelized by ray and
not by pixel. When parallelizing by pixel, some ray paths die out before the maximum
ray depth is reached (either from absorption or lack of intersection) and they are a
drag on the kernel calls, while some paths are traced all the way through.

This implementation uses a pool of rays. Along with the origin and direction, each of
these ray structures stores the (x,y) image coordinate associated with the ray, the
current index of refraction for the ray, a light coefficient for the ray that is a result
of being affected by diffuse absorption, and a flag indicating if the ray is alive or 
dead.

With each new wave of raycasts, the current live rays are pulled from the pool and cast
into the scene. Depending on whether or not geometry was intersected, the ray is marked 
as dead or alive. If there was an intersection and the ray lives on to the next wave of
this iteration, a secondary ray is calculated and inserted back into the pool.

After each such wave, a temporary array structure is computed, having a "true" value if the
corresponding ray in the pool is alive and a "false" value if it is dead. This temp array
is used as the basis for an inclusive scan, which is then shifted to an exclusive scan (the
total number of surviving rays is equal to the last value of the inclusive array before the
shift.) This scan array is used in a scatter call, where the live rays corresponding to these
flags are transferred to a more compact array at the index specified in the scan. This new
array thus includes only the live rays for every wave of raycasts.

All of this code was based on the Parallel Algorithms presentation given in class.

![Stream Compaction Graphic](https://raw.github.com/rarietta/Project2-PathTracer/master/PROJ1_WIN/565Raytracer/README_images/StreamCompaction_graphic.bmp)

-------------------------------------------------------------------------------
Full Global Illumination (Soft shadows, Color bleeding, etc.) by Pathtracing Rays
and Properly Accumulating Luminance Values
-------------------------------------------------------------------------------

Each ray in the scene was traced from the camera until its first bounce within the 
scene. If it hit a diffuse surface, a secondary ray was randomly cast out over the 
cosine weighted hemisphere around the surface normal. The light rays from the camera initially
have an RGB color value of (1.0, 1.0, 1.0). Upon diffuse reflection, some of this light is 
absorbed by the surface, thus the light ray is multiplied by the RGB color of the surface
material. After an intersection and the calculation of this secondary ray, the secondary
ray is placed back in the ray pool in the place of the incoming ray (or marked as
dead if there was no intersection.) This was done for every ray that hits a diffuse
surface. Color is only set when a path happens upon a light source, in which case
the path is terminated and the current color value of the ray is multiplied by the
color of the light source times its emittance.

Unlike ray tracing, we see that this sampling of rays over the surfaces gives us
a lot of features at no additional cost. It computes soft shadows without shadow
feelers, gives us full global illumination (notice the color bleeding from the
walls onto the ceiling/floor/spheres and dark spots on the corners where light gets
trapped), and computes diffuse lighting that varies with the incoming light angle.

![Global Illumination](https://raw.github.com/rarietta/Project2-PathTracer/master/PROJ1_WIN/565Raytracer/README_images/diffuse.bmp)

-------------------------------------------------------------------------------
Perfect Specular Reflection
-------------------------------------------------------------------------------

Unlike in diffuse reflection, when a specular material is intersected, the secondary
ray is not randomly sampled. It is perfectly reflected across the normal and sent out
as a new secondary ray. Furthermore, the incoming ray color value is not changed at all
by the specular coefficient, since in perfect specular reflection all of the light
is reflected and none is absorbed.

The diffuse absorption by specular surfaces does, however, still occur because
the choice of a secondary ray type is sampled over an interval [0,1]. If the material
is specular (or refractive, see below), then a defined float value 0.3 is used as
the probability of diffuse reflection. The probabilities of specular reflection
and refraction are computed via Fresnel equations.

![Perfect Specular Reflection](https://raw.github.com/rarietta/Project2-PathTracer/master/PROJ1_WIN/565Raytracer/README_images/specular.bmp])

-------------------------------------------------------------------------------
Fresnel Refraction
-------------------------------------------------------------------------------

In addition to reflection, the path tracer accounts for refractive surfaces such
as glass. We can see that the refractive glass inverts the view of the scene behind
it, and the path tracer automatically accounts for caustics formed under the glass
in the direction of the light source (note the bright spot on the wall.)

![Fresnel Refraction](https://raw.github.com/rarietta/Project2-PathTracer/master/PROJ1_WIN/565Raytracer/README_images/refraction.bmp)

-------------------------------------------------------------------------------
Addition of Supersampled AntiAliasing
-------------------------------------------------------------------------------

_NOTE: This feature was implemented in Assignment 1 and the code was reused in
this pathtracer. Since the pathttracer images are a little noisier due to random
sampling, I have included the raytraced images from Assignment 1 to illustrate
the supersamples antialiasing. The code in the pathtracer is exactly the same._

With the existing code base described up to this point, it was easy to implement
antialiasing by supersampling the pixel values. Instead of casting the same ray
through the center of the pixel with each iteration, the direction of the ray
within the bounds of the pixel were determined randomly in each iteration, and
the computed intersection illumination values were averaged over the entire series
of iterations.

Compare the following two images. All input and scene data was identical between
the two, except the second version included supersampling of the pixels. You can 
see how smooth the edges are on the spheres and cubes in this version. While there 
are clear "jaggies" in the above version, the below version has none and even 
corrects for tricky edge intersection cases in the corners of the box. 

![Non-Antialiased](https://raw.github.com/rarietta/Project1-RayTracer/master/PROJ1_WIN/565Raytracer/README_images/005_phong_illumination_with_soft_shadows_and_reflections.bmp)

![Antialiased](https://raw.github.com/rarietta/Project1-RayTracer/master/PROJ1_WIN/565Raytracer/README_images/005_phong_illumination_with_soft_shadows_and_reflections_and_supersampled_antialiasing.bmp)

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------

To analyze the performance of the program on the GPU hardware, I decided to run
timing tests on the renders with and without stream compaction for various path
depths.

I ran the program for 50 iterations both with and without stream compaction, 
charting the results for traceDepths over the range [1,6]. As we can see from the
below graph and data chart, the runtime for the program without stream compaction
is initially a little smaller, since there is not the overhead of the scan and
scatter. However, as more waves are traced out, the implementation with stream
compaction becomes faster and the additional time required for each successive
trace depth is smaller than the last. This is because less and less rays are
traced after each bounce, since a number of the pool dies off. Alternatively, 
the version without stream compaction increases linearly with the traceDepth and
quickly surpasses the stream compaction version w.r.t. runtime.

If this was charted over higher and higher traceDepths, we would see the stream
compaction implementation flatten out while the non-compacted version would
continue to climb.

![Chart](https://raw.github.com/rarietta/Project2-PathTracer/master/PROJ1_WIN/565Raytracer/README_images/StreamCompactionVsNot__Chart.bmp)

![Graph](https://raw.github.com/rarietta/Project2-PathTracer/master/PROJ1_WIN/565Raytracer/README_images/StreamCompactionVsNot__Graph.bmp)
 
-------------------------------------------------------------------------------
Runtime Video
-------------------------------------------------------------------------------

Unfortunately, since I was working in Moore 100, I was unable to download or
utilize and screen capture video software for producing runtime videos.

-------------------------------------------------------------------------------
BASE CODE TOUR:
-------------------------------------------------------------------------------
The main files of interest in this prooject, which handle the ray-tracing
algorithm and image generation, are the following:

* raytraceKernel.cu contains the core pathtracing CUDA kernel

* intersections.h contains functions for geometry intersection testing and
  point generation

* interactions.h contains functions for ray-object interactions that define how
  rays behave upon hitting materials and objects
	  
* sceneStructs.h, which contains definitions for how geometry, materials,
  lights, cameras, and animation frames are stored in the renderer.

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
* RGB (float r) (float g) (float b)					//diffuse color
* SPECX (float specx)									//specular exponent
* SPECRGB (float r) (float g) (float b)				//specular color
* REFL (bool refl)									//reflectivity flag, 0 for
  no, 1 for yes
* REFR (bool refr)									//refractivity flag, 0 for
  no, 1 for yes
* REFRIOR (float ior)									//index of refraction
  for Fresnel effects
* SCATTER (float scatter)								//scatter flag, 0 for
  no, 1 for yes
* ABSCOEFF (float r) (float b) (float g)				//absorption
  coefficient for scattering
* RSCTCOEFF (float rsctcoeff)							//reduced scattering
  coefficient
* EMITTANCE (float emittance)							//the emittance of the
  material. Anything >0 makes the material a light source.

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

For meshes, note that the base code will only read in .obj files. For more 
information on the .obj specification see http://en.wikipedia.org/wiki/Wavefront_.obj_file.

An example of a mesh object is as follows:

OBJECT 0
mesh tetra.obj
material 0
frame 0
TRANS       0 5 -5
ROTAT       0 90 0
SCALE       .01 10 10 
