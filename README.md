-------------------------------------------------------------------------------
CIS565: Project 2: CUDA Pathtracer
-------------------------------------------------------------------------------
Fall 2013
![alt tag](https://raw.github.com/mchen15/Project2-Pathtracer/master/renders/screenshot1.png)
![alt tag](https://raw.github.com/mchen15/Project2-Pathtracer/master/renders/screenshot2.png)
-------------------------------------------------------------------------------
PROJECT OVERVIEW
-------------------------------------------------------------------------------
For this project, I built off of my CUDA ray tracer in order to implement a parallel path tracer featuring full global illumination,
color accumulation, supersampled antialiasing, parallelization by ray instead of pixel via stream compaction, perfect specular reflection,
translational motion blur, OBJ mesh loading, and interactive camera. 

One of the first tasks I tackled when working on the project was the ray parallelization. In my ray tracer, I had done parallelization by pixel. 
While this is intuitive, rays tend to terminate at different depths, which leads to many idle threads. One way of fixing this issue is to instead, 
have each thread represent a ray at one particular depth. In other words, a breadth first instead of depth first approach is used. The most difficult
part of this was to properly allocate a pool for each ray and figuring out how to keep track of the active rays. The cuda thrust library provides a nice
way to help with stream compaction. I have included the links that helped me a lot when thinking about how to structure my code. 

Once the parallelization of rays step had been done, I moved on to implementing the core path tracing algorithm. The trickiest part in this portion was
computing the diffuse scattering direction. Choosing a good seed has a drastic effect on how the outcome image looks. Once the path tracer was functional, I added
more features including motion blur (which is done via translating the geometry across certain number of frames at a certain interval), OBJ mesh loading (using
the third party software linked below), and camera controls via keyboard.

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
*Stream compaction comparison:

**Ray Parallelization with Stream Compaction:

**Ray Parallelization without Stream Compaction:


*Number of bounces comparison:

**Bounce 5:

**Bounce 10:

**Bounce 20:



-------------------------------------------------------------------------------
THIRD PARTY CODE & LINKS
-------------------------------------------------------------------------------
Tiny Obj Loader: https://github.com/syoyo/tinyobjloader
Thrust Stream Compaction: http://thrust.github.io/doc/group__stream__compaction.html#gaf01d45b30fecba794afae065d625f94f
Thrust Device Pointers: http://docs.thrust.googlecode.com/hg/classthrust_1_1device__ptr.html
CUDA Parallel Prefix Sum: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html

Project Demo Video: https://github.com/mchen15/Project2-Pathtracer/blob/master/renders/Path%20Tracer%20Demo.mp4
-------------------------------------------------------------------------------
SELF-GRADING
-------------------------------------------------------------------------------
* On the submission date, email your grade, on a scale of 0 to 100, to Liam, liamboone+cis565@gmail.com, with a one paragraph explanation.  Be concise and realistic.  Recall that we reserve 30 points as a sanity check to adjust your grade.  Your actual grade will be (0.7 * your grade) + (0.3 * our grade).  We hope to only use this in extreme cases when your grade does not realistically reflect your work - it is either too high or too low.  In most cases, we plan to give you the exact grade you suggest.
* Projects are not weighted evenly, e.g., Project 0 doesn't count as much as the path tracer.  We will determine the weighting at the end of the semester based on the size of each project.

-------------------------------------------------------------------------------
TODO
-------------------------------------------------------------------------------
* Squash bugs!
* Additional BDSF such as refraction
* Additional BRDF models
* Bounding boxes for triangle meshes
