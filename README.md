-------------------------------------------------------------------------------
CIS565: Project 2: CUDA Pathtracer Fall 2013
-------------------------------------------------------------------------------
Results
-------------------------------------------------------------------------------
![alt tag](https://raw.github.com/mchen15/Project2-Pathtracer/master/renders/screenshot7.png)
![alt tag](https://raw.github.com/mchen15/Project2-Pathtracer/master/renders/screenshot5.png)
![alt tag](https://raw.github.com/mchen15/Project2-Pathtracer/master/renders/screenshot1.png)
![alt tag](https://raw.github.com/mchen15/Project2-Pathtracer/master/renders/screenshot2.png)
![alt tag](https://raw.github.com/mchen15/Project2-Pathtracer/master/renders/screenshot3.png)
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

Overall, here are the features I have implemented so far:
* Full global illumination
* Emittance and color accumulation
* Jittered supersampled antialiasing
* Parallelization by ray with stream compaction
* Perfect specular reflection
* Interactive camera
* Translational motion blur
* OBJ Mesh loading and rendering
* Interactive depth of field
* Simple refraction model

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
Stream compaction and Number of bounces comparison (800x800):
	
	Ray Parallelization with Stream Compaction: 
			Bounce 5:  8fps
			Bounce 10: 7fps
			Bounce 20: 5fps
	Ray Parallelization without Stream Compaction: 
			Bounce 5:  9fps
			Bounce 10: 5fps
			Bounce 20: 3fps

-------------------------------------------------------------------------------
THIRD PARTY CODE & LINKS
-------------------------------------------------------------------------------
	Tiny Obj Loader: 
		https://github.com/syoyo/tinyobjloader
	Thrust Stream Compaction: 
		http://thrust.github.io/doc/group__stream__compaction.html#gaf01d45b30fecba794afae065d625f94f
	Thrust Device Pointers: 
		http://docs.thrust.googlecode.com/hg/classthrust_1_1device__ptr.html
	CUDA Parallel Prefix Sum: 
		http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
	Project Demo Video: 
		https://github.com/mchen15/Project2-Pathtracer/blob/master/renders/Path%20Tracer%20Demo.mp4

-------------------------------------------------------------------------------
TODO
-------------------------------------------------------------------------------
* Squash bugs!
* Find ways to achieve convergence faster.
* Additional BRDF models.
* Bounding boxes for triangle meshes.
