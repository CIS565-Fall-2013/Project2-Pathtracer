-------------------------------------------------------------------------------
CIS565: Project 2: CUDA Pathtracer
-------------------------------------------------------------------------------
For Project 2, I extended my raytracer into a full-blown pathtracer. The effort was made 
easier by the fact that much of the work had been done in the raytracer phase itself, 
including implementation of the geometrical intersection tests. Antialiasing and Motion Blur, 
which I implemented for the raytracer (but did not make it in time for submission), 
became default components of the pathtracer and they satisfy the "two extra features" requirement. 

-------------------------------------------------------------------------------
PATH TRACING:
-------------------------------------------------------------------------------
As in the case of the raytracer, rays are projected into the scene through the projection plane. The ray bounces 
around in the scene, accumulating colours of all the objects that it hits, eventually either hitting the light, 
flying off into the darkness behind the camera or dying after x Bounces. If the ray hits the light, 
it will contribute its colour to the pixel through which it was traced. Otherwise, it contributes noise (black). When 
we do this a sufficient number of times, we get a result close to what the ground truth would be.

-------------------------------------------------------------------------------
IMPLEMENTATION DETAILS:
-------------------------------------------------------------------------------
In contrast to the raytracer, there are two crucial differences:
i. Rays are parallelized, not the pixels. This means that a thread computes the pixel colour contribution of a 
single ray.
ii. Rays can bounce around in the scene, upto a certain maximum number of bounces. Russian roulette is not employed 
to determine when a ray dies. This makes this pathtracer more heavily biased than others.

Unlike implementations of most of my peers, my pathtracer renders the image, accumulating 
the colours in the colour buffer as it goes and outputs a single, final image to the GLUT/GLFW window. 
Because of this, effects like Antialiasing, Motion Blur, Depth of Field or any other effect where the pixel 
colour results from averaging together distinct values from many iterations are "free" for me, just like in 
the case of the raytracer.

For antialiasing, I'm sampling the scene at a rate of 8 samples per pixel.

-------------------------------------------------------------------------------
FEATURES:
-------------------------------------------------------------------------------
Current
-------
As with the raytracer, the pathtracer supports sphere and cube primitives. It implements all of the required 
features and the following optional features:

* Antialiasing (Supersampled at 8x)
* Refraction with Fresnel reflectance
* Motion blur (Translational)
* My own stream compactor (a CPU/GPU hybrid; more on this below)
* Diffuse reflectance

In addition, the submission includes code for Texture Mapping. However, this code isn't ready for primetime yet. 

-------------------------------------------------------------------------------
STREAM COMPACTION
-------------------------------------------------------------------------------
I implemented stream compaction on my own using the Naive Parallel Scan method discussed by Patrick in class. 
However, even though there's nothing apparently wrong about my implementation, the scan fails for indices beyond 
65536. It so happens that log2 (65536) = 16 which is the size of a half-warp, and the algorithm involves a thread 
accessing a location 2^d - 1 spaces before its own index. I highly suspect something is amiss here. But, it could 
even turn out to be a problem with the lab machines (which have old cards with ancient drivers). 

This failure was giving an incorrect result when I was using the GPU solely for stream compaction 
(see below for screenshot). This necessitated using a hybrid approach, where the CPU would do the exclusive scan and 
the GPU would perform the actual compaction. As noted in the performance analysis document, this causes a slight drop
in performance.

Not having nSight installed on lab machines makes the whole process equivalent to shooting in the dark. If it hits, 
well and good. Otherwise, hard luck. Till now, it's been the latter.

-------------------------------------------------------------------------------
SCREENSHOTS
-------------------------------------------------------------------------------
This is the best image I have, rendered with 5000 iterations. Unfortunately, it has got a lot of artifacts:<br /> 
<img src="https://raw.github.com/rohith10/Project2-Pathtracer/master/renders/FinalRender.png" height="350" width="350"/><br />
Artifacts aside, the glass looks amazingly real. The diffuse reflectance on the gold sphere in the centre also looks good.<br />
The same scene with 2000 iterations:<br />
<img src="https://raw.github.com/rohith10/Project2-Pathtracer/master/renders/FinalRender_2000Iter.png" height="350" width="350"/><br />
<img src="https://raw.github.com/rohith10/Project2-Pathtracer/master/renders/CPUStreamCompaction.png" height="350" width="350"/><br />
Incorrect rendering of the same scene with full-GPU stream compaction:<br />
<img src="https://raw.github.com/rohith10/Project2-Pathtracer/master/renders/GPUStreamCompaction.png" height="350" width="350"/><br />
The above two images were rendered at around 3000 iterations. As is evident, when the iteration count goes up, so does the 
artifacts, which has lead me to believe that these are some sort of floating point accumulation errors.

-------------------------------------------------------------------------------
VIDEO
-------------------------------------------------------------------------------
Unfortunately, I wasn't able to use the laptop I used to make a video for the raytracer to run the pathtracer, 
because some of my cudaMalloc calls were erroring out without rhyme or reason. The SIG and Moore Lab machines don't 
have any video capturing/encoding software, and my own laptop has an AMD graphics card. So, a video could not be 
prepared for this.

-------------------------------------------------------------------------------
PERFORMANCE ANALYSIS
-------------------------------------------------------------------------------
A performance analysis was performed for this project and can be found in the root folder with the 
name Project2-PerfAnalysis. It is a Word Document. 
