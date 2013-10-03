-------------------------------------------------------------------------------
CIS565: Project 2: CUDA Pathtracer
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------
Due Wednesday, 10/02/13
-------------------------------------------------------------------------------
Qiong Wang
-------------------------------------------------------------------------------


-------------------------------------------------------------------------------
INTRODUCTION
-------------------------------------------------------------------------------
This is a fast GPU path tracer initialized by a CUDA ray tracer with ray parallelization. It is one project of CIS 565 GPU Programming course at University of Pennsylvania.

The basic algorithm of the path tracing of this project can be summarized as:

1. Build a pool of rays that need to be tested;
2. Construct an image that each color can be accumulated;
3. Launch a kernel to trace one bounce of a ray;
4. Add any new rays to the pool after bounce and check whether the ray hits the light source or not;
5. Remove terminated rays from the pool;
6. Repeat the third step until no ray left in the pool.

Since the number of the ray in the ray pool decreases we need fewer blocks per grid when launching the kernel. Hence, we can have a quite fast execution speed for the path tracer.


-------------------------------------------------------------------------------
FEATURES IMPLEMENTED
-------------------------------------------------------------------------------
Basic features:

* Full global illumination (including soft shadows, color bleeding, etc.) by pathtracing rays through the scene. 
* Properly accumulating emittance and colors to generate a final image
* Supersampled antialiasing
* Parallelization by ray instead of by pixel via string compaction
* Perfect specular reflection

Optional features:

* Translational motion blur
* Depth of field
* Fresnel-based Refraction, i.e. glass (Still tuning)


-------------------------------------------------------------------------------
SCREENSHOTS OF THE FEATURES IMPLEMENTED
-------------------------------------------------------------------------------
* Global illumination

![ScreenShot](https://raw.github.com/GabriellaQiong/Project2-Pathtracer/master/10021534.PNG)

* Properly accumulating emittance and colors to generate a final image

![ScreenShot](https://raw.github.com/GabriellaQiong/Project2-Pathtracer/master/10021636.PNG)

* Antialiasing (Stochastic method)

![ScreenShot](https://raw.github.com/GabriellaQiong/Project2-Pathtracer/master/anti-aliasing.PNG)

* Perfect specular reflection

![ScreenShot](https://raw.github.com/GabriellaQiong/Project2-Pathtracer/master/10021740.PNG)

* Translational motion blur

![ScreenShot](https://raw.github.com/GabriellaQiong/Project2-Pathtracer/master/10022256.PNG)

* Depth of field

With small radius for the circle of confusion

![ScreenShot](https://raw.github.com/GabriellaQiong/Project2-Pathtracer/master/10022047.PNG)

With big radius for the circle of confusion

![ScreenShot](https://raw.github.com/GabriellaQiong/Project2-Pathtracer/master/10022141.PNG)


-------------------------------------------------------------------------------
VIDEOS OF IMPLEMENTATION
-------------------------------------------------------------------------------

This is the video of the rendering process of the path tracer.

[![ScreenShot](https://raw.github.com/GabriellaQiong/Project2-Pathtracer/master/videoScreenShot.PNG)](http://www.youtube.com/watch?v=GcbRUaLgz5A)

The youtube link is here if you cannot open the video in the markdown file: http://www.youtube.com/watch?v=GcbRUaLgz5A


-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
Here is the table for the performance evaluation when changing the size of tile and with stream compaction or not. 

| tileSize  |  with compaction  |      time for running 10 iteration      |  approximate fps  |
|:---------:|:-----------------:|:---------------------------------------:|:-----------------:|
|     8     |        yes        |               0 : 22.72                 |       0.440       |
|    10     |        yes        |               0 : 25.97                 |       0.385       |
|     8     |        no         |               0 : 23.39                 |       0.428       |
|    10     |        no         |               0 : 27.47                 |       0.364       |

We can easily find that when the tile size become larger the fps decrease a little somehow. The fps with stream compaction is higher than that without stream compaction.

-------------------------------------------------------------------------------
REFERENCES
-------------------------------------------------------------------------------
* Snell's law: http://en.wikipedia.org/wiki/Snell%27s_law

* Internal Reflection: http://en.wikipedia.org/wiki/Total_internal_reflection
 
* Fresnel Equation: http://en.wikipedia.org/wiki/Fresnel_equations

* Russian Roulette rule from Peter and Karl: https://docs.google.com/file/d/0B72qrSEH6CGfbFV0bGxmLVJiUlU/edit

* General Phong Specular Model: http://en.wikipedia.org/wiki/Phong_reflection_model

* Depth of Field: http://http.developer.nvidia.com/GPUGems/gpugems_ch23.html

* Stochastic Sampling: http://pages.cpsc.ucalgary.ca/~mario/courses/591-691-W06/PR/3-ray-tracing/3-advanced/readings/Cook_Stochastic_Sampling_TOG86.pdf

* Usage of thrust in stream compaction: http://stackoverflow.com/questions/12201446/converting-thrustiterators-to-and-from-raw-pointers/12236270#12236270

-------------------------------------------------------------------------------
ACKNOWLEDGEMENT
-------------------------------------------------------------------------------
Thanks a lot to Patrick and Liam for the preparation of this project. And special thanks to Liam who helped us debugging so late till 4:30 am in the morning! Thank you all :)
