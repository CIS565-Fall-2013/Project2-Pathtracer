-------------------------------------------------------------------------------
<center>CIS565: Project 2: CUDA Pathtracer
-------------------------------------------------------------------------------
<center>Fall 2013
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
<center>INTRODUCTION:
-------------------------------------------------------------------------------
This project is a CUDA based path tracer that utilizes the GPU to generate path traced images very quickly. The project reads in a text file that specifies the materials, objects, and camera for the scene. For my path tracer, I've implemented intersection testing for spheres and cubes, full global illumination, properly accumulating emittance, supersampled antialiasing, parallelization by ray with my own coded stream compaction, perfect specular reflection, depth of field, an interactive camera, and fresnel refraction. Here’s two sample renders:

<center>![path tracer](https://raw.github.com/josephto/Project2-Pathtracer/master/renders/FinalRender.jpg "refractive spheres")

<center>![path tracer](https://raw.github.com/josephto/Project2-Pathtracer/master/renders/PathTracerRender2.jpg "glass sphere")

-------------------------------------------------------------------------------
<center>SREENSHOTS:
-------------------------------------------------------------------------------

Because my current CUDA compatible computer doesn't have the necessary software installed to screen capture a video of my path tracer in process, here are a bunch of screen shots illustrating the image converging.

<center>![path tracer](https://raw.github.com/josephto/Project2-Pathtracer/master/renders/Screenshot2.jpg "screenshots")

<center>![path tracer](https://raw.github.com/josephto/Project2-Pathtracer/master/renders/Screenshot4.jpg "screenshots")

<center>![path tracer](https://raw.github.com/josephto/Project2-Pathtracer/master/renders/Screenshot6.jpg "screenshots")

<center>![path tracer](https://raw.github.com/josephto/Project2-Pathtracer/master/renders/Screenshot8.jpg "screenshots")

<center>![path tracer](https://raw.github.com/josephto/Project2-Pathtracer/master/renders/Screenshot9.jpg "screenshots")

<center>Final Render

<center>![path tracer](https://raw.github.com/josephto/Project2-Pathtracer/master/renders/Screenshot.jpg "screenshots")


-------------------------------------------------------------------------------
<center>PERFORMANCE REPORT:
-------------------------------------------------------------------------------

Here's a table with some performance analysis that I conducted on my code. I recorded how many seconds per iteration it takes for my path tracer with stream compaction and without stream compaction.

Maximum Ray Depth | With Stream Compaction | No Stream Compaction
------------------|------------------------|---------------------
1    |  0.036 sec/iter | 0.035 sec/iter
2    |  0.066 sec/iter | 0.052 sec/iter
4    |  0.121 sec/iter | 0.089 sec/iter
8    |  0.201 sec/iter | 0.160 sec/iter
16    |  0.291 sec/iter | 0.268 sec/iter
32    |  0.399 sec/iter | 0.336 sec/iter
64    |  0.548 sec/iter | 0.345 sec/iter
128    |  0.825 sec/iter | 0.357 sec/iter
256    |  1.379 sec/iter | 0.383 sec/iter
512    |  2.482 sec/iter | 0.430 sec/iter

As you can see, my stream compaction actually ended slowing down my path tracer. This is probably because I used a lot of memcopy to and from the GPU which seriously slowed down my stream compaction. I plan on going back and re-implementing the stream compaction so that it actually achieves a significant speed up per iteration.

