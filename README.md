-------------------------------------------------------------------------------
CIS565: Project 2: CUDA Pathtracer
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
This is an implementation of path tracing rendering algorithm on the GPU, using CUDA.

Path tracing is considered to be the best known solution to the classic rendering equation. 

The scary looking integral in the rendering equation is approximated beautifully by Monte Carlo sampling which also implies that in order to reduce our approximation errors, we need to take a really large number of samples.

-------------------------------------------------------------------------------
CONTENTS:
-------------------------------------------------------------------------------
The Project2 root directory contains the following subdirectories:
	
* src/ contains the source code for the project. Both the Windows Visual Studio solution and the OSX makefile reference this folder for all source; the base source code compiles on OSX and Windows without modification.
* scenes/ contains some example scene files that were used to create the renders which can be found in the "renders/" subdir
* renders/ contains a few renders showcasing the abilities of my path tracer 
* PROJ1_WIN/ contains a Windows Visual Studio 2010 project and all dependencies needed for building and running on Windows 7.
* PROJ1_OSX/ contains a OSX makefile, run script, and all dependencies needed for building and running on Mac OSX 10.8. 
* PROJ1_NIX/ contains a Linux makefile for building and running on Ubuntu 
  12.04 LTS. Note that you will need to set the following environment
  variables: 
    
  - PATH=$PATH:/usr/local/cuda-5.5/bin
  - LD_LIBRARY_PATH=/usr/local/cuda-5.5/lib64:/lib

-------------------------------------------------------------------------------
FEATURES:
-------------------------------------------------------------------------------
Path tracing lends itself naturally to a lot of generally "sought" after features in rendering.
We can implement these features at almost no extra cost. The following is the list of features of my path tracer. 

Features that you get for free with path tracing
* Supersampling
* Area lighting and soft shadows
* Full global illumination

Features that I implemented additionally
* A generic BRDF model that supports diffuse, specular and transmittive surfaces (or a combination of those)
* Fresnel reflections and refractions
* Translational motion blur
* Depth of field based on aperture and focal plane
* Procedural textures
* User interaction with the scene

-------------------------------------------------------------------------------
FEATURES (in a bit more detail)
-------------------------------------------------------------------------------

* BRDF model : Here is an example of results from a scene having surfaces that are both diffuse and specular at the same time.
The way the BRDF deals with it is a russian roulette based on probabilities. Simply put, if a material is 70% diffuse, at every iteration, draw a random number from 0 to 1, and if its less than 0.7, then calculate diffuse reflection, else specular reflection.
![alt tag](https://raw.github.com/vimanyu/Project2-Pathtracer/master/renders/pool.0.bmp)

* Fresnel reflection and refractions
This physically based model tells you the amount of reflection or transmittance that takes place at given incoming directions of rays on the material.
Notice that the sphere at the bottom refracts and reflects based on angles. If we do not use fresnel, then we will see either of those phenomena on the sphere but not both.
![alt tag](https://raw.github.com/vimanyu/Project2-Pathtracer/master/renders/refr_refl_fresnel.bmp)

* Translational motion blur
The idea is to do temporal sampling in addition to spatial sampling. Given a frame, go back in time, and linearly (could be cubic or higher order) interpolate between the positions of geometry now and then.
![alt tag](https://raw.github.com/vimanyu/Project2-Pathtracer/master/renders/motion_blurred.bmp)

* Depth of field
Based on the aperture and focal plane specified as camera attributes in the config file.
The idea is to jitter the rays in such a manner that the jittering is least at the focal plane region.
![alt tag](https://raw.github.com/vimanyu/Project2-Pathtracer/master/renders/refr_refl_diff_DOF.bmp)

* Procedural textures
I cannot be more fascinated by the beauty of procedural textures (two things that amazed me the most during this assignment are Monte Carlo sampling and Perlin Noise. 
So beautiful!). I have implemented a few basic procedural textures for the path tracer -Vertical stripes texture,Horizontal stripes texture, Checkerboard texture ,Marble texture , and a generic Perlin texture
And then, it was time to play with these parameters and just appreciate mathematics.
![alt tag](https://raw.github.com/vimanyu/Project2-Pathtracer/master/renders/perlin_texture.bmp)
As you can see, this was too much fun. So here is an image with everything textured using perlin textures.
![alt tag](https://raw.github.com/vimanyu/Project2-Pathtracer/master/renders/perlin_earth.bmp)

* User interaction with the path tracer
According to me, this was the coolest part of the assignment. User interaction with photorealistic rendering algorithms are best avoided, but hey, we have the GPU! And we have inherently formulated the path tracing algorithm in a parallel and iterative fashion. This makes it idea for user interaction.
There are three types of user interactions supported.

1. Interactive "look-at" depth of field: The user can focus on any part of the image by using the right mouse button. Just click and let it be in focus! 
2. Interactive positioning of objects on screen: LMB on any object in the scene, and use the following keys to move the object
   + x : move in positive x-direction
   + X : move in negative x-direction
   + y : move in positive y-direction
   + Y : move in negative y-direction
   + z : move in positive z-direction
   + Z : move in negative z-direction
3. Interacitve editing of procedural textures : Press "t" on the keyboard to enter/exit the world of procedural textures. This feature grew out of necessity. It was taking me quite a while to come up with good settings for the "attempted" earth procedural texture above. So, I thought it might be covenient to do this visually.
Finally, I ended up implementing this feature as an integral part of path tracer. We can think of it like this. If you don't like the way, your procedural textures are rendering in your path tracer, you can hit 't' and set your textures as per your liking, and then hit 't' again to path trace with these textures.


-------------------------------------------------------------------------------
README
-------------------------------------------------------------------------------
All students must replace or augment the contents of this Readme.md in a clear 
manner with the following:

* A brief description of the project and the specific features you implemented.
* At least one screenshot of your project running.
* A 30 second or longer video of your project running.  To create the video you
  can use http://www.microsoft.com/expression/products/Encoder4_Overview.aspx 
* A performance evaluation (described in detail below).

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
The performance evaluation is where you will investigate how to make your CUDA
programs more efficient using the skills you've learned in class. You must have
performed at least one experiment on your code to investigate the positive or
negative effects on performance. 

One such experiment would be to investigate the performance increase involved 
with adding a spatial data-structure to your scene data.

Another idea could be looking at the change in timing between various block
sizes.

A good metric to track would be number of rays per second, or frames per 
second, or number of objects displayable at 60fps.

We encourage you to get creative with your tweaks. Consider places in your code
that could be considered bottlenecks and try to improve them. 

Each student should provide no more than a one page summary of their
optimizations along with tables and or graphs to visually explain any
performance differences.

-------------------------------------------------------------------------------
THIRD PARTY CODE POLICY
-------------------------------------------------------------------------------
* Use of any third-party code must be approved by asking on the Google group.  If it is approved, all students are welcome to use it.  Generally, we approve use of third-party code that is not a core part of the project.  For example, for the ray tracer, we would approve using a third-party library for loading models, but would not approve copying and pasting a CUDA function for doing refraction.
* Third-party code must be credited in README.md.
* Using third-party code without its approval, including using another student's code, is an academic integrity violation, and will result in you receiving an F for the semester.

-------------------------------------------------------------------------------
SELF-GRADING
-------------------------------------------------------------------------------
* On the submission date, email your grade, on a scale of 0 to 100, to Liam, liamboone+cis565@gmail.com, with a one paragraph explanation.  Be concise and realistic.  Recall that we reserve 30 points as a sanity check to adjust your grade.  Your actual grade will be (0.7 * your grade) + (0.3 * our grade).  We hope to only use this in extreme cases when your grade does not realistically reflect your work - it is either too high or too low.  In most cases, we plan to give you the exact grade you suggest.
* Projects are not weighted evenly, e.g., Project 0 doesn't count as much as the path tracer.  We will determine the weighting at the end of the semester based on the size of each project.

-------------------------------------------------------------------------------
SUBMISSION
-------------------------------------------------------------------------------
As with the previous project, you should fork this project and work inside of your fork. Upon completion, commit your finished project back to your fork, and make a pull request to the master repository.
You should include a README.md file in the root directory detailing the following

* A brief description of the project and specific features you implemented
* At least one screenshot of your project running, and at least one screenshot of the final rendered output of your pathtracer
* Instructions for building and running your project if they differ from the base code
* A link to your blog post detailing the project
* A list of all third-party code used
