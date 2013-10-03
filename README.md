-------------------------------------------------------------------------------
CIS565: Project 2: CUDA Pathtracer
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------
Due Wednesday, 10/02/13
-------------------------------------------------------------------------------


-------------------------------------------------------------------------------
DESCRIPTION :
-------------------------------------------------------------------------------
In this Project a Path tracer is created by using the concept of GLobal Illumination. In thsi technique a bunch
of rays are shooted into the scene and each ray is tracked until they hit the light, the color at every bounce 
is calculated and multiplied with the previous color. This process is repeated for every iteration ,and the color 
obtained at each iteration is averaged and added to the previous value. 

All the basic requirements such as 
* Full global illumination 
* Properly accumulating emittance and colors to generate a final image
* Supersampled antialiasing
Have been implemented 

* Parallelization by ray instead of by pixel via stream compaction 
This was done using Thrust library and the function if_remove was used to compress the ray pool by removing 
the dead rays.

* Perfect specular reflection
Was achieved by reflecting the rays when they hit the reflective surface 


The extra features that were implemented are : 

* Translational motion blur
* Snell-based Refraction, i.e. glass
* OBJ Mesh loading and rendering with Bounding boxes
* Interactive camera
* Depth of field

-------------------------------------------------------------------------------
ScreenShots :
-------------------------------------------------------------------------------
Here is the first image with the basic diffuse applied on all the materials 

![alt tag](https://raw.github.com/vivreddy/Project2-Pathtracer/master/renders/3.png)


With Reflections applied 
![alt tag](https://raw.github.com/vivreddy/Project2-Pathtracer/master/renders/4.png)

With refractions applied 
![alt tag](https://raw.github.com/vivreddy/Project2-Pathtracer/master/renders/5.png)

With Depth of Field 
![alt tag](https://raw.github.com/vivreddy/Project2-Pathtracer/master/renders/6.png)

With Motion Blur 
![alt tag](https://raw.github.com/vivreddy/Project2-Pathtracer/master/renders/8.png)

With OBJ loader and bounding boxes 
![alt tag](https://raw.github.com/vivreddy/Project2-Pathtracer/master/renders/7.png)

Other renders
![alt tag](https://raw.github.com/vivreddy/Project2-Pathtracer/master/renders/2.png)



-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------

![alt tag](https://raw.github.com/vivreddy/Project2-Pathtracer/master/renders/table.png)

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


