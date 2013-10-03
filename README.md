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
  Here the object position was varied according to linear interpolated values of its start and end 
  position during each iteration

* Snell-based Refraction, i.e. glass
  Here the basic Snells refraction was implemented using the Glm::refract function and a probability 
  was assigned to get both reflections and refractions on the material 

* OBJ Mesh loading and rendering with Bounding boxes
  The Obj loading was implemented earlier for ray tracer , but in this assignmnet the maximum and minimum
  co-ordinates of the mesh were calculated and sent to the GPU and hence a max min box intersection test 
  was done

* Interactive camera
  A interactive mouse has been implemented by imagining a camera attched to the sphere and hence the new 
  positions are calculated based on that. To do this feature CIS 563 cloth simulation assignmnet was referred,
  and it partially acted as a guide for path tracer interaction feature. 

* Depth of field
  DOF was implemneted by selecting a depth plane away from the camera and jittering the camera position during 
  each iteration.  

-------------------------------------------------------------------------------
Working Video :
-------------------------------------------------------------------------------
![Path Tracer Video](http://www.youtube.com/watch?v=bA-7rGa7juM&feature=youtu.be)
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

In this Performance evaluation, the speeds or the time elapsed for with and without stream
campaction is compared. Here initially when the number of bounces are less, without stream 
campaction would work better because of thrust remove_if overhead calculations for smaller
bounces.
Whereas it can be easily seen how the performance improves when the number of threads are reduced 
when we have more number of bounces and hence more number of rays that can be removed from the raypool


