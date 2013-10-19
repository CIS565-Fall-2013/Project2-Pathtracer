#CIS 565 -- Fall 2013

#Project 2 : CUDA Pathtracer
------

##Overview

------
##Features

The features that will be discussed below are those in addition to the ones implemented in the raytracer.  The discussion section will discuss some of the findings and observations when changing from the raytracer to pathtracer.

###OBJ Mesh Loading

We have implemented a simple obj mesh loader that ignores vertex and texture normals.  We have designed it such that the mesh data is stored in separate structs that have indices to point to the start and end index of the mesh's faces in the serialized face array.  We did this because the GPU requires serialized memory.  Instead of adding complex structs and serializing them before sending them over to the GPU, we have opted for a simpler structure with more lookups.

As of now, the loader works; however, we have had trouble with color attenuation on the loaded mesh.  As seen below, the normals and intersection passes seem to be fine.  However, when running the renderer, the loaded object does not follow the normal shading rules.  We have already considered the possibility of the ray internally reflecting, but increasing the offset of the newly cast ray does not seem to remedy this problem.

###Stream Compaction

We have implemented stream compaction using a shared memory verion of naive exclusive prefix sum in hopes of gaining performance from utilizing on-chip shared memory. This version requires a tail-recursion like method to reassemble the final output array from the different blocks by adding partial sums on the recall. 

Much of our implementation was aided and informed by Ch. 39 (Parallel Prefix Sum with CUDA) of GPU Gems 3.  The major modification to the code presentation is that we copy over the buffered array in every iteration of the scan step.  We found that following their pseudocode resulted in inaccurate output arrays because the entries would be inconsistent in the buffer on the swap iteration.


###Fresnel Reflection and Refraction

We have implemented perfect relection, perfect refraction (Snell's law) and Fresnel Reflection/Refraction using Schlick's approximation. Due to its tree like approach to calculating reflection and refraction, unpolarized Fresnel has the potential to grow indefinitely. This would not be helpful when trying to use stream compaction.  Thus, we have used Schlick's approximation to approximate the likelihood of refraction and reflection on each iteration and bounce of a ray.  In line with the motivation behind path tracing, it relies on successive iterations to converge to a physically correct solution.

------
##Discussion

While moving from a raytracer to a pathtracer, there were certain aspects of the code that produced interesting results.  Here, we will present some of these hurdles, the ways in which we belief these things came about, and the subsequent fixes.

###Artifacts in Depth of Field

After implementing naive pathtracing, we added depth of field.  When rendering diffuse objects, there were persistent artifacts that showed up on the spheres, as shown below. 

Debugging showed that it resulted from jittering the position of the camera to achieve the effect.  Interestingly, we found that, with the correct math to compute depth of field, we had odd renderings:

1.	When changing the depth of field, the image would blur in a biased direction and then "restabilize" and become clear in a shifted direction. We found that this happened regardless of the maximum trace depth.


2.	When changing the aperture, the spheres and the image would warp in a biased zig-zag pattern that was reproduced every time.

Because of this, we decided to change the seed input to the hashing function that produced random numbers to jitter our camera position. Originally, the seed was the product of the iteration and the pixel index (iterations * index); however, from the first point, we believe that increasing the seed by a factor of the time may have biased the generation to jitter less in a particular position as large float numbers have less precision from each other as they reach a limit. Thus, we changed the hash to only be seeded by the index, and that, effectively, fixed our problem.


###Obj Mesh loading



------
##Performance Analysis

###Effects of Stream Compaction

------
## Acknowledgements and Citations
