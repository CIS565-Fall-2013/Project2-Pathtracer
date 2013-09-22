#CUDA-based GPU raytracer 
This project implements a GPU raytracer using nVidia CUDA.  
  
The following featues are implemented:

* Parallel Ray casting
* Ray-sphere and Ray-Triangle intersections 
* Phong reflection model
* Area lights and soft shadows
  1. Area lights are sampled in 4x4 grid resolutin; each sample ray is jitterd with randomness to decrease blocking effects.  
  2. cuRAND is used to generate the random values 
* mirror reflection  
  1. An iterative approach is used to gather reflectance energy without the use of costly recursive approach is avoided.  
  2. The default depth of reflection is 4.
* Wavefront obj model rendering

#Rendering Snapshot
 * Recording of execution: http://www.youtube.com/watch?v=46AOCXCAYd8
 * 30-sec. rendered animation: http://www.youtube.com/watch?v=SMAJNEoWPoc
 * Snapshot of execution: https://raw.github.com/otaku690/Project1-RayTracer/master/execute_snapshot.png
 * Snapshot of rendered image: https://raw.github.com/otaku690/Project1-RayTracer/master/render_results.png

#Performance evaluation
 setting: 1024x768 res., 4x4 shadow & lighting samples, 4 bounces of reflection
## Use of shared memory
  Given that each ray test the geometries in the same sequence, we can load the geometries into shared memory before testing them, one at a time.  
    
  This approach increase the performance by about 4~5%. However, further investigation is needed because some blocks would display anormally.
## Block size 
 * without using shared memory
  * 8x8: ~26 sec. per frame
  * 16x16: 25 sec. per frame
  * 32*32: 25 sec. per frame
 * using shared memory
  * 8x8: ~25 sec. per frame
  * 16x16: 24 sec. per frame
  * 32*32: 24 sec. per frame
 * There are no significant difference in 8x8, 16x16 and 32x32 blcok sizes
 



#Third-party code
 * GLM object loader from [Nate Robins](https://user.xmission.com/~nate/tutors.html)

#Third-party libraries
  * [GLEW](http://glew.sourceforge.net/)
  * [Freeglut](http://freeglut.sourceforge.net/)
  * [FreeImage](http://freeimage.sourceforge.net/)
  * [GLM](http://glm.g-truc.net/0.9.4/index.html)  
  
#Development Environment
* Visual Studio 2012 on Windows 7
* How to build
  * Make sure the project has correct INCLUDE and LIBRARY Pathes of the above libraries.
  * Make sure the CUDA 5.5 is selected in the [Build Customization] Setting.
  * Make sure the compute_10/sm_10 compute version is remvoed from the Code Generation setting under the [0CUDA C/C++] setting
  * Place the needed DLL inside the execution folder.
  * Place testScene.scene and model venusv.obj & box.obj in the execution folder.
  * You are good to go.