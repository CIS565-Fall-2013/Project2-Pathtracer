#CUDA-based GPU Pathtracer
This project implements an path tracer on GPU using CUDA.
It has the following features:
 1. Diffuse interreflectance
 2. Supersampling
 3. Interactive camera
 4. Depth of field effect
 5. OBJ model rendering support


##Implementation Details:
 **This work was extended from my previous project, which is a CUDA-based ray tracer.** 
 To make it have a better interactivity, the tracing of an eye ray is broken into 
 multiple kernel invocations, one for each tracing depth.
 Furthermore, the ray-triangle intersection test is accelerated with the use of Bounding Box.  
 
 **Two ray-termination schemes are employed: Russian Roulette and predefined depth limit.**  
 Since recursive invocations within CUDA kernels are slow, and patht racing is inherently a recursive operation, this path tracer traces the rays iteratively and builds a stack to mimic the recursive behavior.
 To prevent the graphics memory from exhaustion, a predefined tracing depth limit is set.  
 Within the depth limit, Russian Roulette is used to determine if a ray should terminate or not. Currenlty 50% termination rate is used.
 
 **For stream compaction, an index array is constructed, each element stores the index of each pixel, that is, each eye ray.**  
  During the path tracing, the index values of 
  rays deemed terminated are replaced with -1, which will then be remove from the index array when stream compaction is performed. With 50% termination rate at each depht,
  the travese time could be greatly reduced, especially on enclosed scenes, where rays bounce around mulitple times.
  
 **For better sampling efficiency, direct illumination and indirect illumination sampling are separated.**
 
 **Supersampling is done by sampling each pixel four times.**  
  A 2x2 rotated grid pattern is used to yield better results than the normal grid pattern.  
  Since this is a path tracer, the sampling results are averaged without special care.
 
 **Depth of field effect is realized by randomly offsetting the eye position at the start of each iteration.**
 
##performance:
  
  