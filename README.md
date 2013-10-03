-------------------------------------------------------------------------------
CIS565: Project 2: CUDA Pathtracer
-------------------------------------------------------------------------------
Fall 2012
-------------------------------------------------------------------------------
Due Friday, 10/12/2012
-------------------------------------------------------------------------------


-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
This is GPU-Based Path tracer! Based on the code of the last project ray-tracer, the path tracer has some new features that make it looks really cool!

Beyond the basic specular, refraction, as well as soft-shadow or anti-aliasing, which is the natural properties of the path tracer, it also supports super-sampling anti-aliasing. Instead of these, Motion Blur and Texture Map are also included in this project.


-------------------------------------------------------------------------------
WHAT ACTUALLY MAKE TROUBLE:
-------------------------------------------------------------------------------
In this project, the path-tracing part is really simple and straight forward. The only thing that make a lot of trouble is the random-number generator. As the thrust random number generator is so much rely on the seed, so how to calculate seed to make it as stochastic as possible is important.

The four key values of calculating seed are: iterations, index, raypool index and the amount of bounce. I tried all sorts of combinations of these three elements and + - * / to try to compute the seed. Finally it turns out that time*index+raypoolidx*bounce is best suitable for the computation of seed. 

-------------------------------------------------------------------------------
STREAM COMPACTION
-------------------------------------------------------------------------------
I¡¯d tried both stream compaction with thrust::exclusive_scan as well as using thrust::remove_if function. As far as I tested, the later one is a little bit faster(not obvious) and easy to implement. The if condition of the remove_if is the ray is terminated. And after the removing process, the pool is thereby shrinked

-------------------------------------------------------------------------------
Texture Map
-------------------------------------------------------------------------------
Because of the time is limited (linkedin phone interview took me so long..) and I under-estimated the complexity of using BMP file for texture, it make me a lot of trouble when wrap the code with EasyBMP, a third party library that we had used in CIS560 Computer Graphics course. Anyway, I¡¯d fixed it after all, but do not have time to make it better support sphere UV map. I¡¯ll do this after the deadline. 

The cube texture-map is support, where the cube surface should align with the bmp boundary.

-------------------------------------------------------------------------------
Motion Blur
-------------------------------------------------------------------------------
The motion blur is even interesting then I thought it would be. There are many ways to calculating a motion blur, and I use offset=v*t^2 to simulate an accelerate object. This make a cool trail after the object, if you refer to my rendering result.

-------------------------------------------------------------------------------
THIRD PARTY CODE
-------------------------------------------------------------------------------
I¡¯d introduced the EasyBMP third party code in my project to analyze BMP file.

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
Again, I don¡¯t have much time for performance evaluation, and I¡¯ll update this evaluation after the due date. 

As far as I used the ray-parallelization technique, it is 1D case so the only factor that may affect the speed is the how many threads is working simultaneously. As far as I do, I set it to the max(512) and when calculating the block amount I just use raypoolsize/threadsnum. Anyway I tried threadperblock=256 and it is a little bit slower. I think the reason is that when there are more blocks, it is more likely that some block or wrap is not working while others are still doing.

In case of bank share, I¡¯ll try it later on, and update the test report as soon as possible

-------------------------------------------------------------------------------
RESULT
-------------------------------------------------------------------------------
Youtube link is here
http://www.youtube.com/watch?v=d4a8aPpC-BM&feature=youtu.be

Screenshot is Here

