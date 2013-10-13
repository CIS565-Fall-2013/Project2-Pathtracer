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
I'd tried both stream compaction with thrust::exclusive_scan as well as using thrust::remove_if function. As far as I tested, the later one is a little bit faster(not obvious) and easy to implement. The if condition of the remove_if is the ray is terminated. And after the removing process, the pool is thereby shrinked

-------------------------------------------------------------------------------
Texture Map
-------------------------------------------------------------------------------
Because of the time is limited (linkedin phone interview took me so long..) and I under-estimated the complexity of using BMP file for texture, it make me a lot of trouble when wrap the code with EasyBMP, a third party library that we had used in CIS560 Computer Graphics course. Anyway, I'd fixed it after all, but do not have time to make it better support sphere UV map. I'll do this after the deadline. 

The cube texture-map is support, where the cube surface should align with the bmp boundary.

-------------------------------------------------------------------------------
Motion Blur
-------------------------------------------------------------------------------
The motion blur is even interesting then I thought it would be. There are many ways to calculating a motion blur, and I use offset=v*t^2 to simulate an accelerate object. This make a cool trail after the object, if you refer to my rendering result.

-------------------------------------------------------------------------------
THIRD PARTY CODE
-------------------------------------------------------------------------------
I'd introduced the EasyBMP third party code in my project to analyze BMP file.

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
I'd tested several factors that may have some affect on the final FPS or rendering speed. Therefore I tested them independently,
and collected the result in a excel file and draw graph that show how the FPS change according to these factors.

Firstly, I must announce that all thte tests below are within 200*200 base. I use 200*200 to save up more time on running, instead of 800*800.

By using raypool, one of the most important thing is how many threads per block, and how many blocks(absolutely is the ray pool size divided by the threads per block,
because there is only one primary hit light for each thread. 

![screenshot](https://github.com/heguanyu/Project2-Pathtracer/blob/master/performance%20analyze/rayperblockgraph.bmp?raw=true)

According to this graph, the FPS reach a peak point when the threads per block is 64. And this result matches my GPU model(GT750m). The program is apparently slow when the threads 
number per block is not enough than the Maximum available threads per block of the GPU.

Another one is the tile size, when I'm dealing with the initialization part(which is, add the intial rays, from camera to the pixel, to the ray pool)
![screenshot](https://github.com/heguanyu/Project2-Pathtracer/blob/master/performance%20analyze/tilesize.bmp?raw=true)

It occurs to me that tilesize of 16 is the best choice. However, as far as this parameter only affect the initialization part, it do not have a heavy weight on the final result. The FPS 
just maintain around 55fps

Anti-Aliasing is another factor that have a great influence on the rendering speed. With 4x super-sampling anti-aliasing on, it can only reach 23fps, 
comparing to the 55 fps of no-super-sampling. It is interesting that it only halved the speed even though 4x of initial rays are being traced.

Coming to the next important factor, max-depth. It turns out that max-depth is very important that affect the speed, but do not necessarily affect the final rendering result, 
undering my current way of rendering.
![screenshot](https://github.com/heguanyu/Project2-Pathtracer/blob/master/performance%20analyze/maxdepth.bmp?raw=true)

It is almost a linear relationship where the FPS decline with the ascendance of the max-depth. However, when using the 2, there are some glitches with the refraction and reflection. Therefore, a minimum max-depth of 5 is suggested
 to avoid unnecessary sin.
 
The last tests surrounds the object amount.
![screenshot](https://github.com/heguanyu/Project2-Pathtracer/blob/master/performance%20analyze/object%20amount.bmp?raw=true)

It is obvious that removing the objects from the scene can enhance the speed of the rendering.

-------------------------------------------------------------------------------
RESULT
-------------------------------------------------------------------------------
Youtube link is here
http://www.youtube.com/watch?v=d4a8aPpC-BM&feature=youtu.be

Screenshot is Here

![screenshot](https://github.com/heguanyu/Project2-Pathtracer/blob/master/renders/pathtracer/with%20texturemap.bmp?raw=true)