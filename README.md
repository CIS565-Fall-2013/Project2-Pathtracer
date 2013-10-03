-------------------------------------------------------------------------------
CIS565: Project 1: CUDA Raytracer
-------------------------------------------------------------------------------
Yingting Xiao
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
Features implemented:
-------------------------------------------------------------------------------

• All required features

• Stream compaction from scratch

• Fresnel refraction


Screenshots:
-------------------------------------------------------------------------------

1) Reflection

![alt tag](https://raw.github.com/YingtingXiao/Project2-PathTracer/master/screenshots/reflection.PNG)

2) Refraction

![alt tag](https://raw.github.com/YingtingXiao/Project2-PathTracer/master/screenshots/refraction.PNG)

3) Reflection and refraction

![alt tag](https://raw.github.com/YingtingXiao/Project2-PathTracer/master/screenshots/reflection_refraction.PNG)
       
4) Diffuse

![alt tag](https://raw.github.com/YingtingXiao/Project2-PathTracer/master/screenshots/diffuse.PNG)


Screen recording:
-------------------------------------------------------------------------------

https://vimeo.com/75074121


Performance analysis:
-------------------------------------------------------------------------------

Fps with stream compaction: 3.08
Fps without stream compaction: 1.67

There is a 2x speed enhancement with my stream compaction. I think the reason that it is still pretty slow is that I used too many cudaMalloc's in my stream compaction. In the future I will try to use less cudaMalloc's and do a performance analysis with that. I also want to do a performance comparison with thrust.