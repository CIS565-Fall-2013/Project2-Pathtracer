-------------------------------------------------------------------------------
CIS565: Project 2: CUDA Pathtracer
-------------------------------------------------------------------------------
Yingting Xiao
-------------------------------------------------------------------------------

![alt tag](https://raw.github.com/YingtingXiao/Project2-PathTracer/master/screenshots/reflection_refraction_blend.PNG)

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

https://vimeo.com/76024508


Performance analysis:
-------------------------------------------------------------------------------

Fps with stream compaction using shared memory: 4.15

Fps with stream compaction: 3.08

Fps without stream compaction: 1.67