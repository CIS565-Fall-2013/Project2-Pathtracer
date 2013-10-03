-------------------------------------------------------------------------------
CIS565: Project 2: CUDA Pathtracer
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------
Due Wednesday, 10/02/13
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
This is a ray-parallel GPU pathtracer. I thought the most challenging part of 
the assignment was definitely stream compaction, which I implemented using the 
double buffer method. For each bounce of the ray, I mark which rays are dead 
(rays are dead if they hit the light or intersect no objects at all), and then
do an inclusive scan based on the ray's status. When I have the scanned array 
that contains the new ray indices and how many rays are alive total, I build a 
new pool of rays based on this data, and then launch the next bounce with less
rays to account for. The number of blocks used in the pathtracer kernel 
decreases each bounce since I calculated the number of blocks needed based 
on how many alive rays there are per bounce. 

I also did fresnel refraction using Schlick's approximation.

-------------------------------------------------------------------------------
SCREENSHOTS
-------------------------------------------------------------------------------
Broken stream compaction

![Alt text](/renders/finished/bad.jpg "early test")

Diffuse

![Alt text](/renders/finished/diffuse.jpg "basic diffuse")

Look how much difference antialiasing makes!

![Alt text](/renders/finished/noAntialias.jpg "no antialiasing" )

![Alt text](/renders/finished/antialias.jpg "antialiasing")

Diffuse, reflection, and Fresnel refraction
![Alt text](/renders/finished/fresnel.bmp "Fresnel refraction")

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------

Performance evaluation of ray parallel pathtracing with stream compaction vs. 
pixel parallel pathtracing. 

![Alt text](/renders/finished/analysis.jpg "Fresnel refraction")

-------------------------------------------------------------------------------
VIDEO
-------------------------------------------------------------------------------
Video link:

https://vimeo.com/user10815579

-------------------------------------------------------------------------------
THIRD PARTY CODE 
-------------------------------------------------------------------------------
For stream compaction, I used the pseudocode from the class slides. 

I referred to http://www.cs.unc.edu/~rademach/xroads-RT/RTarticle.html
to do refraction and the wikipedia page to do Schlick's approximation.  
