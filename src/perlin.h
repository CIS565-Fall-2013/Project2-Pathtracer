#ifndef PERLIN_H
#define PERLIN_H

#include "sceneStructs.h"
#include "cudaMat4.h"
#include "glm/glm.hpp"


#include <cmath>
/*
The following code is copied from perlin noise 
http://www.codermind.com/articles/Raytracer-in-C++-Part-III-Textures.html
*/


__device__ __host__ float fade(float t)
{ 
  return t * t * t * (t * (t * 6 - 15) + 10);
}

__device__ __host__ float lerp_perlin(float t, float a, float b) { 
  return a + t * (b - a);
}

__device__ __host__ float grad(int hash, float x, float y, float z) {
  int h = hash & 15;
  // CONVERT LO 4 BITS OF HASH CODE
  float u = h<8||h==12||h==13 ? x : y, // INTO 12 GRADIENT DIRECTIONS.
  v = h < 4||h == 12||h == 13 ? y : z;
  return ((h & 1) == 0 ? u : -u) + ((h&2) == 0 ? v : -v);
}

__device__ __host__ float noise(float x, float y, float z, int* perlinPerm) {
  int X = (int)floor(x) & 255, // FIND UNIT CUBE THAT
      Y = (int)floor(y) & 255, // CONTAINS POINT.
      Z = (int)floor(z) & 255;
  x -= floor(x);                   // FIND RELATIVE X,Y,Z
  y -= floor(y);                   // OF POINT IN CUBE.
  z -= floor(z);
  float u = fade(x),              // COMPUTE FADE CURVES
         v = fade(y),              // FOR EACH OF X,Y,Z.
         w = fade(z);
  int A = perlinPerm[X]+Y,    // HASH COORDINATES OF
      AA = perlinPerm[A]+Z,   // THE 8 CUBE CORNERS,
      AB = perlinPerm[A+1]+Z, 
      B = perlinPerm[X+1]+Y, 
      BA = perlinPerm[B]+Z, 
      BB = perlinPerm[B+1]+Z;
	

  return 
    lerp_perlin(w, lerp_perlin(v, lerp_perlin(u, grad(perlinPerm[AA], x, y, z),      // AND ADD  
                           grad(perlinPerm[BA], x-1, y, z)),    // BLENDED
                   lerp_perlin(u, grad(perlinPerm[AB], x, y-1, z),     // RESULTS
                           grad(perlinPerm[BB], x-1, y-1, z))), // FROM 8
           lerp_perlin(v, lerp_perlin(u, grad(perlinPerm[AA+1], x, y, z-1),   // CORNERS
                           grad(perlinPerm[BA+1], x-1, y, z-1)),// OF CUBE
                   lerp_perlin(u, grad(perlinPerm[AB+1], x, y-1, z-1 ),
                           grad(perlinPerm[BB+1], x-1, y-1, z-1 ))));
}


__host__ __device__ float turbulence(int l,float x, float y, float z, int* perlinPerm)
{
  float turb = 0.0f;
  for (int level = 1; level < l; level ++)
  {
    turb += (1.0f / level ) 
              * fabsf(float(noise(level * 0.05 *x, 
                                  level * 0.05 *y,
                                  level * 0.05 *z,perlinPerm)));
  }
  return turb;
}



#endif