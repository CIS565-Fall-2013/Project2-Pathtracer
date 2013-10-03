// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>
#include <time.h>
#include <thrust/device_ptr.h> 
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
//#define  printf(f, ...) ((void)(f, __VA_ARGS__),0)  
//#endif
#define DEPTH_OF_FIELD

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, float x, float y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  //TODO: CLEAR UP
  int index = y * resolution.x + x;
  ray r;
  r.origin = eye; 	
  float sx, sy;

   //anti-aliasing 
  thrust::default_random_engine rng(hash(time*index));
  thrust::uniform_real_distribution<float> u01(-0.5,0.5);
  sx = (float)(x+(float)u01(rng))/((float)resolution.x-1);
  sy = (float)(y+(float)u01(rng))/((float)resolution.y-1);

  glm::vec3 C = view;
  glm::vec3 M = eye + C;
  glm::vec3 A = glm::cross(C,up);
  glm::vec3 B = glm::cross(A,C);
  glm::vec3 H = A*glm::length(C)*(float)tan(fov.x*PI/180.0) / glm::length(A);
  glm::vec3 V = B*glm::length(C)*(float)tan(fov.y*PI/180.0) / glm::length(B);

  glm::vec3 P = M + (float)(2.0*sx - 1)*H + (float)(1 - 2.0*sy)*V;
  r.direction = P-eye;
  r.direction = glm::normalize(r.direction);

#ifdef DEPTH_OF_FIELD
  //Depth of field  
  thrust::uniform_real_distribution<float> u02(-0.3,0.3);
  glm::vec3 aimPoint = r.origin + (float)DOFLENGTH * r.direction;
  r.origin += glm::vec3(u02(rng),u02(rng),u02(rng));
  r.direction = aimPoint - r.origin;
  r.direction = glm::normalize(r.direction);
#endif
 
  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}
__host__ __device__ glm::vec3 getReflect(glm::vec3 normal, ray Ri)
{
	return glm::normalize(-2.0f * normal * (float)glm::dot(Ri.direction,normal) + Ri.direction);	
}
__host__ __device__ glm::vec3 getRefractRay(float rand,glm::vec3 normal, ray Ri,float n1, float n2)
{
	//determine whether reflect or refract
	glm::vec3 Rr = getReflect(normal,Ri);
	float cosThetai = -1.0f * glm::dot(Ri.direction,normal);
	float sqrsinThetat = (n1*n1)/(n2*n2) * (1-cosThetai*cosThetai);
	//if n1>n2, there should be a critical angle
	if(sqrsinThetat > 1) return Rr;

	float cosThetat = sqrt(1-sqrsinThetat*sqrsinThetat);
	float rV = (n1*cosThetai - n2*cosThetat)/(n1*cosThetai + n2*cosThetat+EPSILON);
	rV = rV*rV;
	float rP = (n2*cosThetai - n1*cosThetat)/(n2*cosThetai + n1*cosThetat+EPSILON);
	rP = rP*rP;
	if(rand<=rV)
	{
		//reflect
		return getReflect(normal,Ri);
	}
	else
	{
		//transmittance, refract
		glm::vec3 Rt = (float)(n1/n2)*Ri.direction + (float)(n1/n2*cosThetai - sqrt(1-sqrsinThetat))*normal;
		Rt = glm::normalize(Rt);
		return Rt;
	}
	printf("error");
	return glm::vec3(0,0,0);
}

/////shadow check only for raytracer
__host__ __device__ bool ShadowRayUnblocked(glm::vec3 point,glm::vec3 lightPos,staticGeom* geoms, int numberOfGeoms,material* mats)
	//,	glm::vec3* pbo,unsigned short* ibo, glm::vec3* nbo)
	//thrust::device_vector<glm::vec3> pbo,thrust::device_vector<unsigned short> ibo, thrust::device_vector<glm::vec3> nbo)
{
	//return true;
	float tmpDist = -1;
	glm::vec3 tmpnormal;
	glm::vec3 intersectionPoint;
	ray r; r.origin = point;
	r.direction = lightPos-point; r.direction = glm::normalize(r.direction);	
	r.origin += r.direction*0.1f;
	float lightToObjDist = glm::length(lightPos - r.origin)-0.25;
	for(int i = 0;i<numberOfGeoms;++i)
	{
		tmpDist = IntersectionTest(geoms[i],r,intersectionPoint,tmpnormal);//,pbo,ibo,nbo);
		if(tmpDist > -1 && tmpDist < lightToObjDist&&mats[i].emittance == 0)
		{
			return false;
		}
	}
	return true;
}

//recursive raytrace
__device__ void raytrace(ray Ri,glm::vec2 resolution, float time, cameraData cam, int rayDepth,int rayIndex, glm::vec3& color,
                            staticGeom* geoms, int numberOfGeoms,material* mats,int* lightIndex,int lightNum)
							//,glm::vec3* pbo,unsigned short* ibo, glm::vec3* nbo)
							//thrust::device_vector<glm::vec3> pbo,thrust::device_vector<unsigned short> ibo, thrust::device_vector<glm::vec3> nbo)
{
	if(rayIndex > rayDepth)
	{
		color = glm::vec3(bgColorR,bgColorG,bgColorB);
		return;
	}
	color = glm::vec3(0,0,0);
	/////////////variables//////////////	
	ray Rr; //reflect ray
	ray Rrl; // light reflect ray
	glm::vec3 intersectionPoint(0,0,0);
	glm::vec3 normal(0,0,0);		
	glm::vec3 diffuseColor(0,0,0);
	glm::vec3 specularColor(0,0,0);  
	glm::vec3 reflectedColor(1.0,1.0,1.0); 
	glm::vec3 refractColor(0,0,0); // TODO, haven't uesed yet
	glm::vec3 localColor(0,0,0);
	glm::vec3 lightPosition (0,0,0);
	int nearestObjIndex = -1; // nearest intersect object index
	glm::vec3 ambient(ambientColorR,ambientColorG,ambientColorB); ambient *= Kambient;	
	float interPointDist = -1;	
	int nearestLight = -1;
	////////////////////////////////////////////
	glm::vec3 tmpnormal(0,0,0);
	float tmpDist = -1;
	////////////////////////////////////////////
	
	for(int i = 0;i<numberOfGeoms;++i)
	{
		tmpDist = IntersectionTest(geoms[i],Ri,intersectionPoint,tmpnormal);//,pbo,ibo,nbo);
		if(tmpDist!=-1 &&(interPointDist==-1 ||(interPointDist!=-1 && tmpDist<interPointDist)))
		{
			interPointDist = tmpDist;
			normal = tmpnormal;
			nearestObjIndex = i;
		}
	}

	//if first ray didn't hit any object,color set to light / bg color
	if(interPointDist == -1 || (interPointDist != -1 && mats[nearestObjIndex].emittance>0))
	{
		if(interPointDist == -1)
			color = glm::vec3(bgColorR,bgColorG,bgColorB);
		else
			color = mats[nearestObjIndex].color;	
		return;
	}
	else if(interPointDist!= -1 && mats[nearestObjIndex].emittance == 0)
	{						
		// this is the reflect ray
		Rr.direction = normal; 
		Rr.direction *= glm::dot(Ri.direction,normal);
		Rr.direction *= -2.0;
		Rr.direction += glm::normalize(Ri.direction);
		Rr.origin = intersectionPoint;	
		Rr.direction = glm::normalize(Rr.direction);
		////////////////////////////////////////////////

		if(mats[nearestObjIndex].hasReflective>0)
		{
			//raytrace(Rr,resolution, time, cam, rayDepth,rayIndex+1,reflectedColor,geoms,numberOfGeoms,mats,lightIndex,lightNum);
			color = glm::vec3(mats[nearestObjIndex].hasReflective*reflectedColor.x,mats[nearestObjIndex].hasReflective*reflectedColor.y,mats[nearestObjIndex].hasReflective*reflectedColor.z);
			//printf("%f,%f,%f ::",reflectedColor.x,reflectedColor.y,reflectedColor.z);
		}
		color += ambient * mats[nearestObjIndex].color;
		//shadow check
		for(int j = 0;j<lightNum;++j)
		{
			//need a ray for specular highlight	
			glm::vec3 lightVector(0,0,0);											
			lightPosition = geoms[lightIndex[j]].translation;
			//calculate light reflect ray
			lightVector = glm::normalize(intersectionPoint - lightPosition);			
			Rrl.direction = normal;
			Rrl.direction *= -2.0;
			Rrl.direction *= glm::dot(lightVector,normal);
			Rrl.direction += lightVector;		
			localColor = glm::vec3(0,0,0);		
			if(ShadowRayUnblocked(intersectionPoint,lightPosition,geoms,numberOfGeoms,mats) == true)				
			{
				//not in shadow			
				diffuseColor = mats[nearestObjIndex].color;								
				glm::vec3 L = glm::normalize(lightPosition - intersectionPoint);
				float diffuseCon = glm::dot(normal,L);
				if(diffuseCon<0)
					diffuseColor = glm::vec3(0,0,0);
				else
				{
					diffuseColor *= diffuseCon;
					diffuseColor *= Kdiffuse;
					diffuseColor *= mats[lightIndex[j]].color * mats[lightIndex[j]].emittance;
				}			
				localColor += diffuseColor;
				float specularCon = glm::dot(Rrl.direction,glm::normalize(cam.position-intersectionPoint));
				
				if(specularCon < 0 || mats[nearestObjIndex].specularExponent == 0)
				{
					specularCon = 0;
					specularColor = glm::vec3(0,0,0);
				}
				else
				{	
					
					specularCon  = pow((double)specularCon,(double)mats[nearestObjIndex].specularExponent);
					specularColor = mats[lightIndex[j]].color * mats[lightIndex[j]].emittance;
					specularColor *= Kspecular;		
					specularColor *= specularCon;
				}			
				localColor += specularColor;
				//sumShadowColor = localColor;
			}			
		}	
	}
	color += localColor;	
}

//TODO: IMPLEMENT THIS FUNCTION
//iterative raytrace
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms,material* mats,int* lightIndex,int lightNum,ray* rays,int sampleIndex,
							glm::vec3* pbo,unsigned short* ibo, glm::vec3* nbo)
							//thrust::device_vector<glm::vec3> pbo,thrust::device_vector<unsigned short> ibo, thrust::device_vector<glm::vec3> nbo)
{

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  //if(sampleIndex == 0)
	 // colors[index] = glm::vec3(0,0,0);
  int rayIndex = 0;
  if((x<=resolution.x && y<=resolution.y)){
		
		//colors[index] = glm::vec3(rays[index].direction.x,rays[index].direction.y,rays[index].direction.z);
		/////////////variables//////////////	
		ray Ri = raycastFromCameraKernel(resolution, time, x,y,cam.position, cam.view, cam.up, cam.fov);
		//ray Ri = rays[index].rays[sampleIndex];
		ray Rr = Ri; //reflect ray
		ray Rrefra; // refraction ray
		ray Rreflect; // reflect ray when hitting refract object
		ray Rrl; // light reflect ray
		glm::vec3 intersectionPoint(0,0,0);
		glm::vec3 normal(0,0,0);		
		glm::vec3 diffuseColor(0,0,0);
		glm::vec3 specularColor(0,0,0);  
		glm::vec3 refractColor(0,0,0); // TODO, haven't uesed yet
		glm::vec3 reflectColor(1,1,1);
		glm::vec3 localColor(0,0,0);
		glm::vec3 lightPosition (0,0,0);
		int nearestObjIndex = -1; // nearest intersect object index
		glm::vec3 ambient(1.0,1.0,1.0); ambient *= 0.3; // *=kambient;	
		float interPointDist = -1;	
		int nearestLight = -1;
		float hasReflect = 1.0;
		glm::vec3 tmpnormal(0,0,0);
		float tmpDist = -1;
		glm::vec3 color(0,0,0);
		float reflectCoeff = 0;
		glm::vec3 oneColor(0,0,0); // used to store each sample ray's color;

		//Fresnel
		float refraCoff1 = 1.0f;//air
		float refraCoff2 = 0;
		float fractionRefra = 0;
		float fractionRefle = 0;
		float hasRefract = 1.0;
		bool hasShootReflect = false; // refraction use
		int rayType = 0; // 0 for first ray, 1 for reflect ray from non-refraction surface, 2 for refraction ray from refract surface, 3 for reflect ray from refract surface
		////////////////////////////////////////////
		while(rayIndex <= rayDepth &&(abs(hasReflect - 1.0)<EPSILON || abs(hasRefract - 1.0)<EPSILON))
		{
			
			color = glm::vec3(0,0,0);
			localColor = glm::vec3(0,0,0);
			nearestObjIndex = -1;
			interPointDist = -1;
			nearestLight = -1;
			tmpDist = -1;
#pragma region intersect scene
			for(int i = 0;i<numberOfGeoms;++i)
			{
				tmpDist = IntersectionTest(geoms[i],Ri,intersectionPoint,tmpnormal);//,pbo,ibo,nbo);
				if(tmpDist!=-1 &&(interPointDist==-1 ||(interPointDist!=-1 && tmpDist<interPointDist)))
				{
					interPointDist = tmpDist;
					normal = tmpnormal;
					nearestObjIndex = i;
				}
			}
#pragma endregion
#pragma region if first ray didn't hit any object,color set to light / bg color
			if(interPointDist == -1 || (interPointDist != -1 && mats[nearestObjIndex].emittance>0))
			{				
				if(interPointDist == -1)
				{
					//TODO change background color	
					color = glm::vec3(0.25,0.18,0.1);		
				}
				else
				{					
					color = mats[nearestObjIndex].color;	
				}
				hasRefract = 0;
				hasReflect = 0;
			}
#pragma endregion
#pragma region else if hit scene
			else if(interPointDist!= -1 && mats[nearestObjIndex].emittance == 0)
			{						
				// this is the reflect ray
				Rr.direction = normal; 
				Rr.direction *= glm::dot(Ri.direction,normal);
				Rr.direction *= -2.0;
				Rr.direction += glm::normalize(Ri.direction);
				Rr.origin = intersectionPoint;	
				Rr.direction = glm::normalize(Rr.direction);
				////////////////////////////////////////////////
				if(mats[nearestObjIndex].hasReflective>0)
				{										
					/*Ri = Rr;*/	
					reflectCoeff = mats[nearestObjIndex].hasReflective;
					hasReflect = 1.0;
					//rayType = 1;
				}
				else if(mats[nearestObjIndex].hasRefractive>0)
				{
					refraCoff1 = 1.0f;
					refraCoff2 = mats[nearestObjIndex].indexOfRefraction;	
					//if(glm::length(normal)-1.0>0.000001) printf("%f ",glm::length(normal));
					float cosThetai = -1.0f * glm::dot(Ri.direction,normal);
					float squareSinThetat = pow((double)refraCoff1/(double)refraCoff1,(double)2.0) * (1-pow((double)cosThetai,(double)2.0));
					float Rverticle = (refraCoff1*cosThetai - refraCoff2*cosThetai) / (refraCoff1*cosThetai + refraCoff2*cosThetai);
					Rverticle = pow((double)Rverticle,(double)2.0);
					float Rparall = (refraCoff2*cosThetai - refraCoff1*cosThetai) / (refraCoff2*cosThetai + refraCoff1*cosThetai);
					Rparall = pow((double)Rparall,(double)2.0);
					float cosThetat = sqrt(1-squareSinThetat);
					fractionRefle = (Rverticle + Rparall) / 2.0;
					fractionRefra = 1 - fractionRefle;
					Rrefra.origin = intersectionPoint;
					Rrefra.direction = Ri.direction; Rrefra.direction*=(refraCoff1/refraCoff2);
					glm::vec3 tmp = normal;
					tmp *= (refraCoff1/refraCoff2*cosThetai - sqrt(1-squareSinThetat));
					Rrefra.direction += tmp;			
					Rrefra.direction = glm::normalize(Rrefra.direction);
					Rreflect = Rr;
					hasRefract = 1.0;
					//hasReflect = 1.0;					
				}
				else
				{
					hasRefract = 0;
					hasReflect = 0;
				}
				color += ambient * mats[nearestObjIndex].color;
				//shadow check
				for(int j = 0;j<lightNum;++j)
				{
					//need a ray for specular highlight	
					glm::vec3 lightVector(0,0,0);											
					lightPosition = geoms[lightIndex[j]].translation;
					//calculate light reflect ray
					lightVector = glm::normalize(intersectionPoint - lightPosition);			
					Rrl.direction = normal * glm::vec3(-2.0,-2.0,-2.0) * glm::dot(lightVector,normal)+lightVector ;					
					localColor = glm::vec3(0,0,0);		
					if(ShadowRayUnblocked(intersectionPoint,lightPosition,geoms,numberOfGeoms,mats) == true)				
					{
						//not in shadow			
						diffuseColor = mats[nearestObjIndex].color;								
						glm::vec3 L = glm::normalize(lightPosition - intersectionPoint);
						float diffuseCon = glm::dot(normal,L);
						if(diffuseCon<0)
							diffuseColor = glm::vec3(0,0,0);
						else
						{
							diffuseColor *= diffuseCon;
							//TODO change diffuse coefficient
							diffuseColor *= 0.1;
							diffuseColor *= mats[lightIndex[j]].color * mats[lightIndex[j]].emittance;
						}			
						localColor += diffuseColor;
						float specularCon = glm::dot(Rrl.direction,glm::normalize(cam.position-intersectionPoint));
				
						if(specularCon < 0 || mats[nearestObjIndex].specularExponent == 0)
						{
							specularCon = 0;
							specularColor = glm::vec3(0,0,0);
						}
						else
						{	
					
							specularCon  = pow((double)specularCon,(double)mats[nearestObjIndex].specularExponent);
							specularColor = mats[lightIndex[j]].color * mats[lightIndex[j]].emittance;
							//TODO change specular coefficient
							specularColor *= 0.3;		
							specularColor *= specularCon;
						}			
						localColor += specularColor;						
					}			
				}	
				color += localColor;	
			}
	
#pragma endregion		
			if(rayType == 0)
			{										
				if(hasReflect>0)
				{
					oneColor = color;
					Ri = Rr;
					rayType = 1;
				}
				else if(hasRefract >0)
				{					
					Ri = Rrefra;
					rayType = 2;
				}
				else
					oneColor = color;
				rayIndex ++;
			}
			else if(rayType == 1)
			{								 
				oneColor += reflectCoeff * color;	
				
				if(hasReflect>0)
				{
					Ri = Rr;
					rayType = 1;
				}
				else if(hasRefract >0)
				{
					Ri = Rrefra;
					rayType = 2;
				}
				rayIndex ++;			
			}
			else if(rayType == 2)
			{
				oneColor += fractionRefra * color;				
				Ri = Rreflect;
				rayType = 3;
			}
			else if(rayType ==3)
			{
				oneColor += fractionRefle * color;
				Ri = Rr;
				rayType = 1;
				rayIndex ++;
			}
			
		}		
		colors[index] = (colors[index]*(time-1)+oneColor) / time;
   }
   
}

__global__ void generateInitialRays(ray* initialRay,glm::vec3* rayColor, glm::vec2 resolution,float time, cameraData cam)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);	
	ray r = raycastFromCameraKernel(resolution, time, x,y,cam.position, cam.view, cam.up, cam.fov); 
	r.tag = 1; // valid, 0 if invalid
	r.pixelId = index;
	initialRay[index] = r;
	rayColor[index] = glm::vec3(1,1,1);
}

__global__ void pathTracer(float time,cameraData cam,int rayDepth,glm::vec3* colors, 
	staticGeom* geoms,int numberOfGeoms,material* mats,int* lightIndex,int lightNum,ray* rays,int rayNum,glm::vec3* rayColor)
	//,	glm::vec3* pbo,unsigned short* ibo, glm::vec3* nbo)
	//thrust::device_vector<glm::vec3> pbo,thrust::device_vector<unsigned short> ibo, thrust::device_vector<glm::vec3> nbo)
{
	/*int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);*/
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(index <= rayNum)
	{		
		ray Ri = rays[index];
		Ri.origin += INTEROFFSET * Ri.direction;
		int pixelId = Ri.pixelId; // this index is correspond to the index in rayColor buffer
		if(rayDepth >= BOUNCE_DEPTH)
		{
			//if the ray didn't hit any bg or light, but ray depth is overload, then terminate it. 
			colors[pixelId] = (colors[pixelId]*(time-1) + rayColor[pixelId])/time;
			return;
		}
		
		//test intersectoin
		glm::vec3 intersectionPoint(0,0,0);
		glm::vec3 normal(0,0,0);		
		int nearestObjIndex = -1; // nearest intersect object index
		float interPointDist = -1;	
		int nearestLight = -1;
		glm::vec3 tmpnormal(0,0,0);
		for(int i = 0;i<numberOfGeoms;++i)
		{
			float tmpDist = IntersectionTest(geoms[i],Ri,intersectionPoint,tmpnormal);//,pbo,ibo,nbo);
			if(tmpDist!=-1 &&(interPointDist==-1 ||(interPointDist!=-1 && tmpDist<interPointDist)))
			{
				//printf("debug");
				interPointDist = tmpDist;
				normal = tmpnormal;
				nearestObjIndex = i;
			}
		}
		
#pragma region didn't hit object or hit light
		if(interPointDist == -1 || (interPointDist != -1 && mats[nearestObjIndex].emittance>0))
		{				
			if(interPointDist == -1)
			{
				//TODO change background color	
				rayColor[pixelId] *= Ri.color_fraction * glm::vec3(0,0,0);		 // hit background
				
			}
			else
			{					
				rayColor[pixelId] *= mats[nearestObjIndex].emittance * mats[nearestObjIndex].color; // hit light					
			}	
			//set ray dead;
			colors[pixelId] = (colors[pixelId]*(time-1)+rayColor[pixelId])/time;
			//colors[pixelId] += rayColor[pixelId];
			rays[index].tag = -1;
			return;
		}
#pragma endregion
		else // did hit objects in the scene
		{
			ray secondRay; //secondary Ray
			secondRay.origin = intersectionPoint;	
			thrust::default_random_engine rng(hash(time*(rayDepth+1))*hash(pixelId));
			thrust::uniform_real_distribution<float> u01(0,1);			
			if(mats[nearestObjIndex].hasReflective>0)
			{
				//reflect ray, set current ray as this ray
				float rand =(float) u01(rng);
				secondRay.origin = intersectionPoint;
				if(rand<mats[nearestObjIndex].hasReflective)
				{
					//reflect
					secondRay.direction = getReflect(normal,Ri);		
					secondRay.color_fraction = mats[nearestObjIndex].hasReflective;
				}
				else
				{
					//diffuse
					//get random direction over hemisphere
					secondRay.direction = calculateRandomDirectionInHemisphere(normal, (float)u01(rng), (float)u01(rng)); 
					secondRay.direction = glm::normalize(secondRay.direction);
					//colors[index] += mats[nearestObjIndex].color * mats[nearestObjIndex].hasReflective;
					secondRay.color_fraction = 1 - mats[nearestObjIndex].hasReflective;
				}
			}
			else if(mats[nearestObjIndex].hasRefractive>0)
			{
				//refract ray
				//Fresnel law, either reflect or refract
				//thrust to generate cofficient
				float rand =(float)u01(rng);
				//TODO:how to tell whether ray goes in or out ? // add index for ray, 1.0 by default
				if(Ri.m_index == 1.0)
				{
					//going into the object
					secondRay.direction = getRefractRay(rand,normal,Ri,1.0,mats[nearestObjIndex].indexOfRefraction);
					secondRay.m_index = mats[nearestObjIndex].indexOfRefraction;
				}
				else
				{
					//going out from object
					secondRay.direction = getRefractRay(rand,-normal,Ri,mats[nearestObjIndex].indexOfRefraction,1.0);
				}
			}
			else
			{

				secondRay.direction = calculateRandomDirectionInHemisphere(normal, u01(rng), u01(rng)); 
				secondRay.direction = glm::normalize(secondRay.direction);
				//index changes everytime !!!
				rayColor[pixelId] *= mats[nearestObjIndex].color;//*Ri.color_fraction;				
			}
			rays[index] = secondRay;
			secondRay.origin += INTEROFFSET*secondRay.direction;
		}	


	}
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms)
	//,std::vector<glm::vec3> pbo,std::vector<unsigned short> ibo, std::vector<glm::vec3>nbo)
{
  
 // int traceDepth = 3; //determines how many bounces the raytracer traces

  
  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //send mesh info to GPU
 /* glm::vec3* d_pbo = NULL;
  cudaMalloc((void**)&d_pbo,pbo.size()*sizeof(glm::vec3));
  for(int i = 0;i<pbo.size();++i)
  {
	  cudaMemcpy(&d_pbo[i],&pbo[i],sizeof(glm::vec3),cudaMemcpyHostToDevice);
  }

  glm::vec3* d_nbo = NULL;
  cudaMalloc((void**)&d_nbo,nbo.size()*sizeof(glm::vec3));
  for(int i = 0;i<nbo.size();++i)
  {
	  cudaMemcpy(&d_nbo[i],&nbo[i],sizeof(glm::vec3),cudaMemcpyHostToDevice);
  }

  unsigned short* d_ibo = NULL;
  cudaMalloc((void**)&d_nbo,nbo.size()*sizeof(unsigned short));
  for(int i = 0;i<ibo.size();++i)
  {
	  cudaMemcpy(&d_ibo[i],&ibo[i],sizeof(unsigned short),cudaMemcpyHostToDevice);
  }*/

 /* thrust::device_vector<glm::vec3> d_pbo(pbo.size());
  thrust::device_vector<unsigned short> d_ibo(ibo.size());
  thrust::device_vector<glm::vec3> d_nbo(nbo.size());
  for(int i = 0;i<pbo.size();++i)
  {
	  d_pbo[i] = pbo[i];	  
  }
  for(int i = 0;i<ibo.size();++i)
  {
	  d_ibo[i] = ibo[i];
  }
  for(int i = 0; i<nbo.size();++i)
  {
	  d_nbo[i] = nbo[i];
  }*/

   if(iterations <= 3)
  {
	  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,cudaimage);
  }
  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  //material
  material* matList = new material[numberOfGeoms];
  int lightNum = 0;
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
	matList[i] = materials[newStaticGeom.materialid];
	if(matList[i].emittance >0)
		lightNum++;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }
  int* lightIndex = new int[lightNum];
  
  int lin = 0;
  for(int i = 0;i<numberOfGeoms;i++)
  {
	  if(matList[i].emittance ==0) continue;
	  lightIndex[lin] = i;
	  lin++;
  }
  int* cudaLight = NULL;
  cudaMalloc((void**)&cudaLight,lightNum*sizeof(int));
  cudaMemcpy(cudaLight,lightIndex,lightNum*sizeof(int),cudaMemcpyHostToDevice);
  material* cudamat = NULL;
  cudaMalloc((void**)&cudamat,numberOfGeoms*sizeof(material));
  cudaMemcpy(cudamat,matList,numberOfGeoms*sizeof(material),cudaMemcpyHostToDevice);
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;
  
  // initialize rays
  ray* rayPool = NULL;
  int numberOfValidRays = (int)renderCam->resolution.x*(int)renderCam->resolution.y;
  cudaMalloc((void**)&rayPool,numberOfValidRays*sizeof(ray)); 

  glm::vec3* rayColor = NULL; //final color of each ray, this color is independent of each iteration
  cudaMalloc((void**)&rayColor,numberOfValidRays*sizeof(glm::vec3));

  generateInitialRays<<<fullBlocksPerGrid, threadsPerBlock>>>(rayPool,rayColor,renderCam->resolution,(float)iterations,cam);
  

 
  thrust::device_ptr<ray> rayPoolEnd; 
  // change to 1D, blocksize has nothing with resolution now.
  int threadPerBlock = 128;//TODO tweak
  int blockPerGrid = (int)ceil((float)numberOfValidRays/threadPerBlock);
  for(int i = 0;i<BOUNCE_DEPTH;++i)
  {
	  if(numberOfValidRays == 0) break;
	  blockPerGrid = (int)ceil((float)numberOfValidRays/threadPerBlock);
	  pathTracer<<<blockPerGrid, threadPerBlock>>>((float)iterations, cam, i, cudaimage, cudageoms, numberOfGeoms,cudamat,cudaLight,
		  lightNum,rayPool,numberOfValidRays,rayColor);//,d_pbo,d_ibo,d_nbo);

	  //each step, number of valid rays changes
	  thrust::device_ptr<ray> rayPoolStart = thrust::device_pointer_cast(rayPool);
	  rayPoolEnd = thrust::remove_if(rayPoolStart,rayPoolStart+numberOfValidRays,isDead());
	  numberOfValidRays = (int)( rayPoolEnd - rayPoolStart);
	  //printf("%d  ",numberOfValidRays);
  }
  
 
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree (cudaLight);
  cudaFree (cudamat);
  cudaFree (rayPool);
  cudaFree (rayColor);
 /* cudaFree (d_pbo);
  cudaFree (d_ibo);
  cudaFree (d_nbo);*/
  delete[] matList;
  delete[] lightIndex;
  delete[] geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}