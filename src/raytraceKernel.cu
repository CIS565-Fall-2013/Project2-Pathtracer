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

#include <time.h>
#include <vector>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

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
    ray r;

	glm::vec3 A=glm::cross(view,up);
	glm::vec3 B=glm::cross(A,view);
	glm::vec3 M=eye+view;
	glm::vec3 V=B*(glm::length(view)*tan(fov.y)/glm::length(B));
	glm::vec3 H=A*(glm::length(view)*tan(fov.x)/glm::length(A));

	float t1=(x/(resolution.x+0.0f))*2.0f-1.0f;
	float t2=(y/(resolution.y+0.0f))*2.0f-1.0f;
	glm::vec3 P=M-t1*H+t2*V;
	glm::vec3 R=glm::normalize(P-eye);

	r.origin = eye;
	r.direction = R;
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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, int iterations){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;
	  color*=1.0f/(float)iterations;

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

__host__ __device__ float findIntersection(int index, ray r, staticGeom* geoms, int numberOfGeoms ,int& hitidx, glm::vec3& p, glm::vec3& n, glm::vec2& texturecoord)
{
		glm::vec3 intersectPoint;
		glm::vec3 normalValue;

		glm::vec3 final_intersectPoint;
		glm::vec3 final_normal;

		float mindist=1000000000;
		glm::vec3 outColor(0,0,0);
		float tempd=0;
		bool isLight=false;

		glm::vec2 tempcoord;
		for(int i=0;i<numberOfGeoms;i++)
		{
			if(geoms[i].type==0)
			{
				tempd=sphereIntersection(geoms[i], r, intersectPoint, normalValue,0.5,tempcoord);
			}
			else if (geoms[i].type==1)
			{
				tempd=boxIntersection(geoms[i], r, intersectPoint, normalValue,tempcoord);
			}
			if(tempd>0 && tempd<mindist)
			{
				mindist=tempd;
				final_intersectPoint=intersectPoint;
				final_normal=normalValue;
				hitidx=i;
				texturecoord=tempcoord;
			}
		}
		p=final_intersectPoint;
		n=final_normal;
		if(mindist<10000000) return mindist; else return -1;
}


//SECOND step: addRefelctance//!!!!DISCARDED
__global__ void dofBlur(glm::vec2 resolution, float time, glm::vec3* toColors, glm::vec3* colors, hitInfo* cudahitinfo)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if((x<=resolution.x && y<=resolution.y)){
	  toColors[index]=colors[index];
	  if(!cudahitinfo[index].hit) return;
	  
	  float standardDOF=cudahitinfo[(int)(resolution.x*resolution.y/2)].dof;
	  float myDOF=cudahitinfo[index].dof;
	  if(abs(myDOF/standardDOF-1.0f)<0.1f) return;
	  int blurradius=(int)(abs(myDOF/standardDOF-1.0f)/0.1f);
	  if(blurradius>6) blurradius=6;
	  int blurnum=0;
	  toColors[index]=glm::vec3(0,0,0);
	  for(int i=-blurradius;i<=blurradius;i++) for(int j=-blurradius;j<=blurradius;j++)
	  {
		  int xx=x+i, yy=y+j;
		  if(xx<=0 || xx>resolution.x || yy<=0 || yy>resolution.y) continue;
		  int newidx=xx+yy*resolution.x;
		  if(abs(cudahitinfo[newidx].dof/standardDOF-1.0f)<0.1f) continue;
		  blurnum++;
		  toColors[index]+=colors[newidx];
	  }
	  toColors[index]*=(1.0f/(float)blurnum);
  }
}

__global__ void initializeRayPool(glm::vec2 resolution, float time, cameraData cam, parallelRay* raypool, int* flagpool, int subraynum){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if((x>resolution.x || y>resolution.y)) return;
  for(int i=0;i<subraynum;i++)
  {
	  float offsetx=(i%2)*0.66f-0.33f;
	  float offsety=(i/2)*0.66f-0.33f;
	  int indexoffset=i*(int)resolution.x*(int)resolution.y;

	  ray r=raycastFromCameraKernel(resolution, time,x+offsetx,  y+offsetx, cam.position, cam.view, cam.up, cam.fov);
	  parallelRay pr;
	  pr.direction=r.direction;

	  pr.index=index;
	  pr.iters=0;
	  pr.origin=r.origin;
	  pr.coeff=glm::vec3(1,1,1)*(1.0f/(float)subraynum);
	  pr.terminated=false;
	  raypool[index+indexoffset]=pr;
	  flagpool[index+indexoffset]=0;
  }

}

__global__ void computeRaypool(parallelRay* rayPool,int* flagPool, int raypoolsize, float time, int maxDepth, glm::vec3* colors, cameraData cam, BMPInfo* bmps, int numberOfTexture, glm::vec3* textures,
                            staticGeom* geoms, int numberOfGeoms, staticMaterial* materials, ParameterSet ps)
{
	int now=blockIdx.x*blockDim.x+threadIdx.x;
	if(now>=raypoolsize) return;
	parallelRay pr=rayPool[now];
	if(pr.iters>maxDepth) { rayPool[now].terminated=true;return;}

	int index=pr.index;
	ray r;
	r.direction=pr.direction;
	r.origin=pr.origin;

	glm::vec3 final_intersectPoint;
	glm::vec3 final_normal;

	float mindist=1000000000;
	glm::vec3 outColor(0,0,0);
	float tempd=0;
	bool isLight=false;
	staticMaterial targetMat;

	int hitidx,matid;
	glm::vec2 texturecoord;
	tempd=findIntersection(index,r,geoms,numberOfGeoms,hitidx,final_intersectPoint,final_normal,texturecoord);
	
	if(tempd<0)
	{
		flagPool[now]=0;
		rayPool[now].terminated=true;
	}
	else
	{
		flagPool[now]=0;
		targetMat=materials[geoms[hitidx].materialid];
		glm::vec3 hitcolor=targetMat.color;
		if(targetMat.textureidx>=1)
		{
			hitcolor=getTextureColor(texturecoord,targetMat.textureidx,bmps,numberOfTexture,textures);
		}

		rayPool[now].terminated=(targetMat.emittance>0.1f);

		colors[index]+=colorMultiply(hitcolor*(targetMat.emittance),rayPool[now].coeff);

		if(rayPool[now].terminated) return;
		flagPool[now]=1;
		int nextstep=0;		//0=DIFFUSE,1=REFLECTIVE, 2=REFRACTIVE
		if(targetMat.hasRefractive>0.5f && targetMat.hasReflective>0.5f)
		{
			nextstep=getNextStep(index*time,0.01,0.2);
		}
		else if(targetMat.hasRefractive>0.5f)
		{
			nextstep=getNextStep(index*time,0.01,0.01);
		}
		else if(targetMat.hasReflective)
		{
			nextstep=getNextStep(index*time,0.01,1.0);
		}


		if(nextstep==2)	//REFRACTION
		{
			glm::vec3 dir1=calculateTransmissionDirection(final_normal,r.direction,1,targetMat.indexOfRefraction);
			glm::vec2 tempcoord;
			if(glm::length(dir1)<0.5f)
			{
				r.origin=final_intersectPoint+final_normal*0.001f;
				r.direction=calculateReflectionDirection(final_normal,r.direction);
			}
			else
			{
				glm::vec3 hp1=final_intersectPoint+dir1*0.001f;
				glm::vec3 hp2,normal2;
				r.origin=hp1;
				r.direction=dir1;
				if(geoms[hitidx].type==0) tempd=sphereIntersection(geoms[hitidx],r,hp2,normal2,0.5f,tempcoord);
				else if (geoms[hitidx].type==1) tempd=boxIntersection(geoms[hitidx],r,hp2,normal2,tempcoord);
				normal2=-normal2;
				glm::vec3 dir2=calculateTransmissionDirection(normal2, dir1,targetMat.indexOfRefraction,1);
				r.origin=hp2+dir2*0.01f;
				r.direction=dir2;
				r.origin-=normal2*0.001f;
			}
			rayPool[now].origin=r.origin;
			rayPool[now].direction=r.direction;
		}
		else if(nextstep==1)	//REFLECTION
		{
			rayPool[now].origin=final_intersectPoint+final_normal*0.0001f;
			rayPool[now].direction=calculateReflectionDirection(final_normal,r.direction);
		}
		else					//DIFFUSE
		{
			
			if(!rayPool[now].terminated)
			{
				rayPool[now].origin=final_intersectPoint+final_normal*0.001f;
				int theseed=(int)time*index+now*rayPool[now].iters;
				//theseed=(int)(time*index*rayPool[now].iters/maxDepth);
				//theseed=(int)(time*(index+rayPool[max(now-1,0)].index));
				//theseed=(int)time*now;
				rayPool[now].direction=getRandomDirectionInSphere(final_normal, theseed);
				rayPool[now].coeff*=glm::dot(final_normal,rayPool[now].direction);
			}
		}
		rayPool[now].iters++;
		rayPool[now].coeff=colorMultiply(hitcolor,rayPool[now].coeff);
	}
}
struct terminated
{
	__host__ __device__
		bool operator()(const parallelRay pr)
	{
		return pr.terminated;
	}
};

parallelRay* raypool;
int raypoolsize;
glm::vec3* cudashadow;
glm::vec3* cudaTexture;
BMPInfo* textureInfo;



//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos,camera* renderCam, ParameterSet* pSet, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, m_BMP* textures, int numberOfTextures){
//pre-process, including memcpy and initialization
	int traceMaxDepth = (int)pSet->ks; //determines how many bounces the raytracer traces
  // set up crucial magic
  int tileSize = (int)pSet->ka;	//don't care about this var name. it is tilesize from input file
  int numberOfLights=0;

  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;
  cam.ambient=renderCam->ambient;

  if(iterations<1.5f)
  {
	  
	  cudaMalloc((void**)&cudashadow, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	  cudaMemcpy( cudashadow, renderCam->shadowVal, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

	  BMPInfo* localInfo=new BMPInfo[numberOfTextures];
	  printf("number of textures: %d\n", numberOfTextures);
	  
	  int total=0;
	  for(int i=0;i<numberOfTextures;i++)
	  {
		  localInfo[i].offset=total;
		  localInfo[i].width=textures[i].resolution.x;
		  localInfo[i].height=textures[i].resolution.y;
		  total+=(int)textures[i].resolution.x*(int)textures[i].resolution.y;
		  
	  }
	  
	  glm::vec3* localtextures=new glm::vec3[total];
	  int now=0;
	  for(int i=0;i<numberOfTextures;i++)
	  {
		  for(int j=0;j<(int)textures[i].resolution.x*(int)textures[i].resolution.y;j++)
		  {
			  localtextures[now]=textures[i].colors[j];
			  now++;
		  }
	  }
	  
	  cudaMalloc((void**)&textureInfo,numberOfTextures*sizeof(BMPInfo));
	  cudaMemcpy(textureInfo,localInfo,numberOfTextures*sizeof(BMPInfo),cudaMemcpyHostToDevice);
	  cudaMalloc((void**)&cudaTexture,now*sizeof(glm::vec3));
	  cudaMemcpy(cudaTexture,localtextures,now*sizeof(glm::vec3),cudaMemcpyHostToDevice);
	  delete localtextures;
	  delete localInfo;
  }


  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
	newStaticGeom.moblur=geoms[i].moblur[frame];
	if(glm::length(newStaticGeom.moblur)>0.1f)
	{
		
		float t=(iterations%pSet->shadowRays)/(float)pSet->shadowRays;
		float mbcoeff=t*t/2;
		newStaticGeom.translation+=mbcoeff*newStaticGeom.moblur;
		glm::mat4 transform = utilityCore::buildTransformationMatrix(newStaticGeom.translation, newStaticGeom.rotation, newStaticGeom.scale);
		newStaticGeom.transform = utilityCore::glmMat4ToCudaMat4(transform);
		newStaticGeom.inverseTransform = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));
	}
	else
	{
		newStaticGeom.transform = geoms[i].transforms[frame];
		newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
	}
    geomList[i] = newStaticGeom;
  }
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);


  staticMaterial* matList = new staticMaterial[numberOfMaterials];
  for(int i=0; i<numberOfMaterials; i++){
    staticMaterial newStaticMat;

    newStaticMat.color = materials[i].color; 
    newStaticMat.specularExponent = materials[i].specularExponent;
    newStaticMat.specularColor = materials[i].specularColor;
    newStaticMat.hasReflective = materials[i].hasReflective;
    newStaticMat.hasRefractive = materials[i].hasRefractive;
    newStaticMat.indexOfRefraction = materials[i].indexOfRefraction;
	newStaticMat.hasScatter = materials[i].hasScatter;
	newStaticMat.absorptionCoefficient = materials[i].absorptionCoefficient;
	newStaticMat.reducedScatterCoefficient = materials[i].reducedScatterCoefficient;
	newStaticMat.emittance = materials[i].emittance;
	newStaticMat.textureidx=materials[i].textureidx;
    matList[i] = newStaticMat;
  }

  staticMaterial* cudamats = NULL;
  cudaMalloc((void**)&cudamats, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamats, matList, numberOfMaterials*sizeof(staticMaterial), cudaMemcpyHostToDevice);

  //package camera


  ParameterSet ps;
  ps.ka=pSet->ka;
  ps.kd=pSet->kd;
  ps.ks=pSet->ks;
  ps.shadowRays=pSet->shadowRays;
  ps.hasSubray=pSet->hasSubray;
  

  //kernel launches
//  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaimage);
  int subrays=pSet->hasSubray;
  float subraycoeff=1.0f/(float)subrays;

  raypoolsize=(int)renderCam->resolution.x*(int)renderCam->resolution.y*pSet->hasSubray;
  if(iterations<1.5f){
	  cudaMalloc((void**)&raypool, raypoolsize*sizeof(parallelRay));
  }
  int* flagPool;
  cudaMalloc((void**)&flagPool, raypoolsize*sizeof(int));
  initializeRayPool<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,(float)iterations,cam,raypool,flagPool, pSet->hasSubray);
  

	for(int nowturn=0;nowturn<traceMaxDepth*2;nowturn++)
	{
	  if(raypoolsize<=1) break;
	  int threadperblock=(int)pSet->kd;
	  int blocknum=raypoolsize/threadperblock;
	  blocknum=max(blocknum,1);

		//first step, get the first bounce and put them into the ray pool
		computeRaypool<<<blocknum, threadperblock>>>(raypool, flagPool,raypoolsize, (float)iterations, traceMaxDepth, cudaimage, cam,textureInfo,numberOfTextures,cudaTexture,
			cudageoms, numberOfGeoms , cudamats, ps);
		//cudaThreadSynchronize();
	//	parallelRay* raypoollocal=new parallelRay[raypoolsize];
	//	cudaMemcpy(raypoollocal,raypool,raypoolsize*sizeof(parallelRay),cudaMemcpyDeviceToHost);

	//	thrust::device_ptr<int> d=thrust::device_pointer_cast(flagPool);  
	//	thrust::device_vector<int> v(raypoolsize);                    
	//	thrust::exclusive_scan(d, d+raypoolsize, v.begin());


	////	thrust::exclusive_scan(flagpoollocal, flagpoollocal+raypoolsize,flagpoollocal);
	//	//second step, remove terminated rays and make a new raypool



	//	int head=0;
	//	int tail=raypoolsize-1;
	//	while(head<tail)
	//	{
	//		while(head<raypoolsize && !raypoollocal[head].terminated)head++;
	//		while(tail>=0 && raypoollocal[tail].terminated) tail--;
	//		if(head>=tail) break;
	//		raypoollocal[head]=raypoollocal[tail];
	//		tail--;
	//		head++;
	//	}
	//	raypoolsize=tail+1;
	//	cudaMemcpy(raypool,raypoollocal,raypoolsize*sizeof(parallelRay),cudaMemcpyHostToDevice);
	//	delete raypoollocal;
	//	printf("bounce: %d, remain: %d\n",nowturn, raypoolsize);
		thrust::device_ptr<parallelRay> iteratorStart(raypool);
		thrust::device_ptr<parallelRay> iteratorEnd = iteratorStart + raypoolsize;
		iteratorEnd = thrust::remove_if(iteratorStart, iteratorEnd, terminated());
		raypoolsize = (int)(iteratorEnd - iteratorStart);
	}
//  cudaMemcpy(renderCam->shadowVal, cudashadow,(int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  glm::vec3* blurredimage = NULL;
  cudaMalloc((void**)&blurredimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
//  dofBlur<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,(float)iterations,blurredimage, cudaimage, cudahitinfo);
//  cudaMemcpy( cudaimage, blurredimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage,iterations);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  
  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamats );
  cudaFree( blurredimage);
  cudaFree( flagPool);
//  cudaFree( raypool);
//  cudaFree( cudashadow);

  delete geomList;
  delete matList;



  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
