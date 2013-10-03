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
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>
#include "glm/glm.hpp"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/remove.h>

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
	std::cin.get();
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

//Kernel that does the initial raycast from the camera.
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
   
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);
  
  //standard camera raycast stuff
  glm::vec3 E = eye;
  glm::vec3 C = view;
  glm::vec3 U = up;
  float fovx = fov.x;
  float fovy = fov.y;
  
  float CD = glm::length(C);
  
  glm::vec3 A = glm::cross(C, U);
  glm::vec3 B = glm::cross(A, C);
  glm::vec3 M = E+C;
  glm::vec3 H = (A*float(CD*tan(fovx*(PI/180))))/float(glm::length(A));
  glm::vec3 V = (B*float(CD*tan(-fovy*(PI/180))))/float(glm::length(B));
  
  float sx = (x)/(resolution.x-1);
  float sy = (y)/(resolution.y-1);
  
  glm::vec3 P = M + (((2*sx)-1)*H) + (((2*sy)-1)*V);
  glm::vec3 PmE = P-E;
  glm::vec3 R = E + (float(200)*(PmE))/float(glm::length(PmE));
  
  glm::vec3 direction = glm::normalize(R);
  //major performance cliff at this point, TODO: find out why!
  ray r;
  r.origin = eye;
  r.direction = direction;
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

__global__ void raytoColorbouncecopy(glm::vec2 resolution,glm::vec3* colBounce, ray* r, int num,int blockdim)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * blockdim* blockDim.x );
  if(index < num)
  {
  int xx = r[index].x;
  int yy = r[index].y;
  int newindex =  xx + (yy * resolution.x);

  if(r[index].life == true && r[index].rcolor[0] !=0  && r[index].rcolor[1] !=0 && r[index].rcolor[2] !=0)
	colBounce[newindex] = r[index].rcolor ;
	//if(r[newindex].life == true)
	///  r[newindex].rcolor = glm::vec3(1,1,1);
  }

}


__global__ void finalizeraycolor(glm::vec2 resolution,glm::vec3* colBounce,glm::vec3* colors,float iters)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  //int xx = r[index].x;
  //int yy = r[index].y;
 // int newindex =  xx + (yy * resolution.x);
  if(iters < 0.05)
	  iters = 1.0f;
  if((x<=resolution.x && y<=resolution.y)){
	  /*colIters[index] = colIters[index] + colBounce[index];
	  colIters[index][0] = ((colIters[index][0] * (iters - 1)) + colBounce[index][0])  / iters ;
	  colIters[index][1] = ((colIters[index][1] * (iters - 1)) + colBounce[index][1])  / iters ;
	  colIters[index][2] = ((colIters[index][2] * (iters - 1)) + colBounce[index][2])  / iters ;*/
	  //colors[index ][0] = (colors[index][0] + colBounce[index][0] )/(iters+1) ; //colBounce[index] ;//
	  //colors[index ][1] = (colors[index][1] + colBounce[index][1] )/(iters+1)  ;
	  //colors[index ][2] = (colors[index][2] + colBounce[index][2] )/(iters+1)  ;

	  colors[index ] = (colors[index ] *(iters-1) + colBounce[index])/ (iters)  ;
	 // colBounce[index ] = glm::vec3(1,1,1);
	  //r[index ].rcolor = glm::vec3(1,1,1);
  }
}

__global__ void initializeray(glm::vec2 resolution, float time,cameraData cam, ray* r,glm::vec3* colBounce,glm::vec3* colIters){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if((x<=resolution.x && y<=resolution.y)){
  ray rnew = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
  
  //Depth of Field 
  glm::vec3 dofRayPoint = rnew.origin + 14.0f * glm::normalize(rnew.direction) ;	
  thrust::default_random_engine rng (hash (time ));
  thrust::uniform_real_distribution<float> xi6(-1,1);
  thrust::uniform_real_distribution<float> xi7(-1,1);
  thrust::uniform_real_distribution<float> r1(-1.0,1.0);
//	srand(time);		
			float dx =  r1(rng) ;//* cos(xi6(rng));   //((int)xi6(rng) % 100 + 1 )/1000;//
			float dy =  r1(rng) ;//* sin(xi7(rng));    //((int)xi7(rng)  % 100 + 1 )/1000; //
			
			rnew.origin    =  rnew.origin  + glm::vec3(dx,dy,0.0f);	
			rnew.direction =  glm::normalize(dofRayPoint - rnew.origin );
			
			
		


  r[index].direction = glm::normalize(rnew.direction);
  r[index].origin = rnew.origin;
  r[index].x = x ;
  r[index].y = y ;
  r[index].life = false ;
  r[index].rcolor = glm::vec3(1,1,1);
  colBounce[index] = glm::vec3(0,0,0);
  colIters[index] = glm::vec3(0,0,0);
  }
}
//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, float bounce, cameraData cam, int rayDepth, glm::vec3* colors, 
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials,ray* newr, glm::vec3* colBounce, int bou,int num,int blockdim,glm::vec3* myvertex, int numVertices,float *m ){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * blockdim* blockDim.x );//


if ( index < num )
{
 
	//if(bounce < 1.5f)
	//{
	//geoms[4].translation[0]+=0.1;
	//glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
	//geoms[4].transform = utilityCore::glmMat4ToCudaMat4(utilityCore::buildTransformationMatrix(geoms[4].translation,geoms[4].rotation,geoms[4].scale));
	//geoms[4].inverseTransform = utilityCore::glmMat4ToCudaMat4(glm::inverse(utilityCore::cudaMat4ToGlmMat4(geoms[4].transform)));
	//}
  //ray r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
  ray r = newr[index];

  int xx = r.x;
  int yy = r.y;
//  int newindex =  xx + (yy *  blockdim);

  glm::vec3 curIps;
  glm::vec3 curNorm;
  if((x<=resolution.x && y<=resolution.y)){
	
    float MAX_DEPTH = 100000000000000000;
    float depth = MAX_DEPTH;
	int geoIndex = -1; 
    for(int i=0; i<numberOfGeoms; i++){
        glm::vec3 intersectionPoint;
        glm::vec3 intersectionNormal;
       if(geoms[i].type==SPHERE){
           depth = sphereIntersectionTest(geoms[i], r, intersectionPoint, intersectionNormal);
        }else if(geoms[i].type==CUBE){
            depth = boxIntersectionTest(geoms[i], r, intersectionPoint, intersectionNormal);
        }else if(geoms[i].type==MESH){
			depth = boxIntersectionTest( glm::vec3(m[1],m[5],m[3]) ,glm::vec3(m[0],m[4],m[2]),geoms[i],r,intersectionPoint, intersectionNormal);
			if (depth != -1)
				depth = meshIntersectionTest(geoms[i],r,myvertex,numVertices,intersectionPoint, intersectionNormal);
        }else{
            //lol?
        }
        if(depth<MAX_DEPTH && depth>-EPSILON){
          MAX_DEPTH = depth;
		  geoIndex = i;
		  curIps  =  intersectionPoint;
		  curNorm =  intersectionNormal;
        }
    }
	// If you are hitting a object that is not light
	if(geoIndex != -1 && materials[geoms[geoIndex].materialid].emittance < 0.01f && (r.life == false))
	{
	
		thrust::default_random_engine rng (hash (time * index * bou));
		thrust::uniform_real_distribution<float> xi1(0,1);
		thrust::uniform_real_distribution<float> xi2(0,1);

		// If the object that you hit is not reflective
		if ( materials[geoms[geoIndex].materialid].hasReflective < 0.01f &&  materials[geoms[geoIndex].materialid].hasRefractive < 0.01f)
		{
			newr[index].direction =  glm::normalize(calculateRandomDirectionInHemisphere(glm::normalize(curNorm),  (float)xi1(rng),(float)xi2(rng)));
			newr[index].origin    =  curIps + newr[index].direction  * 0.001f ; //glm::vec3 neyep = dips + ref1 * 0.001f ;
			newr[index].rcolor    =  newr[index].rcolor * materials[geoms[geoIndex].materialid].color;
		}
		// If the object that you hit is reflective
		else if ( materials[geoms[geoIndex].materialid].hasReflective > 0.01f &&  materials[geoms[geoIndex].materialid].hasRefractive < 0.01f) 
		{
			// Reflectitivity works based on probabbility of the random number generated 
			thrust::uniform_real_distribution<float> xi3(0,1);
			float rtest =  (float)xi3(rng) ;
			if( rtest < materials[geoms[geoIndex].materialid].hasReflective)
			{
				glm::vec3 inc = glm::normalize(newr[index].direction)  ; 
				newr[index].direction = inc - (2.0f * glm::normalize(curNorm) * (glm::dot(glm::normalize(curNorm),inc))); //glm::vec3 ref1  =  lig - (2.0f * dnorm * (glm::dot(dnorm,lig))); 
				newr[index].rcolor    =  newr[index].rcolor * materials[geoms[geoIndex].materialid].specularColor;
			}
			else
			{
				newr[index].direction =  glm::normalize(calculateRandomDirectionInHemisphere(glm::normalize(curNorm),  (float)xi1(rng),(float)xi2(rng)));
				newr[index].rcolor    =  newr[index].rcolor * materials[geoms[geoIndex].materialid].color;
			}
			newr[index].origin    =  curIps + newr[index].direction  * 0.001f ; //glm::vec3 neyep = dips + ref1 * 0.001f ;
	
		}

		// If the object that you hit is refractive
		if ( materials[geoms[geoIndex].materialid].hasRefractive > 0.01f)
		{
			thrust::uniform_real_distribution<float> xi4(0,1);
			float rfr = (float)xi4(rng) ;
			if (rfr < 0.7)//materials[geoms[geoIndex].materialid].hasRefractive )
			{
				float n1 = 1.0f;
				float n2 = materials[geoms[geoIndex].materialid].hasRefractive;
				float angleofincidence  = acos(glm::dot(newr[index].direction ,glm::normalize(curNorm))/(glm::length(newr[index].direction) * glm::length(newr[index].direction)));
				angleofincidence  = abs(angleofincidence * (180.0f/PI));
				//float angleofreflection = asin(sin(angleofincidence) * (n1/n2));
				float io = glm::dot( glm::normalize(newr[index].direction),glm::normalize(curNorm));
				float criticalAngle = asin(n2/n1);// * (180.0f/PI) ;

				if(io < 0.0f  )
				{
					glm::vec3 refractedray = glm::refract(glm::normalize(newr[index].direction),glm::normalize(curNorm),(n1/n2));
					newr[index].direction  = glm::normalize(refractedray);
					newr[index].origin     =  curIps + newr[index].direction  * 0.001f ;	
					//newr[index].rcolor    =  newr[index].rcolor * materials[geoms[geoIndex].materialid].color;
				}
				else if(io >= 0.0f  ) // && (angleofincidence < criticalAngle )
				{
					glm::vec3 refractedray = glm::refract(glm::normalize(newr[index].direction),-1.0f * glm::normalize(curNorm),(n2/n1));
					newr[index].direction  = glm::normalize(refractedray);
					newr[index].origin     =  curIps + newr[index].direction  * 0.001f ;	
					//newr[index].rcolor    =  newr[index].rcolor * materials[geoms[geoIndex].materialid].color;
				}

			}
			else
			{
				glm::vec3 inc = glm::normalize(newr[index].direction)  ; 
				newr[index].direction = inc - (2.0f * glm::normalize(curNorm) * (glm::dot(glm::normalize(curNorm),inc))); //glm::vec3 ref1  =  lig - (2.0f * dnorm * (glm::dot(dnorm,lig))); 
				newr[index].rcolor    =  newr[index].rcolor * materials[geoms[geoIndex].materialid].specularColor;
				newr[index].origin    =  curIps + newr[index].direction  * 0.001f ;	
			}
		}



	
	}
	// If the ray hits an object that is light
	else if(geoIndex != -1 && materials[geoms[geoIndex].materialid].emittance > 0.01f && (r.life == false))
	{
	
		newr[index].rcolor =  newr[index].rcolor  * materials[geoms[geoIndex].materialid].emittance;
		newr[index].life = true;

	}
	// If the ray keeps hitting the light once it dies - This case actually never happens
	else if(geoIndex != -1 && materials[geoms[geoIndex].materialid].emittance > 0.01f && (r.life == true))
	{
	
		newr[index].rcolor =  newr[index].rcolor ;
		newr[index].life = true;

	}
	// The final case where the ray does not hit any object at all 
	else
	{
	
		newr[index].rcolor =  newr[index].rcolor * glm::vec3(0,0,0);
		newr[index].life = true;

	}
	

   }
}
}

  //A thrust based structure 
   struct is_dead
  {
    __host__ __device__
    bool operator()(const ray r)
    {
		return r.life;
    }
  };



//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms,std::vector<glm::vec3> mypoints,float *maxmin ){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  int numVertices = mypoints.size();
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)) , (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //Send vertices of the mesh to GPU
  glm::vec3* mvertex = NULL;
  cudaMalloc((void**)&mvertex,mypoints.size() * sizeof(glm::vec3));
 
  for(int i=0; i < mypoints.size(); i++){
	   
	   cudaMemcpy( &mvertex[i] , &mypoints[i], sizeof(glm::vec3), cudaMemcpyHostToDevice);
  }
  

    //Send maxmins of the mesh to GPU
  float* mami = NULL;
  cudaMalloc((void**)&mami,6 * sizeof(float));
   if(maxmin != NULL)
  {
  for(int i=0; i < 6; i++){
	   
	   cudaMemcpy( &mami[i] , &maxmin[i], sizeof(float), cudaMemcpyHostToDevice);
  }
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
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  material* cudamaterials = NULL;
  cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

   //Allocate memory for ray pool
  ray* raypool = NULL;
  cudaMalloc((void**)&raypool, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(ray));
 
  //Allocate memory to store color for bounces
  glm::vec3* colorBounce = NULL;
  cudaMalloc((void**)&colorBounce, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));

  //Allocate memory to store color for each iteration accumulation
  glm::vec3* colorIters = NULL;
  cudaMalloc((void**)&colorIters , (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));

  //Initialize the ray values
  initializeray<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,(float)iterations,cam,raypool,colorBounce,colorIters);
  cudaThreadSynchronize();
  ////kernel launches

 // const int N = 6;
 // int A[N] = {1, 4, 2, 8, 5, 7};
 // int *new_end = thrust::remove_if(A, A + N, is_even());
   // ray* raystart = new ray[N] ;
  //cudaMemcpy( raystart, raypool, N*sizeof(ray), cudaMemcpyDeviceToHost);
  //for(int j=0 ; j < N ; j++)
	 // std::cout << raystart[j].life ;
  //delete [] raystart;

  //Super-sampled antialiasing code 
  srand(iterations);
  float x = 0.0f , y = 0.0f ;
   
  if(iterations%20 == 0 )
  {
	x = (rand() % 100 + 1)/1000.0f;
	y = (rand() % 100 + 1)/1000.0f;
	cam.position[0] +=x;
	cam.position[1] +=y;
  }

  // Motion blur
  int mID = 5;
  float raa = (rand() % 10 + 1 )/ 10.0f ;
  float xtrans =  (2.0f * (1.0f - raa)) + (3.0f * raa) ;
  geoms[mID].translations[0][0] = xtrans;
  glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
  geoms[mID].transforms[0] = utilityCore::glmMat4ToCudaMat4(utilityCore::buildTransformationMatrix(geoms[mID].translations[0],geoms[mID].rotations[0],geoms[mID].scales[0]));
  geoms[mID].inverseTransforms[0] = utilityCore::glmMat4ToCudaMat4(glm::inverse(utilityCore::cudaMat4ToGlmMat4(geoms[mID].transforms[0])));


  int N  = ((int)renderCam->resolution.x*(int)renderCam->resolution.y);
  dim3 StreamBlocksPerGrid = fullBlocksPerGrid ;
  int blockdim = fullBlocksPerGrid.x ;
  for(int bounce = 1; bounce <=5; ++bounce)
  {   
 
  raytraceRay<<<StreamBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, (float)bounce, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials,raypool,colorBounce,bounce,N,blockdim,mvertex,numVertices,mami);
  raytoColorbouncecopy<<<StreamBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,colorBounce,raypool,N,blockdim);

  thrust::device_ptr<ray> rptr = thrust::device_pointer_cast(raypool);  
  ray *endrptr = thrust::remove_if(rptr,rptr + N , is_dead()).get();
  N =  endrptr - raypool  ;
  int numofBlocks = ceil((float)N / (float)(tileSize * tileSize)) ;
  blockdim = ceil(sqrt((float)numofBlocks));
  StreamBlocksPerGrid = dim3(blockdim,blockdim);


  //int rows = (int)N / ( (int)renderCam->resolution.x)    ;
  //int rem = N %  ( (int)renderCam->resolution.x) ;
  //if ( rem != 0 )
	 // rows = rows + 1 ;
  ////StreamBlocksPerGrid = dim3((int)ceil((float)N / (float)tileSize ),(int)((float)N / (float)tileSize )) ;
  //StreamBlocksPerGrid = dim3((int)ceil(float(renderCam->resolution.x)/float(tileSize)) , (int)ceil(float(rows)/float(tileSize)));
 // StreamBlocksPerGrid = dim3(rows , rows ) ;
  cudaThreadSynchronize();
  }
  //int j = 5 ;
  finalizeraycolor<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,colorBounce,cudaimage,(float)iterations);
  cudaThreadSynchronize();
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamaterials );
  cudaFree(colorBounce);
  cudaFree(colorIters);
  cudaFree(raypool);
  delete [] geomList;

  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}


float __device__ meshIntersectionTest(staticGeom curGeom,ray s,glm::vec3* myvertex, int numVertices, glm::vec3& mintersect, glm::vec3& mnormal)
{
		glm::vec3 ipss,normss;
		float t , at = 12345.0;
		glm::vec3 curnorm , curipss;

		for(int k=0 ;k < numVertices - 2 ; k= k+3)          
		{
			t = triangleIntersectionTest(curGeom,s,myvertex[k],myvertex[k+1],myvertex[k+2], ipss, normss);
			if(t != -1  && t<at)
			{
				curnorm  = normss;
				curipss  = ipss;
				at = t;
			}
		}  

		mnormal    = curnorm;
		mintersect = curipss;
		if (at == 12345.0)
			return -1;
		else
			return  at ;

}
