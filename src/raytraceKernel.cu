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

using namespace glm;

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
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
	//vec3 jitter = 2.0f*generateRandomNumberFromThread(resolution, time, x, y);
	vec3 jitter = vec3(0,0,0);	//no antialiasing 
	
	float NDCx = ((float)x +jitter.x)/resolution.x;
	float NDCy = ((float)y +jitter.y )/resolution.y;
	
	//float NDCx = ((float)x )/resolution.x;
	//float NDCy = ((float)y )/resolution.y;

	vec3 A = cross(view, up);
	vec3 B = cross(A, view);

	vec3 M = eye+view;
	vec3 V = B * (1.0f/length(B)) * length(view)*tan(radians(fov.y));
	vec3 H = A * (1.0f/length(A)) * length(view)*tan(radians(fov.x));

	vec3 point = M + (2*NDCx -1)*H + (1-2*NDCy)*V;

	ray r;
	r.origin = eye;
	r.direction = normalize(point-eye);
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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, float iterations){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x/iterations*255.0;
      color.y = image[index].y/iterations*255.0;
      color.z = image[index].z/iterations*255.0;

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

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(ray* cudarays, glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, int numberOfCubes, int numberOfSpheres, material* cudamaterials, 
							int numberOfMaterials, int numBounce, int* cudaalive, int initialMaxRays){

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int y = cudarays[index].pixelID/resolution.y;
	int x = cudarays[index].pixelID-resolution.x*y;

	if (index < initialMaxRays){

		float tempLength, closest = 1e26, indexOfRefraction = 0;
		int closestObjectid = -1;
		vec3 tempIntersectionPoint = vec3(0,0,0), tempNormal = vec3(0,0,0), normal = vec3(0,0,0), intersectionPoint = vec3(0,0,0);
		vec3 objectColor = vec3(0,0,0), specColor = vec3(0,0,0);
		float specExponent = 0, 
		bool isReflective = 0, isRefractive = 0;
		bool inside = false, tempInside = false;

		//input text file must load cubes first before loading spheres

		for (int i = 0; i < numberOfCubes; i++){
			if(geoms[i].type == CUBE){
				tempLength = boxIntersectionTest( geoms[i], cudarays[index], tempIntersectionPoint, tempNormal, tempInside);
			}

			if (tempLength < closest && tempLength >= 0){
				closest = tempLength;
				normal = tempNormal;
				intersectionPoint = tempIntersectionPoint;
				closestObjectid = i;
				inside = tempInside;
			}
		}

		for(int i = numberOfCubes; i < numberOfGeoms; i++){
			if(geoms[i].type == SPHERE){
				tempLength = sphereIntersectionTest( geoms[i], cudarays[index], tempIntersectionPoint, tempNormal, tempInside);
			}

			if (tempLength < closest && tempLength >= 0){
				closest = tempLength;
				normal = tempNormal;
				intersectionPoint = tempIntersectionPoint;
				closestObjectid = i;
				inside = tempInside;
			}
		}
			 
		if (closest < 1e26 && closest >= 0){

			objectColor = cudamaterials[geoms[closestObjectid].materialid].color;
			specExponent = cudamaterials[geoms[closestObjectid].materialid].specularExponent;
			specColor = cudamaterials[geoms[closestObjectid].materialid].specularColor;
			isReflective = cudamaterials[geoms[closestObjectid].materialid].hasReflective;
			isRefractive = cudamaterials[geoms[closestObjectid].materialid].hasRefractive;
			indexOfRefraction = cudamaterials[geoms[closestObjectid].materialid].indexOfRefraction;

			vec3 reflectedDir = cudarays[index].direction - vec3(2*vec4(normal*(dot(cudarays[index].direction,normal)),0));
			reflectedDir = normalize(reflectedDir);
			vec3 refractedDir = vec3(0,0,0);
			
			
			if (cudamaterials[geoms[closestObjectid].materialid].emittance > 0){
				cudarays[index].color *= cudamaterials[geoms[closestObjectid].materialid].color*cudamaterials[geoms[closestObjectid].materialid].emittance;
				cudarays[index].alive = false;
				colors[cudarays[index].pixelID] += cudarays[index].color;
				cudaalive[index] = 0; //dead
				return;
			}

			float n1 = 0, n2 = 0;
			float costheta_i = 0; float costheta_t = 0;
			float sin2theta_t = 0;
			float R = 0;
			bool TIR = false;
			float schlicksR = 0;
			float random = 0;

			if (isRefractive){

				//graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf

				if (inside){
					n1 = indexOfRefraction;
					n2 = 1.0f;
					normal = -normal;
				}else{
					n1 = 1.0f;
					n2 = indexOfRefraction;
				}

				costheta_i = glm::dot(-1.0f*cudarays[index].direction, normal);
				sin2theta_t = pow(n1/n2,2)*(1-pow(costheta_i,2));
				R = pow((n1-n2)/(n1+n2),2);
				if (sin2theta_t > 1){
					TIR = true;
				}else{
					costheta_t = sqrt(1-sin2theta_t);
					refractedDir = (n1/n2)*cudarays[index].direction + ((n1/n2)*costheta_i - sqrt(1-sin2theta_t))*normal;
				}

				if (n1 <= n2){
					schlicksR = R + (1-R)*(1-costheta_i)*(1-costheta_i)*(1-costheta_i)*(1-costheta_i)*(1-costheta_i);
				}else if (n1 > n2 && !TIR){
					schlicksR = R + (1-R)*(1-costheta_t)*(1-costheta_t)*(1-costheta_t)*(1-costheta_t)*(1-costheta_t);
				}else{
					schlicksR = 1;
				}
  
				thrust::default_random_engine rng(hash((cudarays[index].pixelID)*time));
				thrust::uniform_real_distribution<float> u01(0,1);

				random = (float) u01(rng);
					
				cudarays[index].origin = intersectionPoint+0.01f*refractedDir;
				cudarays[index].direction = refractedDir;
					
				if (random <= schlicksR){
					cudarays[index].origin = intersectionPoint+0.0001f*reflectedDir;
					cudarays[index].direction = reflectedDir;
				}

			}else if (isReflective){
				cudarays[index].origin = intersectionPoint+0.01f*reflectedDir;
				cudarays[index].direction = reflectedDir;
			}else{ //just diffuse
				//vec3 rand = generateRandomNumberFromThread(resolution, time*(numBounce+1), x, y);
				thrust::default_random_engine rng1(hash(time*(numBounce + 1)* index));
                thrust::uniform_real_distribution<float> u02(0,time);
				thrust::default_random_engine rng(hash((float)u02(rng1)*(numBounce + 1)* index));
                thrust::uniform_real_distribution<float> u01(0,1);
				
				if((float)u01(rng) < 0.1) // russian roulette rule: ray is absorbed
                {
					cudarays[index].color *= cudamaterials[geoms[closestObjectid].materialid].color*cudamaterials[geoms[closestObjectid].materialid].emittance;
					cudarays[index].alive = false;
					colors[cudarays[index].pixelID] += cudarays[index].color;
					cudaalive[index] = 0; //dead
					return;
                }
                else
                {
					vec3 outgoingDir = calculateRandomDirectionInHemisphere(normal, (float)u01(rng), (float)u01(rng));
                    cudarays[index].origin = intersectionPoint+0.01f*outgoingDir;
                    cudarays[index].direction = outgoingDir;
				}

				/*vec3 outgoingDir = calculateRandomDirectionInHemisphere(normal, (float)u01(rng), (float)u01(rng));
				cudarays[index].origin = intersectionPoint+0.001f*outgoingDir;
				cudarays[index].direction = outgoingDir;*/
			}

			cudarays[index].color *= objectColor;

		}//if intersects with anything
		else{
			cudarays[index].color *= vec3(0,0,0);
			cudarays[index].alive = false;
			colors[cudarays[index].pixelID] += cudarays[index].color;
			cudaalive[index] = 0; //dead
			//numAliveRays[0]--;
			return;
		}

	}//end of ifstatement
}

//INITIALIZES A POOL OF RAYS
__global__ void initializeRays(glm::vec2 resolution, float time, cameraData cam, ray* cudarays){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if((x<=resolution.x && y<=resolution.y)){

		ray rayFromCamera = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);

		//find aim point
		vec3 aimPoint = rayFromCamera.origin + cam.focalLength*rayFromCamera.direction;

		//jittered ray (DOF)
		float degOfJitter = 1;
		vec3 jitter = generateRandomNumberFromThread(resolution, time, x, y);
		ray jitteredRay;
		jitteredRay.origin = vec3(rayFromCamera.origin.x+degOfJitter*jitter.x, rayFromCamera.origin.y+degOfJitter*jitter.y, rayFromCamera.origin.z);	
		jitteredRay.direction = normalize(aimPoint-jitteredRay.origin);

		ray currentRay = rayFromCamera; //jitteredRay;
		currentRay.pixelID = index;
		currentRay.color = vec3(1,1,1);
		currentRay.alive = true;
		cudarays[index] = currentRay;	//stores ray

	}
}

__global__ void scan(int* cudacondition, int* cudatemp, int d){
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index == 0)
		cudatemp[0] = cudacondition[0];

	int e = pow(2.0f,d-1);	//speed up this later
	if (index >= e){
		cudatemp[index] = cudacondition[index-e] + cudacondition[index];
	}else{
		cudatemp[index]  = cudacondition[index];
	}

}

__global__ void streamCompact( int* cudaalive, ray* cudarays, ray* cudaraysTemp, int numRays){
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numRays){
		if(cudarays[index].alive){						//compare to see if ray is alive or dead
			cudaraysTemp[cudaalive[index]-1] = cudarays[index];
		}
	}
}

__global__ void resetAliveConditionArray( int* cudaalive){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	cudaalive[index] = 1;
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, int numberOfCubes, int numberOfSpheres, bool cameraMoved){
  
  int traceDepth = 200; //determines how many bounces the pathtracer traces
  std::vector<int> lightsid;

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock2d(tileSize, tileSize);
  dim3 fullBlocksPerGrid2d((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  dim3 threadsPerBlock1d(tileSize*tileSize);
  float s = renderCam->resolution.x*renderCam->resolution.y;
  dim3 fullBlocksPerGrid1d((int)ceil((float(renderCam->resolution.x)/float(tileSize))*(float(renderCam->resolution.y)/float(tileSize))));



  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
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

  int numberOfPixels = renderCam->resolution.x*renderCam->resolution.y;
  ray* cudarays = NULL;
  cudaMalloc((void**)&cudarays, numberOfPixels*sizeof(ray));
    
  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;
  cam.focalLength = renderCam->focalLengths[frame];

  //clear image
  if (cameraMoved)
	clearImage<<<fullBlocksPerGrid2d, threadsPerBlock2d>>>(renderCam->resolution,cudaimage);

  if (numberOfGeoms != numberOfCubes+numberOfSpheres){
	  std::cout<<"ERROR numberOfGeoms != numberOfCubes+numberOfSpheres"<<std::endl;
	  std::cout<<numberOfGeoms<<", "<<numberOfCubes<<", "<<numberOfSpheres<<std::endl;
  }

  //initial pool of rays
  initializeRays<<<fullBlocksPerGrid2d, threadsPerBlock2d>>>(renderCam->resolution, (float)iterations, cam, cudarays);

  //intialize the alive array 
  int* cudaalive = NULL;
  cudaMalloc((void**)&cudaalive, numberOfPixels*sizeof(int));
  resetAliveConditionArray<<<fullBlocksPerGrid1d, threadsPerBlock1d>>>( cudaalive);

  int* cudatemp = NULL;
  cudaMalloc((void**)&cudatemp, numberOfPixels*sizeof(int));


  ray* cudaraysTemp = NULL;
  cudaMalloc((void**)&cudaraysTemp, numberOfPixels*sizeof(ray));

  int numRays = renderCam->resolution.x*renderCam->resolution.y;

  //kernel launches
  for (int i = 0; i < traceDepth && numRays > 0; i++){

	raytraceRay<<<fullBlocksPerGrid1d, threadsPerBlock1d>>>(cudarays, renderCam->resolution, (float)iterations, cam, traceDepth, 
	cudaimage, cudageoms, numberOfGeoms, numberOfCubes, numberOfSpheres, cudamaterials, numberOfMaterials, i, cudaalive, numRays);

	int log2n = (int)ceil(log(float(numRays)) / log(2.0f));
	for (int d = 1; d <= log2n; d++){
		scan<<<fullBlocksPerGrid1d, threadsPerBlock1d>>>( cudaalive, cudatemp, d);			//scan
		int* temp = cudaalive;
		cudaalive = cudatemp;
		cudatemp = temp;
	}

    int numAliveRaysCPU = 0;

	//cudaalive now has the summed corresponding new indices for the alive rays
	cudaMemcpy(&numAliveRaysCPU, &cudaalive[numRays-1], sizeof(int), cudaMemcpyDeviceToHost);
	
	streamCompact<<<fullBlocksPerGrid1d, threadsPerBlock1d>>>( cudaalive, cudarays, cudaraysTemp, numRays);

	ray* tempR = cudarays;
	cudarays = cudaraysTemp;
	cudaraysTemp = tempR;

	resetAliveConditionArray<<<fullBlocksPerGrid1d, threadsPerBlock1d>>>( cudaalive);

	numRays = numAliveRaysCPU;
	fullBlocksPerGrid1d = dim3((int)ceil((float(numRays)/float(tileSize*tileSize))));
  }

  sendImageToPBO<<<fullBlocksPerGrid2d, threadsPerBlock2d>>>(PBOpos, renderCam->resolution, cudaimage, (float)iterations);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudarays  );
  cudaFree(cudaalive);
  cudaFree(cudatemp);
  cudaFree(cudaraysTemp);
  delete geomList;



  // make certain the kernel has completed
  cudaThreadSynchronize();
  checkCUDAError("Kernel failed!");

}