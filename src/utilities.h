//  UTILITYCORE- A Utility Library by Yining Karl Li
//  This file is part of UTILITYCORE, Coyright (c) 2012 Yining Karl Li
//  
//  File: utilities.h
//  Header for utilities.cpp

#ifndef Pathtracer_utilities_h
#define Pathtracer_utilities_h

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include "cudaMat4.h"

#define PI                          3.1415926535897932384626422832795028841971
#define TWO_PI                      6.2831853071795864769252867665590057683943
#define SQRT_OF_ONE_THIRD           0.5773502691896257645091487805019574556476
#define NATURAL_E                   2.7182818284590452353602874713526624977572
#define EPSILON                     .000000001
#define ZERO_ABSORPTION_EPSILON     0.00001
#define RAY_BIAS_AMOUNT             0.0002
#define MAX_DEPTH					3
#define SHADOWRAY_NUM				2												// controls how many shadow rays to send. Set to 1 for hard shadows
#define ANTIALIASING_SWITCH		    0												// 0 for no anti-aliasing
#define PATHTRACING_SWITCH			1												// 0 for ray tracing 1 for path tracing
#define MOTION_BLUR_SWITCH			1												// 0 for no motion blur.
#define SAMPLES_PER_PIXEL			1												// Number of samples to use per pixel. Min 1.
#define USE_STREAM_COMPACTION		1												// Use thrust::remove_if to manage ray pool
#define MAX_BOUNCE					10												// maximum number of times a path tracing ray can bounce

namespace utilityCore {
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str); 
    extern cudaMat4 glmMat4ToCudaMat4(glm::mat4 a);
    extern glm::mat4 cudaMat4ToGlmMat4(cudaMat4 a);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern void printCudaMat4(cudaMat4 m);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413

    //-----------------------------
    //-------GLM Printers----------
    //-----------------------------
    extern void printMat4(glm::mat4);
    extern void printVec4(glm::vec4);
    extern void printVec3(glm::vec3);
}
#endif
