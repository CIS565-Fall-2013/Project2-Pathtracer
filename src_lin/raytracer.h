#pragma once

#include "sceneDesc.h"
#include "ColorImage.h"

class RayTracer
{
public: 
    RayTracer(){}
    virtual ~RayTracer(){}
    virtual void renderImage( const SceneDesc &scene, ColorImage &img ){};

};