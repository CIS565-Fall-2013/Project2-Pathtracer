#include "sceneDesc.h"

SceneDesc::SceneDesc(void)
{
    width = 0;
    height = 0;
    modelCount = 0;
    this->rayDepth = 5;
}


SceneDesc::~SceneDesc(void)
{
    while( !primitives.empty() )
    {
        Shape* p = primitives.back();
        if( p )
        {
            delete p;
            p = NULL;
        }
        primitives.pop_back();

    }

    for( int i = 0; i < modelCount; ++i )
        glmDelete( model[i] );
}