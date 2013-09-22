#pragma once
#include "sceneDesc.h"

class FileParser
{
public:
    FileParser();
    ~FileParser();

    static int parse( const char input[], SceneDesc& sceneDesc );

};
