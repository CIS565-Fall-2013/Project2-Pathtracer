#include "FileParser.h"
#include "Light.h"
#include "shape.h"
#include "sphere.h"
#include "triangle.h"
#include "bbox.h"
#include "material.h"
#include "transform.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <array>
#include "glm/glm.hpp"

//glm wavefront obj loader
#include "glm.h"


using namespace std;
using namespace glm;

// Function to read the input data values
// Use is optional, but should be very helpful in parsing.  
bool floatParse(stringstream &s, const int numvals, float* values) 
{
    for (int i = 0; i < numvals; i++) {
        s>>values[i]; 
        if (s.fail()) 
        {
            cout << "Failed reading value " << i << " will skip\n"; 
            return false;
        }
    }
    return true; 
}

bool stringParse( stringstream &s, const int numvals, string* str )
{
    for( int i = 0; i < numvals; i++ )
    {
        s>>str[i];
        if (s.fail()) 
        {
            cout << "Failed reading value " << i << " will skip\n"; 
            return false;
        }
    }
}

void rightmultiply(const mat4 & M, vector<mat4> &transfstack) 
{
    mat4 &T = transfstack.back(); 
    T = T * M; 
}

FileParser::FileParser(void)
{
}


FileParser::~FileParser(void)
{
}

int FileParser::parse( const char input[], SceneDesc& sceneDesc )
{
    fstream inFile;
    string itemName;
    string lineStr;
    string output;
    //stringstream lineSStr;
    char lineBuf[256];
    float param[40];
    string modelName;

    vec3 diffuse(0.0f);
    vec3 specular(0.0f);
    vec3 emission(0.0f);
    vec3 ambient(0.0f);
    float shininess = 0;
    
    float attenu_const = 0;
    float attenu_linear = 0;
    float attenu_quadratic = 1;

    vector<mat4> transtack;
    int maxvert = 0;
    int maxnorm = 0;
    vector<vec3> vertices;  //vertices without normal vectors
    vector<vec3> vertnorms; //vertices with normal vectors

    GLMmodel* model;

    transtack.push_back( mat4(1.0) );
    
    inFile.open( input );
    if( !inFile.is_open() )
        return -1;

    while(1)
    {
        inFile.getline( lineBuf, 255 );
        
        lineStr = lineBuf;
        stringstream lineSStr( lineStr );
        
        if( inFile.eof() )
            break;
        if( lineStr.find_first_not_of( " \t\r\n" ) == string::npos || lineStr[0] == '#' )
            continue;

        lineSStr>>itemName;

        if( itemName == "camera" )
        {
            if( floatParse( lineSStr, 10, param ) )
            {
                sceneDesc.eyePos = vec3( param[0], param[1], param[2] );
                sceneDesc.eyePosHomo = vec4( param[0], param[1], param[2], 1.0f );
                sceneDesc.center = vec3( param[3], param[4], param[5] );
                sceneDesc.up = vec3( param[6], param[7], param[8] );
                sceneDesc.fovy = param[9];

                //convert the fovy to radian unit
                sceneDesc.fovy = sceneDesc.fovy * pi /180.0;
            }


        }
        else if( itemName == "size" )
        {
            if( floatParse( lineSStr, 2, param ) )
            {
                sceneDesc.width = param[0];
                sceneDesc.height = param[1];
            }
        }
        else if( itemName == "maxdepth" )
        {
            if( floatParse( lineSStr, 1, param ) )
            {
                sceneDesc.rayDepth = param[0];
            }
        }
        else if( itemName == "output" )
        {
            lineSStr>>output;
            cout<<output<<endl;
        }
        else if( itemName == "mtl" )
        {
            if( floatParse( lineSStr, 13, param ) )// emission ambient diffuse specular shininess
            {
                Material mtl;
                mtl.emission = vec3( param[0], param[1], param[2] );
                mtl.ambient = vec3( param[3], param[4], param[5] );
                mtl.diffuse = vec3( param[6], param[7], param[8] );
                mtl.specular = vec3( param[9], param[10], param[11] );
                mtl.shininess = param[12];
                sceneDesc.mtls.push_back( mtl );
            }
        }
        else if( itemName == "directional" || itemName == "point" )
        {
            if( floatParse( lineSStr, 6, param ) )
            {
                Light light;
                light.pos = vec4( param[0], param[1], param[2], 1 );
                

                if( itemName == "directional" )
                {
                    light.pos[3] = 0;
                    light.type = 1;
                }
                else
                    light.type = 0;

                light.pos = transtack.back() * light.pos ;
                light.color = vec3( param[3], param[4], param[5] );

                light.attenu_const = attenu_const;
                light.attenu_linear = attenu_linear;
                light.attenu_quadratic = attenu_quadratic;

                sceneDesc.lights.push_back( light );
            }
        }
        else if(  itemName == "area"  ) //area light
        {
            if( floatParse( lineSStr, 10, param ) )
            {
                Light light;
                light.type = 2;
                light.pos = vec4( param[0], param[1], param[2], 1 );
                light.pos = transtack.back() * light.pos;

                light.width = param[3];
                light.normal = vec3( param[4], param[5], param[6] );
                light.normal = mat3( transpose( inverse( transtack.back() ) ) ) * light.normal;

                light.color = vec3( param[7], param[8], param[9] );

                light.attenu_const = attenu_const;
                light.attenu_linear = attenu_linear;
                light.attenu_quadratic = attenu_quadratic;
                sceneDesc.lights.push_back( light );

            }
        }
        else if( itemName == "attenuation" )
        {
            if( floatParse( lineSStr, 3, param ) )
            {
                attenu_const = param[0];
                attenu_linear = param[1];
                attenu_quadratic = param[2];

            }
        }
        else if( itemName == "translate" )
        {
            if( floatParse( lineSStr, 3, param ) )
            {
                mat4 m = Transform::translate( param[0], param[1], param[2] ) ;
                rightmultiply( m, transtack );
            }
        }
        else if( itemName == "rotate" )
        {
            if( floatParse( lineSStr, 4, param ) )
            {
                mat3 m = Transform::rotate( param[3], vec3( param[0], param[1], param[2] ) ) ;
                mat4 m4x4 = mat4( m );
                rightmultiply( m4x4, transtack );
            }
        }
        else if( itemName == "scale" )
        {
            if( floatParse( lineSStr, 3, param ) )
            {
                mat4 m = Transform::scale( param[0], param[1], param[2] ) ;
                rightmultiply( m, transtack );
            }
        }
        else if( itemName == "pushTransform" )
        {
            transtack.push_back( transtack.back() );
        }
        else if( itemName == "popTransform" )
        {
            if( transtack.size() <= 1 )
                cout<<"No more transform matrix could be poped\n";
            else
                transtack.pop_back();
        }
        else if( itemName == "sphere" )
        {
            if( floatParse( lineSStr, 4, param ) )
            {
                Sphere *pSphere = new Sphere();
                pSphere->center = vec4( param[0], param[1], param[2], 1.0 );
                pSphere->radius = param[3];

                pSphere->center = transtack.back() * pSphere->center;
                pSphere->radius *= transtack.back()[0][0];

                pSphere->mtl_idx = sceneDesc.mtls.size() - 1;

                sceneDesc.primitives.push_back(pSphere);
             
            }
        }
        else if( itemName == "maxverts" )
        {
            lineSStr>>maxvert;
            vertices.reserve( maxvert );
        }
        else if( itemName == "maxvertnorms" )
        {
            lineSStr>>maxnorm;
            vertnorms.reserve( maxnorm );
        }
        else if( itemName == "vertex" )
        {
            if( maxvert == 0 ) //No vertices, skip the parsing
                continue;
            if( floatParse( lineSStr, 3, param ) )
            {
                vec3 vertex( param[0], param[1], param[2] );
                vertices.push_back( vertex );
            }

        }
        else if( itemName == "vertexnormal" )
        {
            if( maxnorm == 0 ) //No vertices, skip the parsing
                continue;
            if( floatParse( lineSStr, 6, param ) )
            {
                vec3 vertex( param[0], param[1], param[2] );
                vec3 normal( param[3], param[4], param[5] );
                vertnorms.push_back( vertex );
                vertnorms.push_back( normal );
            }

        }
        else if( itemName == "tri" )
        {
           if( floatParse( lineSStr, 3, param ) )
           {
               Triangle *pTri = new Triangle();

               pTri->v[0] = vec3( transtack.back() * vec4( vertices[ (int)param[0] ], 1 ) );
               pTri->v[1] = vec3( transtack.back() * vec4( vertices[ (int)param[1] ], 1 ) );
               pTri->v[2] = vec3( transtack.back() * vec4( vertices[ (int)param[2] ], 1 ) );
               //calculate plane normal
               pTri->pn = normalize( cross( pTri->v[1] - pTri->v[0], pTri->v[2] - pTri->v[0] ) );


               pTri->mtl_idx = sceneDesc.mtls.size() - 1;



               sceneDesc.primitives.push_back( pTri );
           }
        }
        else if( itemName == "trinormal" )
        {
           if( floatParse( lineSStr, 6, param ) )
           {
               Triangle *pTri = new Triangle();

               pTri->v[0] = vec3( transtack.back() * vec4( vertices[ (int)param[0] ], 1 ) );
               pTri->v[1] = vec3( transtack.back() * vec4( vertices[ (int)param[1] ], 1 ) );
               pTri->v[2] = vec3( transtack.back() * vec4( vertices[ (int)param[2] ], 1 ) );
               pTri->pn = normalize( cross( pTri->v[1] - pTri->v[0], pTri->v[2] - pTri->v[0] ) );


               pTri->mtl_idx = sceneDesc.mtls.size() - 1;


               sceneDesc.primitives.push_back( pTri );
              
           }
        }
        else if( itemName == "model" )
        {
           // string modelName;
            if( stringParse( lineSStr, 1, &modelName ) )
            {   
                model = sceneDesc.model[sceneDesc.modelCount] = glmReadOBJ( const_cast<char*>(modelName.c_str()) );
                if( sceneDesc.model[sceneDesc.modelCount] != NULL )
                {
                    mat4 transform = transtack.back();
                    sceneDesc.modelCount+=1;
                    glmUnitize( model );

                    //make a bounding box for this obj  
                    Bbox* pBbox = new Bbox();
                    pBbox->min = vec3( transform * glm::vec4( -1, -1, -1, 1 ) );
                    pBbox->max = vec3( transform * glm::vec4( 1, 1, 1, 1 ) );
                    pBbox->polyNum = model->numtriangles;
                    sceneDesc.primitives.push_back( pBbox );

                    //parse triangles
                    GLMgroup* group = model->groups;
			        while( group )
                    {
                        GLMtriangle* triangle;
                        Triangle *pTri = new Triangle();
                        for( int i = 0; i < group->numtriangles; ++i )
                        {
                            Triangle *pTri = new Triangle();
                            triangle = &model->triangles[group->triangles[i]];

                            pTri->v[0] = vec3( transform * vec4( model->vertices[ 3 * triangle->vindices[0]],
                                                                 model->vertices[ 3 * triangle->vindices[0]+1],
                                                                 model->vertices[ 3 * triangle->vindices[0]+2], 1) );
                            pTri->v[1] = vec3( transform * vec4( model->vertices[ 3 * triangle->vindices[1]],
                                                                 model->vertices[ 3 * triangle->vindices[1]+1],
                                                                 model->vertices[ 3 * triangle->vindices[1]+2], 1) );
                            pTri->v[2] = vec3( transform * vec4( model->vertices[ 3 * triangle->vindices[2]],
                                                                 model->vertices[ 3 * triangle->vindices[2]+1],
                                                                 model->vertices[ 3 * triangle->vindices[2]+2], 1) );
                            pTri->pn = normalize( cross( pTri->v[0] - pTri->v[1], pTri->v[0] - pTri->v[2] ) );     
                            pTri->mtl_idx = sceneDesc.mtls.size() - 1;

                            sceneDesc.primitives.push_back( pTri );
                        }
                        group = group->next;
                    }
                }
                glmDelete( model );
            }
        }
    }

    return 0;
}
