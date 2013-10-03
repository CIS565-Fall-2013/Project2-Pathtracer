// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com
// Edited by Liam Boone for use with CUDA v5.5

#include <iostream>
#include "scene.h"
#include <cstring>

scene::scene(string filename){
	cout << "Reading scene from " << filename << " ..." << endl;
	cout << " " << endl;
	char* fname = (char*)filename.c_str();
	fp_in.open(fname);
	if(fp_in.is_open()){
		while(fp_in.good()){
			string line;
            utilityCore::safeGetline(fp_in,line);
			if(!line.empty()){
				vector<string> tokens = utilityCore::tokenizeString(line);
				if(strcmp(tokens[0].c_str(), "MATERIAL")==0){
				    loadMaterial(tokens[1]);
				    cout << " " << endl;
				}else if(strcmp(tokens[0].c_str(), "OBJECT")==0){
				    loadObject(tokens[1]);
				    cout << " " << endl;
				}else if(strcmp(tokens[0].c_str(), "CAMERA")==0){
				    loadCamera();
				    cout << " " << endl;
				}
			}
		}
	}
}

int scene::loadObject(string objectid){
    int id = atoi(objectid.c_str());
    if(id!=objects.size()){
        cout << "ERROR: OBJECT ID does not match expected number of objects" << endl;
        return -1;
    }else{
        cout << "Loading Object " << id << "..." << endl;
        geom newObject;
        string line;
       maxmin = NULL;  // Change this for obj loading and no obj loading
        //load object type 
        utilityCore::safeGetline(fp_in,line);
        if (!line.empty() && fp_in.good()){
            if(strcmp(line.c_str(), "sphere")==0){
                cout << "Creating new sphere..." << endl;
				newObject.type = SPHERE;
            }else if(strcmp(line.c_str(), "cube")==0){
                cout << "Creating new cube..." << endl;
				newObject.type = CUBE;
            }else{
				string objline = line;
                string name;
                string extension;
                istringstream liness(objline);
                getline(liness, name, '.');
                getline(liness, extension, '.');
                if(strcmp(extension.c_str(), "obj")==0){ //
					loadmesh(objline );
                    cout << "Creating new mesh..." << endl;
                    cout << "Reading mesh from " << line << "... " << endl;
		    		newObject.type = MESH;
                }else{
                    cout << "ERROR: " << line << " is not a valid object type!" << endl;
                    return -1;
                }
            }
        }
       
	//link material
    utilityCore::safeGetline(fp_in,line);
	if(!line.empty() && fp_in.good()){
	    vector<string> tokens = utilityCore::tokenizeString(line);
	    newObject.materialid = atoi(tokens[1].c_str());
	    cout << "Connecting Object " << objectid << " to Material " << newObject.materialid << "..." << endl;
        }
        
	//load frames
    int frameCount = 0;
    utilityCore::safeGetline(fp_in,line);
	vector<glm::vec3> translations;
	vector<glm::vec3> scales;
	vector<glm::vec3> rotations;
    while (!line.empty() && fp_in.good()){
	    
	    //check frame number
	    vector<string> tokens = utilityCore::tokenizeString(line);
        if(strcmp(tokens[0].c_str(), "frame")!=0 || atoi(tokens[1].c_str())!=frameCount){
            cout << "ERROR: Incorrect frame count!" << endl;
            return -1;
        }
	    
	    //load tranformations
	    for(int i=0; i<3; i++){
            glm::vec3 translation; glm::vec3 rotation; glm::vec3 scale;
            utilityCore::safeGetline(fp_in,line);
            tokens = utilityCore::tokenizeString(line);
            if(strcmp(tokens[0].c_str(), "TRANS")==0){
                translations.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }else if(strcmp(tokens[0].c_str(), "ROTAT")==0){
                rotations.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }else if(strcmp(tokens[0].c_str(), "SCALE")==0){
                scales.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }
	    }
	    
	    frameCount++;
        utilityCore::safeGetline(fp_in,line);
	}
	
	//move frames into CUDA readable arrays
	newObject.translations = new glm::vec3[frameCount];
	newObject.rotations = new glm::vec3[frameCount];
	newObject.scales = new glm::vec3[frameCount];
	newObject.transforms = new cudaMat4[frameCount];
	newObject.inverseTransforms = new cudaMat4[frameCount];
	for(int i=0; i<frameCount; i++){
		newObject.translations[i] = translations[i];
		newObject.rotations[i] = rotations[i];
		newObject.scales[i] = scales[i];
		glm::mat4 transform = utilityCore::buildTransformationMatrix(translations[i], rotations[i], scales[i]);
		newObject.transforms[i] = utilityCore::glmMat4ToCudaMat4(transform);
		newObject.inverseTransforms[i] = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));
	}
	
        objects.push_back(newObject);
	
	cout << "Loaded " << frameCount << " frames for Object " << objectid << "!" << endl;
        return 1;
    }
}

int scene::loadCamera(){
	cout << "Loading Camera ..." << endl;
        camera newCamera;
	float fovy;
	
	//load static properties
	for(int i=0; i<4; i++){
		string line;
        utilityCore::safeGetline(fp_in,line);
		vector<string> tokens = utilityCore::tokenizeString(line);
		if(strcmp(tokens[0].c_str(), "RES")==0){
			newCamera.resolution = glm::vec2(atoi(tokens[1].c_str()), atoi(tokens[2].c_str()));
		}else if(strcmp(tokens[0].c_str(), "FOVY")==0){
			fovy = atof(tokens[1].c_str());
		}else if(strcmp(tokens[0].c_str(), "ITERATIONS")==0){
			newCamera.iterations = atoi(tokens[1].c_str());
		}else if(strcmp(tokens[0].c_str(), "FILE")==0){
			newCamera.imageName = tokens[1];
		}
	}
        
	//load time variable properties (frames)
    int frameCount = 0;
	string line;
    utilityCore::safeGetline(fp_in,line);
	vector<glm::vec3> positions;
	vector<glm::vec3> views;
	vector<glm::vec3> ups;
    while (!line.empty() && fp_in.good()){
	    
	    //check frame number
	    vector<string> tokens = utilityCore::tokenizeString(line);
        if(strcmp(tokens[0].c_str(), "frame")!=0 || atoi(tokens[1].c_str())!=frameCount){
            cout << "ERROR: Incorrect frame count!" << endl;
            return -1;
        }
	    
	    //load camera properties
	    for(int i=0; i<3; i++){
            //glm::vec3 translation; glm::vec3 rotation; glm::vec3 scale;
            utilityCore::safeGetline(fp_in,line);
            tokens = utilityCore::tokenizeString(line);
            if(strcmp(tokens[0].c_str(), "EYE")==0){
                positions.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }else if(strcmp(tokens[0].c_str(), "VIEW")==0){
                views.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }else if(strcmp(tokens[0].c_str(), "UP")==0){
                ups.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }
	    }
	    
	    frameCount++;
        utilityCore::safeGetline(fp_in,line);
	}
	newCamera.frames = frameCount;
	
	//move frames into CUDA readable arrays
	newCamera.positions = new glm::vec3[frameCount];
	newCamera.views = new glm::vec3[frameCount];
	newCamera.ups = new glm::vec3[frameCount];
	for(int i=0; i<frameCount; i++){
		newCamera.positions[i] = positions[i];
		newCamera.views[i] = views[i];
		newCamera.ups[i] = ups[i];
	}

	//calculate fov based on resolution
	float yscaled = tan(fovy*(PI/180));
	float xscaled = (yscaled * newCamera.resolution.x)/newCamera.resolution.y;
	float fovx = (atan(xscaled)*180)/PI;
	newCamera.fov = glm::vec2(fovx, fovy);

	renderCam = newCamera;
	
	//set up render camera stuff
	renderCam.image = new glm::vec3[(int)renderCam.resolution.x*(int)renderCam.resolution.y];
	renderCam.rayList = new ray[(int)renderCam.resolution.x*(int)renderCam.resolution.y];
	for(int i=0; i<renderCam.resolution.x*renderCam.resolution.y; i++){
		renderCam.image[i] = glm::vec3(0,0,0);
	}
	
	cout << "Loaded " << frameCount << " frames for camera!" << endl;
	return 1;
}

int scene::loadMaterial(string materialid){
	int id = atoi(materialid.c_str());
	if(id!=materials.size()){
		cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
		return -1;
	}else{
		cout << "Loading Material " << id << "..." << endl;
		material newMaterial;
	
		//load static properties
		for(int i=0; i<10; i++){
			string line;
            utilityCore::safeGetline(fp_in,line);
			vector<string> tokens = utilityCore::tokenizeString(line);
			if(strcmp(tokens[0].c_str(), "RGB")==0){
				glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
				newMaterial.color = color;
			}else if(strcmp(tokens[0].c_str(), "SPECEX")==0){
				newMaterial.specularExponent = atof(tokens[1].c_str());				  
			}else if(strcmp(tokens[0].c_str(), "SPECRGB")==0){
				glm::vec3 specColor( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
				newMaterial.specularColor = specColor;
			}else if(strcmp(tokens[0].c_str(), "REFL")==0){
				newMaterial.hasReflective = atof(tokens[1].c_str());
			}else if(strcmp(tokens[0].c_str(), "REFR")==0){
				newMaterial.hasRefractive = atof(tokens[1].c_str());
			}else if(strcmp(tokens[0].c_str(), "REFRIOR")==0){
				newMaterial.indexOfRefraction = atof(tokens[1].c_str());					  
			}else if(strcmp(tokens[0].c_str(), "SCATTER")==0){
				newMaterial.hasScatter = atof(tokens[1].c_str());
			}else if(strcmp(tokens[0].c_str(), "ABSCOEFF")==0){
				glm::vec3 abscoeff( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
				newMaterial.absorptionCoefficient = abscoeff;
			}else if(strcmp(tokens[0].c_str(), "RSCTCOEFF")==0){
				newMaterial.reducedScatterCoefficient = atof(tokens[1].c_str());					  
			}else if(strcmp(tokens[0].c_str(), "EMITTANCE")==0){
				newMaterial.emittance = atof(tokens[1].c_str());					  
			
			}
		}
		materials.push_back(newMaterial);
		return 1;
	}
}



// Object loader
struct coordinate{
	float x,y,z;
	coordinate(float a, float b, float c) : x(a),y(b),z(c)
		 {
		/*	 x = a ;
			 y = b ;
			 z = c ;*/
	     } ;
};

struct face
{
int facenum ;
bool three;
int faces[3] ;
face(int f1,int f2,int f3)
{
	faces[0] = f1;
	faces[1] = f2;
	faces[2] = f3;
	three = true;
}
};

std::vector<std::string*> coord;

std::vector<coordinate*> vertex;

std::vector<face*> faces;

std::vector<coordinate*> normals;

//std::vector<glm::vec3> mymainpoints ;


int scene::loadmesh(string filename ) 
{
	//for(int f=1 ; f < 48 ; f++)
//	{
    //string result,actualfile;
	//ostringstream convert ;
	//convert << f ;
	//result = convert.str();
	// assigning a number to the output filename 
	//actualfile = filename  ; // result+".obj";
	std::ifstream in("bunny.obj" );//
	char c[256];
	cout << filename <<endl ;
	char buf[256];
	while(!in.eof())
	{
		in.getline(c,256);
//		cout << "reading" << std::string(c) << endl ;
		if ( std::string(c) == "")
			continue;
		coord.push_back(new std::string(c));
	}

	for(int i=0 ; i < coord.size() ; i++)
	{
		if ( (*coord[i])[0] == '#')
			continue;
		else if( (*coord[i])[0] == 'v') 
		{
			char tmp;
			float tmpx,tmpy,tmpz;
			sscanf(coord[i]->c_str(),"%c %f %f %f",&tmp ,&tmpx ,&tmpy , &tmpz);
			vertex.push_back(new coordinate(tmpx , tmpy,tmpz));
			cout << "vertext is " << i <<endl ;// vertex[i]->x << " "<< vertex[i]->y << " " << vertex[i]->z  << endl;
		}
		else if((*coord[i])[0] == 'f')
		{
			char tmp;
			int a,b,c;
			if (count(coord[i]->begin(), coord[i]->end(),' ')==3)
			{
			 sscanf(coord[i]->c_str(),"%c %d %d %d",&tmp ,&a ,&b , &c);
			 faces.push_back(new face(a,b,c));
			}
		}
		else
		{
		
		}
    }



	

	glm::vec3 v1(0,0,0),v2(0,0,0),v3(0,0,0);
	//glm::mat3 a(150,0,0,0,150,0,0,0,150);// scale
	//glm::vec3 p(1.0,6.0,0); // position  

	float inf = 10000000000.0f;
	//maxmin = new float[6]  ;
	//maxmin[6] = {-inf,inf,-inf,inf,-inf,inf};
	maxmin = new float[6]; 
	maxmin[0] = -inf ; 
	maxmin[1] =  inf ; 
	maxmin[2] = -inf ; 
	maxmin[3] =  inf ; 
	maxmin[4] = -inf ; 
	maxmin[5] =  inf ; 

	for(int i=0;i<faces.size();i++)
	{

		v1 = glm::vec3(vertex[faces[i]->faces[0]-1]->x,vertex[faces[i]->faces[0]-1]->y,vertex[faces[i]->faces[0]-1]->z);
		v2 = glm::vec3(vertex[faces[i]->faces[1]-1]->x,vertex[faces[i]->faces[1]-1]->y,vertex[faces[i]->faces[1]-1]->z);
		v3 = glm::vec3(vertex[faces[i]->faces[2]-1]->x,vertex[faces[i]->faces[2]-1]->y,vertex[faces[i]->faces[2]-1]->z) ;
	
		mymainpoints.push_back(glm::vec3(v1[0],v1[2],v1[1]));
		mymainpoints.push_back(glm::vec3(v2[0],v2[2],v2[1]));
		mymainpoints.push_back(glm::vec3(v3[0],v3[2],v3[1]));

		if(v1[0] > maxmin[0] )  maxmin[0] = v1[0]; // maximum x of v1 is stored 
		if(v2[0] > maxmin[0] )  maxmin[0] = v2[0]; // maximum x of v2 is stored 
		if(v3[0] > maxmin[0] )  maxmin[0] = v3[0]; // maximum x of v3 is stored 

		if(v1[0] < maxmin[1] )  maxmin[1] = v1[0]; // minimum x of v1 is stored 
		if(v2[0] < maxmin[1] )  maxmin[1] = v2[0]; // minimum x of v2 is stored 
		if(v3[0] < maxmin[1] )  maxmin[1] = v3[0]; // minimum x of v3 is stored 


		if(v1[1] > maxmin[2] )  maxmin[2] = v1[1]; // maximum x of v1 is stored 
		if(v2[1] > maxmin[2] )  maxmin[2] = v2[1]; // maximum x of v2 is stored 
		if(v3[1] > maxmin[2] )  maxmin[2] = v3[1]; // maximum x of v3 is stored 

		if(v1[1] < maxmin[3] )  maxmin[3] = v1[1]; // minimum x of v1 is stored 
		if(v2[1] < maxmin[3] )  maxmin[3] = v2[1]; // minimum x of v2 is stored 
		if(v3[1] < maxmin[3] )  maxmin[3] = v3[1]; // minimum x of v3 is stored 


		if(v1[2] > maxmin[4] )  maxmin[4] = v1[2]; // maximum x of v1 is stored 
		if(v2[2] > maxmin[4] )  maxmin[4] = v2[2]; // maximum x of v2 is stored 
		if(v3[2] > maxmin[4] )  maxmin[4] = v3[2]; // maximum x of v3 is stored 

		if(v1[2] < maxmin[5] )  maxmin[5] = v1[2]; // minimum x of v1 is stored 
		if(v2[2] < maxmin[5] )  maxmin[5] = v2[2]; // minimum x of v2 is stored 
		if(v3[2] < maxmin[5] )  maxmin[5] = v3[2]; // minimum x of v3 is stored 
	}


	//meanVertices->push_back(mymainpoints);
	//mymainpoints.clear();

	coord.clear();
	vertex.clear();
	faces.clear();

	cout << "bunny is loaded " << endl;
	
return 0 ;	

}