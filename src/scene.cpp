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
	int numTotalFaces = 0;
    int id = atoi(objectid.c_str());
    if(id!=objects.size()){
        cout << "ERROR: OBJECT ID does not match expected number of objects" << endl;
        return -1;
    }else{
        cout << "Loading Object " << id << "..." << endl;
        geom newObject;
        string line;
        
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
                if(strcmp(extension.c_str(), "obj")==0){
                    cout << "Creating new mesh..." << endl;
                    cout << "Reading mesh from " << line << "... " << endl;
		    		newObject.type = MESH;
					newObject.meshId = loadMesh(line, numTotalFaces);
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
		if(materials.at(newObject.materialid).emittance > .001f) lightIds.push_back(objects.size() - 1);
	
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

int scene::loadMesh(string filename, int& numTotalFaces){
	char* fname = (char*)filename.c_str();
	ifstream file;
	file.open(fname);

	glm::vec3 min_v = glm::vec3(1000000, 1000000, 1000000);
	glm::vec3 max_v = glm::vec3(-1000000, -1000000, -1000000);
	
	if(file.is_open()){
		while(file.good()){
			string line;
			utilityCore::safeGetline(file, line);
			if(!line.empty()){
				vector<string> tokens = utilityCore::tokenizeString(line);
				if(tokens.size() > 0){
					if(strcmp(tokens[0].c_str(), "v") == 0){
						glm::vec3 v = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
						min_v = glm::vec3(min(min_v.x, v.x), min(min_v.y, v.y), min(min_v.z, v.z));
						max_v = glm::vec3(max(max_v.x, v.x), min(max_v.y, v.y), max(max_v.z, v.z));
						vertices.push_back(v);
					}else if(strcmp(tokens[0].c_str(), "f") == 0){
						// Tokenize '/' to get texture and normal coordinates
						int* indices = tokenizeFaceVerts(tokens);

						// Fan triangulation of faces with 3+ vertices
						for(int i = 0; i < tokens.size() - 3; i++){
							face f;
							// Read vertex index
							// TODO: Add texture and normal support
							f.p1 = indices[0] - 1;
							f.p2 = indices[i+1] - 1;
							f.p3 = indices[i+2] - 1;
							faces.push_back(f);
						}

						if(indices) delete(indices);
					}
				}
			}
		}
	}

	mesh m;
	m.numberOfFaces = faces.size();
	m.min = min_v;
	m.max = max_v;
	m.startFaceIdx = numTotalFaces;

	meshes.push_back(m);

	numTotalFaces += m.numberOfFaces;

	return meshes.size() - 1;
}


int* scene::tokenizeFaceVerts(vector<std::string> tokens_vec){
	int* indices = new int[tokens_vec.size() - 1];
	int i;

	for(i = 1; i < tokens_vec.size(); i++){
		char* str = (char*)tokens_vec.at(i).c_str();
		char* tok = strtok(str, "/");
		indices[i-1] = atoi(tok);
	}
	return indices;
}