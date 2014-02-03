// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef SCENE_H
#define SCENE_H

#include "glm/glm.hpp"
#include "utilities.h"
#include <vector>
#include "sceneStructs.h"
#include <sstream>
#include <fstream>
#include <iostream>

using namespace std;

class scene{
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadObject(string objectid);
    int loadCamera();
	int loadMesh(string line, int& numTotalFaces);
	int* tokenizeFaceVerts(vector<string> token_vec);
public:
    scene(string filename);
    ~scene();

	vector<mesh> meshes;
    vector<geom> objects;
    vector<material> materials;
	vector<face> faces;
	vector<glm::vec3> vertices;
	vector<int> lightIds;
    camera renderCam;
};

#endif
