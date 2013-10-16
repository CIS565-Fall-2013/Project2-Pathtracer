#ifndef MESHLOADER
#define MESHLOADER

#include <vector>
#include <iostream>
#include <fstream>
#include "glm/glm.hpp"

class meshLoader{
	
public:
	meshLoader();
	~meshLoader();

	void loadObj(char* fileName);
	void printVerts();
	void printFaces();
	void printNormals();

	std::vector<glm::vec3> verts;
	std::vector<glm::vec3> faces;
	std::vector<glm::vec3> normals;

private:

};

#endif