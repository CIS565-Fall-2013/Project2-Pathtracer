#include "meshLoader.h"

using namespace std;

meshLoader::meshLoader(){

}

meshLoader::~meshLoader(){
	verts.clear();
	faces.clear();
	normals.clear();
}

void meshLoader::loadObj(char* fileName){
	cout<<"loading "<<fileName<<endl;

	fstream stream;
	stream.open(fileName);

	bool noRepeat=false;

	if(stream.is_open()){
		char line[4096];
		stream.getline(line, 4096);
		stream.getline(line, 4096);		//skip first 2 lines

		stream.getline(line, 4096);
		while(!stream.eof()){
			char* tok=strtok(line, " ");
			//cout<<"vert vector"<<endl;
			if(strcmp(tok, "v")== 0){		//read in vertex position
				if(noRepeat)			//already finished reading all vert/inds needed, so break
					break;
				glm::vec3 point;
				for(int i=0; i<3; i++){
					tok=strtok(NULL, " ");	
					point[i] = atof(tok);
					//cout<<atof(tok)<<" ";
				}
				verts.push_back(point);
				//cout<<endl;
			}
			else if(strcmp(tok, "f")==0){
				glm::vec3 faceInd;
				for(int i=0; i<3; i++){
					tok=strtok(NULL, " /");
					faceInd[i] = atoi(tok) - 1;	//subtrace 1 since it's 1 indexed
					//cout<<tok<<" ";
					tok=strtok(NULL, " /");		//skip 2nd index for vt
				}
				faces.push_back(faceInd);
				//cout<<endl;
			}
			else if(!noRepeat && strcmp(tok, "vt")==0){
				noRepeat=true;
			}
			stream.getline(line, 4096);
		}

		//calculate normals for all faces
		for(int i = 0; i < faces.size() ; i++){
			glm::vec3 pt1 = verts[faces[i].x];
			glm::vec3 pt2 = verts[faces[i].y];
			glm::vec3 pt3 = verts[faces[i].z];

			glm::vec3 normal = glm::normalize(glm::cross(pt2-pt1, pt3-pt1));
			normals.push_back(normal);
			//cout<<normal.x<<" "<<normal.y<<" "<<normal.z<<endl;
		}
	}

	else{
		cout<<"failed to open "<<fileName<<endl;
	}
}

void meshLoader::printVerts(){

	for(int i=0; i<verts.size(); i++)
		cout<<verts[i].x<<" "<<verts[i].y<<" "<<verts[i].z<<endl;

}

void meshLoader:: printFaces(){

	for(int i=0; i<faces.size(); i++)
		cout<<faces[i].x<<" "<<faces[i].y<<" "<<faces[i].z<<endl;
}