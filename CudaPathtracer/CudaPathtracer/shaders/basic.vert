# version 400

layout (location = 0) in vec4 glVertex;
layout (location = 1 ) in vec2 glTexcoord;
//uniform mat4 ModelViewMatrix;
//uniform mat4 ProjectionMatrix;
out vec2 texcoord;

void main() {

	gl_Position = glVertex ; 
	texcoord = glTexcoord;
}