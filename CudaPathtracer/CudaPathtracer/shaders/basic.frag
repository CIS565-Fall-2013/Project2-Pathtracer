# version 400

layout (location = 0) out vec4 gl_FragColor;

in vec2 texcoord;
uniform sampler2D tex1;

void main (void) 
{       
	gl_FragColor = texture2D( tex1, texcoord );
}
