// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Varun Sampath and Patrick Cozzi for GLSL Loading, from CIS565 Spring 2012 HW5 at the University of Pennsylvania: http://cis565-spring-2012.github.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include "main.h"
#include <set>

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv){

  #ifdef __APPLE__
	  // Needed in OSX to force use of OpenGL3.2 
	  glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
	  glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
	  glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	  glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  #endif

  // Set up pathtracer stuff
  bool loadedScene = false;
  finishedRender = false;

  targetFrame = 0;
  singleFrameMode = false;

  // Load scene file
  for(int i=1; i<argc; i++){
    string header; string data;
    istringstream liness(argv[i]);
    getline(liness, header, '='); getline(liness, data, '=');
    if(strcmp(header.c_str(), "scene")==0){
      renderScene = new scene(data);
      loadedScene = true;
    }else if(strcmp(header.c_str(), "frame")==0){
      targetFrame = atoi(data.c_str());
      singleFrameMode = true;
    }
  }

  if(!loadedScene){
    cout << "Error: scene file needed!" << endl;
    return 0;
  }

  // Set up camera stuff from loaded pathtracer settings
  iterations = 0;
  renderCam = &renderScene->renderCam;
  width = renderCam->resolution[0];
  height = renderCam->resolution[1];

  if(targetFrame>=renderCam->frames){
    cout << "Warning: Specified target frame is out of range, defaulting to frame 0." << endl;
    targetFrame = 0;
  }

	sendDataToGPU();

  // Launch CUDA/GL

  #ifdef __APPLE__
	init();
  #else
	init(argc, argv);
  #endif

  initCuda();

  initVAO();
  initTextures();

  GLuint passthroughProgram;
  passthroughProgram = initShader("shaders/passthroughVS.glsl", "shaders/passthroughFS.glsl");

  glUseProgram(passthroughProgram);
  glActiveTexture(GL_TEXTURE0);

  #ifdef __APPLE__
	  // send into GLFW main loop
	  while(1){
		display();
		if (glfwGetKey(GLFW_KEY_ESC) == GLFW_PRESS || !glfwGetWindowParam( GLFW_OPENED )){
				exit(0);
		}
	  }

	  glfwTerminate();
  #else
	  glutDisplayFunc(display);
	  glutKeyboardFunc(keyboard);

	  glutMainLoop();
  #endif
  return 0;
}

// Send static data, such as geometry, lights, materials at the first frame to GPU
void sendDataToGPU() {
	// the image we are rendering to
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy(cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

	// pack geom and material arrays
	numberOfGeoms = renderScene->objects.size();
	numberOfMaterials = renderScene->materials.size();
	numberOfFaces = renderScene->faces.size();

	material* materials = new material[numberOfMaterials];
	for(int i=0; i<numberOfMaterials; i++){
    materials[i] = renderScene->materials[i];
  }
  
	// allocate memory for geometry and faces
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
	cudaMalloc((void**)&cudafaces, numberOfFaces*sizeof(transformedTriangle));

	// copy materials to CUDA
	cudaMalloc((void**)&cudamtls, numberOfMaterials*sizeof(material));
  cudaMemcpy(cudamtls, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);
  
  // package camera
  cam.resolution = renderCam->resolution;
  cam.fov = renderCam->fov;
	cam.focal = renderCam->focal;
	cam.aperture = renderCam->aperture;

	uploadDataOfCurrentFrame();

	// set up ray pools and scan arrays
	numberOfRays = (int)cam.resolution.x * (int)cam.resolution.y;
	cudaMalloc((void**)&raypool1, (int)cam.resolution.x * (int)cam.resolution.y * sizeof(ray));
	cudaMalloc((void**)&raypool2, (int)cam.resolution.x * (int)cam.resolution.y * sizeof(ray));
	cudaMalloc((void**)&scanArray, numberOfRays * sizeof(int));
	cudaMalloc((void**)&sumArray1, ceil((float)numberOfRays/(float)64) * sizeof(int));
	cudaMalloc((void**)&sumArray2, ceil((float)numberOfRays/(float)64) * sizeof(int));
}

// update geometry and camera with data of current frame
void uploadDataOfCurrentFrame() {
	geom* geoms = new geom[numberOfGeoms];
	triangle* faces = new triangle[numberOfFaces];

	for(int i=0; i<numberOfGeoms; i++){
    geoms[i] = renderScene->objects[i];
  }
	for (int i=0; i<numberOfFaces; ++i) {
		faces[i] = *(renderScene->faces[i]);
	}

	staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
		newStaticGeom.translation = geoms[i].translations[targetFrame];
    newStaticGeom.rotation = geoms[i].rotations[targetFrame];
    newStaticGeom.scale = geoms[i].scales[targetFrame];
    newStaticGeom.transform = geoms[i].transforms[targetFrame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[targetFrame];
    geomList[i] = newStaticGeom;
  }

	// package transformed faces and send to GPU
	transformedTriangle* faceList = new transformedTriangle[numberOfFaces];
	for (int i=0; i<numberOfFaces; ++i) {
		int geomId = faces[i].geomId;
		transformedTriangle face;
		face.materialid = geomList[geomId].materialid;
		face.v1 = multiplyMVdup(geomList[geomId].transform, glm::vec4(faces[i].v1, 1.0f));
		face.v2 = multiplyMVdup(geomList[geomId].transform, glm::vec4(faces[i].v2, 1.0f));
		face.v3 = multiplyMVdup(geomList[geomId].transform, glm::vec4(faces[i].v3, 1.0f));
		face.n1 = glm::normalize(multiplyMVdup(geomList[geomId].transform, glm::vec4(faces[i].n1, 0.0f)));
		face.n2 = glm::normalize(multiplyMVdup(geomList[geomId].transform, glm::vec4(faces[i].n2, 0.0f)));
		face.n3 = glm::normalize(multiplyMVdup(geomList[geomId].transform, glm::vec4(faces[i].n3, 0.0f)));
		faceList[i] = face;
	}

	// update geometry data on GPU
	cudaMemcpy(cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

	// copy faces to CUDA
  cudaMemcpy(cudafaces, faceList, numberOfFaces*sizeof(transformedTriangle), cudaMemcpyHostToDevice);

	// update camera info
	cam.position = renderCam->positions[targetFrame];
  cam.view = renderCam->views[targetFrame];
  cam.up = renderCam->ups[targetFrame];

	// delete pointers
	delete [] geomList;
	delete [] faceList;
}

// helper
glm::vec3 multiplyMVdup(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda(){
  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

  if(iterations<renderCam->iterations){

    uchar4 *dptr=NULL;
    iterations++;
    cudaGLMapBufferObject((void**)&dptr, pbo);
  
    // execute the kernel
		cudaRaytraceCore(dptr, renderCam, cam, iterations, cudageoms, numberOfGeoms, cudafaces, numberOfFaces,
			cudamtls, cudaimage, raypool1, raypool2, numberOfRays, scanArray, sumArray1, sumArray2);
    
    // unmap buffer object
    cudaGLUnmapBufferObject(pbo);
  }else{

    if(!finishedRender){
			//retrieve image from GPU
			cudaMemcpy(renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

      //output image file
      image outputImage(renderCam->resolution.x, renderCam->resolution.y);

      for(int x=0; x<renderCam->resolution.x; x++){
        for(int y=0; y<renderCam->resolution.y; y++){
          int index = x + (y * renderCam->resolution.x);
          outputImage.writePixelRGB(renderCam->resolution.x-1-x,y,renderCam->image[index]);
        }
      }
      
      gammaSettings gamma;
      gamma.applyGamma = true;
      gamma.gamma = 1.0;
      gamma.divisor = renderCam->iterations;
      outputImage.setGammaSettings(gamma);
      string filename = renderCam->imageName;
      string s;
      stringstream out;
      out << targetFrame;
      s = out.str();
      utilityCore::replaceString(filename, ".bmp", "."+s+".bmp");
      utilityCore::replaceString(filename, ".png", "."+s+".png");
      outputImage.saveImageRGB(filename);
      cout << "Saved frame " << s << " to " << filename << endl;
      finishedRender = true;
      if(singleFrameMode==true || targetFrame==renderCam->frames-1){
        cudaDeviceReset();
				// free cuda memory
				cudaFree(cudageoms);
				cudaFree(cudamtls);
				cudaFree(cudaimage);
				cudaFree(raypool1);
				cudaFree(raypool2);
				cudaFree(scanArray);
				cudaFree(sumArray1);
				cudaFree(sumArray2);

				checkCUDAError("Kernel failed!");
        exit(0);
      }
    }
    if(targetFrame<renderCam->frames-1){
			//clear image buffer and move onto next frame
			targetFrame++;
			uploadDataOfCurrentFrame();
			clearImage();
			cudaDeviceReset(); 
			finishedRender = false;
    }
  }
  
}

// Clear the image on CPU and set iterations to 0
void clearImage() {
  iterations = 0;
  for(int i=0; i<renderCam->resolution.x*renderCam->resolution.y; i++){
    renderCam->image[i] = glm::vec3(0,0,0);
  }
	cudaMemcpy(cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
}

#ifdef __APPLE__

	void display(){
		runCuda();

		string title = "CIS565 Render | " + utilityCore::convertIntToString(iterations) + " Iterations";
		glfwSetWindowTitle(title.c_str());

		glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
			  GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		glClear(GL_COLOR_BUFFER_BIT);   

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

		glfwSwapBuffers();
	}

#else

	void display(){
		// Keep track of time
    //theFpsTracker.timestamp();

		runCuda();

		string title = "565Raytracer | " + utilityCore::convertIntToString(iterations) + " Iterations";
		glutSetWindowTitle(title.c_str());

		glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
			  GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		glClear(GL_COLOR_BUFFER_BIT);   

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

		glutPostRedisplay();
		glutSwapBuffers();

		//cout << "Framerate: " << theFpsTracker.fpsAverage() << endl;
	}

	void keyboard(unsigned char key, int x, int y)
	{
		float step = 0.1; // camera step
		std::cout << key << std::endl;
		switch (key) 
		{
		   case(27): //Esc
			   exit(1);
			   break;
			 case(97): //a
				 cam.position += glm::vec3(1, 0, 0) * step;
				 clearImage();
				 break;
			 case(119): //w
				 cam.position += glm::vec3(0, 1, 0) * step;
				 clearImage();
				 break;
			 case(100): //d
				 cam.position += glm::vec3(-1, 0, 0) * step;
				 clearImage();
				 break;
			 case(115): //s
				 cam.position += glm::vec3(0, -1, 0) * step;
				 clearImage();
				 break;
			 case(122): //z
				 cam.position += glm::vec3(0, 0, -1) * step;
				 clearImage();
				 break;
			 case(120): //x
				 cam.position += glm::vec3(0, 0, 1) * step;
				 clearImage();
				 break;
			 //case(91): //[
				// cam.focal -= 0.5;
				// clearImage();
				// break;
			 //case(93): //]
				// cam.focal += 0.5;
				// clearImage();
				// break;
		}
	}

#endif




//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

#ifdef __APPLE__
	void init(){

		if (glfwInit() != GL_TRUE){
			shut_down(1);      
		}

		// 16 bit color, no depth, alpha or stencil buffers, windowed
		if (glfwOpenWindow(width, height, 5, 6, 5, 0, 0, 0, GLFW_WINDOW) != GL_TRUE){
			shut_down(1);
		}

		// Set up vertex array object, texture stuff
		initVAO();
		initTextures();
	}
#else
	void init(int argc, char* argv[]){
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
		glutInitWindowSize(width, height);
		glutCreateWindow("565Raytracer");

		// Init GLEW
		glewInit();
		GLenum err = glewInit();
		if (GLEW_OK != err)
		{
			/* Problem: glewInit failed, something is seriously wrong. */
			std::cout << "glewInit failed, aborting." << std::endl;
			exit (1);
		}

		initVAO();
		initTextures();
	}
#endif

void initPBO(GLuint* pbo){
  if (pbo) {
    // set up vertex data parameter
    int num_texels = width*height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    
    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1,pbo);
    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject( *pbo );
  }
}

void initCuda(){
  // Use device with highest Gflops/s
  cudaGLSetGLDevice( compat_getMaxGflopsDeviceId() );

  initPBO(&pbo);

  // Clean up on program exit
  atexit(cleanupCuda);

  runCuda();
}

void initTextures(){
    glGenTextures(1,&displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
        GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void){
    GLfloat vertices[] =
    { 
        -1.0f, -1.0f, 
         1.0f, -1.0f, 
         1.0f,  1.0f, 
        -1.0f,  1.0f, 
    };

    GLfloat texcoords[] = 
    { 
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);
    
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader(const char *vertexShaderPath, const char *fragmentShaderPath){
    GLuint program = glslUtility::createProgram(vertexShaderPath, fragmentShaderPath, attributeLocations, 2);
    GLint location;

    glUseProgram(program);
    
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda(){
  if(pbo) deletePBO(&pbo);
  if(displayImage) deleteTexture(&displayImage);
}

void deletePBO(GLuint* pbo){
  if (pbo) {
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(*pbo);
    
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers(1, pbo);
    
    *pbo = (GLuint)NULL;
  }
}

void deleteTexture(GLuint* tex){
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}
 
void shut_down(int return_code){
  #ifdef __APPLE__
	glfwTerminate();
  #endif
  exit(return_code);
}
