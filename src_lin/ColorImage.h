#pragma once
/** 
 *   Output module.
**/

struct Pixel {
	unsigned char R, G, B;  // Blue, Green, Red
};

class ColorImage 
{
	Pixel *pPixel;
	int xRes, yRes;
public:
	ColorImage();
	~ColorImage();
	void init(int xSize, int ySize);
	void clear(Pixel &background);
	Pixel readPixel(int x, int y);
	void writePixel(int x, int y, const Pixel &p);
	void outputPPM(const char *filename);
};

void textOutput( char** pixel, int w, int h );
