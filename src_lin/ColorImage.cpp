#include "ColorImage.h"

#include <fstream>
#include <iostream>
#include <cassert>

using namespace std;

ColorImage::ColorImage()
{
	pPixel = 0;
}

ColorImage::~ColorImage()
{
	if (pPixel) delete[] pPixel;
	pPixel = 0;
}

void ColorImage::init(int xSize, int ySize)
{
	Pixel p = {0,0,0};
	xRes = xSize;
	yRes = ySize;
	pPixel = new Pixel[xSize*ySize];
	clear(p);
}

void ColorImage::clear(Pixel &background)
{
	int i;
	
	if (! pPixel) return;
	for (i=0; i<xRes*yRes; i++) pPixel[i] = background;
}

Pixel ColorImage::readPixel(int x, int y)
{
	assert(pPixel); // die if image not initialized
	//return pPixel[x + y*yRes];
    return pPixel[ x + y*xRes ];
}

void ColorImage::writePixel(int x, int y, const Pixel &p)
{
	assert(pPixel); // die if image not initialized
	//pPixel[x + y*yRes] = p;
    pPixel[x + y*xRes] = p;
}

void ColorImage::outputPPM( const char *filename)
{
    FILE *outFile = fopen(filename, "wb");

	assert(outFile); // die if file can't be opened

	fprintf(outFile, "P6 %d %d 255\n", xRes, yRes);
	fwrite(pPixel, 1, 3*xRes*yRes, outFile );

	fclose(outFile);
}


//Output final pixel values to a text file
void textOutput( char** pixel, int w, int h )
{
	std::ofstream output( "HW1_2_output.txt" );
	if( ! output.is_open() )
	{
		std::cout<<"Open output file failed!\n";
		return;
	}	

	for( int i = 0; i < h; ++i )
	{
		for( int j = 0; j < w; ++j )
		{
			if( pixel[i][j] == 0 )
				output<<" ";
			else
				output<<"*";
		}
		output<<"\n";
	}

	std::cout<<"Output to file 'HW1_output.txt'\n";
	output.close();
}