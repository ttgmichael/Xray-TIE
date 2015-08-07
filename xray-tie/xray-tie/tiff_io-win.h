//	tiff_io-win.h
//  orginal author: Justin Blair
//  original institution: Lawrence Berkeley National Laboratory
/* Class to read and write tiff files */

#ifndef TIFF_IO_WIN_H
#define TIFF_IO_WIN_H

#include <cstdlib>
#include <cstring>
#include <string>
#include "tiff.h"
#include "tiffio.h"

using namespace std;

 class TiffIO
 {
 private:
	float min_val;
	float max_val;
 public:
	TiffIO();
	~TiffIO();
<<<<<<< HEAD
	float* readFloatImage(std::string input_name, int* w_ptr, int* h_ptr);
	float* read16bitImage(std::string input_name, int* w_ptr, int* h_ptr);
=======
	float** readFloatImage(std::string input_name, int* w_ptr, int* h_ptr);
	float** read16bitImage(std::string input_name, int* w_ptr, int* h_ptr);
>>>>>>> bf31d68f9ba4f08251c11785798d5cf592377dae
	void writeFloatImage(float** image, std::string output_name, int width, int height);
	void write16bitImage(float** image, std::string output_name, int width, int height);
 };
 
 #endif
