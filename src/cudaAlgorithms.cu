#include "cudaAlgorithms.h"


//Does a inclusive scan in CUDA for a single block
//Based on http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
//Allows in place scans by setting datain == dataout
template<typename DataPtr, typename BinaryOperation>
__device__ DataPtr inclusive_scan_block(DataPtr datain, DataPtr dataout, int N, BinaryOperation op)
{  
	extern __shared__ float temp[];
	int index = threadIdx.x;  
	int offset = 1;  

	//Shared memory for access speed
	temp[2*index] = datain[2*index]; // load input into shared memory  
	temp[2*index+1] = datain[2*index+1];  

	// build sum in place up the tree  
	// d limits the number of active threads, halving it each iteration.
	for (int d = N>>1; d > 0; d >>= 1)                  
	{   
		__syncthreads();  //Make sure previous step has completed
		if (index < d)  
		{
			int ai = offset*(2*index+1)-1;  
			int bi = offset*(2*index+2)-1;  

			temp[bi] = op(temp[ai], temp[bi]);  
		}  
		offset *= 2;  //Adjust offset
	}
	//Reduction step complete. 

	if (index == 0) { temp[N - 1] = 0; } // clear the last element in prep for down scan

	//
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan  
	{  
		offset >>= 1;  
		__syncthreads();  //wait for previous step to finish
		if (index < d)                       
		{  
			int ai = offset*(2*index+1)-1;  
			int bi = offset*(2*index+2)-1;  


			float t = temp[ai];  
			temp[ai] = temp[bi];  
			temp[bi] = op(temp[bi], t);   
		}  
	}  
	__syncthreads();  

	//Store block scan result back to memory.
	dataout[2*index] = temp[2*index]; // write results to device memory  
	dataout[2*index+1] = temp[2*index+1];  

}  