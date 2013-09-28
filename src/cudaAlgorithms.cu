#include "cudaAlgorithms.h"

//For easy avoidance of bank conflicts
#define NUM_BANKS 32  
#define LOG_NUM_BANKS 5  
#define CONFLICT_FREE_OFFSET(n)    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))  



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
	int ai = index;
	int bi = index + N/2;
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	temp[ai + bankOffsetA] = datain[ai]; // load input into shared memory  
	temp[bi + bankOffsetB] = datain[bi];  

	// build sum in place up the tree  
	// d limits the number of active threads, halving it each iteration.
	for (int d = N>>1; d > 0; d >>= 1)                  
	{   
		__syncthreads();  //Make sure previous step has completed
		if (index < d)  
		{
			ai = offset*(2*index+1)-1;  
			bi = offset*(2*index+2)-1;  
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] = op(temp[ai], temp[bi]);  
		}  
		offset *= 2;  //Adjust offset
	}
	//Reduction step complete. 

	if (index == 0) { temp[(N - 1) + CONFLICT_FREE_OFFSET(N-1)] = 0; } // clear the last element in prep for down scan

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
	dataout[ai] = temp[ai+bankOffsetA]; // write results to device memory  
	dataout[bi] = temp[bi+bankOffsetB];  

}  