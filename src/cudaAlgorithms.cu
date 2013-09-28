#include "cudaAlgorithms.h"

template<typename DataType, typename BinaryOperation>
__global__ void exclusive_scan_kernel(DataType* datain, DataType* dataout, DataType* blockResults, int N, BinaryOperation op)
{
	int blockIndex = blockIdx.x + blockIdx.y*gridDim.x;
	int dataIndex = threadIdx.x + blockIndex*blockDim.x;
	//Remember that we have two elements per thread.
	int blockOffset = blockIndex * (blockDim.x*2);
	
	int fullElements = N - blockOffset;
	if(fullElements > (blockDim.x*2))
		fullElements = blockDim.x*2;

	DataType blockResult = exclusive_scan_block(&datain[blockOffset], &dataout[blockOffset], fullElements, op);
	//Wait for results to come in
	__syncthreads();

	if(threadIdx.x == 0)
	{
		//Only have one thread write back the answer
		blockResults[blockIndex] = blockResult;
	}

}

template<typename DataType, typename BinaryOperation>
__global__ 	void scan_reintegrate_blocks(DataType* dataout, DataType* blockResults, int N, BinaryOperation op)
{
	int blockIndex = blockIdx.x + blockIdx.y*gridDim.x;
	int dataIndex = threadIdx.x + blockIndex*blockDim.x;
	
	//First block stays as is.
	if(blockIndex > 0 && dataIndex < N){
		dataout[dataIndex] += blockResults[blockIndex-1];
	}
}

//Generic 
template<typename DataType, typename BinaryOperation>
__host__ DataType exclusive_scan(DataType* datain, DataType* dataout, int N, BinaryOperation op)
{
	//Divide array into blocks
	//TODO: Get this dynamically
	int blockSize = MAX_BLOCK_DIM_X;
	dim3 threadsPerBlock(blockSize);;
	dim3 fullBlocksPerGrid;

	int numBlocks = ceil(float(N)/(blockSize*2));//2 data elements per thread
	if(numBlocks > MAX_GRID_DIM_X){
		fullBlocksPerGrid = dim3(MAX_GRID_DIM_X, (int)ceil( numBlocks / float(MAX_GRID_DIM_X)));
	}else{
		fullBlocksPerGrid = dim3(numBlocks);
	}

	//Create an array to store results from each block
	DataType* blockResults;
	cudaMalloc((void**)&blockResults, numBlocks*sizeof(DataType));
	exclusive_scan_kernel<<<fullBlocksPerGrid, threadsPerBlock, N*sizeof(DataType)>>>(datain, dataout, blockResults, N, op);

	DataType result;
	if(numBlocks == 1)
	{
		//We've reached the bottom of the stack, grab the answer. Just one element
		cudaMemcpy( &result, blockResults, sizeof(DataType), cudaMemcpyDeviceToHost);
	}else{
		result = exclusive_scan(blockResults, blockResults, numBlocks, op);

		//sum in blockResults
		scan_reintegrate_blocks<<<fullBlocksPerGrid, threadsPerBlock>>>(dataout, blockResults, N, op);
	}
	//Free block
	cudaFree(blockResults);
	return result;
}


template<typename DataType>
__host__ DataType exclusive_scan_sum(DataType* datain, DataType* dataout, int N)
{
	Add add;
	return exclusive_scan(datain, dataout, N, add);
}

template<typename DataType>
__host__ DataType exclusive_scan_mult(DataType* datain, DataType* dataout, int N)
{
	
	Multiply mult;
	return exclusive_scan(datain, dataout, N, mult);
}


//Does a inclusive scan in CUDA for a single block
//Based on http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
//Allows in place scans by setting datain == dataout
//Only works for an array ptr to device mem.
template<typename DataType, typename BinaryOperation>
__device__ DataType exclusive_scan_block(DataType* datain, DataType* dataout, int N, BinaryOperation op)
{  
	extern __shared__ float temp[];
	int index = threadIdx.x;  
	int offset = 1;  
	int bdimx = blockDim.x;//get actual temp padded array length

	//Shared memory for access speed
	//Get modified temp access
	int ai = index;
	int bi = index + bdimx;
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	if(ai < N){
		temp[ai + bankOffsetA] = datain[ai]; // load input into shared memory  
	}else{
		temp[ai + bankOffsetA] = 0;
	}
	if(bi < N){
		temp[bi + bankOffsetB] = datain[bi];  
	}else{
		temp[bi + bankOffsetB] = 0;//if out of range, pad shared memory with zeros.
	}
	__syncthreads();
	
	//Pre load last element in block in case it gets overwritten later
	DataType total =  temp[(N - 1) + CONFLICT_FREE_OFFSET(N-1)];

	// build sum in place up the tree  
	// d limits the number of active threads, halving it each iteration.
	for (int d = bdimx>>1; d > 0; d >>= 1)                  
	{   
		__syncthreads();  //Make sure previous step has completed
		if (index < d)  
		{
			int ai2 = offset*(2*index+1)-1;  
			int bi2 = offset*(2*index+2)-1;  
			ai2 += CONFLICT_FREE_OFFSET(ai2);
			bi2 += CONFLICT_FREE_OFFSET(bi2);

			temp[bi2] = op(temp[ai2], temp[bi2]);  
		}  
		offset *= 2;  //Adjust offset
	}
	//Reduction step complete. 

	if (index == 0) { temp[(bdimx*2 - 1) + CONFLICT_FREE_OFFSET(bdimx*2-1)] = 0; } // clear the last element in prep for down scan

	//
	for (int d = 1; d < bdimx; d *= 2) // traverse down tree & build scan  
	{  
		offset >>= 1;  
		__syncthreads();  //wait for previous step to finish
		if (index < d)                       
		{  
			int ai2 = offset*(2*index+1)-1;  
			int bi2 = offset*(2*index+2)-1;  


			float t = temp[ai2];  
			temp[ai2] = temp[bi2];  
			temp[bi2] = op(temp[bi2], t);   
		}  
	}  
	__syncthreads();  

	//Store block scan result back to memory.
	if(ai < N)//Don't write back if out of range
		dataout[ai] = temp[ai+bankOffsetA]; // write results to device memory  
	if(bi < N)
		dataout[bi] = temp[bi+bankOffsetB];  


	//Return last element of shared memory plus the last element of the array.
	return total + temp[(bdimx*2 - 1) + CONFLICT_FREE_OFFSET(bdimx*2-1)];
}

///Explicit template instantiations. Do this to avoid code bloat in .h file.
template int exclusive_scan_sum<int>(int*, int*, int);