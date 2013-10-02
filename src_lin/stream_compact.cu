#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <thrust/count.h>
#include "util.h"
#include "stream_compact.h"

struct is_nonnegative{
    __host__ __device__
    bool operator()(const int x)
    {
        return x >= 0;
    }
};

void compactNaturalNum( int* in, int* result, int size )
{
    thrust::device_ptr<int> before(in);
    thrust::device_ptr<int> after(result);
    thrust::remove_copy( before, before+size, after, -1 );
    cudaDeviceSynchronize(); 
}


int countValidPath( int* in, int size )
{
    thrust::device_ptr<int> array(in);
    return thrust::count_if( array, array+size, is_nonnegative() );
    cudaDeviceSynchronize(); 
}

