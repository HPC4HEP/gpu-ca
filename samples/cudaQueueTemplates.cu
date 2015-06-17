/*
 * cudaQueueTemplates.cu
 *
 *  Created on: Jun 17, 2015
 *      Author: fpantale
 */


#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>



// CUDAQueue is a single-block queue.
// One may want to use it as a __shared__ struct, and have multiple threads
// pushing data into it.
template< int maxSize, class T>
struct CUDAQueue
{
	__inline__ __device__
	void push(const T& element) { auto oldvalue = atomicAdd (&size, 1); data[oldvalue] = element;   };

	T data[maxSize];
	int size;
};


__global__ void Find3(int* a, int* results, int* N)
{

	__shared__ CUDAQueue<1024, int> queue ;
	int index  = threadIdx.x;

	if (threadIdx.x ==0 )
		queue.size= 0;
	__syncthreads();
	if(a[index] ==3)
		queue.push(index);

	__syncthreads();

	if(threadIdx.x < queue.size)
		results[index] = queue.data[index];
	if (threadIdx.x ==0 )
		*N = queue.size;



}



int main ()
{
	int* a;
	int* d_a, *d_results;
	int* d_numberOfThrees;
	int N = 1024;

	a= (int*)malloc(N*sizeof(int));

	cudaMalloc((void**)&d_a, N*sizeof(int));
	cudaMalloc((void**)&d_results, N*sizeof(int));
	cudaMalloc((void**)&d_numberOfThrees, sizeof(int));

	std::vector<int> idOf3;
	for (int i =0; i< N; ++i)
	{
		a[i] = i % 4;
		if(a[i] == 3)
			idOf3.push_back(i);

	}
	cudaMemcpy(d_a, a, sizeof(int)*N, cudaMemcpyHostToDevice);




	Find3<<<1, 1024>>> (d_a, d_results, d_numberOfThrees);
	int results[1024];
	int numberOfThrees;
	cudaMemcpy(results, d_results, sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(&numberOfThrees, d_numberOfThrees, sizeof(int), cudaMemcpyDeviceToHost);

	std::sort(idOf3.begin(), idOf3.end());

	std::vector<int> idOf3_gpu(numberOfThrees);
	for(int i =0; i< numberOfThrees; ++i)
	{
		idOf3_gpu[i] = results[i];
	}
	std::sort(idOf3_gpu.begin(), idOf3_gpu.end());

	for(int i =0; i< numberOfThrees; ++i)
	{
		std::cout << idOf3_gpu[i] << "\t" << idOf3[i] << std::endl;

	}
	cudaFree(d_a);
	cudaFree(d_results);
	cudaFree(d_numberOfThrees);
	free(a);


}
