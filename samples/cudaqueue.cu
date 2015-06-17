#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
__inline__
__device__ int push(int* array, int* num, const int& element) 
{
    int oldvalue = atomicAdd(num, 1);
    array[oldvalue] = element;

}
__global__ void Find3(int* a, int* results, int* N)
{
    
    __shared__ int s_threes[1024];
    __shared__ int threes_num;

    int index  = threadIdx.x;
    
    if (threadIdx.x ==0 )
        threes_num = 0;
    __syncthreads();
    if(a[index] ==3)
       push(s_threes, &threes_num, index);
    
    __syncthreads();
    
    if(threadIdx.x < threes_num)
    results[index] = s_threes[index];

    *N = threes_num;

    

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