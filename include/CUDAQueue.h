/*
 * CUDAQueue.h
 *
 *  Created on: Jun 17, 2015
 *      Author: fpantale
 */

#ifndef INCLUDE_CUDAQUEUE_H_
#define INCLUDE_CUDAQUEUE_H_

// CUDAQueue is a single-block queue.
// One may want to use it as a __shared__ struct, and have multiple threads
// pushing data into it.
template< int maxSize, class T>
struct CUDAQueue
{
	__inline__ __device__
	void push(const T& element) { auto previousSize = atomicAdd (&m_size, 1); m_data[previousSize] = element;   };

	__inline__ __device__
	void insertArray(T* array, int numElements) {
		if(numElements + m_size <= maxSize && threadIdx.x < numElements)
		{
			auto previousSize = atomicAdd(&m_size,1);
			m_data[previousSize+threadIdx.x] = array[threadIdx.x];
		}
	}
	T m_data[maxSize];
	int m_size;
};



#endif /* INCLUDE_CUDAQUEUE_H_ */
