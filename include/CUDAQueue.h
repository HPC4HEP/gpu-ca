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
	void push(const T& element) { auto oldvalue = atomicAdd (&size, 1); data[oldvalue] = element;   };

	T data[maxSize];
	int size;
};



#endif /* INCLUDE_CUDAQUEUE_H_ */
