/*
 * Cell.h
 *
 *  Created on: Jun 17, 2015
 *      Author: fpantale
 */

#ifndef INCLUDE_CELL_H_
#define INCLUDE_CELL_H_

#include "CUDAQueue.h"

template< int maxSize>
struct Cell
{
	template< int queueMaxSize, class T>
	__inline__
	__host__ __device__ int neighbourSearch(const CUDAQueue<queueMaxSize, T>&);

	CUDAQueue<maxSize, int> leftNeighbours;
	CUDAQueue<maxSize, int> rightNeighbours;

	int2 hitsIds;




};



#endif /* INCLUDE_CELL_H_ */
