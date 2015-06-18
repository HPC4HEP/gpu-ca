/*
 * Cell.h
 *
 *  Created on: Jun 17, 2015
 *      Author: fpantale
 */

#ifndef INCLUDE_CELL_H_
#define INCLUDE_CELL_H_

#include "CUDAQueue.h"


// Maximum relative difference (par1_A - par1_B)/par1_A for each parameters
constexpr float c_maxParRelDifference[]{0.1, 0.1, 0.1};
constexpr int c_numParameters = sizeof(c_maxParRelDifference)/sizeof(c_maxParRelDifference[0]);

// maxSize is the maximum number of neighbors that a Cell can have
// parNum is the number of parameters to check for the neighbors conditions

template<int maxSize, int parNum>
class Cell
{
public:
	template< int queueMaxSize, class T>
	__inline__
	__device__ int neighborSearch(const CUDAQueue<queueMaxSize, T>& rightCells)
	{
		int neighborNum = 0;
		for (auto i= 0; i < rightCells.m_size; ++i)
		{
			auto j = 0;
			while ( (fabs(2*(m_params.m_data[j] - rightCells.m_data[i].m_params[j]) /(m_params.m_data[j]+rightCells.m_data[i].m_params[j]))  < c_maxParRelDifference[j]) && (j < m_params.m_size) )
			{
				++j;
			}

			// if all the parameters are inside the range the right cell is a right neighbor.
			if (j == m_params.m_size)
			{
				rightCells.m_data[i].m_leftNeighbors.push(m_id);
				m_rightNeighbors.m_rightNeighbors.push(i);
				++neighborNum;
			}

		}
		return neighborNum;
	}

	__inline__
	__device__ void increaseState() { atomicAdd(&m_CAState,1); }

	CUDAQueue<maxSize, int> m_leftNeighbors;
	CUDAQueue<maxSize, int> m_rightNeighbors;
	CUDAQueue<parNum, float> m_params;

	int m_id;
	int2 m_hitsIds;
	int m_CAState;

};

#endif /* INCLUDE_CELL_H_ */
