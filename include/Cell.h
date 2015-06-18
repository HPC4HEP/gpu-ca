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
			bool isNeighbor = true;
			for (auto j =0; j < m_params.m_size; ++j )
			{
				isNeighbor = isNeighbor & (fabs(2*(m_params.m_data[j] - rightCells.m_data[i].m_params[j]) /(m_params.m_data[j]+rightCells.m_data[i].m_params[j]))  < c_maxParRelDifference[j]);
				if(!isNeighbor)
					break;

			}

			// if all the parameters are inside the range the right cell is a right neighbor.
			// viceversa this cell will be the left neighbors for rightNeighbor(i)
			if (isNeighbor)
			{
				rightCells.m_data[i].m_leftNeighbors.push(m_id);
				m_rightNeighbors.m_rightNeighbors.push(i);
				++neighborNum;
			}

		}
		return neighborNum;
	}


// if there is at least one left neighbor with the same state (friend), the state has to be increased by 1.
	__inline__
	__device__ void evolve() {


		auto hasFriends = false;
		for(auto i =0; i < m_leftNeighbors.m_size; ++i)
		{
			if(m_leftNeighbors.m_data[i].m_CAState == m_CAState)
			{
				hasFriends = true;
				break;
			}
		}
		__syncthreads();
		if(hasFriends)
			m_CAState++;
	}

	CUDAQueue<maxSize, int> m_leftNeighbors;
	CUDAQueue<maxSize, int> m_rightNeighbors;
	CUDAQueue<parNum, float> m_params;

	int m_id;
	int2 m_hitsIds;
	int m_CAState;

};

#endif /* INCLUDE_CELL_H_ */
