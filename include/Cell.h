/*
 * Cell.h
 *
 *  Created on: Jun 17, 2015
 *      Author: fpantale
 */

#ifndef INCLUDE_CELL_H_
#define INCLUDE_CELL_H_

#include "CUDAQueue.h"
#include "Track.h"

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
			if(rightCells.m_data[i].m_leftHitId != m_rightHitId)
				continue;
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

//
//	void GPUGeometry::trieTraverse(std::vector<int>& tmpChain, const int segment, const int zBin)
//	{
//		if(getNumNeighbours(segment,zBin)==0)
//		{
//			for(const auto element : tmpChain)
//				std::cout << " segment : " << element ;
//			std::cout << std::endl;
//			nChain.index.push_back(nChain.chain.size());
//			nChain.chain.insert(nChain.chain.end(), tmpChain.begin(), tmpChain.end());
//		}
//
//		std::cout << "segment " << segment << " in zBin " << zBin << " has " <<  getNumNeighbours(segment,zBin) << " neighbours :" << std::endl;
//
//		for(int neighbourId =0 ; neighbourId< getNumNeighbours(segment,zBin); ++neighbourId)
//		{
//			std::cout << "segment " << getNeighbour( segment, zBin,neighbourId)  << std::endl;
//
//			tmpChain.push_back(getNeighbour( segment, zBin,neighbourId));
//			trieTraverse(tmpChain, getNeighbour( segment, zBin,neighbourId), zBin);
//			tmpChain.pop_back();
//		}
//
//
//	}

	// trying to free the track building process from hardcoded layers, leaving the visit of the graph
	// based on the neighborhood connections between cells.
	template<int maxTracksNum, int maxHitsNum>
	__inline__
	__device__ void findTracks ( CUDAQueue<maxTracksNum, Track<maxHitsNum> >& foundTracks, Track<maxHitsNum>& tmpTrack, const Cell& thisCell) {

		// the building process for a track ends if:
		// it has no right neighbor
		// it has no compatible neighbor

		// the track is then saved if the number of hits it contains is greater than a threshold
//		if(thisCell.m_rightNeighbors.m_size == 0)
//		{
//			foundTracks.push()
//		}





	}

	CUDAQueue<maxSize, int> m_leftNeighbors;
	CUDAQueue<maxSize, int> m_rightNeighbors;
	CUDAQueue<parNum, float> m_params;

	int m_id;
	int m_leftHitId;
	int m_rightHitId;
	int m_CAState;

};

#endif /* INCLUDE_CELL_H_ */
