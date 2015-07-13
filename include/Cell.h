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
#include "eclipse_parser.h"




// Maximum relative difference (par1_A - par1_B)/par1_A for each parameters
constexpr int c_numParameters = sizeof(c_maxParAbsDifference)/sizeof(c_maxParAbsDifference[0]);

// maxSize is the maximum number of neighbors that a Cell can have
// parNum is the number of parameters to check for the neighbors conditions

template<int maxSize, int parNum>
class Cell
{
public:
	__host__ __device__ Cell() : m_CAState(0), m_id(0), m_innerHitId(0), m_outerHitId(0), m_cellsArray(0), m_layerId(0) { }
	__host__ __device__ Cell(int innerHitId, int outerHitId, int layerId, Cell* cellsArray) : m_CAState(0), m_id(this - cellsArray), m_innerHitId(innerHitId), m_outerHitId(outerHitId), m_cellsArray(cellsArray), m_layerId(layerId) { }

	template< int queueMaxSize>
	__inline__
	__device__ int neighborSearch(const CUDAQueue<queueMaxSize, int>& rightCells)
	{
		constexpr float c_maxParAbsDifference[]{0.1, 0.1, 0.1};

		int neighborNum = 0;

		for (auto i= 0; i < rightCells.m_size; ++i)
		{
			if(m_cellsArray[rightCells.m_data[i]].m_innerHitId != m_outerHitId)
				continue;
			bool isNeighbor = true;

			for (auto j =0; j < m_params.m_size; ++j )
			{
				isNeighbor = isNeighbor & (fabs((m_params.m_data[j] - m_cellsArray[rightCells.m_data[i]].m_params.m_data[j]))  < c_maxParAbsDifference[j]);
				if(!isNeighbor)
					break;

			}

			// if all the parameters are inside the range the right cell is a right neighbor.
			// viceversa this cell will be the left neighbors for rightNeighbor(i)
			if (isNeighbor)
			{
				m_cellsArray[rightCells.m_data[i]].m_leftNeighbors.push(m_id);
				m_rightNeighbors.push(m_cellsArray[rightCells.m_data[i]].m_id);
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
			if(m_cellsArray[m_leftNeighbors.m_data[i]].m_CAState == m_CAState)
			{
				hasFriends = true;
				break;
			}
		}
		__syncthreads();
		if(hasFriends)
			m_CAState++;
	}




//check whether a Cell and the root have compatible parameters.
	__inline__
	__device__
	bool areCompatible(Cell* a, Cell* root)
	{
		for (auto j =0; j < a->m_params.m_size; ++j )
		{
			bool isCompatible = (m_CAState < a->m_CAState) &&
					(fabs((a->m_params.m_data[j] - root->m_params.m_data[j]))  < c_maxParAbsDifference[j]);
			if(!isCompatible)
				return false;

		}
		return true;
	}

	// trying to free the track building process from hardcoded layers, leaving the visit of the graph
	// based on the neighborhood connections between cells.
	template<int maxTracksNum, int maxHitsNum>
	__inline__
	__device__ void findTracks ( CUDAQueue<maxTracksNum, Track<maxHitsNum> >& foundTracks, Track<maxHitsNum>& tmpTrack) {

		// the building process for a track ends if:
		// it has no right neighbor
		// it has no compatible neighbor

		// the track is then saved if the number of hits it contains is greater than a threshold

		if(m_rightNeighbors.m_size == 0 )
		{
			if( tmpTrack.m_cells.m_size >= c_minHitsPerTrack-1)
				foundTracks.push(tmpTrack);
			else
				return;
		}
		else{
			bool hasOneCompatibleNeighbor = false;
			for( auto i=0 ; i < m_rightNeighbors.m_size; ++i)
			{
				if(areCompatible(&m_cellsArray[m_rightNeighbors.m_data[i]], &m_cellsArray[tmpTrack.m_cells[0]]) )
				{
					hasOneCompatibleNeighbor = true;
					tmpTrack.m_cells.push(m_rightNeighbors.m_data[i]);
					m_cellsArray[m_rightNeighbors.m_data[i]]->findTracks<maxTracksNum, maxHitsNum>(foundTracks, tmpTrack);
					tmpTrack.m_cells.pop_back();

				}

			}
			if (!hasOneCompatibleNeighbor && tmpTrack.m_cells.m_size >= c_minHitsPerTrack-1)
			{
				foundTracks.push(tmpTrack);
			}
		}

	}

	CUDAQueue<maxSize, int>  m_leftNeighbors;
	CUDAQueue<maxSize, int>  m_rightNeighbors;
	CUDAQueue<parNum, float> m_params;

	int m_id;
	int m_layerId;
	int m_innerHitId;
	int m_outerHitId;
	int m_CAState;
	Cell * m_cellsArray;


};

#endif /* INCLUDE_CELL_H_ */
