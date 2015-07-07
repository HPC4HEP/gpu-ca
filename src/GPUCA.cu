/*
 * GPUCA.cu
 *
 *  Created on: Jun 17, 2015
 *      Author: fpantale
 */

#include <cuda.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <vector>
#include "Cell.h"
#include "CUDAQueue.h"
#include "SimpleHit.h"
#include <iostream>
#include "eclipse_parser.h"


// Maximum relative difference (par1_A - par1_B)/par1_A for each parameters
constexpr float c_maxDoubletRelDifference[]{0.1, 0.1};
constexpr int c_doubletParametersNum = sizeof(c_maxDoubletRelDifference)/sizeof(c_maxDoubletRelDifference[0]);
constexpr int maxCellsNumPerLayer  = 256;
constexpr int maxNeighborsNumPerCell = 32;

__inline__
__device__
bool isADoublet(const SimpleHit* __restrict__ hits, const int idOrigin, const int idTarget)
{
	float relEtaDiff = 2*fabs((hits[idOrigin].eta - hits[idTarget].eta)/(hits[idOrigin].eta+hits[idTarget].eta));
	if(relEtaDiff > c_maxDoubletRelDifference[0]) return false;
	float relPhiDiff = 2*fabs((hits[idOrigin].phi - hits[idTarget].phi)/(hits[idOrigin].phi+hits[idTarget].phi));
	if(relPhiDiff > c_maxDoubletRelDifference[1]) return false;

	return true;
}


// this will become a global kernel in the offline CA
template< int maxCellsNum, int warpSize >
__device__ void makeCells (const SimpleHit* __restrict__ hits, CUDAQueue<maxCellsNum,Cell<maxNeighborsNumPerCell, c_doubletParametersNum> >& outputCells,
			int hitId, int layerId, int firstHitIdOnNextLayer, int numHitsOnNextLayer, int threadId )
{
	auto nSteps = (numHitsOnNextLayer+warpSize-1)/warpSize;

	for (auto i = 0; i < nSteps; ++i)
	{
		auto targetHitId = i*warpSize + threadId;
		if(targetHitId < numHitsOnNextLayer)
		{
			if(isADoublet(hits, hitId, targetHitId))
			{
				auto cellId = outputCells.push(Cell(hitId, targetHitId, layerId, outputCells.m_data));
				if(cellId == -1)
					break;

			}

		}

	}

}



__global__ void singleBlockCA (Cell<20, c_numParameters>** arrayOfLayers, int numberOfLayers, int* numberOfCellsPerLayer )
{



}




int main()
{
	constexpr auto numLayers = 5;
	constexpr auto numHitsPerLayer = 100;

	srand (time(NULL));
	std::pair<float, float> range_eta(0.1, 0.3);
	std::pair<float, float> range_phi(0.5, 0.6);

	std::vector<SimpleHit> hitsVector(numLayers*numHitsPerLayer);



	for (auto i = 0; i< numLayers; ++i)
	{
		for(auto j =0; j<numHitsPerLayer; ++j)
		{
			hitsVector[i*numHitsPerLayer + j].eta = range_eta.first + (range_eta.second - range_eta.first)*(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
			hitsVector[i*numHitsPerLayer + j].phi = range_phi.first + (range_phi.second - range_phi.first)*(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
			hitsVector[i*numHitsPerLayer + j].layerId = i;
		}
	}












}
