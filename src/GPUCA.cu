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
#include "PacketHeader.h"
#include <iostream>
#include "eclipse_parser.h"
#include <assert.h>     /* assert */


// Maximum relative difference (par1_A - par1_B)/par1_A for each parameters
constexpr float c_maxDoubletRelDifference[]{0.1, 0.05};
constexpr float c_maxDoubletAbsDifference[]{0.1, 0.1};
constexpr int c_doubletParametersNum = sizeof(c_maxDoubletRelDifference)/sizeof(c_maxDoubletRelDifference[0]);
constexpr int c_maxCellsNumPerLayer  = 64;
constexpr int c_maxNeighborsNumPerCell = 6;
constexpr int c_maxHitsNumPerTrack = 5;
constexpr int c_maxTracksNum = 64;

template <int maxNumLayersInPacket>
__inline__
__device__
int getNumHitsInLayer(const PacketHeader<maxNumLayersInPacket>* __restrict__ packetHeader, const int layer )
{
	int numHitsInLayer = 0;
	if(layer < packetHeader->numLayers)
	{
		numHitsInLayer = (layer == packetHeader->numLayers -1) ?
				packetHeader->size - packetHeader->firstHitIdOnLayer[layer]:
				packetHeader->firstHitIdOnLayer[layer+1] - packetHeader->firstHitIdOnLayer[layer];
	}

	return numHitsInLayer;


}


__inline__
__device__
bool isADoublet(const SimpleHit* __restrict__ hits, const int idOrigin, const int idTarget)
{
	//	float maxDoubletRelDifference[]{0.1, 0.1};
	//	float relEtaDiff = 2*fabs((hits[idOrigin].eta - hits[idTarget].eta)/(hits[idOrigin].eta+hits[idTarget].eta));
	//	if(relEtaDiff > maxDoubletRelDifference[0]) return false;
	//	float relPhiDiff = 2*fabs((hits[idOrigin].phi - hits[idTarget].phi)/(hits[idOrigin].phi+hits[idTarget].phi));
	//	if(relPhiDiff > maxDoubletRelDifference[1]) return false;

	float maxDoubletAbsDifference[]{0.5, 0.5};
	float relEtaDiff = fabs(hits[idOrigin].eta - hits[idTarget].eta);
	if(relEtaDiff > maxDoubletAbsDifference[0]) return false;
	float relPhiDiff = fabs(hits[idOrigin].phi - hits[idTarget].phi);
	if(relPhiDiff > maxDoubletAbsDifference[1]) return false;
	return true;
}


// this will become a global kernel in the offline CA
template< int maxCellsNum, int maxNumLayersInPacket,int maxCellsPerLayer, int warpSize >
__device__ void makeCells (const PacketHeader<maxNumLayersInPacket>* __restrict__ packetHeader, const SimpleHit* __restrict__ hits,
		CUDAQueue<maxCellsNum, Cell<c_maxNeighborsNumPerCell, c_doubletParametersNum> >& outputCells, CUDAQueue<maxCellsPerLayer, int >* outputCellsIdOnLayer, int hitId )
{
	auto threadInWarpIdx = threadIdx.x%32;
	auto layerId = hits[hitId].layerId;
	auto firstHitIdOnNextLayer = packetHeader->firstHitIdOnLayer[layerId+1];
	auto numHitsOnNextLayer = getNumHitsInLayer(packetHeader, layerId+1 );
	auto nSteps = (numHitsOnNextLayer+warpSize-1)/warpSize;
	for (auto i = 0; i < nSteps; ++i)
	{
		auto targetHitId = firstHitIdOnNextLayer + i*warpSize + threadInWarpIdx;
		if(targetHitId-firstHitIdOnNextLayer < numHitsOnNextLayer)
		{

			if(isADoublet(hits, hitId, targetHitId))
			{

				auto cellId = outputCells.push(Cell<c_maxNeighborsNumPerCell, c_doubletParametersNum>(hitId, targetHitId, layerId, outputCells.m_data));
				if(cellId == -1)
					break;
				outputCells.m_data[cellId].m_id = cellId;
				outputCellsIdOnLayer[layerId].push(cellId);

				//				printf("cellId pushed: %d, size of outputCells:%d \n", cellId, outputCells.m_size);
			}

		}

	}



}


template <int maxNumLayersInPacket, int maxCellsPerLayer, int maxNeighborsNumPerCell, int doubletParametersNum, int maxHitsNum, int maxTracksNum, int warpSize>
__global__ void singleBlockCA (const PacketHeader<maxNumLayersInPacket>* __restrict__ packetHeader, const SimpleHit* __restrict__ packetPayload, Track<maxHitsNum>* outputTracks )
{
	auto warpIdx = (blockDim.x*blockIdx.x + threadIdx.x)/warpSize;
	auto warpNum = blockDim.x/warpSize;
	auto threadInWarpIdx = threadIdx.x%warpSize;
	constexpr const auto maxCellsNum = maxCellsPerLayer*maxNumLayersInPacket;
	__shared__ CUDAQueue<maxCellsNum, Cell<maxNeighborsNumPerCell, doubletParametersNum> > foundCells;
	__shared__ CUDAQueue<maxCellsPerLayer, int > cellsOnLayer[maxNumLayersInPacket];
	__shared__ CUDAQueue<maxTracksNum, Track<maxHitsNum> > foundTracks;
	__shared__ CUDAQueue<2*maxCellsPerLayer, int> rootCells;
	//We will now create cells with the inner hit on each layer except the last one, which does not have a layer next to it.
	auto numberOfOriginHitsInInnerLayers = packetHeader->firstHitIdOnLayer[packetHeader->numLayers-1];

	auto nSteps = (numberOfOriginHitsInInnerLayers+warpNum-1)/warpNum;
	__syncthreads();

	for (auto i = 0; i < nSteps; ++i)
	{
		auto hitIdx = warpIdx + warpNum*i;
		if(hitIdx < numberOfOriginHitsInInnerLayers)
		{
			makeCells< maxCellsNum, maxNumLayersInPacket, maxCellsPerLayer, warpSize > (packetHeader, packetPayload, foundCells, cellsOnLayer, hitIdx);

		}

	}
	__syncthreads();
	if(threadIdx.x ==0)
		printf("size of outputCells:%d \n", foundCells.m_size);


	//	auto copyOutputCellsSteps = (foundCells[0].m_size + blockDim.x - 1) / blockDim.x;
	//	for(auto i = 0; i<copyOutputCellsSteps; ++i)
	//	{
	//		auto cellIdx = threadIdx.x + blockDim.x *i;
	//		if(cellIdx < foundCells.m_size)
	//		{
	//			foundCells.m_data[cellIdx].m_id = cellIdx;
	//			outputCells[cellIdx] = foundCells.m_data[cellIdx];
	//
	//		}
	//	}
	//
	//	__syncthreads();
	//	if(threadIdx.x == 0){
	//
	////     printf("number of cells=%d numberOfOriginHitsInInnerLayers=%d copyOutputCellsSteps=%d \n", foundCells.m_size, numberOfOriginHitsInInnerLayers);
	//     for(auto i =0 ; i< foundCells.m_size; ++i)
	//     {
	//    	 printf("foundCells m_id = %d foundCells m_layerId = %d foundCells m_innerhit = %d foundCells m_outerHit = %d "
	//    			 "foundCells m_CAstate = %d \n", foundCells.m_data[i].m_id, foundCells.m_data[i].m_layerId,
	//    			 foundCells.m_data[i].m_innerHitId, foundCells.m_data[i].m_outerHitId, foundCells.m_data[i].m_CAState);
	//     }
	//	}

	// now that we have the cells, it is time to match them and find neighboring cells


	auto cellsLoopingNumSteps = (foundCells.m_size + blockDim.x - 1) / blockDim.x;
	for (auto i = 0; i < cellsLoopingNumSteps; ++i)
	{
		auto cellIdx = threadIdx.x + i*blockDim.x;
		if(cellIdx < foundCells.m_size && foundCells.m_data[cellIdx].m_layerId < packetHeader->numLayers -1)
		{
			foundCells.m_data[cellIdx].neighborSearch(cellsOnLayer[foundCells.m_data[cellIdx].m_layerId+1]);
			//			printf("cell %d m_id %d on layer:%d num rightneighbours: %d first 3: %d %d %d  num leftneighbours: %d first 3:%d %d %d\n",cellIdx, foundCells.m_data[cellIdx].m_id,foundCells.m_data[cellIdx].m_layerId, foundCells.m_data[cellIdx].m_rightNeighbors.m_size,
			//					foundCells.m_data[cellIdx].m_rightNeighbors.m_data[0],foundCells.m_data[cellIdx].m_rightNeighbors.m_data[1],foundCells.m_data[cellIdx].m_rightNeighbors.m_data[2],foundCells.m_data[cellIdx].m_leftNeighbors.m_size,
			//					foundCells.m_data[cellIdx].m_leftNeighbors.m_data[0],foundCells.m_data[cellIdx].m_leftNeighbors.m_data[1],foundCells.m_data[cellIdx].m_leftNeighbors.m_data[2] );

		}

	}
	__syncthreads();



	auto CAevolutionIterationsNum = packetHeader->numLayers - 2;
	for(auto l = 0; l < CAevolutionIterationsNum; ++l)
	{
		for (auto i = 0; i < cellsLoopingNumSteps; ++i)
		{
			auto cellIdx = threadIdx.x + i*blockDim.x;
			if(cellIdx < foundCells.m_size)
			{

//				auto printstate=
						foundCells.m_data[cellIdx].evolve();
//				printf("cell %d on layer:%d innerHitId:%d outerHitId:%d state: %d \n",
//						cellIdx,foundCells.m_data[cellIdx].m_layerId, foundCells.m_data[cellIdx].m_innerHitId, foundCells.m_data[cellIdx].m_outerHitId, printstate);

						if(l == CAevolutionIterationsNum)
						{
							foundCells.m_data[cellIdx].isRootCell(rootCells);

						}
			}


		}
		__syncthreads();
	}



	// only 1 thread per cell on the first layer will now look for tracks
	auto cellsOnInnermostLayer = (rootCells.m_size + blockDim.x - 1) / blockDim.x;
	for (auto i = 0; i < cellsOnInnermostLayer; ++i)
	{
		//
		auto cellIdx = threadIdx.x + i*blockDim.x;

		if(cellIdx < rootCells.m_size)
		{
			Track<maxHitsNum> tmpTrack;
			tmpTrack.m_cells.push_singleThread(rootCells.m_data[cellIdx]);
			tmpTrack.m_hits.push_singleThread(foundCells.m_data[rootCells.m_data[cellIdx]].m_innerHitId);

//			if(cellIdx ==0)
//			{
//				printf("tmpTrack size %d \n",tmpTrack.m_cells.m_size );
//
//				for(auto j =0; j<tmpTrack.m_cells.m_size; ++j)
//				{
//					printf("\t cellIdx: %d cellid: %d on layer:%d innerHitId:%d outerHitId:%d\n",cellIdx,  foundCells.m_data[tmpTrack.m_cells.m_data[j]].m_id, foundCells.m_data[tmpTrack.m_cells.m_data[j]].m_layerId, foundCells.m_data[tmpTrack.m_cells.m_data[j]].m_innerHitId, foundCells.m_data[tmpTrack.m_cells.m_data[j]].m_outerHitId);
//				}
//
//			}
			foundCells.m_data[rootCells.m_data[cellIdx]].findTracks(foundTracks,tmpTrack);
//			if(cellIdx ==0)
//				{
//					printf("tmpTrack size %d \n",tmpTrack.m_cells.m_size );
//
//					for(auto j =0; j<tmpTrack.m_cells.m_size; ++j)
//					{
//						printf("\t cellid: %d on layer:%d innerHitId:%d outerHitId:%d\n",foundCells.m_data[tmpTrack.m_cells.m_data[j]].m_id, foundCells.m_data[tmpTrack.m_cells.m_data[j]].m_layerId, foundCells.m_data[tmpTrack.m_cells.m_data[j]].m_innerHitId, foundCells.m_data[tmpTrack.m_cells.m_data[j]].m_outerHitId);
//					}
//
//				}
		}
		//
	}
	//
	__syncthreads();

	auto tracksSteps = (foundTracks.m_size + blockDim.x - 1) / blockDim.x;
	for (auto i = 0; i < tracksSteps; ++i)
	{
		//
		auto trackIdx = threadIdx.x + i*blockDim.x;
		if(trackIdx < foundTracks.m_size)
		{

			for(auto cellInTrack = 0; cellInTrack<foundTracks.m_data[trackIdx].m_cells.m_size; ++cellInTrack)
			{
				foundTracks.m_data[trackIdx].m_hits.push(foundCells.m_data[foundTracks.m_data[trackIdx].m_cells.m_data[cellInTrack]].m_outerHitId);
			}

			outputTracks[trackIdx] = foundTracks.m_data[trackIdx];
		}
		//
	}




}




int main()
{
	constexpr auto numLayers = 5;
	constexpr auto numHitsPerLayer = 2;

	srand (time(NULL));
	std::pair<float, float> range_eta(0.1, 0.11);
	std::pair<float, float> range_phi(0.4, 0.41);

	std::vector<SimpleHit> hitsVector(numLayers*numHitsPerLayer);



	for (auto i = 0; i< numLayers; ++i)
	{
		for(auto j =0; j<numHitsPerLayer; ++j)
		{
			hitsVector[i*numHitsPerLayer + j].eta = range_eta.first + (range_eta.second - range_eta.first)*(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
			hitsVector[i*numHitsPerLayer + j].phi = range_phi.first + (range_phi.second - range_phi.first)*(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
			hitsVector[i*numHitsPerLayer + j].layerId = i;
			std::cout << i*numHitsPerLayer + j << " "<<  hitsVector[i*numHitsPerLayer + j].eta << " " << hitsVector[i*numHitsPerLayer + j].phi << " " << hitsVector[i*numHitsPerLayer + j].layerId << std::endl;

		}
	}


	PacketHeader<c_maxNumberOfLayersInPacket>* host_packetHeader;
	PacketHeader<c_maxNumberOfLayersInPacket>* device_Packet;
	auto packetSize = sizeof(PacketHeader<c_maxNumberOfLayersInPacket>) + hitsVector.size()*sizeof(SimpleHit);

	Track<c_maxHitsNumPerTrack>* device_outputTracks;
	Track<c_maxHitsNumPerTrack>* host_outputTracks;
	cudaMallocHost((void**)&host_packetHeader, packetSize);
	cudaMalloc((void**)&device_Packet, packetSize);
	cudaMalloc((void**)&device_outputTracks, c_maxTracksNum*sizeof(Track<c_maxHitsNumPerTrack>));
	cudaMallocHost((void**)&host_outputTracks, c_maxTracksNum*sizeof(Track<c_maxHitsNumPerTrack>));

	SimpleHit* host_packetPayload = (SimpleHit*)((char*)host_packetHeader + sizeof(PacketHeader<c_maxNumberOfLayersInPacket>));


	//initialization of the Packet to send to the GPU
	host_packetHeader->size = hitsVector.size();
	host_packetHeader->numLayers = numLayers;
	for(auto i = 0; i<numLayers; ++i)
		host_packetHeader->firstHitIdOnLayer[i] = i*numHitsPerLayer;
	memcpy(host_packetPayload, hitsVector.data(), hitsVector.size()*sizeof(SimpleHit));

	// end of the initialization



	for (auto i = 0; i< numLayers; ++i)
	{
		for(auto j =0; j<numHitsPerLayer; ++j)
		{
			assert(hitsVector[i*numHitsPerLayer + j].eta == host_packetPayload[i*numHitsPerLayer + j].eta);
			assert(hitsVector[i*numHitsPerLayer + j].phi == host_packetPayload[i*numHitsPerLayer + j].phi);
			assert(hitsVector[i*numHitsPerLayer + j].layerId == host_packetPayload[i*numHitsPerLayer + j].layerId);

		}
	}
	cudaMemcpyAsync(device_Packet, host_packetHeader, packetSize, cudaMemcpyHostToDevice, 0);

	singleBlockCA<c_maxNumberOfLayersInPacket, c_maxCellsNumPerLayer,c_maxNeighborsNumPerCell, c_doubletParametersNum,c_maxHitsNumPerTrack,c_maxTracksNum, 32><<<1,1024,0,0>>>(device_Packet, (SimpleHit*)((char*)device_Packet+sizeof(PacketHeader<c_maxNumberOfLayersInPacket>)),device_outputTracks);



	cudaMemcpyAsync(host_outputTracks, device_outputTracks, c_maxTracksNum*sizeof(Track<c_maxHitsNumPerTrack>), cudaMemcpyDeviceToHost, 0);


	cudaStreamSynchronize(0);

		for (auto i = 0; i<c_maxTracksNum; ++i)
		{
			std::cout << "hits in track:" << host_outputTracks[i].m_hits.m_size << std::endl;
			for (auto j = 0; j<c_maxHitsNumPerTrack ; ++j)
			{
				std::cout << "\t hit " << j << " : " << host_outputTracks[i].m_hits.m_data[j] << std::endl;
			}
		}

	cudaFreeHost(host_packetHeader);
	cudaFree(device_Packet);

}
