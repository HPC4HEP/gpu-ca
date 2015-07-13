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
constexpr float c_maxDoubletRelDifference[]{0.5, 0.5};
constexpr float c_maxDoubletAbsDifference[]{0.1, 0.2};
constexpr int c_doubletParametersNum = sizeof(c_maxDoubletRelDifference)/sizeof(c_maxDoubletRelDifference[0]);
constexpr int c_maxCellsNumPerLayer  = 64;
constexpr int c_maxNeighborsNumPerCell = 8;

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

	float maxDoubletAbsDifference[]{0.1, 0.1};
	float relEtaDiff = fabs(hits[idOrigin].eta - hits[idTarget].eta);
	if(relEtaDiff > maxDoubletAbsDifference[0]) return false;
	float relPhiDiff = fabs(hits[idOrigin].phi - hits[idTarget].phi);
	if(relPhiDiff > maxDoubletAbsDifference[1]) return false;
	return true;
}


// this will become a global kernel in the offline CA
template< int maxNumLayersInPacket,int maxCellsNum, int warpSize >
__device__ void makeCells (const PacketHeader<maxNumLayersInPacket>* __restrict__ packetHeader, const SimpleHit* __restrict__ hits,
		CUDAQueue<maxCellsNum, Cell<c_maxNeighborsNumPerCell, c_doubletParametersNum> >* outputCells, CUDAQueue<maxCellsPerLayer, int >* outputCellsIdOnLayer, int hitId )
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
				outputCellsIdOnLayer[layerId].push(cellId);
				if(cellId == -1)
					break;

			}

		}

	}

}


template <int maxNumLayersInPacket, int maxCellsPerLayer, int maxNeighborsNumPerCell, int doubletParametersNum, int warpSize>
__global__ void singleBlockCA (const PacketHeader<maxNumLayersInPacket>* __restrict__ packetHeader, const SimpleHit* __restrict__ packetPayload, Cell<maxNeighborsNumPerCell, doubletParametersNum>* outputCells )
{
	auto warpIdx = (blockDim.x*blockIdx.x + threadIdx.x)/warpSize;
	auto warpNum = blockDim.x/warpSize;
	auto threadInWarpIdx = threadIdx.x%warpSize;
	constexpr const auto maxCellsNum = maxCellsPerLayer*maxNumLayersInPacket;
	__shared__ CUDAQueue<maxCellsNum, Cell<maxNeighborsNumPerCell, doubletParametersNum> > foundCells;
	__shared__ CUDAQueue<maxCellsPerLayer, int > cellsOnLayer[maxNumLayersInPacket];

	//We will now create cells with the inner hit on each layer except the last one, which does not have a layer next to it.
	auto numberOfOriginHitsInInnerLayers = packetHeader->firstHitIdOnLayer[packetHeader->numLayers-1];

	auto nSteps = (numberOfOriginHitsInInnerLayers+warpNum-1)/warpNum;

	for (auto i = 0; i < nSteps; ++i)
	{
		auto hitIdx = warpIdx + warpNum*i;
		if(hitIdx < numberOfOriginHitsInInnerLayers)
		{
			makeCells< maxNumLayersInPacket, maxCellsNum, warpSize > (packetHeader, packetPayload, foundCells, cellsOnLayer, hitIdx);

		}

	}
	__syncthreads();


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


	auto neighborFindingNumSteps = (foundCells.m_size + blockDim.x - 1) / blockDim.x;
	for (auto i = 0; i < neighborFindingNumSteps; ++i)
	{
		auto cellIdx = threadIdx + i*blockDim.x;
		if(cellIdx < foundCells.m_size && foundCells.m_data[cellIdx].m_layerId < packetHeader->numLayers -1)
		{
			foundCells.m_data[cellIdx].neighborSearch(cellsOnLayer[foundCells.m_data[cellIdx].m_layerId+1]);
		}

	}

	__syncthreads();


}




int main()
{
	constexpr auto numLayers = 5;
	constexpr auto numHitsPerLayer = 7;

	srand (time(NULL));
//	srand(42);
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
			std::cout << i*numHitsPerLayer + j << " "<<  hitsVector[i*numHitsPerLayer + j].eta << " " << hitsVector[i*numHitsPerLayer + j].phi << " " << hitsVector[i*numHitsPerLayer + j].layerId << std::endl;

		}
	}


	PacketHeader<c_maxNumberOfLayersInPacket>* host_packetHeader;
	PacketHeader<c_maxNumberOfLayersInPacket>* device_Packet;
	auto packetSize = sizeof(PacketHeader<c_maxNumberOfLayersInPacket>) + hitsVector.size()*sizeof(SimpleHit);

	Cell<c_maxNeighborsNumPerCell, c_doubletParametersNum>* device_outputCells;
	Cell<c_maxNeighborsNumPerCell, c_doubletParametersNum>* host_outputCells;
	cudaMallocHost((void**)&host_packetHeader, packetSize);
	cudaMalloc((void**)&device_Packet, packetSize);
	cudaMalloc((void**)&device_outputCells, c_maxCellsNumPerLayer*numLayers*sizeof(Cell<c_maxNeighborsNumPerCell, c_doubletParametersNum>));
	cudaMallocHost((void**)&host_outputCells, c_maxCellsNumPerLayer*numLayers*sizeof(Cell<c_maxNeighborsNumPerCell, c_doubletParametersNum>));

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

	singleBlockCA<c_maxNumberOfLayersInPacket, c_maxCellsNumPerLayer,c_maxNeighborsNumPerCell, c_doubletParametersNum, 32><<<1,1024,0,0>>>(device_Packet, (SimpleHit*)((char*)device_Packet+sizeof(PacketHeader<c_maxNumberOfLayersInPacket>)),device_outputCells);


	cudaMemcpyAsync(host_outputCells, device_outputCells, c_maxCellsNumPerLayer*numLayers*sizeof(Cell<c_maxNeighborsNumPerCell, c_doubletParametersNum>), cudaMemcpyDeviceToHost, 0);


	cudaStreamSynchronize(0);

//	for (auto i = 0; i<c_maxCellsNumPerLayer*numLayers; ++i)
//	{
//		std::cout << host_outputCells->m_id << " " << host_outputCells->m_layerId << " " << host_outputCells->m_innerHitId << std::endl;
//	}

	cudaFreeHost(host_packetHeader);
	cudaFree(device_Packet);

}
