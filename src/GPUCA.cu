/*
 * GPUCA.cu
 *
 *  Created on: Jun 17, 2015
 *      Author: fpantale
 */

#include <cuda.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include "Cell.h"
#include "CUDAQueue.h"
#include <iostream>

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
		}


	}










}
