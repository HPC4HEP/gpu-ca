/*
 * Track.h
 *
 *  Created on: Jun 19, 2015
 *      Author: fpantale
 */

#ifndef INCLUDE_TRACK_H_
#define INCLUDE_TRACK_H_

#include "eclipse_parser.h"

#include "CUDAQueue.h"


constexpr int c_minHitsPerTrack = 3;

template<int maxHitsNum>
class Track {
public:
	__host__ __device__ Track(): m_cells() { }

// track constructor should be passed the pointer of the root Cell
	//

	CUDAQueue<maxHitsNum, int> m_cells;
	CUDAQueue<maxHitsNum, int> m_hits;
	float m_trkParameters[5];
};

//

#endif /* INCLUDE_TRACK_H_ */
