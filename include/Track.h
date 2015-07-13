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


constexpr int c_minHitsPerTrack = 4;

template<int maxHitsNum>
struct Track {

	__host__ __device__ Track(int rootHitId) : { m_cells.push(rootHitId);}

// track constructor should be passed the pointer of the root Cell
	//
	CUDAQueue<maxHitsNum-1, int> m_cells;
	float m_trkParameters[5];
};

//

#endif /* INCLUDE_TRACK_H_ */
