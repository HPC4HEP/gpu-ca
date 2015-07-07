/*
 * PacketHeader.h
 *
 *  Created on: Jul 7, 2015
 *      Author: fpantale
 */

#ifndef GPUCA_INCLUDE_PACKETHEADER_H_
#define GPUCA_INCLUDE_PACKETHEADER_H_

const constexpr int c_maxNumberOfLayersInPacket = 6;
template <int maxLayersInPacket>
struct PacketHeader {

	int size;      // size in bytes
	int numLayers; // number of layers used. Cannot be larger than maxNumberOfLayersInPacket
	int layer[maxLayersInPacket]; // bytes from the beginning of the payload where the hits from each layer are stored



};


#endif /* GPUCA_INCLUDE_PACKETHEADER_H_ */
