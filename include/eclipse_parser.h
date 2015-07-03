/*
 * eclipse_parser.h
 *
 *  Created on: Jul 3, 2015
 *      Author: fpantale
 */

#ifndef GPUCA_INCLUDE_ECLIPSE_PARSER_H_
#define GPUCA_INCLUDE_ECLIPSE_PARSER_H_


#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __host__
#define __shared__
#define CUDA_KERNEL_DIM(...)

#else
#define CUDA_KERNEL_DIM(...)  <<< __VA_ARGS__ >>>

#endif



#endif /* GPUCA_INCLUDE_ECLIPSE_PARSER_H_ */
