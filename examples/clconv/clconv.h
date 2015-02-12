#ifndef CLCONV_H
#define CLCONV_H

#include<initializer_list>
#include<cstdint>

void seperable_conv_3D_sep(const void* datain,void* dataout,
	std::initializer_list<std::size_t> dims,unsigned int num_channels,const float* kernels,unsigned int kernel_length,bool use_gpu=true);

#endif
