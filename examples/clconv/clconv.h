#ifndef CLCONV_H
#define CLCONV_H

void seperable_conv_3D_sep(const void* datain,void* dataout,
	std::array<size_t,3> dims,const std::vector<std::vector<double> >& kernels);

#endif
