#include<OpenCL.hpp>

struct SeperableConvolver
{
	cl::Device dev;
	cl::Context ctx;
	cl::Program pgm;
	cl::CommandQueue queue;
	cl::Kernel conv3;
	
	static const std::string pgmsrc;
	SeperableConvolver(bool use_gpu):
		dev(cl::Platform::GetIDs()[0].GetDeviceIDs(use_gpu ? CL_DEVICE_TYPE_GPU :CL_DEVICE_TYPE_CPU)[0]),
		ctx({dev}),
		pgm(ctx,{pgmsrc}),
		conv3(pgm,"conv3"),
		queue(ctx,dev)
	{}
};

const std::string SeperableConvolver::pgmsrc=R"CLSRC(
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
const sampler_t reader_samp =
CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;
__kernel void conv3(__write_only image_3d_t dst,__read_only image_2d_t src,__constant float* kerns,unsigned int kernsize,unsigned int kernselect)
{
	float4 color=(float4){0.0f,0.0f,0.0f,0.0f};
	int4 coorddst=(int4){get_global_size(0),get_global_size(1),get_global_size(2),0};
	if(any(isgreaterthanequal(coorddst,get_image_dim(src))))
	{
		return;
	}
	int4 coordsrc=coorddst;
	coordsrc[kernselect]-=kernsize/2;
	__constant float* K=kerns+kernselect*kernsize;
	for(unsigned int i=0;i<kernsize;i++)
	{
		float4 tc=read_imagef(src,reader_samp,coordsrc);
		color+=tc*K[i];
		coordsrc[kernselect]++;
	}
	write_imagef(dst,coorddst,color);
}

)CLSRC";

void seperable_conv_3D_sep(const void* datain,void* dataout,
	std::initializer_list<std::size_t> dims,unsigned int num_channels,const float* kernels,unsigned int kernel_length,bool use_gpu)
{
	static SeperableConvolver sc(use_gpu);
	static const cl_channel_order channels[4]={CL_R,CL_RG,CL_RGB,CL_RGBA};
	cl::Image src(sc.ctx,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,{channels[num_channels],CL_FLOAT},dims,{0,0},const_cast<void*>(datain));
	cl::Image dst(sc.ctx,CL_MEM_READ_WRITE,{channels[num_channels],CL_FLOAT},dims);
	cl::Buffer convkernel(sc.ctx,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,3*kernel_length,(void*)kernels);
	
	std::vector<cl::Event> evs(1);
	cl::Image* srcp=&src;
	cl::Image* dstp=&dst;

	for(unsigned int i=0;i<3;i++)
	{
		sc.conv3.setArgs(*dstp,*srcp,convkernel,kernel_length,i);
		evs[i]=sc.queue.EnqueueNDRangeKernel(sc.conv3,{},dims,{},evs);
		std::swap(srcp,dstp);		
	}
	sc.queue.EnqueueReadImage(dst,true,{},dims,{},dataout,evs);
}
