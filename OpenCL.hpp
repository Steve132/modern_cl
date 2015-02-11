#ifndef OPENCL_ALT_HPP
#define OPENCL_ALT_HPP
#include<CL/cl.h>
#include<string>
#include<unordered_map>
#include<string>
#include<stdexcept>
#include<vector>
#include<functional>

#define CL_ALT_HPP_EXCEPTIONS

namespace cl
{
class Platform;
class Device;
class Context;
class CommandQueue;
class MemObject;
class Program;
class Kernel;
class Event;
class Image;
class Sampler;

class Exception: public std::runtime_error
{
private:
	static const std::string& error_strings(int a)
	{
		static std::unordered_map<int,std::string> es{
			{CL_SUCCESS,"CL_SUCCESS"},
			{CL_DEVICE_NOT_FOUND,"CL_DEVICE_NOT_FOUND"},
			{CL_DEVICE_NOT_AVAILABLE,"CL_DEVICE_NOT_AVAILABLE"},
			{CL_COMPILER_NOT_AVAILABLE,"CL_COMPILER_NOT_AVAILABLE"},
			{CL_MEM_OBJECT_ALLOCATION_FAILURE,"CL_MEM_OBJECT_ALLOCATION_FAILURE"},
			{CL_OUT_OF_RESOURCES,"CL_OUT_OF_RESOURCES"},
			{CL_OUT_OF_HOST_MEMORY,"CL_OUT_OF_HOST_MEMORY"},
			{CL_PROFILING_INFO_NOT_AVAILABLE,"CL_PROFILING_INFO_NOT_AVAILABLE"},
			{CL_MEM_COPY_OVERLAP,"CL_MEM_COPY_OVERLAP"},
			{CL_IMAGE_FORMAT_MISMATCH,"CL_IMAGE_FORMAT_MISMATCH"},
			{CL_IMAGE_FORMAT_NOT_SUPPORTED,"CL_IMAGE_FORMAT_NOT_SUPPORTED"},
			{CL_BUILD_PROGRAM_FAILURE,"CL_BUILD_PROGRAM_FAILURE"},
			{CL_MAP_FAILURE,"CL_MAP_FAILURE"},
			{CL_INVALID_VALUE,"CL_INVALID_VALUE"},
			{CL_INVALID_DEVICE_TYPE,"CL_INVALID_DEVICE_TYPE"},
			{CL_INVALID_PLATFORM,"CL_INVALID_PLATFORM"},
			{CL_INVALID_DEVICE,"CL_INVALID_DEVICE"},
			{CL_INVALID_CONTEXT,"CL_INVALID_CONTEXT"},
			{CL_INVALID_QUEUE_PROPERTIES,"CL_INVALID_QUEUE_PROPERTIES"},
			{CL_INVALID_COMMAND_QUEUE,"CL_INVALID_COMMAND_QUEUE"},
			{CL_INVALID_HOST_PTR,"CL_INVALID_HOST_PTR"},
			{CL_INVALID_MEM_OBJECT,"CL_INVALID_MEM_OBJECT"},
			{CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"},
			{CL_INVALID_IMAGE_SIZE,"CL_INVALID_IMAGE_SIZE"},
			{CL_INVALID_SAMPLER,"CL_INVALID_SAMPLER"},
			{CL_INVALID_BINARY,"CL_INVALID_BINARY"},
			{CL_INVALID_BUILD_OPTIONS,"CL_INVALID_BUILD_OPTIONS"},
			{CL_INVALID_PROGRAM,"CL_INVALID_PROGRAM"},
			{CL_INVALID_PROGRAM_EXECUTABLE,"CL_INVALID_PROGRAM_EXECUTABLE"},
			{CL_INVALID_KERNEL_NAME,"CL_INVALID_KERNEL_NAME"},
			{CL_INVALID_KERNEL_DEFINITION,"CL_INVALID_KERNEL_DEFINITION"},
			{CL_INVALID_KERNEL,"CL_INVALID_KERNEL"},
			{CL_INVALID_ARG_INDEX,"CL_INVALID_ARG_INDEX"},
			{CL_INVALID_ARG_VALUE,"CL_INVALID_ARG_VALUE"},
			{CL_INVALID_ARG_SIZE,"CL_INVALID_ARG_SIZE"},
			{CL_INVALID_KERNEL_ARGS,"CL_INVALID_KERNEL_ARGS"},
			{CL_INVALID_WORK_DIMENSION,"CL_INVALID_WORK_DIMENSION"},
			{CL_INVALID_WORK_GROUP_SIZE,"CL_INVALID_WORK_GROUP_SIZE"},
			{CL_INVALID_WORK_ITEM_SIZE,"CL_INVALID_WORK_ITEM_SIZE"},
			{CL_INVALID_GLOBAL_OFFSET,"CL_INVALID_GLOBAL_OFFSET"},
			{CL_INVALID_EVENT_WAIT_LIST,"CL_INVALID_EVENT_WAIT_LIST"},
			{CL_INVALID_EVENT,"CL_INVALID_EVENT"},
			{CL_INVALID_OPERATION,"CL_INVALID_OPERATION"},
			{CL_INVALID_GL_OBJECT,"CL_INVALID_GL_OBJECT"},
			{CL_INVALID_BUFFER_SIZE,"CL_INVALID_BUFFER_SIZE"},
			{CL_INVALID_MIP_LEVEL,"CL_INVALID_MIP_LEVEL"},
			{CL_INVALID_GLOBAL_WORK_SIZE,"CL_INVALID_GLOBAL_WORK_SIZE"},

		};
		return es[a];
	}
public:
	Exception(int errcode,const std::string& what=""):
		std::runtime_error("OpenCL Produced error: "+error_strings(errcode)+"\n: "+what)
	{}	
};

namespace impl
{
	void check_result(cl_int result,cl_int check,const std::string& what="")
	{
#ifdef CL_ALT_HPP_EXCEPTIONS
		if(result != CL_SUCCESS)
		{
			if(check==result || check!=CL_SUCCESS)
			{
				throw Exception(result,what);
			}
		}
#endif
	}
	template <typename T>
	struct has_value_type {
		typedef char yes[1];
		typedef char no[2];
	
		template <typename C>
		static yes& test(typename C::value_type*);
	
		template <typename>
		static no& test(...);
	
		static const bool value = sizeof(test<T>(0)) == sizeof(yes);
	};
	template<class ReturnType>
	typename std::enable_if<has_value_type<ReturnType>::value,ReturnType>::type
		GetInfo_parse_from_data(const uint8_t* data,size_t n)
	{
		typedef typename ReturnType::value_type Value;
		//static_assert(std::is_copy_constructible<Value>::value
		size_t num_datatypes=n/sizeof(Value);
		const Value* dptr=reinterpret_cast<const Value*>(data);
		return ReturnType(dptr,dptr+num_datatypes);
	}
	template<class ReturnType>
	typename std::enable_if<!has_value_type<ReturnType>::value,ReturnType>::type
		GetInfo_parse_from_data(const uint8_t* data,size_t n)
	{
		//static_assert(std::is_copy_constructible<ReturnType>::value
		size_t num_datatypes=n/sizeof(ReturnType);
		const ReturnType* dptr=reinterpret_cast<const ReturnType*>(data);
		return *dptr;
	}
	template<class InfoReturnType,cl_int invalid_id_error,class id_type,class info_type,class Func>
	InfoReturnType GenericGetInfo(const id_type& id,const info_type& paramname,const Func& gif)
	{
		std::size_t required;
		cl_int result=gif(id,paramname,0,NULL,&required);
		check_result(result,invalid_id_error,"The given id is not valid");
		check_result(result,CL_INVALID_VALUE,"The param name given is not supported");
		
		std::vector<uint8_t> s(required);
		gif(id,paramname,required,s.data(),NULL);
		return GetInfo_parse_from_data<InfoReturnType>(&s[0],required);
	}
	
	template<class id_type>
	struct obj_func_computer
	{
		static cl_int nullfunc(id_type) {return CL_SUCCESS;}
	};

	template<	class id_type,
			class info_type, 
			cl_int (*info_func)(id_type,info_type,size_t,void*,size_t*),
			cl_int invalid_id_error,
			cl_int (*retain_func)(id_type),
			cl_int (*release_func)(id_type)
		>
	class ObjectBase
	{
	public:
		id_type id;
		
		ObjectBase(const id_type& nid):id(nid)
		{}
		
		operator id_type() const { return id; } 
		
		template<class InfoReturnType>
		InfoReturnType GetInfo(info_type paramname) const
		{
			return GenericGetInfo<InfoReturnType,invalid_id_error>(id,paramname,info_func);
		}
		void Retain()
		{
			cl_int result=retain_func(id);
			impl::check_result(result,CL_SUCCESS,"Error retaining the device");
		}
		void Release()
		{
			cl_int result=release_func(id);
			impl::check_result(result,CL_SUCCESS,"Error releasing the device");
		}
	};
	
	template<class Type>
	class Object;
	
}

namespace impl 
{
	template<>
	class Object<Platform>: public ObjectBase<
			cl_platform_id,
			cl_platform_info,
			clGetPlatformInfo,
			CL_INVALID_PLATFORM,
			impl::obj_func_computer<cl_platform_id>::nullfunc,
			impl::obj_func_computer<cl_platform_id>::nullfunc>
	{
	public:
		Object(const cl_platform_id& id):
			ObjectBase<
			cl_platform_id,
			cl_platform_info,
			clGetPlatformInfo,
			CL_INVALID_PLATFORM,
			impl::obj_func_computer<cl_platform_id>::nullfunc,
			impl::obj_func_computer<cl_platform_id>::nullfunc>(id)
		{}
	};
}
class Platform: public impl::Object<Platform>
{
public:	
	Platform(const cl_platform_id& pid):
		impl::Object<Platform>(pid)
	{}
	static std::vector<Platform> GetIDs()
	{
		cl_uint num_platforms;
		cl_int result=clGetPlatformIDs(0,NULL,&num_platforms);
		impl::check_result(result,CL_SUCCESS,"Error enumerating platforms");
		std::vector<cl_platform_id> ids(num_platforms);

		result=clGetPlatformIDs(num_platforms,&ids[0],NULL);
		impl::check_result(result,CL_SUCCESS,"Error enumerating platforms");
		
		return std::vector<Platform>(ids.cbegin(),ids.cend());
	}

	std::vector<Device> GetDeviceIDs(cl_device_type dt=CL_DEVICE_TYPE_GPU)
	{
		cl_uint num_devices;
		cl_int result=clGetDeviceIDs(id,dt,0,NULL,&num_devices);
		impl::check_result(result,CL_SUCCESS,"Error enumerating devices");
		std::vector<cl_device_id> ids(num_devices);
		
		result=clGetDeviceIDs(id,dt,num_devices,&ids[0],NULL);
		impl::check_result(result,CL_SUCCESS,"Error enumerating devices");

		return std::vector<Device>(ids.cbegin(),ids.cend());
	}
};






namespace impl 
{
	template<>
	class Object<Device>: public ObjectBase<
			cl_device_id,
			cl_device_info,
			clGetDeviceInfo,
			CL_INVALID_DEVICE,
#ifdef CL1_2
			clRetainDevice,
			clReleaseDevice,
#else
			impl::obj_func_computer<cl_device_id>::nullfunc,
			impl::obj_func_computer<cl_device_id>::nullfunc
#endif
		>
	{
	public:
		Object(const cl_device_id& id):
			ObjectBase<
						cl_device_id,
			cl_device_info,
			clGetDeviceInfo,
			CL_INVALID_DEVICE,
#ifdef CL1_2
			clRetainDevice,
			clReleaseDevice,
#else
			impl::obj_func_computer<cl_device_id>::nullfunc,
			impl::obj_func_computer<cl_device_id>::nullfunc
#endif
		>(id)
		{}
	};
}
class Device:public impl::Object<Device>
{
public:
	Device(const cl_device_id& did):
		impl::Object<Device>(did)
	{}
	
	static std::vector<Device> GetIDs(cl_device_type dt=CL_DEVICE_TYPE_GPU)
	{	return Platform(0).GetDeviceIDs(dt); }
	
#ifdef CL1_2
	std::vector<Device> CreateSubDevices(std::vector<cl_device_partition_property> properties=std::vector<cl_device_partition_property>())
	{
		properties.push_back(0);
		cl_uint num_devices;
		cl_int result=clCreateSubDevices(device_id,&properties[0],0,NULL,&num_devices);
		impl::check_result(result,CL_SUCCESS,"Error in clCreateSubDevices");

		std::vector<cl_device_id> ids(num_devices);
		result=clCreateSubDevices(device_id,&properties[0],num_devices,&ids[0],NULL);
		impl::check_result(result,CL_SUCCESS,"Error in clCreateSubDevices");
		return std::vector<Device>(ids.cbegin(),ids.cend());
	}
#endif

};

namespace impl 
{
	template<>
	class Object<Context>: public ObjectBase<
			cl_context,
			cl_context_info,
			clGetContextInfo,
			CL_INVALID_CONTEXT,
			clRetainContext,
			clReleaseContext>
	{
	public:
		Object(const cl_context& id):
			ObjectBase<
			cl_context,
			cl_context_info,
			clGetContextInfo,
			CL_INVALID_CONTEXT,
			clRetainContext,
			clReleaseContext>(id)
		{}
	};
}
class Context: public impl::Object<Context>
{
public:
	typedef std::function<void (const std::string&,const void*,size_t)> NotifyFunction;
	static NotifyFunction default_notify;
private:
	NotifyFunction context_notify;
	
	static void context_notify_bind(const char *errinfo,const void *private_info, size_t cb, void *user_data) 
	{
		Context* ctx=reinterpret_cast<Context*>(user_data);
		ctx->context_notify(errinfo,private_info,cb);
	}
public:	
	Context(const std::vector<Device>& devices,
		std::vector<cl_context_properties> properties=std::vector<cl_context_properties>(),
		const NotifyFunction& notify=Context::default_notify):
			impl::Object<Context>(cl_context())
	{
		properties.push_back(0);
		cl_int result;
		std::vector<cl_device_id> device_ids(devices.cbegin(),devices.cend());
		//std::transform(devices.cbegin(),devices.cend(),device_ids.begin(),[](const Device& d){ return d.)
		id=clCreateContext(&properties[0],device_ids.size(),&device_ids[0],&Context::context_notify_bind,this,&result);
		impl::check_result(result,CL_SUCCESS,"Error creating the context");
	}


	Context(const cl_device_type& dt=CL_DEVICE_TYPE_GPU,
		std::vector<cl_context_properties> properties=std::vector<cl_context_properties>(),
		const NotifyFunction& notify=Context::default_notify):
			impl::Object<Context>(cl_context())
	{
		properties.push_back(0);
		cl_int result;
		id=clCreateContextFromType(&properties[0],dt,&Context::context_notify_bind,this,&result);
		impl::check_result(result,CL_SUCCESS,"Error creating the context");
	}
	
};





namespace impl 
{
	template<>
	class Object<CommandQueue>: public ObjectBase<
			cl_command_queue,
			cl_command_queue_info,
			clGetCommandQueueInfo,
			CL_INVALID_COMMAND_QUEUE,
			clRetainCommandQueue,
			clReleaseCommandQueue>
	{
	public:
		Object(const cl_command_queue& id):
			ObjectBase<
			cl_command_queue,
			cl_command_queue_info,
			clGetCommandQueueInfo,
			CL_INVALID_COMMAND_QUEUE,
			clRetainCommandQueue,
			clReleaseCommandQueue>(id)
		{}
	};
}
class CommandQueue: public impl::Object<CommandQueue>
{
public:

public:
	CommandQueue(const Context& context,const Device& device,std::vector<cl_command_queue_properties> properties=std::vector<cl_command_queue_properties>()):
		impl::Object< CommandQueue >(cl_command_queue())
	{
		int result;
		properties.push_back(0);

#ifdef CL2
		id=clCreateCommandQueueWithProperties(context,device,&properties[0],&result);
#else
		cl_bitfield bf=0;
		for(auto p: properties)
			bf|=p;
		id=clCreateCommandQueue(context,device,bf,&result);
#endif
		impl::check_result(result,CL_SUCCESS,"Error building command queue");
	}
	
	void Flush()
	{
		cl_int result=clFlush(id);
		impl::check_result(result,CL_SUCCESS,"Error flushing the command queue");
	}
	
	void Finish()
	{
		cl_int result=clFinish(id);
		impl::check_result(result,CL_SUCCESS,"Error finishing the command queue");
	}

};

namespace impl 
{
	template<>
	class Object<MemObject>: public ObjectBase<
			cl_mem,
			cl_mem_info,
			clGetMemObjectInfo,
			CL_INVALID_MEM_OBJECT,
			clRetainMemObject,
			clReleaseMemObject>
	{
	public:
		Object(const cl_mem& id):
			ObjectBase<
			cl_mem,
			cl_mem_info,
			clGetMemObjectInfo,
			CL_INVALID_MEM_OBJECT,
			clRetainMemObject,
			clReleaseMemObject>(id)
		{}
	};
}
class MemObject: public impl::Object<MemObject>
{
public:
	typedef std::function<void (const MemObject& o)> DestructorFunction;
	static DestructorFunction default_destruct;
private:
	DestructorFunction destruct_notify;
	static void destruct_bind(cl_mem memobj,void* userdata) 
	{
		MemObject* mo=reinterpret_cast<MemObject*>(userdata);
		mo->destruct_notify(*mo);
	}
protected:
	MemObject():
		impl::Object<MemObject>(cl_mem())
	{}
public:
	void SetDestructorCallback(const DestructorFunction& df)
	{
		destruct_notify=df;
		cl_int result=clSetMemObjectDestructorCallback(id,&MemObject::destruct_bind,this);
	}
};

class Buffer: public MemObject
{
public:
	Buffer(const Context& context,cl_mem_flags flags,size_t size,void* host_ptr=NULL):
		MemObject()
	{
		cl_int result;
		id=clCreateBuffer(context,flags,size,host_ptr,&result);
		impl::check_result(result,CL_SUCCESS,"Error creating buffer");
	}
	//Buffer(const Buffer& buf,cl_mem_flags flags,size_t size,cl_buffer_create_type bct
};

class Image: public MemObject
{
public:
	typedef cl_image_format Format;
	
	Image(const Context& ctx,cl_mem_flags flags,const Format& image_format,
		const std::initializer_list<size_t>& dims_list,const std::initializer_list<size_t>& pitch_list={0,0},void* host_ptr=NULL)
	{
		cl_int result;
		const size_t* dims=dims_list.begin();
		const size_t* pitch=pitch_list.begin();
		switch(dims_list.size())
		{
			case 2:
				id=clCreateImage2D(ctx,flags,&image_format,dims[0],dims[1],pitch[0],host_ptr,&result);
				break;
			case 3:
				id=clCreateImage3D(ctx,flags,&image_format,dims[0],dims[1],dims[2],pitch[0],pitch[1],host_ptr,&result);
				break;
			default:
				throw Exception(CL_INVALID_VALUE,"Dimensions for an image must be either 2 or 3!");
		};
		impl::check_result(result,CL_SUCCESS,"Creating image failed");
	}
	template<class ReturnType>
	ReturnType GetImageInfo(const Device& dev,cl_image_info paramname)
	{
		return impl::GenericGetInfo<ReturnType,CL_INVALID_KERNEL>(id,paramname,clGetImageInfo);
	}
	
	static std::vector<Format> GetSupportedFormats(const Context& ctx,cl_mem_flags flags,cl_mem_object_type mot)
	{
		cl_uint required;
		cl_int result=clGetSupportedImageFormats(ctx,flags,mot,0,NULL,&required);
		impl::check_result(result,CL_SUCCESS,"Error getting supported image formats");
		
		std::vector<cl_image_format> data(required);
		result=clGetSupportedImageFormats(ctx,flags,mot,required,&data[0],NULL);
		impl::check_result(result,CL_SUCCESS,"Error getting supported image formats");
	}
};

namespace impl 
{
	template<>
	class Object<Program>: public ObjectBase<
			cl_program,
			cl_program_info,
			clGetProgramInfo,
			CL_INVALID_PROGRAM,
			clRetainProgram,
			clReleaseProgram>
	{
	public:
		Object(const cl_program& id):
			ObjectBase<
			cl_program,
			cl_program_info,
			clGetProgramInfo,
			CL_INVALID_PROGRAM,
			clRetainProgram,
			clReleaseProgram>(id)
		{}
	};
}
class Program: public impl::Object<Program>
{
private:
	std::function<void (const Program&)> notify;
	static void build_notify_bind(cl_program pgm,void* userdata) 
	{
		Program* ptr=reinterpret_cast<Program*>(userdata);
		ptr->notify(*ptr);
	}

public:
	Program(const Context& ctx,std::vector<std::string>& sources):
		impl::Object<Program>(cl_program())
	{
		cl_int result;
		std::vector<const char*> ptrs(sources.size());
		std::vector<size_t> lengths(sources.size());
		for(size_t i=0;i<sources.size();i++)
		{
			ptrs[i]=sources[i].data();
			lengths[i]=sources[i].size();
		}
		id=clCreateProgramWithSource(ctx,sources.size(),&ptrs[0],&lengths[0],&result);
		impl::check_result(result,CL_SUCCESS,"Error creating program from source");
	}
	Program(const Context& ctx,const std::vector<Device>& devices,std::vector<std::vector<uint8_t> >& binaries):
		impl::Object<Program>(cl_program())
	{
		if(devices.size() != binaries.size())
		{
			throw Exception(CL_INVALID_VALUE,"The number of binaries and devices does not match!");
		}
		cl_int result;
		std::vector<const uint8_t*> ptrs(binaries.size());
		std::vector<size_t> lengths(binaries.size());
		std::vector<cl_int> status(binaries.size());
		std::vector<cl_device_id> device_ids(devices.cbegin(),devices.cend());
		for(size_t i=0;i<binaries.size();i++)
		{
			ptrs[i]=binaries[i].data();
			lengths[i]=binaries[i].size();
		}
		id=clCreateProgramWithBinary(ctx,device_ids.size(),&device_ids[0],&lengths[0],&ptrs[0],&status[0],&result);
		impl::check_result(result,CL_SUCCESS,"Error creating program from source");
		for(size_t i=0;i<status.size();i++)
		{
			impl::check_result(status[i],CL_SUCCESS,std::string()+"Error attaching binary to device"+devices[i].GetInfo<std::string>(CL_DEVICE_NAME));
		}
	}
	
	std::string Build(const std::vector<Device>& devices,const std::string& options="")//,const std::function<void (const Program&)>& callback=[](const Program& p){return CL_SUCCESS;})
	{
		std::vector<cl_device_id> device_ids(devices.cbegin(),devices.cend());
		//notify=callback;
		//cl_int result=clBuildProgram(id,device_ids.size(),&device_ids[0],options.c_str(),&Program::build_notify_bind,this);
		cl_int result=clBuildProgram(id,device_ids.size(),&device_ids[0],options.c_str(),NULL,this);
		std::string log;
		for(const Device& d : devices)
		{
			log+="\nDevice Log:"+d.GetInfo<std::string>(CL_DEVICE_NAME);
			log+=GetBuildInfo<std::string>(d,CL_PROGRAM_BUILD_LOG);
		}
		if(result==CL_BUILD_PROGRAM_FAILURE)
		{
			throw Exception(CL_BUILD_PROGRAM_FAILURE,std::string("Program compilation failed.  Build log reads:\n")+log);
		}
		impl::check_result(result,CL_SUCCESS,"Program complation failed");
		return log;
	}
	
	template<class ReturnType>
	ReturnType GetBuildInfo(const Device& dev,cl_program_build_info paramname)
	{
		return impl::GenericGetInfo<ReturnType,CL_INVALID_PROGRAM>(id,paramname,
			[dev](cl_program id,cl_program_build_info paramname,size_t as,void* data,size_t* aso)
			{	return	clGetProgramBuildInfo(id,dev.id,paramname,as,data,aso); }
		);
	}
	std::vector<Kernel> CreateKernels();
};


namespace impl 
{
	template<>
	class Object<Kernel>: public ObjectBase<
			cl_kernel,
			cl_kernel_info,
			clGetKernelInfo,
			CL_INVALID_KERNEL,
			clRetainKernel,
			clReleaseKernel>
	{
	public:
		Object(const cl_kernel& id):
			ObjectBase<
			cl_kernel,
			cl_kernel_info,
			clGetKernelInfo,
			CL_INVALID_KERNEL,
			clRetainKernel,
			clReleaseKernel>(id)
		{}
	};
}
class Kernel: public impl::Object<Kernel>
{
public:
	Kernel(const Program& p,const std::string& name):
		impl::Object<Kernel>(cl_kernel())
	{
		cl_int result;
		id=clCreateKernel(p,name.c_str(),&result);
		impl::check_result(result,CL_SUCCESS,"Error creating kernel object");
	}
	
	//this works for all POD types..just a straightup memory copy
	//template<class T,typename std::enable_if<std::is_trivially_copyable<T>::value && std::is_standard_layout<T>::value>::type* = nullptr>
	template<class T,typename std::enable_if<std::is_pod<T>::value>::type* = nullptr>
	void SetArg(cl_uint index,const T& arg)
	{
		cl_int result=clSetKernelArg(id,index,sizeof(T),&arg);
		impl::check_result(result,CL_SUCCESS,"Error setting kernel argument");
	}
	void SetArg(cl_uint index, const MemObject& mem)
	{
		SetArg(index,&mem.id);
	}
protected:
	template<unsigned int top>
	void priv_arg()
	{}

	template<unsigned int argindex,class Head,class ...Tail>
	void priv_arg(const Head& h,const Tail&... tail)
	{
		SetArg(argindex,h);
		priv_arg<argindex+1>(tail...);
	}

public:
	template<class ...Args>
	Kernel& setArgs(const Args&... a)
	{
		priv_arg<0>(a...);
		return *this;
	}
	/*template<Args...>
	Kernel& operator (Args&&... a)
	{
		return setArgs(a...);
	}*/
	
	//sets all the args then returns it.  This allows enqueueNdRange(f()) syntax
	
	template<class ReturnType>
	ReturnType GetWorkGroupInfo(const Device& dev,cl_kernel_work_group_info paramname)
	{
		return impl::GenericGetInfo<ReturnType,CL_INVALID_KERNEL>(id,paramname,
			[dev](cl_kernel id,cl_kernel_work_group_info paramname,size_t as,void* data,size_t* aso)
			{	return	clGetKernelWorkGroupInfo(id,dev.id,paramname,as,data,aso); }
		);
	}
	
};


namespace impl 
{
	template<>
	class Object<Event>: public ObjectBase<
			cl_event,
			cl_event_info,
			clGetEventInfo,
			CL_INVALID_EVENT,
			clRetainEvent,
			clReleaseEvent>
	{
	public:
		Object(const cl_event& id):
			ObjectBase<
			cl_event,
			cl_event_info,
			clGetEventInfo,
			CL_INVALID_EVENT,
			clRetainEvent,
			clReleaseEvent>(id)
		{}
	};
}

class Event: public impl::Object<Event>
{
protected:
	static void callback_default(const Event& e,cl_int status) 
	{};
	std::function<void (const Event&,cl_int)> callback;
	static void callback_bind(cl_event e,cl_int status,void* userdata)
	{
		Event* eptr=reinterpret_cast<Event*>(userdata);
		eptr->callback(*eptr,status);
	}
public:
	Event(const Context& ctx):
		impl::Object<Event>(cl_event()),
		callback(&Event::callback_default)
	{
		cl_int result;
		id=clCreateUserEvent(ctx,&result);
		impl::check_result(result,CL_SUCCESS,"Error creating user event");
	}
	Event(const cl_event& e):
		impl::Object<Event>(e),
		callback(&Event::callback_default)
	{}
	void SetUserStatus(cl_int execution_status)
	{
		cl_int result=clSetUserEventStatus(id,execution_status);
		impl::check_result(result,CL_SUCCESS,"Error setting the user status");
	}
	void SetCallback(cl_int callback_type,const std::function<void (const Event&,cl_int)>& func)
	{
		callback=func;
		cl_int result=clSetEventCallback(id,callback_type,&Event::callback_bind,this);
		impl::check_result(result,CL_SUCCESS,"Error etting the callback for the event");
	}
	void Wait()
	{
		cl_int result=clWaitForEvents(1,&id);
		impl::check_result(result,CL_SUCCESS,"Error waiting for events");
	}
	
	static void WaitForEvents(const std::vector<Event>& events)
	{
		std::vector<cl_event> event_ids(events.cbegin(),events.cend());
		cl_int result=clWaitForEvents(event_ids.size(),&event_ids[0]);
		impl::check_result(result,CL_SUCCESS,"Error waiting for events");
	}
	
	template<class ReturnType>
	ReturnType GetProfilingInfo(const Device& dev,cl_profiling_info paramname)
	{
		return impl::GenericGetInfo<ReturnType,CL_INVALID_KERNEL>(id,paramname,clGetEventProfilingInfo);
	}
};

namespace impl 
{
	template<>
	class Object<Sampler>: public ObjectBase<
			cl_sampler,
			cl_sampler_info,
			clGetSamplerInfo,
			CL_INVALID_SAMPLER,
			clRetainSampler,
			clReleaseSampler>
	{
	public:
		Object(const cl_sampler& id):
			ObjectBase<
			cl_sampler,
			cl_sampler_info,
			clGetSamplerInfo,
			CL_INVALID_SAMPLER,
			clRetainSampler,
			clReleaseSampler>(id)
		{}
	};
}

class Sampler: public impl::Object<Sampler>
{
public:
	Sampler(	const Context& ctx,
			bool normalized_coords,
			cl_addressing_mode addr_mode,
			cl_filter_mode filt_mode
		):
			impl::Object<Sampler>(cl_sampler())
	{
		cl_int result;
		id=clCreateSampler(ctx,normalized_coords,addr_mode,filt_mode,&result);
		impl::check_result(result,CL_SUCCESS,"Error with sampler");
	}	
};

}

#endif //ifndef OPENCL_ALT_HPP
