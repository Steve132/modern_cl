

add_library(clconv clconv.cpp clconv.h)
target_link_libraries(clconv OpenCL)
add_executable(clconv_test clconv_test.cpp)
target_link_libraries(clconv_test clconv)