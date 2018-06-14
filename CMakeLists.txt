# CMake minimun version
cmake_minimum_required(VERSION 2.8)

# executable name
set(TARGET_NAME sampleDebayer)

# solution name
project(${TARGET_NAME})

if(MSVC)
    add_definitions("/wd4265 /wd4266 /wd4350 /wd4365 /wd4435 /wd4514 /wd4625 /wd4626 /wd4640 /wd4668 /wd4710 /wd4819 /wd4820 /wd4946 /wd5026 /wd5027")
	add_definitions("/Wall")
endif(MSVC)

# find CUDA
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# include dir
include_directories(${OpenCV_INCLUDE_DIRS})

# executable
cuda_add_executable(${TARGET_NAME} main.cpp kernel.cu)

target_link_libraries(${TARGET_NAME} ${CUDA_LIBRARIES} ${OpenCV_LIBS})

