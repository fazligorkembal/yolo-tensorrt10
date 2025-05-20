cmake_minimum_required(VERSION 3.10)

project(tensorrtforge)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
  message("embed_platform off")

  # cuda
  include_directories(/usr/local/cuda/include)
  link_directories(/usr/local/cuda/lib64)

  # tensorrt
  #include_directories(/workspace/shared/TensorRT-10/include)
  #link_directories(/workspace/shared/TensorRT-10/lib)
endif()

add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

file(GLOB_RECURSE SRCS_SCRATCH src/scratch/*.cpp src/scratch/*.cu)
add_library(model_segmentation SHARED ${SRCS_SCRATCH})
target_link_libraries(model_segmentation nvinfer cudart myplugins ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(main_test src/main_test.cpp)
target_link_libraries(main_test model_segmentation)