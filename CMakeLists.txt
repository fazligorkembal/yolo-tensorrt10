cmake_minimum_required(VERSION 3.10)

project(tensorrtforge)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# CUDA ve TensorRT yollarını değişkenlere ata
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  set(CUDA_INCLUDE_DIR /usr/local/cuda/targets/aarch64-linux/include)
  set(CUDA_LIB_DIR /usr/local/cuda/targets/aarch64-linux/lib)
else()
  message("embed_platform off")
  set(CUDA_INCLUDE_DIR /usr/local/cuda/include)
  set(CUDA_LIB_DIR /usr/local/cuda/lib64)
  
  # TensorRT include dizini (sisteminize göre ayarlayın)
  set(TENSORRT_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)



include_directories(include)
include_directories(plugin)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIR})

if(DEFINED TENSORRT_INCLUDE_DIR)
  include_directories(${TENSORRT_INCLUDE_DIR})
endif()

link_directories(${CUDA_LIB_DIR})


add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)



if(NOT DEFINED PYTHON_INSTALL_DIR)
    set(PYTHON_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
endif()




pybind11_add_module(
  pybind_tracker
  MODULE
  src/tracker/tracker_bind.cpp
)



add_library(tracker SHARED src/tracker/trackerlib.cpp)
add_compile_definitions(MODULE_PATH="${PROJECT_SOURCE_DIR}/src/tracker")
target_link_libraries(tracker pybind11::embed pybind11::module ${OpenCV_LIBS} fmt::fmt)




file(GLOB_RECURSE SRCS_SCRATCH src/tensorrtforge.cpp src/scratch/*.cpp src/scratch/*.cu)
add_library(${PROJECT_NAME} SHARED ${SRCS_SCRATCH})
target_link_libraries(${PROJECT_NAME} nvinfer cudart myplugins ${OpenCV_LIBS} ${Boost_LIBRARIES} fmt::fmt)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${TENSORRT_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CUDA_LIB_DIR}>
)

install(
  TARGETS ${PROJECT_NAME} tracker
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

add_executable(pipeline_benchmarks tests/pipeline_benchmarks.cpp)
set_source_files_properties(tests/pipeline_benchmarks.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(pipeline_benchmarks nvinfer cudart)


add_library(tensorrt_cpp_api SHARED experimental/engine.cpp)
target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(tensorrt_cpp_api PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES} nvonnxparser fmt::fmt)

add_executable(infer1 experimental/infer1.cpp)
target_link_libraries(infer1 tensorrt_cpp_api nvinfer cudart fmt::fmt ${OpenCV_LIBS} ${Boost_LIBRARIES})