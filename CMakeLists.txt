cmake_minimum_required(VERSION 4.0)

project(tensorrtforge)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

include_directories(include)
include_directories(plugin)

#set(CUDA_DIR "/usr/local/cuda-12") 

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

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
if(NOT DEFINED PYTHON_INSTALL_DIR)
    set(PYTHON_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
endif()

find_package(pybind11 REQUIRED)
pybind11_add_module(
  pybind_tracker
  MODULE
  src/tracker/tracker_bind.cpp
)

add_library(tracker SHARED src/tracker/trackerlib.cpp)
add_compile_definitions(MODULE_PATH="${PROJECT_SOURCE_DIR}/src/tracker")
target_link_libraries(tracker pybind11::embed pybind11::module ${OpenCV_LIBS})

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

file(GLOB_RECURSE SRCS_SCRATCH src/tensorrtforge.cpp src/scratch/*.cpp src/scratch/*.cu)
add_library(${PROJECT_NAME} SHARED ${SRCS_SCRATCH})
target_link_libraries(${PROJECT_NAME} nvinfer cudart myplugins ${OpenCV_LIBS} ${Boost_LIBRARIES})

target_include_directories(${PROJECT_NAME}
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(
  TARGETS ${PROJECT_NAME} tracker
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)