# Define a new CMake project for the demo plugin
cmake_minimum_required(VERSION 3.5.0)
project(UltraNerfPlugin)
add_compile_options("-fPIC")
#add_compile_options("-fno-sized-deallocation")

#Locate torch

# Locate the ImFusion SDK.
find_package(ImFusionLib REQUIRED)

# Enable automatic MOC, RCC and UIC preprocessing for Qt
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(UltraNerfTorch)

# Define and configure the CMake target
set(Sources
  src/UltraNerfTrainingAlgorithm.cpp
  src/UltraNerfTrainingController.cpp
  src/UltraNerfInferenceAlgorithm.cpp
  src/UltraNerfInferenceController.cpp
  src/UltraNerfFactory.cpp
  src/UltraNerfPlugin.cpp
  )
set(Headers
  include/UltraNerfTrainingAlgorithm.h
  include/UltraNerfTrainingController.h
  include/UltraNerfInferenceAlgorithm.h
  include/UltraNerfInferenceController.h
  include/UltraNerfFactory.h
  include/UltraNerfPlugin.h
  )
set(UiSources
	src/UltraNerfTrainingController.ui
  src/UltraNerfInferenceController.ui
)

# define torch library files ------------------------------------------------
list(APPEND CMAKE_PREFIX_PATH "UltraNerfTorch/libtorch")
find_package(Torch REQUIRED)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
set(TORCH_STATIC ON)

# Define target library
add_library(UltraNerfPlugin SHARED ${Sources} ${UiHeaders} ${Headers})
target_include_directories(UltraNerfPlugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
# Link against the ImFusionLib and selected modules/plugins
target_link_libraries(UltraNerfPlugin PUBLIC
  ImFusionLib
  UltraNerfTorch
  ${TORCH_LIBRARIES}
)
set_property(TARGET UltraNerfPlugin PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Define output target directories and provide instructions on how to launch
# the ImFusion Suite with the built custom plugin.
# These functions are provided by the ImFusionLib target config. 
imfusion_set_common_target_properties()
imfusion_provide_ide_instructions()

