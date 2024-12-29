# Define a new CMake project for the demo plugin
cmake_minimum_required(VERSION 3.5.0)
project(UltraNerfPlugin)
add_compile_options("-fPIC")
add_compile_options("-fno-sized-deallocation")

#Locate torch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Locate the ImFusion SDK.
find_package(ImFusionLib REQUIRED)

add_subdirectory(UltraNerfTorch)
# Enable automatic MOC, RCC and UIC preprocessing for Qt
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
# Define and configure the CMake target
set(Sources
  UltraNerfTrainingAlgorithm.cpp
  UltraNerfTrainingController.cpp
  UltraNerfInferenceAlgorithm.cpp
  UltraNerfInferenceController.cpp
  UltraNerfFactory.cpp
  UltraNerfPlugin.cpp
  )
set(Headers
  UltraNerfTrainingAlgorithm.h
  UltraNerfTrainingController.h
  UltraNerfInferenceAlgorithm.h
  UltraNerfInferenceController.h
  UltraNerfFactory.h
  UltraNerfPlugin.h
  )
set(UiSources
	UltraNerfTrainingController.ui
  UltraNerfInferenceController.ui
)

# define torch library files ------------------------------------------------
# Define target library
add_library(UltraNerfPlugin SHARED ${Sources} ${UiHeaders} ${Headers})
target_include_directories(UltraNerfPlugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link against the ImFusionLib and selected modules/plugins
target_link_libraries(UltraNerfPlugin PUBLIC
  ${TORCH_LIBRARIES}
  UltraNerfTorch
  ImFusionLib
)
set_property(TARGET UltraNerfPlugin PROPERTY CXX_STANDARD 17)
# Define output target directories and provide instructions on how to launch
# the ImFusion Suite with the built custom plugin.
# These functions are provided by the ImFusionLib target config. 
imfusion_set_common_target_properties()
imfusion_provide_ide_instructions()

