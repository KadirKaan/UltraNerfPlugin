# Define a new CMake project for the demo plugin
project(UltraNerfPlugin)
cmake_minimum_required(VERSION 3.2.0)

add_compile_options("-fno-sized-deallocation")

# Locate the ImFusion SDK.
find_package(ImFusionLib REQUIRED COMPONENTS ImFusionUS)

# Enable automatic MOC, RCC and UIC preprocessing for Qt
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)


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

# Define target library
add_library(UltraNerfPlugin SHARED ${Sources} ${UiHeaders} ${Headers})
target_include_directories(UltraNerfPlugin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
# Link against the ImFusionLib and selected modules/plugins
target_link_libraries(UltraNerfPlugin PRIVATE
  ImFusionLib
  ImFusionUS
  )

# Define output target directories and provide instructions on how to launch
# the ImFusion Suite with the built custom plugin.
# These functions are provided by the ImFusionLib target config. 
imfusion_set_common_target_properties()
imfusion_provide_ide_instructions()

