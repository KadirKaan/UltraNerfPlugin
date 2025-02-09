cmake_minimum_required(VERSION 3.5.0)
project(UltraNerfPlugin)

# Compiler options
add_compile_options("-fPIC")
add_compile_options("-fno-sized-deallocation")

# Force static linking for Torch before anything else
set(TORCH_STATIC ON)
set(BUILD_SHARED_LIBS OFF)

# Make sure we're using the same runtime library
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

# Locate the ImFusion SDK
find_package(ImFusionLib REQUIRED)

# Qt settings
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

# Add UltraNerfTorch static library
add_subdirectory(UltraNerfTorch)

# Source files
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

# Torch configuration
set(TORCH_INSTALL_PREFIX "UltraNerfTorch/libtorch")
set(CMAKE_PREFIX_PATH ${TORCH_INSTALL_PREFIX})
find_package(Torch REQUIRED)

find_package(Python3 COMPONENTS Development REQUIRED)

# Make sure TORCH_API_INCLUDE_EXTENSION_H is defined
add_definitions(-DTORCH_API_INCLUDE_EXTENSION_H)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find OpenCV if not already found
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED)
endif()

# Define plugin as shared library
add_library(UltraNerfPlugin SHARED ${Sources} ${UiHeaders} ${Headers})

target_include_directories(UltraNerfPlugin 
    PRIVATE ${Python3_INCLUDE_DIRS}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${OpenCV_INCLUDE_DIRS}
    PRIVATE ${TORCH_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(UltraNerfPlugin
    PRIVATE Python3::Python
    PRIVATE ${TORCH_LIBRARIES}
    PRIVATE ${OpenCV_LIBS}
    PRIVATE ${TORCH_CUDA_LIBRARIES}
    PRIVATE UltraNerfTorch
    PRIVATE ImFusionLib
)

# C++ standard settings
set_property(TARGET UltraNerfPlugin PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ImFusion specific settings
imfusion_set_common_target_properties()
imfusion_provide_ide_instructions()
