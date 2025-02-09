cmake_minimum_required(VERSION 3.5.0)
project(UltraNerfTorch)

find_package(Python3 REQUIRED COMPONENTS Development)
# Add compile options
add_compile_options("-fPIC")
add_compile_options("-fno-sized-deallocation")

# Change to STATIC and ensure proper visibility
add_library(${PROJECT_NAME} STATIC)

# Add source files
target_sources(${PROJECT_NAME} PRIVATE
    src/NeRFModel.cpp
    src/UltraNeRFRenderer.cpp
    src/NeRFUtils.cpp
)

# Force static linking for Torch
set(TORCH_STATIC ON)
set(BUILD_SHARED_LIBS OFF)
find_package(Torch REQUIRED)

# Make sure we're using the same runtime library as the main application
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

find_package(Python3 COMPONENTS Development REQUIRED)
# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# Include directories
target_include_directories(${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PUBLIC
    Python3::Python
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    ${TORCH_CUDA_LIBRARIES}
)