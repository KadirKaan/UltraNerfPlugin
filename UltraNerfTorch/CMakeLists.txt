# Set the project name
cmake_minimum_required(VERSION 3.5.0)
project(UltraNerfTorch)

add_compile_options("-fPIC")
add_compile_options("-fno-sized-deallocation")

# Add a library with the above sources
add_library(${PROJECT_NAME} src/NeRFModel.cpp src/UltraNeRFRenderer.cpp src/NeRFUtils.cpp)

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
find_package(Torch REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "/libtorch")
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})