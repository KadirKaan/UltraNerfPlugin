# Set the project name
project(UltraNerfTorch)

add_compile_options("-fPIC")

# Add a library with the above sources
add_library(${PROJECT_NAME} src/NeRFModel.cpp)

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
find_package(Torch REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "libtorch")
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})