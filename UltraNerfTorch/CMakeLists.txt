# Set the project name
cmake_minimum_required(VERSION 3.5.0)
project(UltraNerfTorch)

add_compile_options("-fPIC")
add_compile_options("-fno-sized-deallocation")

# Add a library with the above sources
add_executable(${PROJECT_NAME} src/main.cpp src/NeRFModel.cpp src/UltraNeRFRenderer.cpp src/NeRFUtils.cpp)
find_package(Torch REQUIRED)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS} )