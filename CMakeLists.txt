cmake_minimum_required(VERSION 3.25)
project(MC-RayTracing VERSION 1.0 LANGUAGES CXX CUDA)

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

# find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

file(GLOB_RECURSE LIB_HEADERS lib/*.h)
foreach(FILE ${LIB_HEADERS})
    get_filename_component(DIR_PATH ${FILE} PATH)
    include_directories(${DIR_PATH})
endforeach()

file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu lib/*.cpp lib/*.cu)
add_executable(MC-RayTracing ${SRC_FILES})
target_link_libraries(MC-RayTracing ${SDL2_LIBRARIES})

set_property(TARGET MC-RayTracing 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)