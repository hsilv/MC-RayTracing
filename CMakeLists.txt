cmake_minimum_required(VERSION 3.25)
  project(MC-RayTracing VERSION 1.0)

  find_package(SDL2 REQUIRED)
  include_directories(${SDL2_INCLUDE_DIRS})

  add_executable(MC-RayTracing src/main.cpp)
  target_link_libraries(MC-RayTracing ${SDL2_LIBRARIES})
