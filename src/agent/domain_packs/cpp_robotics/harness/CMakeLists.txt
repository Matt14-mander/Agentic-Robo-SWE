cmake_minimum_required(VERSION 3.16)
project(robotics_cpp_repair LANGUAGES CXX)
if(NOT DEFINED WORKSPACE_HEADER OR NOT DEFINED ROBOTICS_CASE)
  message(FATAL_ERROR "WORKSPACE_HEADER and ROBOTICS_CASE are required")
endif()
add_executable(robotics_cpp_validator main.cpp)
target_compile_features(robotics_cpp_validator PRIVATE cxx_std_17)
target_compile_definitions(robotics_cpp_validator PRIVATE
  WORKSPACE_HEADER="${WORKSPACE_HEADER}" ROBOTICS_CASE=${ROBOTICS_CASE})
