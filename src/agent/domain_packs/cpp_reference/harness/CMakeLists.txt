cmake_minimum_required(VERSION 3.16)
project(agentic_robo_swe_cpp_reference LANGUAGES CXX)

if(NOT DEFINED WORKSPACE_HEADER)
  message(FATAL_ERROR "WORKSPACE_HEADER is required")
endif()

add_executable(cpp_reference_validator main.cpp)
target_compile_features(cpp_reference_validator PRIVATE cxx_std_17)
target_compile_definitions(
  cpp_reference_validator
  PRIVATE WORKSPACE_HEADER="${WORKSPACE_HEADER}"
)

