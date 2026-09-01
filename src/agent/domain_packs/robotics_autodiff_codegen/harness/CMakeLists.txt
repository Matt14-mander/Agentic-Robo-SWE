cmake_minimum_required(VERSION 3.16)
project(agentic_robo_swe_autodiff_codegen LANGUAGES CXX)

foreach(required WORKSPACE_HEADER AUTODIFF_PREFIX)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()

add_executable(autodiff_codegen_validator main.cpp)
target_compile_features(autodiff_codegen_validator PRIVATE cxx_std_17)
target_include_directories(autodiff_codegen_validator SYSTEM PRIVATE "${AUTODIFF_PREFIX}/include")
target_compile_definitions(
  autodiff_codegen_validator
  PRIVATE
    WORKSPACE_HEADER="${WORKSPACE_HEADER}"
    OUTPUT_TOLERANCE=${OUTPUT_TOLERANCE}
    JACOBIAN_RELATIVE_TOLERANCE=${JACOBIAN_RELATIVE_TOLERANCE}
)
target_link_libraries(autodiff_codegen_validator PRIVATE ${CMAKE_DL_LIBS})
