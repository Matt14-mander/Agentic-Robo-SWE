cmake_minimum_required(VERSION 3.18)
project(robo_swe_pinocchio_rnea LANGUAGES CXX)
find_package(pinocchio 3.4.0 EXACT REQUIRED CONFIG)
add_executable(pinocchio_rnea_validator main.cpp)
target_compile_features(pinocchio_rnea_validator PRIVATE cxx_std_17)
target_link_libraries(pinocchio_rnea_validator PRIVATE
  pinocchio::pinocchio ${CMAKE_DL_LIBS})
find_library(CPPAD_RUNTIME NAMES cppad_lib HINTS "${AUTODIFF_PREFIX}/lib" "${AUTODIFF_PREFIX}/lib64" REQUIRED)
target_link_libraries(pinocchio_rnea_validator PRIVATE "${CPPAD_RUNTIME}")
target_include_directories(pinocchio_rnea_validator SYSTEM PRIVATE "${AUTODIFF_PREFIX}/include")
target_compile_definitions(pinocchio_rnea_validator PRIVATE
  CPPAD_DEBUG_AND_RELEASE
  WORKSPACE_HEADER="${WORKSPACE_HEADER}" ROBOT_URDF="${ROBOT_URDF}"
  URDF_SHA256="${URDF_SHA256}")
