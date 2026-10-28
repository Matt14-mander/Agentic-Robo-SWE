cmake_minimum_required(VERSION 3.16)
project(robo_workspace_example LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
add_executable(controller main.cpp controller.cpp)
