cmake_minimum_required(VERSION 3.14)
project(matmul_openvino_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_executable(matmul_openvino main.cpp)
target_link_libraries(matmul_openvino PRIVATE openvino::runtime)
