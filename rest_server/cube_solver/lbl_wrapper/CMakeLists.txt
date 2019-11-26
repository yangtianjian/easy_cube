cmake_minimum_required(VERSION 3.15)
project(lbl_wrapper)

#set(PYBIND11_CPP_STANDARD -std=c++11)

#include_directories(/Users/apple/anaconda3/envs/deep_learning_36/include/python3.6m)
#include_directories(/Users/apple/anaconda3/envs/deep_learning_36/include/c++/v1)
#link_directories(/Users/apple/anaconda3/envs/deep_learning_36/lib)
#find_package(pybind11 REQUIRED)
#pybind11_add_module(lblcube lblcube.cpp)
add_executable(lblcube lblcube.cpp)