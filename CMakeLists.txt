cmake_minimum_required(VERSION 3.20)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# Export compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Prefer Python from the venv
set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Find pybind11 installed in venv
find_package(pybind11 CONFIG REQUIRED)

# Find Eigen3 system headers, fallback to FetchContent if missing
find_package(Eigen3 QUIET)
if (NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(eigen)
endif()

# --- Build pybind11 module ---
pybind11_add_module(ginterp src/pyftle/grid_interp/GridInterp.cpp)

# Link pybind11 and Eigen
target_link_libraries(ginterp PRIVATE pybind11::module Eigen3::Eigen)

# Set C++ standard and optimization
target_compile_features(ginterp PRIVATE cxx_std_17)
if(MSVC)
    # Flags for Windows (Visual Studio)
    target_compile_options(ginterp PRIVATE /O2 /MP)
else()
    # Flags for Linux/macOS (GCC/Clang)
    target_compile_options(ginterp PRIVATE -O3)
endif()

# --- Copy compile_commands.json to project root for clangd ---
# add_custom_target(
#     copy_compile_commands ALL
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#         ${CMAKE_BINARY_DIR}/compile_commands.json
#         ${CMAKE_SOURCE_DIR}/compile_commands.json
#     COMMENT "Copying compile_commands.json to project root for clangd"
# )

# --- Install target ---
install(TARGETS ginterp DESTINATION pyftle)
