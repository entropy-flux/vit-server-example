cmake_minimum_required(VERSION 3.28)
project(vit-server LANGUAGES CXX)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Import the pre-built libraries
add_library(tannic STATIC IMPORTED)
set_target_properties(tannic PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/external/tannic-nn/external/tannic/build/libtannic.a
)

add_library(tannic-nn STATIC IMPORTED)
set_target_properties(tannic-nn PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/external/tannic-nn/build/libtannic-nn.a
)

# Set include directories for the libraries
target_include_directories(tannic INTERFACE
    ${CMAKE_SOURCE_DIR}/external/tannic-nn/external/tannic/include
)

target_include_directories(tannic-nn INTERFACE
    ${CMAKE_SOURCE_DIR}/external/tannic-nn/include
    ${CMAKE_SOURCE_DIR}/external/tannic-nn/external/tannic/include
)

# Find OpenBLAS
find_library(OPENBLAS_LIB openblas)
if(OPENBLAS_LIB)
    message(STATUS "Found OpenBLAS: ${OPENBLAS_LIB}")
endif()

# Main executable
add_executable(vit-server main.cpp)

# Link libraries in correct order (tannic-nn depends on tannic)
target_link_libraries(vit-server PRIVATE tannic-nn tannic)

# Add other dependencies
if(OPENBLAS_LIB)
    target_link_libraries(vit-server PRIVATE ${OPENBLAS_LIB})
endif()

target_link_libraries(vit-server PRIVATE pthread dl)

# Include directories for the executable
target_include_directories(vit-server PRIVATE
    include/
    model/
    ${CMAKE_SOURCE_DIR}/external/tannic-nn/include
    ${CMAKE_SOURCE_DIR}/external/tannic-nn/external/tannic/include
)