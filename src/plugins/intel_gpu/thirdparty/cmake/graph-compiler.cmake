# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include_guard()
include("${CMAKE_CURRENT_LIST_DIR}/llvm.cmake")

set(GRAPH_COMPILER_SOURCE_DIR "" CACHE PATH "Local GraphCompiler source directory")
if(NOT GRAPH_COMPILER_SOURCE_DIR)
  find_package(GraphCompiler QUIET CONFIG)
endif()

if (GRAPH_COMPILER_SOURCE_DIR OR NOT GraphCompiler_FOUND)
  option(GRAPH_COMPILER_DYLINK "Use dynamic linking with GraphCompiler" OFF)
  set(GRAPH_COMPILER_REPO "https://github.com/dchigarev/graph-compiler" CACHE STRING "GraphCompiler repository URL")
  set(GRAPH_COMPILER_TAG "ov_pin/0.1.1" CACHE STRING "GraphCompiler git tag/branch")

  include(FetchContent)
  if(GRAPH_COMPILER_SOURCE_DIR)
    FetchContent_Declare(GraphCompiler SOURCE_DIR "${GRAPH_COMPILER_SOURCE_DIR}")
  else()
    message(STATUS "GraphCompiler not found, fetching from: ${GRAPH_COMPILER_REPO}")
    FetchContent_Declare(
        GraphCompiler
        GIT_REPOSITORY ${GRAPH_COMPILER_REPO}
        GIT_TAG ${GRAPH_COMPILER_TAG}
        GIT_SHALLOW TRUE
    )
  endif()
  set(GC_ENABLE_IGC ${ENABLE_IGC_FOR_GPU} CACHE BOOL "Build the Intel GPU backend" FORCE)
  set(GC_ENABLE_TEST OFF CACHE BOOL "" FORCE)
  set(GC_ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
  set(GC_ENABLE_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
  set(GC_DYLINK ${GRAPH_COMPILER_DYLINK} CACHE BOOL "" FORCE)
  set(_ov_build_shared_libs ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS ${GRAPH_COMPILER_DYLINK})
  FetchContent_MakeAvailable(GraphCompiler)
  set(BUILD_SHARED_LIBS ${_ov_build_shared_libs})
endif()

if(NOT ENABLE_IGC_FOR_GPU AND NOT TARGET GcGpuOclSourceRuntime)
  message(FATAL_ERROR "This GraphCompiler has no OpenCL source backend. Set GRAPH_COMPILER_SOURCE_DIR to a GraphCompiler checkout with OclSourceModule support.")
endif()

if(NOT ENABLE_IGC_FOR_GPU AND GraphCompiler_ENABLE_IGC)
  message(FATAL_ERROR "Rebuild GraphCompiler with GC_ENABLE_IGC=OFF to remove Intel compiler dependencies.")
endif()
