# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include_guard()
include(CheckLinkerFlag)
include("${CMAKE_CURRENT_LIST_DIR}/llvm.cmake")

find_package(GraphCompiler QUIET CONFIG)

if (NOT GraphCompiler_FOUND)
  option(GRAPH_COMPILER_DYLINK "Use dynamic linking with GraphCompiler" OFF)
  set(GRAPH_COMPILER_REPO "https://github.com/aobolensk/graph-compiler" CACHE STRING "GraphCompiler repository URL")
  set(GRAPH_COMPILER_TAG "aafa51519f13e701de2c701bf37fba209ae4a6aa" CACHE STRING "GraphCompiler git revision")
  message(STATUS "GraphCompiler not found, fetching from: ${GRAPH_COMPILER_REPO}")
  include(FetchContent)
  FetchContent_Declare(
      GraphCompiler
      GIT_REPOSITORY ${GRAPH_COMPILER_REPO}
      GIT_TAG ${GRAPH_COMPILER_TAG}
  )
  if(WIN32 OR APPLE OR NOT X86_64)
    set(GC_ENABLE_IGC OFF CACHE BOOL "" FORCE)
    set(GC_ENABLE_OCL ON CACHE BOOL "" FORCE)
  endif()
  set(GC_ENABLE_TEST OFF CACHE BOOL "" FORCE)
  set(GC_ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
  set(GC_ENABLE_ZE OFF CACHE BOOL "" FORCE)
  set(GC_DYLINK ${GRAPH_COMPILER_DYLINK} CACHE BOOL "" FORCE)
  set(_ov_build_shared_libs ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS ${GRAPH_COMPILER_DYLINK})
  FetchContent_MakeAvailable(GraphCompiler)
  set(BUILD_SHARED_LIBS ${_ov_build_shared_libs})
  set(GraphCompiler_ENABLE_OCL ${GC_ENABLE_OCL})
endif()

check_linker_flag(CXX "LINKER:--icf=safe" OV_GPU_LINKER_HAS_ICF)
if(OV_GPU_LINKER_HAS_ICF)
  target_link_options(GraphCompiler INTERFACE $<$<CONFIG:Release>:LINKER:--icf=safe >)
endif()

check_linker_flag(CXX "LINKER:-z,pack-relative-relocs" OV_GPU_LINKER_HAS_PACK_RELATIVE_RELOCS)
if(OV_GPU_LINKER_HAS_PACK_RELATIVE_RELOCS)
  include(CheckCXXSourceCompiles)
  check_cxx_source_compiles("#include <features.h>
static_assert(__GLIBC_PREREQ(2, 36), \"\");
int main() {}" OV_GPU_GLIBC_HAS_DT_RELR)
  if(OV_GPU_GLIBC_HAS_DT_RELR)
    target_link_options(GraphCompiler INTERFACE $<$<CONFIG:Release>:LINKER:-z,pack-relative-relocs>)
  endif()
endif()
