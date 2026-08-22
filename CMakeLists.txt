#
# top-level CMake configuration file for PDAL
#
# (based originally on the libLAS files copyright Mateusz Loskot)

cmake_minimum_required(VERSION 3.13)

project(PDAL VERSION 2.10.0 LANGUAGES CXX C)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)

# GPUPDAL versions independently from the pinned/upstream-compatible PDAL ABI.
# Keep this value aligned with packages/npm/package.json at release time.
set(GPUPDAL_VERSION_MAJOR 0)
set(GPUPDAL_VERSION_MINOR 1)
set(GPUPDAL_VERSION_PATCH 0)
set(GPUPDAL_VERSION
    "${GPUPDAL_VERSION_MAJOR}.${GPUPDAL_VERSION_MINOR}.${GPUPDAL_VERSION_PATCH}-dev"
    CACHE STRING "GPUPDAL distribution version")

option(BUILD_SHARED_LIBS "Build pdal as a shared library" ON)

set(ROOT_DIR "${PROJECT_SOURCE_DIR}")
include(${ROOT_DIR}/cmake/common.cmake NO_POLICY_SCOPE)

#------------------------------------------------------------------------------
# internal cmake settings
#------------------------------------------------------------------------------

set(CMAKE_COLOR_MAKEFILE ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
include(FeatureSummary)

# Allow advanced users to generate Makefiles printing detailed commands
mark_as_advanced(CMAKE_VERBOSE_MAKEFILE)

#------------------------------------------------------------------------------
# PDAL general settings
#------------------------------------------------------------------------------

#
# Must be changed if there is an ABI change.  This builds the SONAME
# that's embedded in the library and any plugins.
include(${ROOT_DIR}/cmake/libraries.cmake NO_POLICY_SCOPE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#
# Name of C++ library
#
set(PDAL_LIB_NAME pdalcpp)

set(PDAL_KAZHDAN_LIB_NAME pdal_kazhdan)
set(PDAL_LEPCC_LIB_NAME pdal_lepcc)
set(PDAL_TEST_SUPPORT_OBJS pdal_test_support)

set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)

set(PDAL_OUTPUT_LIB_DIR "${PROJECT_BINARY_DIR}/lib")
set(PDAL_OUTPUT_BIN_DIR "${PROJECT_BINARY_DIR}/bin")


if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PDAL_OUTPUT_LIB_DIR}")
endif()

if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PDAL_OUTPUT_BIN_DIR}")
endif()

# allow override of PDAL_PLUGIN_INSTALL_PATH path
if (NOT PDAL_PLUGIN_INSTALL_PATH)
    if (WIN32)
        set(PDAL_PLUGIN_INSTALL_PATH "${CMAKE_INSTALL_BINDIR}" CACHE PATH "PDAL Plugin install location")
    else()
        set(PDAL_PLUGIN_INSTALL_PATH "${CMAKE_INSTALL_LIBDIR}" CACHE PATH "PDAL Plugin install location")
    endif()
endif()

include(${PDAL_CMAKE_DIR}/rpath.cmake)

# wipe lib/ drectory on clean. It will have plugins that could be out of date
# in the next build
set_directory_properties(PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    "${PDAL_OUTPUT_LIB_DIR}/*")

if(WIN32)
  add_definitions("-DPDAL_DLL_EXPORT=1")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE} "${PDAL_OUTPUT_LIB_DIR}" )
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE} "${PDAL_OUTPUT_BIN_DIR}" )
  # ---[ Windows requires DLLs (shared libraries) to be installed in
  # ---[ the same directory as executables
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE} "${PDAL_OUTPUT_BIN_DIR}" )
endif(WIN32)


# Choose package components

include(${PDAL_CMAKE_DIR}/options.cmake)

#------------------------------------------------------------------------------
# General build settings
#------------------------------------------------------------------------------

set(PDAL_BUILD_TYPE ${CMAKE_BUILD_TYPE})

#------------------------------------------------------------------------------
#  Dependencies.
#------------------------------------------------------------------------------

include(${PDAL_CMAKE_DIR}/backtrace.cmake)
include(${PDAL_CMAKE_DIR}/proj.cmake)
include(${PDAL_CMAKE_DIR}/gdal.cmake)
include(${PDAL_CMAKE_DIR}/geotiff.cmake)  # Optional (not really)
include(${PDAL_CMAKE_DIR}/threads.cmake)
include(${PDAL_CMAKE_DIR}/zlib.cmake)
include(${PDAL_CMAKE_DIR}/lzma.cmake)
include(${PDAL_CMAKE_DIR}/zstd.cmake)
include(${PDAL_CMAKE_DIR}/test.cmake)
include(${PDAL_CMAKE_DIR}/ctest.cmake)
include(${PDAL_CMAKE_DIR}/libxml2.cmake)
include(${PDAL_CMAKE_DIR}/dimension.cmake)
include(${PDAL_CMAKE_DIR}/arbiter.cmake)
include(${PDAL_CMAKE_DIR}/h3.cmake)
include(${PDAL_CMAKE_DIR}/nlohmann.cmake)
include(${PDAL_CMAKE_DIR}/schema-validator.cmake)
include(${PDAL_CMAKE_DIR}/lazperf.cmake)
include(${PDAL_CMAKE_DIR}/gcs.cmake) # Optional
include(${PDAL_CMAKE_DIR}/utfcpp.cmake)

#------------------------------------------------------------------------------
# generate the pdal_features.hpp header
#------------------------------------------------------------------------------

# from http://stackoverflow.com/questions/1435953/how-can-i-pass-git-sha1-to-compiler-as-definition-using-cmake
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
if (WITH_PDG)
    include(${PDAL_CMAKE_DIR}/pdg-oracle.cmake)
    # The compatibility oracle's observable writer metadata belongs to the
    # pinned upstream source, not to later commits on this implementation
    # branch.
    set(GIT_SHA1 "${PDG_ORACLE_COMMIT}")
endif()

# needs to come before configuration of pdal_features
if(APPLE)
    option(PDAL_BUNDLE "Create PDAL as Application Bundle on OSX" FALSE)
    if (PDAL_BUNDLE)
        set(PDAL_APP_BUNDLE 1)
    endif()
endif()

set(pdal_features_hpp_in "${CMAKE_CURRENT_SOURCE_DIR}/pdal_features.hpp.in")
set(pdal_features_hpp
    "${CMAKE_CURRENT_BINARY_DIR}/include/pdal/pdal_features.hpp")
configure_file(${pdal_features_hpp_in} ${pdal_features_hpp})

#------------------------------------------------------------------------------
# subdirectory controls
#------------------------------------------------------------------------------

# PDAL_TARGET_OBJECTS is used to collect the driver object libraries
set(PDAL_TARGET_OBJECTS "")

if (WITH_TESTS)
    enable_testing()
endif()

add_subdirectory(plugins)

if (BUILD_DOCS)
    add_subdirectory(doc)
endif()
if (WITH_TESTS)
    include (${PDAL_CMAKE_DIR}/gtest.cmake)
    add_subdirectory(test)
endif()

add_subdirectory(dimbuilder)
add_subdirectory(vendor/arbiter)
add_subdirectory(vendor/schema-validator)
add_subdirectory(vendor/kazhdan)
add_subdirectory(vendor/lazperf)
add_subdirectory(vendor/utfcpp)
add_subdirectory(vendor/lepcc)
add_subdirectory(vendor/h3)
add_subdirectory(tools)
add_subdirectory(apps)

file(GLOB BASE_SRCS
    ${PDAL_FILTERS_DIR}/*.cpp
    ${PDAL_IO_DIR}/*.cpp
    ${PDAL_KERNELS_DIR}/*.cpp
    ${PDAL_SRC_DIR}/*.cpp
    ${PDAL_SRC_DIR}/compression/*.cpp
    ${PDAL_SRC_DIR}/util/*.cpp)
file(GLOB_RECURSE PRIVATE_SRCS
    ${PDAL_FILTERS_DIR}/private/*.cpp
    ${PDAL_IO_DIR}/private/*.cpp
    ${PDAL_KERNELS_DIR}/private/*.cpp
    ${PDAL_SRC_DIR}/private/*.cpp
    ${BACKTRACE_SOURCE})
list(APPEND SRCS ${BASE_SRCS} ${PRIVATE_SRCS})

#
# Remove stuff we don't want to build.
#
if (NOT PDAL_HAVE_LIBXML2)
    file(GLOB XML_SRCS
        io/Ilvis2MetadataReader.cpp
        io/Ilvis2Metadata.cpp
        io/Ilvis2Reader.cpp
        ${PDAL_SRC_DIR}/DbWriter.cpp
        ${PDAL_SRC_DIR}/DbReader.cpp
        ${PDAL_SRC_DIR}/XMLSchema.cpp)
    list(REMOVE_ITEM SRCS ${XML_SRCS})
endif()
if (NOT PDAL_HAVE_ZSTD)
    file(GLOB ZSTD_SRCS
        ${PDAL_SRC_DIR}/compression/ZstdCompression.cpp)
    list(REMOVE_ITEM SRCS ${ZSTD_SRCS})
endif()
if (NOT PDAL_HAVE_ZLIB)
    file(GLOB ZLIB_SRCS
        ${PDAL_SRC_DIR}/compression/DeflateCompression.cpp)
    list(REMOVE_ITEM SRCS ${ZLIB_SRCS})
endif()
if (NOT PDAL_HAVE_LZMA)
    file(GLOB LZMA_SRCS
        ${PDAL_SRC_DIR}/compression/LzmaCompression.cpp)
    list(REMOVE_ITEM SRCS ${LZMA_SRCS})
endif()

PDAL_ADD_LIBRARY(${PDAL_LIB_NAME} ${SRCS})
if (WIN32)
    # fix for "error C4005: '_XKEYCHECK_H': macro redefinition"
    # C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\
    # include\xkeycheck.h(55): fatal error C1189: #error:  The C++ Standard Library forbids
    # macroizing the keyword "bool". Enable warning C4005 to find the forbidden define.
    target_compile_definitions(${PDAL_LIB_NAME} PRIVATE _XKEYCHECK_H=1)
endif()
add_library(PDAL::PDAL ALIAS ${PDAL_LIB_NAME})
#
# Interface include directories allow downstream project to get the directory
# without specification.
#



target_include_directories(${PDAL_LIB_NAME}
    SYSTEM BEFORE PRIVATE
        ${PDAL_VENDOR_EIGEN_DIR}
        ${PDAL_VENDOR_DIR}/kazhdan
        ${PDAL_VENDOR_DIR}/h3/include
)
target_include_directories(${PDAL_LIB_NAME}
    PRIVATE
        ${ROOT_DIR}
        ${PROJECT_BINARY_DIR}/include
        ${PDAL_VENDOR_DIR}
        ${NLOHMANN_INCLUDE_DIR}
        ${GEOTIFF_INCLUDE_DIR}
        ${UTFCPP_INCLUDE_DIR}
)


target_link_libraries(${PDAL_LIB_NAME}
    PRIVATE
        ${CMAKE_THREAD_LIBS_INIT}
        ${CMAKE_DL_LIBS}
        GDAL::GDAL
        PROJ::proj
        ${GEOTIFF_LIBRARY}
        ${PDAL_REEXPORT}
        ${PDAL_ARBITER_LIB_NAME}
        ${PDAL_H3_LIB_NAME}
        ${JSON_SCHEMA_LIB_NAME}
        ${PDAL_KAZHDAN_LIB_NAME}
        ${PDAL_LEPCC_LIB_NAME}
        ${PDAL_LAZPERF_LIB_NAME}
        ${BACKTRACE_LIBRARIES}
       "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:-lstdc++fs>"
    PUBLIC
        ${WINSOCK_LIBRARY}
    INTERFACE
        ${PDAL_LIBDIR}
        ${WINSOCK_LIBRARY}
)

# The process matrices compare the public candidate against the forked host
# CLI, which shares the candidate's library. Fork-side changes to upstream
# stage code can therefore only be exposed by the separately built pinned
# oracle (preset pdal-upstream-tests). When it exists, differential matrices
# that mark cases pinned_oracle compare those cases against it (D0260).
set(PDG_PINNED_ORACLE_EXECUTABLE "" CACHE FILEPATH
    "pinned upstream PDAL executable for pinned-oracle differential cases")
if (NOT PDG_PINNED_ORACLE_EXECUTABLE AND
        EXISTS "${PROJECT_SOURCE_DIR}/build/pdal-upstream-tests/bin/pdal")
    set(PDG_PINNED_ORACLE_EXECUTABLE
        "${PROJECT_SOURCE_DIR}/build/pdal-upstream-tests/bin/pdal")
endif()

if (PDG_BUILD_TESTS)
    set_property(SOURCE io/LasWriter.cpp io/LasReader.cpp io/CopcReader.cpp
        io/GDALWriter.cpp
        filters/OutlierFilter.cpp
        filters/NeighborClassifierFilter.cpp pdal/PointView.cpp
        APPEND PROPERTY COMPILE_DEFINITIONS PDG_ENABLE_TEST_HOOKS=1)
endif()

if (ZSTD_FOUND)
    target_link_libraries(${PDAL_LIB_NAME}
        PRIVATE
        zstd::libzstd)
endif()

if (ZLIB_FOUND)
    target_link_libraries(${PDAL_LIB_NAME}
        PRIVATE
        ZLIB::ZLIB)
endif()

if (LIBLZMA_FOUND)
    target_link_libraries(${PDAL_LIB_NAME}
        PRIVATE
        LibLZMA::LibLZMA)
endif()

if (LibXml2_FOUND)
    target_link_libraries(${PDAL_LIB_NAME}
        PRIVATE
        LibXml2::LibXml2)
endif()



#
# Allow downstream cmake projects to find PDAL header files without
# being explicit.
#
target_include_directories(${PDAL_LIB_NAME}
    INTERFACE
        $<INSTALL_INTERFACE:include>)

target_compile_definitions(${PDAL_LIB_NAME}
    PRIVATE
    H3_PREFIX=PDALH3)

if(WIN32)
    target_compile_definitions(${PDAL_LIB_NAME}
        PUBLIC
            PDAL_DLL_EXPORT=1
    )
endif(WIN32)

set(PDALCPP_LIB_OUTPUT_NAME "pdalcpp" CACHE STRING "Name of the PDALCPP library")

set_target_properties(${PDAL_LIB_NAME} PROPERTIES
    VERSION ${PDAL_BUILD_VERSION}
    SOVERSION ${PDAL_API_VERSION}
    OUTPUT_NAME ${PDALCPP_LIB_OUTPUT_NAME}
    CLEAN_DIRECT_OUTPUT 1)

# shut off -Wpedantic selectively
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" )
    set_source_files_properties(filters/PoissonFilter.cpp PROPERTIES COMPILE_FLAGS "-Wno-pedantic -Wno-array-parameter")
endif()

#
# Installation
#

#
# Only install compression headers if we're building with the particular
# compression type.
#
if (NOT PDAL_HAVE_ZSTD)
    set(ZSTD_EXCLUDES PATTERN pdal/compression/Zstd* EXCLUDE)
endif()
if (NOT PDAL_HAVE_ZLIB)
    set(ZLIB_EXCLUDES PATTERN pdal/compression/Deflate* EXCLUDE)
endif()
if (NOT PDAL_HAVE_LZMA)
    set(LZMA_EXCLUDES PATTERN pdal/compression/Lzma* EXCLUDE)
endif()

install(DIRECTORY ${PDAL_INCLUDE_DIR}/pdal/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/pdal"
    FILES_MATCHING PATTERN "*.hpp"
    PERMISSIONS
        GROUP_EXECUTE WORLD_EXECUTE OWNER_EXECUTE
        GROUP_READ WORLD_READ OWNER_READ
        OWNER_WRITE
#    PATTERN "pdal/private" EXCLUDE
#    PATTERN "pdal/io/private" EXCLUDE
#    PATTERN "pdal/pdal/private" EXCLUDE
#    PATTERN "pdal/util/private" EXCLUDE
#    PATTERN "pdal/filters/private" EXCLUDE
#    ${ZSTD_EXCLUDES}
#    ${ZLIB_EXCLUDES}
#    ${LZMA_EXCLUDES}
)

install(DIRECTORY ${PDAL_KERNELS_DIR}/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/pdal/kernels"
    FILES_MATCHING PATTERN "*.hpp"
    PERMISSIONS
        GROUP_EXECUTE WORLD_EXECUTE OWNER_EXECUTE
        GROUP_READ WORLD_READ OWNER_READ
        OWNER_WRITE
#    PATTERN "private" EXCLUDE
)
install(DIRECTORY ${PDAL_IO_DIR}/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/pdal/io"
    FILES_MATCHING PATTERN "*.hpp"
    PERMISSIONS
        GROUP_EXECUTE WORLD_EXECUTE OWNER_EXECUTE
        GROUP_READ WORLD_READ OWNER_READ
        OWNER_WRITE
#    PATTERN "private" EXCLUDE
)
install(DIRECTORY ${PDAL_FILTERS_DIR}/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/pdal/filters"
    FILES_MATCHING PATTERN "*.hpp"
    PERMISSIONS
        GROUP_EXECUTE WORLD_EXECUTE OWNER_EXECUTE
        GROUP_READ WORLD_READ OWNER_READ
        OWNER_WRITE
#    PATTERN "private" EXCLUDE
)

install(FILES ${DIMENSION_OUTFILE} ${pdal_features_hpp}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/pdal"
    PERMISSIONS
        GROUP_EXECUTE WORLD_EXECUTE OWNER_EXECUTE
        GROUP_READ WORLD_READ OWNER_READ
        OWNER_WRITE
)
install(FILES ${PDAL_CMAKE_DIR}/pluginmacros.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PDAL
    PERMISSIONS
        GROUP_EXECUTE WORLD_EXECUTE OWNER_EXECUTE
        GROUP_READ WORLD_READ OWNER_READ
        OWNER_WRITE
)

# PDG is added after upstream targets so its differential tests and host
# fallback can refer to the pinned `pdal` executable directly.
if (WITH_PDG)
    if (CMAKE_VERSION VERSION_LESS 3.28)
        message(FATAL_ERROR "PDG requires CMake 3.28 or newer")
    endif()

    if (GPUPDAL_ENABLE_CUDA)
        enable_language(CUDA)
        find_package(CUDAToolkit 12.4 REQUIRED)
        if (NOT PDG_CCCL_INCLUDE_DIR)
            include(cmake/pdg-cccl.cmake)
            pdg_find_cccl_include(PDG_CCCL_INCLUDE_DIR)
        endif()
        if (NOT PDG_CCCL_INCLUDE_DIR)
            message(FATAL_ERROR
                "GPUPDAL CUDA builds require CCCL headers (CUB, Thrust, and libcu++)")
        endif()
        message(STATUS "PDG CCCL include directory: ${PDG_CCCL_INCLUDE_DIR}")

        # CUDA 13 removed the legacy pre-Turing code-generation targets.
        # A CUDA 12.x builder therefore produces the legacy-compatible release
        # fatbin; newer toolchains produce a current-generation artifact.  The
        # special `all` value asks CMake for every real architecture supported
        # by the selected compiler and PTX for its newest architecture.
        if (CUDAToolkit_VERSION VERSION_LESS 13.0)
            set(_pdg_cuda_architectures_default "all")
            set(_pdg_cuda_required_sass "50-real")
            set(_pdg_cuda_architecture_policy
                "CUDA 12.x legacy-compatible release (all compiler targets)")
        else()
            set(_pdg_cuda_architectures_default "all")
            set(_pdg_cuda_required_sass "75-real")
            set(_pdg_cuda_architecture_policy
                "CUDA 13+ current release (all compiler targets)")
        endif()

        option(PDG_REQUIRE_PORTABLE_CUDA_ARCHITECTURES
            "Reject a reduced CUDA architecture override in release builds"
            ON)
        set(PDG_CUDA_ARCHITECTURES "${_pdg_cuda_architectures_default}"
            CACHE STRING
            "PDG CUDA fatbin targets; use all for every compiler-supported GPU")

        # Keep explicit cache overrides honest.  A plain numeric entry creates
        # both real and virtual code; an `all` build delegates the complete
        # real-code matrix and newest PTX image to CMake.  Developers may turn
        # the guard off deliberately for a small native-only iteration build.
        if (PDG_REQUIRE_PORTABLE_CUDA_ARCHITECTURES AND
                NOT PDG_CUDA_ARCHITECTURES STREQUAL "all")
            string(REGEX REPLACE "-real$" "" _pdg_cuda_required_number
                "${_pdg_cuda_required_sass}")
            list(FIND PDG_CUDA_ARCHITECTURES
                "${_pdg_cuda_required_sass}"
                _pdg_cuda_required_sass_index)
            list(FIND PDG_CUDA_ARCHITECTURES
                "${_pdg_cuda_required_number}"
                _pdg_cuda_required_plain_index)
            if (_pdg_cuda_required_sass_index EQUAL -1 AND
                    _pdg_cuda_required_plain_index EQUAL -1)
                message(FATAL_ERROR
                    "PDG_CUDA_ARCHITECTURES must retain "
                    "${_pdg_cuda_required_sass} for the "
                    "${_pdg_cuda_architecture_policy} policy; set "
                    "PDG_REQUIRE_PORTABLE_CUDA_ARCHITECTURES=OFF only for "
                    "an explicitly non-release developer build")
            endif()

            set(_pdg_cuda_has_virtual FALSE)
            foreach(_pdg_cuda_architecture IN LISTS PDG_CUDA_ARCHITECTURES)
                if (_pdg_cuda_architecture MATCHES "^[0-9]+(-virtual)?$")
                    set(_pdg_cuda_has_virtual TRUE)
                endif()
            endforeach()
            if (NOT _pdg_cuda_has_virtual)
                message(FATAL_ERROR
                    "PDG_CUDA_ARCHITECTURES must retain a PTX-generating "
                    "numeric or -virtual entry for forward compatibility; "
                    "set PDG_REQUIRE_PORTABLE_CUDA_ARCHITECTURES=OFF only "
                    "for an explicitly non-release developer build")
            endif()
        endif()
        message(STATUS "PDG CUDA architecture policy: "
            "${_pdg_cuda_architecture_policy}")
        message(STATUS "PDG CUDA architectures: ${PDG_CUDA_ARCHITECTURES}")
        message(STATUS "PDG portable-architecture guard: "
            "${PDG_REQUIRE_PORTABLE_CUDA_ARCHITECTURES}")
    endif()

    if (PDG_BUILD_TESTS)
        enable_testing()
        if (NOT TARGET GTest::gtest)
            include(${PDAL_CMAKE_DIR}/gtest.cmake)
        endif()
    endif()

    add_subdirectory(src)
    if (PDG_BUILD_BENCHMARKS)
        add_subdirectory(bench)
    endif()
    if (PDG_BUILD_TESTS)
        add_subdirectory(tests/unit)
        add_subdirectory(tests/differential)
    endif()
endif()

#
# CPACK
#
include (${PDAL_CMAKE_DIR}/cpack.cmake)

add_custom_target(dist COMMAND ${CMAKE_MAKE_PROGRAM} package_source)

if (WITH_PDG AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
        CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    add_custom_target(gpupdal_linux_bundle
        COMMAND ${CMAKE_COMMAND} -E env GPUPDAL_SKIP_BUILD=1
            ${PROJECT_SOURCE_DIR}/scripts/release/build_linux_bundle.sh
            ${PROJECT_BINARY_DIR}
            ${PROJECT_SOURCE_DIR}/dist
        DEPENDS pdg_cli pdg_engine pdg_frozen_time pdg_verify_helpers pdal
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Building the GPUPDAL Linux x86-64 native bundle"
        USES_TERMINAL)
endif()

if (WITH_PDG AND WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    find_program(POWERSHELL_EXECUTABLE powershell REQUIRED)
    set(GPUPDAL_SOURCE_REVISION "" CACHE STRING
        "Full Git revision recorded in a Windows release artifact")
    if (NOT GPUPDAL_SOURCE_REVISION)
        execute_process(
            COMMAND git -C ${PROJECT_SOURCE_DIR} rev-parse HEAD
            OUTPUT_VARIABLE GPUPDAL_DISCOVERED_SOURCE_REVISION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE GPUPDAL_GIT_REVISION_RESULT
            ERROR_QUIET)
        if (GPUPDAL_GIT_REVISION_RESULT EQUAL 0 AND
                GPUPDAL_DISCOVERED_SOURCE_REVISION MATCHES "^[0-9a-f]{40}$")
            set(GPUPDAL_SOURCE_REVISION
                "${GPUPDAL_DISCOVERED_SOURCE_REVISION}" CACHE STRING
                "Full Git revision recorded in a Windows release artifact"
                FORCE)
        endif()
    endif()
    if (GPUPDAL_SOURCE_REVISION MATCHES "^[0-9a-f]{40}$")
        add_custom_target(gpupdal_windows_bundle
            COMMAND ${POWERSHELL_EXECUTABLE} -NoProfile -ExecutionPolicy Bypass
                -File ${PROJECT_SOURCE_DIR}/scripts/release/build_windows_bundle.ps1
                -Version ${GPUPDAL_VERSION}
                -SourceRevision ${GPUPDAL_SOURCE_REVISION}
                -SourceRoot ${PROJECT_SOURCE_DIR}
                -BuildDirectory ${PROJECT_BINARY_DIR}
                -OutputDirectory ${PROJECT_SOURCE_DIR}/dist
            DEPENDS pdg_cli pdg_engine pdg_verify_helpers pdal
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMENT "Building the GPUPDAL Windows x64 native bundle"
            USES_TERMINAL)
    else()
        message(STATUS "GPUPDAL Windows bundle target disabled: configure "
            "with -DGPUPDAL_SOURCE_REVISION=<full-commit>")
    endif()
endif()

export(
    TARGETS
        ${PDAL_LIB_NAME}
    FILE
        "${PDAL_BINARY_DIR}/PDALTargets.cmake")

if (${PDAL_LIB_TYPE} STREQUAL "STATIC")
  export(
      TARGETS
          ${JSON_SCHEMA_LIB_NAME}
          ${PDAL_ARBITER_LIB_NAME}
          ${PDAL_H3_LIB_NAME}
          ${PDAL_KAZHDAN_LIB_NAME}
          ${PDAL_LAZPERF_LIB_NAME}
          ${PDAL_LEPCC_LIB_NAME}
      FILE
          "${PDAL_BINARY_DIR}/PDALTargets.cmake")
endif ()

install(
    EXPORT
        PDALTargets
    DESTINATION
        "${CMAKE_INSTALL_LIBDIR}/cmake/PDAL")
include(${PDAL_CMAKE_DIR}/config.cmake)



feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES)

# TODO: move under scripts/bash-completion ?
if (WITH_COMPLETION)
    if (IS_DIRECTORY ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/bash-completion/completions)
        install(
            FILES
                "${PROJECT_SOURCE_DIR}/scripts/bash-completion/pdal"
            DESTINATION
                "${CMAKE_INSTALL_DATAROOTDIR}/bash-completion/completions")
    elseif (IS_DIRECTORY /etc/bash_completion.d)
        install(
            FILES
                "${PROJECT_SOURCE_DIR}/scripts/bash-completion/pdal"
            DESTINATION
                "${CMAKE_INSTALL_SYSCONFDIR}/bash_completion.d")
    endif()
endif()
