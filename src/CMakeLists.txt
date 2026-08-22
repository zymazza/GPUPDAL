if(PDG_QUALIFY_AUTOMATIC_APPROXIMATECOPLANAR AND NOT GPUPDAL_ENABLE_CUDA)
    message(FATAL_ERROR
        "PDG_QUALIFY_AUTOMATIC_APPROXIMATECOPLANAR requires GPUPDAL_ENABLE_CUDA")
endif()

set(PDG_CORE_SOURCES
    core/compaction.cpp
    core/coordinate.cpp
    core/dimension.cpp
    core/ExecutionStats.cpp
    core/FastMode.cpp
    core/memory.cpp
    core/point_batch.cpp
    core/scheduler.cpp
    core/SyntheticCloud.cpp
    expr/Expression.cpp
    index/PinnedCovariance.cpp
    index/SpatialIndex.cpp
    index/RasterGrid.cpp
    index/SpatialTile.cpp
    io/las/Las.cpp
    io/las/LasChunkDecoder.cpp
    io/las/LasFerry.cpp
    io/las/LasPointIo.cpp
    io/las/LasTranslate.cpp
    plan/Hybrid.cpp
    plan/LocalProfile.cpp
    plan/Placement.cpp
    plan/PlacementCalibration.cpp
    plan/PlacementProfile.cpp
    plan/ResidentPipeline.cpp
    plan/RuntimePlacement.cpp
    plan/Pipeline.cpp
    stages/Assign.cpp
    stages/ColorMap.cpp
    stages/Crop.cpp
    stages/Csf.cpp
    stages/Elm.cpp
    stages/Ferry.cpp
    stages/Histogram.cpp
    stages/Information.cpp
    stages/LabelDuplicates.cpp
    stages/Locate.cpp
    stages/Morton.cpp
    stages/Ordering.cpp
    stages/Ordinal.cpp
    stages/Partition.cpp
    stages/Pmf.cpp
    stages/Range.cpp
    stages/Robust.cpp
    stages/Smrf.cpp
    stages/Skewness.cpp
    stages/Summary.cpp
    stages/Transformation.cpp)

if (GPUPDAL_ENABLE_CUDA)
    list(APPEND PDG_CORE_SOURCES
        core/allocator.cu
        core/CalibrationProbes.cu
        core/compaction.cu
        core/packed_point_batch.cu
        index/SpatialIndexKernels.cu
        index/SpatialTileCuda.cu
        io/las/LasKernels.cu
        io/las/LasTranslateKernels.cu
        stages/AssignKernels.cu
        stages/ColorMapKernels.cu
        stages/CsfKernels.cu
        stages/ElmKernels.cu
        stages/FerryKernels.cu
        stages/HistogramKernels.cu
        stages/InformationKernels.cu
        stages/LabelDuplicatesKernels.cu
        stages/LocateKernels.cu
        stages/MortonKernels.cu
        stages/NeighborhoodKernels.cu
        stages/OrderingKernels.cu
        stages/OrdinalKernels.cu
        stages/PartitionKernels.cu
        stages/PmfKernels.cu
        stages/RobustKernels.cu
        stages/SmrfKernels.cu
        stages/SummaryKernels.cu
        stages/TransformationKernels.cu)
else()
    list(APPEND PDG_CORE_SOURCES
        core/CalibrationProbesCudaStub.cpp
        core/compaction_stub.cpp
        stages/AssignCudaStub.cpp
        stages/ColorMapCudaStub.cpp
        stages/CsfCudaStub.cpp
        stages/ElmCudaStub.cpp
        core/cuda_stub.cpp
        index/SpatialIndexCudaStub.cpp
        index/SpatialTileCudaStub.cpp
        stages/FerryCudaStub.cpp
        stages/HistogramCudaStub.cpp
        stages/InformationCudaStub.cpp
        stages/LabelDuplicatesCudaStub.cpp
        stages/LocateCudaStub.cpp
        stages/MortonCudaStub.cpp
        stages/NeighborhoodCudaStub.cpp
        stages/OrderingCudaStub.cpp
        stages/OrdinalCudaStub.cpp
        stages/PartitionCudaStub.cpp
        stages/PmfCudaStub.cpp
        stages/RobustCudaStub.cpp
        stages/SmrfCudaStub.cpp
        stages/SummaryCudaStub.cpp
        stages/TransformationCudaStub.cpp)
endif()

# D0279: embed data/placement-profiles/*.json as shipped placement profiles.
file(GLOB PDG_SHIPPED_PROFILE_FILES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/data/placement-profiles/*.json)
# The profile text is read at configure time, so a content change (not only
# an added/removed file) must re-run configure to re-embed it.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${PDG_SHIPPED_PROFILE_FILES})
set(PDG_SHIPPED_PROFILES_CPP
    ${PROJECT_BINARY_DIR}/generated/pdg/ShippedProfilesData.cpp)
set(_pdg_shipped_body "")
set(_pdg_shipped_entries "")
set(_pdg_shipped_index 0)
foreach(_pdg_profile IN LISTS PDG_SHIPPED_PROFILE_FILES)
    get_filename_component(_pdg_profile_name ${_pdg_profile} NAME)
    file(READ ${_pdg_profile} _pdg_profile_text)
    file(SHA256 ${_pdg_profile} _pdg_profile_sha256)
    string(LENGTH "${_pdg_profile_text}" _pdg_profile_bytes)
    set(_pdg_profile_chunks "")
    set(_pdg_profile_chunk_count 0)
    set(_pdg_profile_offset 0)
    while(_pdg_profile_offset LESS _pdg_profile_bytes)
        math(EXPR _pdg_profile_remaining
            "${_pdg_profile_bytes} - ${_pdg_profile_offset}")
        if(_pdg_profile_remaining GREATER 8192)
            set(_pdg_profile_chunk_bytes 8192)
        else()
            set(_pdg_profile_chunk_bytes ${_pdg_profile_remaining})
        endif()
        string(SUBSTRING "${_pdg_profile_text}" ${_pdg_profile_offset}
            ${_pdg_profile_chunk_bytes} _pdg_profile_chunk)
        string(APPEND _pdg_profile_chunks
            "    R\"pdgprofile(${_pdg_profile_chunk})pdgprofile\",\n")
        math(EXPR _pdg_profile_offset
            "${_pdg_profile_offset} + ${_pdg_profile_chunk_bytes}")
        math(EXPR _pdg_profile_chunk_count
            "${_pdg_profile_chunk_count} + 1")
    endwhile()
    string(APPEND _pdg_shipped_body
        "const std::string Profile${_pdg_shipped_index} = []\n"
        "{\n"
        "    constexpr std::array<std::string_view, ${_pdg_profile_chunk_count}> Chunks{{\n"
        "${_pdg_profile_chunks}"
        "    }};\n"
        "    std::string text;\n"
        "    text.reserve(${_pdg_profile_bytes});\n"
        "    for (std::string_view chunk : Chunks)\n"
        "        text.append(chunk);\n"
        "    return text;\n"
        "}();\n")
    string(APPEND _pdg_shipped_entries
        "    {\"${_pdg_profile_name}\", \"${_pdg_profile_sha256}\", std::string_view(Profile${_pdg_shipped_index})},\n")
    math(EXPR _pdg_shipped_index "${_pdg_shipped_index} + 1")
endforeach()
file(WRITE ${PDG_SHIPPED_PROFILES_CPP}.in
"// Generated by src/CMakeLists.txt from data/placement-profiles/*.json.
#include <pdg/LocalProfile.hpp>

#include <array>
#include <span>
#include <string>
#include <string_view>

namespace pdg
{
namespace
{
${_pdg_shipped_body}
const std::array<ShippedProfileSource, ${_pdg_shipped_index}> Sources{{
${_pdg_shipped_entries}}};
} // unnamed namespace

std::span<const ShippedProfileSource> shippedProfileSources()
{
    return Sources;
}

} // namespace pdg
")
configure_file(${PDG_SHIPPED_PROFILES_CPP}.in ${PDG_SHIPPED_PROFILES_CPP} COPYONLY)
list(APPEND PDG_CORE_SOURCES ${PDG_SHIPPED_PROFILES_CPP})

add_library(pdg_core STATIC ${PDG_CORE_SOURCES})
add_library(PDG::core ALIAS pdg_core)
set_target_properties(pdg_core PROPERTIES
    OUTPUT_NAME pdg
    POSITION_INDEPENDENT_CODE ON)

target_compile_features(pdg_core PUBLIC cxx_std_20)
target_link_libraries(pdg_core PUBLIC Threads::Threads)
# B0150/D0212: LAZ decode reuses the lazperf already vendored and built by the
# fork, so no new third-party code enters the tree and AGENTS.md priority 5's
# LASzip prohibition is satisfied by construction.
target_link_libraries(pdg_core PRIVATE ${PDAL_LAZPERF_LIB_NAME})
# SYSTEM: lazperf's vendored headers trip the engine's -Werror -Wshadow
# -Wpedantic. Their warnings are not ours to fix and must not be relaxed
# for our own code.
target_include_directories(pdg_core SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/vendor)
target_include_directories(pdg_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${NLOHMANN_INCLUDE_DIR}
        ${PDAL_VENDOR_EIGEN_DIR})
target_compile_definitions(pdg_core
    PUBLIC
        PDG_HAS_CUDA=$<BOOL:${GPUPDAL_ENABLE_CUDA}>
        PDG_QUALIFY_AUTOMATIC_APPROXIMATECOPLANAR=$<BOOL:${PDG_QUALIFY_AUTOMATIC_APPROXIMATECOPLANAR}>
        GPUPDAL_VERSION="${GPUPDAL_VERSION}"
        PDG_ORACLE_COMMIT="${PDG_ORACLE_COMMIT}"
        PDG_ORACLE_VERSION="${PDG_ORACLE_VERSION}")

if (MSVC)
    target_compile_definitions(pdg_core PRIVATE
        NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    target_compile_options(pdg_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4;/EHsc;/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,/EHsc,/Zc:preprocessor>)
    if (PDG_WARNINGS_AS_ERRORS)
        target_compile_options(pdg_core PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/WX>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>)
    endif()
else()
    target_compile_options(pdg_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>
        # nvcc's generated host stubs use non-standard line directives, so
        # -Wpedantic cannot be promoted to an error for CUDA translation units.
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow>)
    if (PDG_WARNINGS_AS_ERRORS)
        target_compile_options(pdg_core PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Werror>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
    endif()
endif()

if (GPUPDAL_ENABLE_CUDA)
    target_include_directories(pdg_core SYSTEM PRIVATE
        ${PDG_CCCL_INCLUDE_DIR})
    # Exact mode must not depend on whether nvcc finds a contraction opportunity
    # in a new translation unit or on an architecture's denormal policy.
    target_compile_options(pdg_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
        $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>
        $<$<COMPILE_LANGUAGE:CUDA>:--prec-div=true>
        $<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=true>)
    if (MSVC)
        set_source_files_properties(index/SpatialIndexKernels.cu PROPERTIES
            COMPILE_OPTIONS "--expt-relaxed-constexpr")
    else()
        set_source_files_properties(index/SpatialIndexKernels.cu PROPERTIES
            COMPILE_OPTIONS
                "--expt-relaxed-constexpr;-Xcompiler=-Wno-unknown-pragmas")
    endif()
    set_source_files_properties(stages/NeighborhoodKernels.cu PROPERTIES
        COMPILE_OPTIONS "--expt-relaxed-constexpr")
    set_target_properties(pdg_core PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${PDG_CUDA_ARCHITECTURES}")
    target_link_libraries(pdg_core PUBLIC CUDA::cudart CUDA::nvtx3
        CUDA::cuda_driver ${CMAKE_DL_LIBS})
# B0155: NVRTC is resolved with dlopen at first fused-JIT compilation instead of
# being a DT_NEEDED dependency, keeping its 5.4 ms load off every invocation
# that never compiles a kernel.
endif()

add_library(pdg_dispatch STATIC
    cli/Dispatch.cpp)
target_compile_features(pdg_dispatch PUBLIC cxx_std_20)
target_include_directories(pdg_dispatch PRIVATE ${NLOHMANN_INCLUDE_DIR})
if (MSVC)
    target_compile_definitions(pdg_dispatch PRIVATE
        NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    target_compile_options(pdg_dispatch PRIVATE /EHsc /Zc:preprocessor)
endif()

add_executable(pdg_engine
    cli/Calibrate.cpp
    cli/HybridPipeline.cpp
    cli/ResidentPipeline.cpp
    cli/main.cpp
    pdal/PdgApproximateCoplanarFilter.cpp
    pdal/PdgColorinterpFilter.cpp
    pdal/PdgCsfFilter.cpp
    pdal/PdgElmFilter.cpp
    pdal/PdgCovarianceFeaturesFilter.cpp
    pdal/PdgDividerFilter.cpp
    pdal/PdgEigenvaluesFilter.cpp
    pdal/PdgEstimateRankFilter.cpp
    pdal/PdgGroupByFilter.cpp
    pdal/PdgHagDelaunayFilter.cpp
    pdal/PdgHagNnFilter.cpp
    pdal/PdgInfoFilter.cpp
    pdal/PdgLabelDuplicatesFilter.cpp
    pdal/PdgLocateFilter.cpp
    pdal/PdgLofFilter.cpp
    pdal/PdgMergeFilter.cpp
    pdal/PdgMortonOrderFilter.cpp
    pdal/PdgNNDistanceFilter.cpp
    pdal/PdgNeighborhood.cpp
    pdal/PdgNeighborClassifierFilter.cpp
    pdal/PdgNormalFilter.cpp
    pdal/PdgOptimalNeighborhoodFilter.cpp
    pdal/PdgOrderFilter.cpp
    pdal/PdgOutlierFilter.cpp
    pdal/PdgPointProgramFilter.cpp
    pdal/PdgRadialDensityFilter.cpp
    pdal/PdgRadiusAssignFilter.cpp
    pdal/PdgResidentBoundaryFilter.cpp
    pdal/PdgResidentContext.cpp
    pdal/PdgResidentLasSourceReader.cpp
    pdal/PdgPmfFilter.cpp
    pdal/PdgReturnsFilter.cpp
    pdal/PdgRobustFilter.cpp
    pdal/PdgSmrfFilter.cpp
    pdal/PdgSkewnessBalancingFilter.cpp
    pdal/PdgStatsFilter.cpp
    pdal/PdgExpressionStatsFilter.cpp
    ${PROJECT_SOURCE_DIR}/filters/private/DimRange.cpp
    ${PROJECT_SOURCE_DIR}/filters/private/expr/AssignParser.cpp
    ${PROJECT_SOURCE_DIR}/filters/private/expr/AssignStatement.cpp
    ${PROJECT_SOURCE_DIR}/filters/private/expr/BaseParser.cpp
    ${PROJECT_SOURCE_DIR}/filters/private/expr/ConditionalExpression.cpp
    ${PROJECT_SOURCE_DIR}/filters/private/expr/ConditionalParser.cpp
    ${PROJECT_SOURCE_DIR}/filters/private/expr/Expression.cpp
    ${PROJECT_SOURCE_DIR}/filters/private/expr/IdentExpression.cpp
    ${PROJECT_SOURCE_DIR}/filters/private/expr/Lexer.cpp
    ${PROJECT_SOURCE_DIR}/filters/private/expr/MathExpression.cpp
    ${PROJECT_SOURCE_DIR}/filters/private/expr/MathParser.cpp)
target_sources(pdg_engine PRIVATE pdal/PdgSplitterFilter.cpp)
set_target_properties(pdg_engine PROPERTIES OUTPUT_NAME pdg-engine)
target_compile_features(pdg_engine PRIVATE cxx_std_20)
target_link_libraries(pdg_engine PRIVATE PDG::core PDAL::PDAL GDAL::GDAL)
target_include_directories(pdg_engine PRIVATE ${NLOHMANN_INCLUDE_DIR})
target_include_directories(pdg_engine PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}/include
    ${PDAL_VENDOR_EIGEN_DIR})
if (MSVC)
    target_compile_definitions(pdg_engine PRIVATE
        NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    target_compile_options(pdg_engine PRIVATE /EHsc /Zc:preprocessor)
endif()

add_executable(pdg_cli cli/DispatchMain.cpp)
# Keep the internal target name stable for the existing build and differential
# infrastructure.  GPUPDAL is the public, PDAL-compatible command name.
set_target_properties(pdg_cli PROPERTIES OUTPUT_NAME gpupdal)
target_compile_features(pdg_cli PRIVATE cxx_std_20)
target_compile_definitions(pdg_cli PRIVATE
    PDG_HAS_CUDA=$<BOOL:${GPUPDAL_ENABLE_CUDA}>)
target_link_libraries(pdg_cli PRIVATE pdg_dispatch ${CMAKE_DL_LIBS})
if (MSVC)
    target_compile_definitions(pdg_cli PRIVATE
        NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    target_compile_options(pdg_cli PRIVATE /EHsc /Zc:preprocessor)
endif()
add_dependencies(pdg_cli pdg_engine)

if (NOT WIN32)
    add_library(pdg_frozen_time SHARED cli/FrozenTime.c)
    target_compile_definitions(pdg_frozen_time PRIVATE _GNU_SOURCE)
    target_compile_options(pdg_frozen_time PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(pdg_frozen_time PRIVATE ${CMAKE_DL_LIBS})
    add_dependencies(pdg_cli pdg_frozen_time)
endif()

set(PDG_VERIFY_HELPER "${PDAL_OUTPUT_BIN_DIR}/pdg-verify.py")
set(PDG_BENCHMARK_HELPER "${PDAL_OUTPUT_BIN_DIR}/pdg-benchmark-reference.py")
add_custom_command(
    OUTPUT "${PDG_VERIFY_HELPER}" "${PDG_BENCHMARK_HELPER}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PROJECT_SOURCE_DIR}/scripts/pdg/verify.py
        "${PDG_VERIFY_HELPER}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PROJECT_SOURCE_DIR}/scripts/pdg/benchmark_reference.py
        "${PDG_BENCHMARK_HELPER}"
    DEPENDS
        ${PROJECT_SOURCE_DIR}/scripts/pdg/verify.py
        ${PROJECT_SOURCE_DIR}/scripts/pdg/benchmark_reference.py
    COMMENT "Packaging GPUPDAL verify helpers beside the build-tree launcher")
add_custom_target(pdg_verify_helpers
    DEPENDS "${PDG_VERIFY_HELPER}" "${PDG_BENCHMARK_HELPER}")
add_dependencies(pdg_cli pdg_verify_helpers)

find_package(Python3 QUIET COMPONENTS Interpreter)
if (Python3_Interpreter_FOUND)
    add_custom_target(pdg_check_dimensions
        COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/generate_dimensions.py --check
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Checking PDG dimension table against upstream PDAL")
endif()

install(TARGETS pdg_core pdg_cli pdg_engine
    EXPORT PDGTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if (NOT WIN32)
    install(TARGETS pdg_frozen_time
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/pdg)
endif()
install(PROGRAMS
    ${PROJECT_SOURCE_DIR}/scripts/pdg/verify.py
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/pdg
    RENAME pdg-verify.py)
install(PROGRAMS
    ${PROJECT_SOURCE_DIR}/scripts/pdg/benchmark_reference.py
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/pdg
    RENAME pdg-benchmark-reference.py)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/pdg
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
