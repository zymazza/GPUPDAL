add_executable(pdg_placement_audit placement_model_audit.cpp)
target_compile_features(pdg_placement_audit PRIVATE cxx_std_20)
target_link_libraries(pdg_placement_audit PRIVATE PDG::core)
target_include_directories(pdg_placement_audit PRIVATE
    ${NLOHMANN_INCLUDE_DIR})

if(PDG_BUILD_TESTS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME pdg_placement_model_matrix_audit
        COMMAND pdg_placement_audit --calibration
            ${PROJECT_SOURCE_DIR}/test/data/pdg/placement-calibration-sm89.json)
    set_tests_properties(pdg_placement_model_matrix_audit
        PROPERTIES LABELS "pdg;placement;benchmark-audit")

    # D0281: the compiled direction/model audit is not provenance validation.
    # Independently require every case to name a report and verify every raw
    # report that is present in the checkout.  Large historical local reports
    # may be absent; newly retained reports under docs/reports are always
    # present and therefore hash-checked in CI.
    add_test(NAME pdg_placement_calibration_provenance
        COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/verify_placement_calibration.py
            ${PROJECT_SOURCE_DIR}/test/data/pdg/placement-calibration-sm89.json
            --repo-root ${PROJECT_SOURCE_DIR}
            --require-retained)
    set_tests_properties(pdg_placement_calibration_provenance
        PROPERTIES LABELS "pdg;placement;benchmark-audit;provenance")
endif()

if (MSVC)
    target_compile_options(pdg_placement_audit PRIVATE /W4)
    if (PDG_WARNINGS_AS_ERRORS)
        target_compile_options(pdg_placement_audit PRIVATE /WX)
    endif()
else()
    target_compile_options(pdg_placement_audit PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    if (PDG_WARNINGS_AS_ERRORS)
        target_compile_options(pdg_placement_audit PRIVATE -Werror)
    endif()
endif()

add_executable(pdg_laz_chunk_compression_benchmark
    laz_chunk_compression_benchmark.cpp)
target_compile_features(pdg_laz_chunk_compression_benchmark PRIVATE cxx_std_20)
target_link_libraries(pdg_laz_chunk_compression_benchmark PRIVATE PDAL::PDAL)
target_include_directories(pdg_laz_chunk_compression_benchmark SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR})
target_include_directories(pdg_laz_chunk_compression_benchmark PRIVATE
    ${NLOHMANN_INCLUDE_DIR})

if (MSVC)
    target_compile_options(pdg_laz_chunk_compression_benchmark PRIVATE /W4)
    if (PDG_WARNINGS_AS_ERRORS)
        target_compile_options(pdg_laz_chunk_compression_benchmark PRIVATE /WX)
    endif()
else()
    target_compile_options(pdg_laz_chunk_compression_benchmark PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    if (PDG_WARNINGS_AS_ERRORS)
        target_compile_options(pdg_laz_chunk_compression_benchmark PRIVATE
            -Werror)
    endif()
endif()

add_executable(pdg_laz_chunk_decode_benchmark
    laz_chunk_decode_benchmark.cpp)
target_compile_features(pdg_laz_chunk_decode_benchmark PRIVATE cxx_std_20)
target_link_libraries(pdg_laz_chunk_decode_benchmark PRIVATE
    PDAL::PDAL ${PDAL_LAZPERF_LIB_NAME})
target_include_directories(pdg_laz_chunk_decode_benchmark SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR})
target_include_directories(pdg_laz_chunk_decode_benchmark PRIVATE
    ${NLOHMANN_INCLUDE_DIR})

if (MSVC)
    target_compile_options(pdg_laz_chunk_decode_benchmark PRIVATE /W4)
    if (PDG_WARNINGS_AS_ERRORS)
        target_compile_options(pdg_laz_chunk_decode_benchmark PRIVATE /WX)
    endif()
else()
    target_compile_options(pdg_laz_chunk_decode_benchmark PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    if (PDG_WARNINGS_AS_ERRORS)
        target_compile_options(pdg_laz_chunk_decode_benchmark PRIVATE
            -Werror)
    endif()
endif()

add_executable(pdg_r11_neighborhood_attribution
    r11_neighborhood_attribution.cpp)
target_compile_features(pdg_r11_neighborhood_attribution PRIVATE cxx_std_20)
target_link_libraries(pdg_r11_neighborhood_attribution PRIVATE
    PDAL::PDAL Threads::Threads ${CMAKE_DL_LIBS})
target_include_directories(pdg_r11_neighborhood_attribution SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR})
target_include_directories(pdg_r11_neighborhood_attribution PRIVATE
    ${NLOHMANN_INCLUDE_DIR})

if (MSVC)
    target_compile_options(pdg_r11_neighborhood_attribution PRIVATE /W4)
    if (PDG_WARNINGS_AS_ERRORS)
        target_compile_options(pdg_r11_neighborhood_attribution PRIVATE /WX)
    endif()
else()
    target_compile_options(pdg_r11_neighborhood_attribution PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    if (PDG_WARNINGS_AS_ERRORS)
        target_compile_options(pdg_r11_neighborhood_attribution PRIVATE
            -Werror)
    endif()
endif()

if (PDG_ENABLE_CUDA)
    add_executable(pdg_spatial_benchmark spatial_index_benchmark.cpp)
    target_compile_features(pdg_spatial_benchmark PRIVATE cxx_std_20)
    target_link_libraries(pdg_spatial_benchmark PRIVATE
        PDG::core CUDA::cudart)
    target_include_directories(pdg_spatial_benchmark PRIVATE
        ${NLOHMANN_INCLUDE_DIR})

    if (MSVC)
        target_compile_options(pdg_spatial_benchmark PRIVATE /W4)
        if (PDG_WARNINGS_AS_ERRORS)
            target_compile_options(pdg_spatial_benchmark PRIVATE /WX)
        endif()
    else()
        target_compile_options(pdg_spatial_benchmark PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        if (PDG_WARNINGS_AS_ERRORS)
            target_compile_options(pdg_spatial_benchmark PRIVATE -Werror)
        endif()
    endif()

    add_executable(pdg_pmf_reuse_benchmark pmf_reuse_benchmark.cpp)
    target_sources(pdg_pmf_reuse_benchmark PRIVATE
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgPmfFilter.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgResidentContext.cpp)
    target_compile_features(pdg_pmf_reuse_benchmark PRIVATE cxx_std_20)
    target_link_libraries(pdg_pmf_reuse_benchmark PRIVATE
        PDG::core
        PDAL::PDAL
        CUDA::cudart)
    target_include_directories(pdg_pmf_reuse_benchmark SYSTEM PRIVATE
        ${PROJECT_SOURCE_DIR})
    target_include_directories(pdg_pmf_reuse_benchmark PRIVATE
        ${NLOHMANN_INCLUDE_DIR})

    if (MSVC)
        target_compile_options(pdg_pmf_reuse_benchmark PRIVATE /W4)
        if (PDG_WARNINGS_AS_ERRORS)
            target_compile_options(pdg_pmf_reuse_benchmark PRIVATE /WX)
        endif()
    else()
        target_compile_options(pdg_pmf_reuse_benchmark PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        if (PDG_WARNINGS_AS_ERRORS)
            target_compile_options(pdg_pmf_reuse_benchmark PRIVATE -Werror)
        endif()
    endif()

    add_executable(pdg_pmf_tiled_benchmark pmf_tiled_benchmark.cpp)
    target_compile_features(pdg_pmf_tiled_benchmark PRIVATE cxx_std_20)
    target_link_libraries(pdg_pmf_tiled_benchmark PRIVATE
        PDG::core
        PDAL::PDAL
        CUDA::cudart)
    target_include_directories(pdg_pmf_tiled_benchmark SYSTEM PRIVATE
        ${PROJECT_SOURCE_DIR})
    target_include_directories(pdg_pmf_tiled_benchmark PRIVATE
        ${NLOHMANN_INCLUDE_DIR})

    if (MSVC)
        target_compile_options(pdg_pmf_tiled_benchmark PRIVATE /W4)
        if (PDG_WARNINGS_AS_ERRORS)
            target_compile_options(pdg_pmf_tiled_benchmark PRIVATE /WX)
        endif()
    else()
        target_compile_options(pdg_pmf_tiled_benchmark PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        if (PDG_WARNINGS_AS_ERRORS)
            target_compile_options(pdg_pmf_tiled_benchmark PRIVATE -Werror)
        endif()
    endif()
endif()
