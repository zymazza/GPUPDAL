# The math helpers are private libpdal symbols, so link the pinned upstream
# implementation directly into the test executable as the compatibility
# oracle.  Keep its language and code-generation settings aligned with PDAL.
add_library(pdg_pinned_pdal_math_oracle OBJECT
    ${PROJECT_SOURCE_DIR}/pdal/private/MathUtils.cpp)
pdal_lib_compile_settings(pdg_pinned_pdal_math_oracle)
set_target_properties(pdg_pinned_pdal_math_oracle PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(pdg_pinned_pdal_math_oracle PRIVATE PDAL::PDAL)
target_include_directories(pdg_pinned_pdal_math_oracle PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}/include
    ${PDAL_VENDOR_EIGEN_DIR}
    ${PROJECT_SOURCE_DIR}/vendor/nlohmann
    ${PROJECT_SOURCE_DIR}/vendor)

add_executable(pdg_unit_tests
    color_map_test.cpp
    compaction_test.cpp
    coordinate_test.cpp
    csf_test.cpp
    elm_test.cpp
    skewness_test.cpp
    pmf_test.cpp
    crop_test.cpp
    dispatcher_test.cpp
    dimension_test.cpp
    execution_stats_test.cpp
    expression_test.cpp
    ferry_test.cpp
    hybrid_test.cpp
    histogram_test.cpp
    information_test.cpp
    label_duplicates_test.cpp
    laz_chunk_compression_test.cpp
    laz_chunk_decode_test.cpp
    las_chunk_decoder_test.cpp
    las_ferry_test.cpp
    las_test.cpp
    locate_test.cpp
    memory_test.cpp
    morton_test.cpp
    neighborhood_test.cpp
    ordering_test.cpp
    ordinal_test.cpp
    kd3_concurrency_test.cpp
    partition_test.cpp
    plan_test.cpp
    predicate_residency_test.cpp
    point_batch_test.cpp
    raster_grid_test.cpp
    resident_pipeline_test.cpp
    runtime_placement_test.cpp
    spatial_tile_test.cpp
    spatial_index_test.cpp
    summary_test.cpp)
target_sources(pdg_unit_tests PRIVATE
    $<TARGET_OBJECTS:pdg_pinned_pdal_math_oracle>)
target_sources(pdg_unit_tests PRIVATE transformation_test.cpp)
target_sources(pdg_unit_tests PRIVATE robust_test.cpp)
target_sources(pdg_unit_tests PRIVATE scheduler_test.cpp)
target_sources(pdg_unit_tests PRIVATE placement_test.cpp)
target_sources(pdg_unit_tests PRIVATE smrf_test.cpp)
# D0268: las::Summary is a private (unexported) writer helper; the merge
# test compiles it directly.
target_sources(pdg_unit_tests PRIVATE las_summary_merge_test.cpp
    ${PROJECT_SOURCE_DIR}/io/private/las/Summary.cpp)
target_sources(pdg_unit_tests PRIVATE las_reader_unpack_test.cpp)
target_sources(pdg_unit_tests PRIVATE fast_mode_test.cpp)
target_sources(pdg_unit_tests PRIVATE local_profile_test.cpp)
target_compile_features(pdg_unit_tests PRIVATE cxx_std_20)
target_link_libraries(pdg_unit_tests PRIVATE
    PDG::core PDAL::PDAL pdg_dispatch
    GTest::gtest_main)
target_include_directories(pdg_unit_tests PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}/include
    ${PDAL_VENDOR_EIGEN_DIR}
    ${PROJECT_SOURCE_DIR}/vendor/nlohmann
    ${PROJECT_SOURCE_DIR}/vendor)
target_compile_definitions(pdg_unit_tests PRIVATE
    PDG_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/test/data")

if(GPUPDAL_ENABLE_CUDA)
    target_compile_definitions(pdg_unit_tests PRIVATE PDG_UNIT_TESTING=1)
    target_sources(pdg_unit_tests PRIVATE
        approximatecoplanar_pdal_test.cpp
        estimaterank_pdal_test.cpp
        hag_delaunay_pdal_test.cpp
        hag_nn_pdal_test.cpp
        nndistance_pdal_test.cpp
        lof_pdal_test.cpp
        label_duplicates_pdal_test.cpp
        neighborclassifier_pdal_test.cpp
        pmf_pdal_test.cpp
        csf_pdal_test.cpp
        elm_pdal_test.cpp
        radiusassign_pdal_test.cpp
        smrf_pdal_test.cpp
        optimalneighborhood_pdal_test.cpp
        skewness_pdal_test.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgApproximateCoplanarFilter.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgEstimateRankFilter.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgHagDelaunayFilter.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgHagNnFilter.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgResidentBoundaryFilter.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgNNDistanceFilter.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgOutlierFilter.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgOptimalNeighborhoodFilter.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgNeighborClassifierFilter.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgPmfFilter.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgCsfFilter.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgElmFilter.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgSkewnessBalancingFilter.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgRadiusAssignFilter.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgSmrfFilter.cpp
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
        ${PROJECT_SOURCE_DIR}/filters/private/expr/MathParser.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgLofFilter.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgLabelDuplicatesFilter.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgNeighborhood.cpp
        color_map_cuda_test.cu
        compaction_cuda_test.cu
        expression_cuda_test.cu
        ferry_cuda_test.cu
        histogram_cuda_test.cu
        information_cuda_test.cu
        label_duplicates_cuda_test.cu
        las_cuda_test.cu
        locate_cuda_test.cu
        morton_cuda_test.cu
        ordering_cuda_test.cu
        ordinal_cuda_test.cu
        partition_cuda_test.cu
        plan_cuda_test.cu
        packed_point_batch_cuda_test.cu
        resident_context_cuda_test.cpp
        ${PROJECT_SOURCE_DIR}/src/pdal/PdgResidentContext.cpp
        robust_cuda_test.cu
        spatial_index_cuda_test.cu
        summary_cuda_test.cu
        transformation_cuda_test.cu)
    set_target_properties(pdg_unit_tests PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${PDG_CUDA_ARCHITECTURES}")
endif()

include(GoogleTest)
gtest_discover_tests(pdg_unit_tests
    DISCOVERY_TIMEOUT 30
    PROPERTIES LABELS "pdg;unit")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME pdg_dimension_generator_is_current
    COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/generate_dimensions.py --check)
set_tests_properties(pdg_dimension_generator_is_current
    PROPERTIES LABELS "pdg;unit")

add_test(NAME pdg_dispatcher_process_boundary
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dispatcher_process_test.py
        $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdg_engine> $<TARGET_FILE:pdal>)
set_tests_properties(pdg_dispatcher_process_boundary
    PROPERTIES LABELS "pdg;unit;dispatcher;process")
add_test(NAME pdg_resident_pipeline_process
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/resident_pipeline_process_test.py
        $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdal>
        ${PROJECT_SOURCE_DIR}/test/data/las/simple.las)
set_tests_properties(pdg_resident_pipeline_process
    PROPERTIES LABELS "pdg;unit;resident;process")

if(GPUPDAL_ENABLE_CUDA)
    foreach(validation IN ITEMS v8 v7 v4 v5 v6 v3 v2 v1 direct)
        add_test(NAME pdg_resident_placement_${validation}_gpu
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/resident_placement_gpu_test.py
                ${validation}
                $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdal>
                ${PROJECT_SOURCE_DIR}/test/data/las/simple.las)
        set_tests_properties(pdg_resident_placement_${validation}_gpu
            PROPERTIES
                LABELS "pdg;resident;process;differential;cuda;gpu;exact;${validation}"
                RUN_SERIAL TRUE
                SKIP_RETURN_CODE 77
                TIMEOUT 300)
    endforeach()
    # The v2/v3 gates run their shared-index oracles (host KD3 kNN over the
    # full 21.9M-point fixture; v3 adds upstream LOF's adjacency map), which
    # alone exceed the standard budget.
    set_tests_properties(pdg_resident_placement_v2_gpu PROPERTIES TIMEOUT 900)
    set_tests_properties(pdg_resident_placement_v3_gpu PROPERTIES TIMEOUT 3600)
    set_tests_properties(pdg_resident_placement_v6_gpu PROPERTIES TIMEOUT 3600)
    add_test(NAME pdg_resident_eigen_family_gpu
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/resident_eigen_family_gpu_test.py
            $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdal>)
    set_tests_properties(pdg_resident_eigen_family_gpu
        PROPERTIES
            LABELS "pdg;resident;process;differential;cuda;gpu;exact;eigen-family"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            TIMEOUT 3600)
    add_test(NAME pdg_resident_rank_optimal_gpu
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/resident_rank_optimal_gpu_test.py
            $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdal>)
    set_tests_properties(pdg_resident_rank_optimal_gpu
        PROPERTIES
            LABELS "pdg;resident;process;differential;cuda;gpu;exact;rank-optimal"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            TIMEOUT 3600)
    add_test(NAME pdg_resident_outlier_direct_gpu
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/resident_outlier_direct_gpu_test.py
            $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdal>
            ${PROJECT_SOURCE_DIR}/test/data/las/simple.las)
    set_tests_properties(pdg_resident_outlier_direct_gpu
        PROPERTIES
            LABELS "pdg;resident;process;differential;cuda;gpu;exact;outlier"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            # This matrix now includes standalone statistical/radius lanes,
            # the radius-outlier/radial-density composition, automatic
            # admission/refusal, format conversion, and source/publisher
            # failure controls. A serialized physical run is longer than the
            # historical 120-second bound even when every case is healthy.
            TIMEOUT 300)
    add_test(NAME pdg_resident_neighborclassifier_direct_gpu
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/resident_neighborclassifier_direct_gpu_test.py
            $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdal>
            ${PROJECT_SOURCE_DIR}/test/data/las/simple.las)
    set_tests_properties(pdg_resident_neighborclassifier_direct_gpu
        PROPERTIES
            LABELS "pdg;resident;process;differential;cuda;gpu;exact;neighborclassifier"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            TIMEOUT 120)
    add_test(NAME pdg_resident_neighborclassifier_automatic_gpu
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/resident_neighborclassifier_direct_gpu_test.py
            --automatic $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdal>)
    set_tests_properties(pdg_resident_neighborclassifier_automatic_gpu
        PROPERTIES
            LABELS "pdg;resident;process;differential;cuda;gpu;exact;neighborclassifier;automatic"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            TIMEOUT 600)
    add_test(NAME pdg_resident_skewness_direct_gpu
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/resident_skewness_direct_gpu_test.py
            $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdal>
            ${PROJECT_SOURCE_DIR}/test/data/las/simple.las)
    set_tests_properties(pdg_resident_skewness_direct_gpu
        PROPERTIES
            LABELS "pdg;resident;process;differential;cuda;gpu;exact;skewness"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            TIMEOUT 300)
    add_test(NAME pdg_resident_skewness_automatic_gpu
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/resident_skewness_direct_gpu_test.py
            --automatic $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdal>)
    set_tests_properties(pdg_resident_skewness_automatic_gpu
        PROPERTIES
            LABELS "pdg;resident;process;differential;cuda;gpu;exact;skewness;automatic"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            TIMEOUT 600)
    add_test(NAME pdg_resident_hag_nn_count1_automatic_gpu
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/resident_hag_nn_direct_gpu_test.py
            --automatic $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdal>)
    set_tests_properties(pdg_resident_hag_nn_count1_automatic_gpu
        PROPERTIES
            LABELS "pdg;resident;process;differential;cuda;gpu;exact;hag_nn;automatic"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            TIMEOUT 900)
    add_test(NAME pdg_resident_hag_delaunay_count3_automatic_gpu
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/resident_hag_delaunay_direct_gpu_test.py
            --automatic $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdal>)
    set_tests_properties(pdg_resident_hag_delaunay_count3_automatic_gpu
        PROPERTIES
            LABELS "pdg;resident;process;differential;cuda;gpu;exact;hag_delaunay;automatic"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            TIMEOUT 1200)
    add_test(NAME pdg_resident_sort_direct_gpu
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/resident_sort_direct_gpu_test.py
            $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdal>
            ${PROJECT_SOURCE_DIR}/test/data/las/simple.las)
    set_tests_properties(pdg_resident_sort_direct_gpu
        PROPERTIES
            LABELS "pdg;resident;process;differential;cuda;gpu;exact;sort"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            TIMEOUT 300)
    add_test(NAME pdg_resident_sort_automatic_gpu
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/resident_sort_direct_gpu_test.py
            --automatic $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdal>)
    set_tests_properties(pdg_resident_sort_automatic_gpu
        PROPERTIES
            LABELS "pdg;resident;process;differential;cuda;gpu;exact;sort;automatic"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            TIMEOUT 600)
    add_test(NAME pdg_resident_radiusassign_direct_gpu
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/resident_radiusassign_direct_gpu_test.py
            $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdal>)
    set_tests_properties(pdg_resident_radiusassign_direct_gpu
        PROPERTIES
            LABELS "pdg;resident;process;differential;cuda;gpu;exact;radiusassign"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            TIMEOUT 300)
    add_test(NAME pdg_resident_radialdensity_direct_gpu
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/resident_radialdensity_direct_gpu_test.py
            $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdal>
            ${PROJECT_SOURCE_DIR}/test/data/las/simple.las)
    set_tests_properties(pdg_resident_radialdensity_direct_gpu
        PROPERTIES
            LABELS "pdg;resident;process;differential;cuda;gpu;exact;radialdensity"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            TIMEOUT 120)
    add_test(NAME pdg_resident_extra_double_direct_gpu
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/resident_extra_double_direct_gpu_test.py
            $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdal>)
    set_tests_properties(pdg_resident_extra_double_direct_gpu
        PROPERTIES
            LABELS "pdg;resident;process;differential;cuda;gpu;exact;extra-bytes"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            TIMEOUT 120)
    add_test(NAME pdg_resident_coplanar_direct_gpu
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/resident_coplanar_direct_gpu_test.py
            $<TARGET_FILE:pdg_cli> $<TARGET_FILE:pdal>)
    set_tests_properties(pdg_resident_coplanar_direct_gpu
        PROPERTIES
            LABELS "pdg;resident;process;differential;cuda;gpu;exact;approximatecoplanar"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            TIMEOUT 300)
endif()
