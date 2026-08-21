find_package(Python3 REQUIRED COMPONENTS Interpreter)

# The frozen-time shim is intentionally preloaded by differential.py so that
# candidate and oracle metadata remain deterministic.  ASan requires its
# runtime to appear before every other preloaded library, however.  Pass its
# absolute path through the CTest environment and let differential.py add it
# through its existing --{candidate,oracle}-preload ordering before the shim.
set(PDG_DIFFERENTIAL_ASAN_RUNTIME "")
if(CMAKE_CXX_FLAGS MATCHES "(^| )-fsanitize=[^ ]*address" OR
   CMAKE_EXE_LINKER_FLAGS MATCHES "(^| )-fsanitize=[^ ]*address")
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libasan.so
        OUTPUT_VARIABLE PDG_DIFFERENTIAL_ASAN_RUNTIME
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PDG_DIFFERENTIAL_ASAN_RUNTIME_RESULT)
    if(NOT PDG_DIFFERENTIAL_ASAN_RUNTIME_RESULT EQUAL 0 OR
       NOT EXISTS "${PDG_DIFFERENTIAL_ASAN_RUNTIME}")
        message(FATAL_ERROR
            "ASan differential configuration could not locate libasan.so")
    endif()
endif()

add_test(NAME pdg_benchmark_runner_multi_output_contract
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_runner_test.py
        --runner ${PROJECT_SOURCE_DIR}/scripts/pdg/benchmark_translate.py)
set_tests_properties(pdg_benchmark_runner_multi_output_contract
    PROPERTIES LABELS "pdg;benchmark;exact;multi-view;infrastructure")

add_test(NAME pdg_reference_runner_environment_contract
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_runner_test.py
        --runner ${PROJECT_SOURCE_DIR}/scripts/pdg/benchmark_reference.py)
set_tests_properties(pdg_reference_runner_environment_contract
    PROPERTIES LABELS "pdg;benchmark;reference;exact;environment;infrastructure")

add_test(NAME pdg_reference_suite_manifest_contract
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_suite_test.py
        --runner ${PROJECT_SOURCE_DIR}/scripts/pdg/reference_suite.py
        --manifest ${PROJECT_SOURCE_DIR}/bench/pipelines/reference/manifest.json
        --repo-root ${PROJECT_SOURCE_DIR})
set_tests_properties(pdg_reference_suite_manifest_contract
    PROPERTIES LABELS "pdg;benchmark;reference;suite;manifest;aggregate;infrastructure")

add_test(NAME pdg_conformance_runner_contract
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/conformance_runner_test.py
        --runner ${PROJECT_SOURCE_DIR}/scripts/pdg/conformance.py
        --recipes ${PROJECT_SOURCE_DIR}/tests/conformance/bounded-recipes-v1.json
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py)
set_tests_properties(pdg_conformance_runner_contract
    PROPERTIES LABELS "pdg;conformance;differential;generated;infrastructure")

add_test(NAME pdg_copc_semantic_comparator_contract
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/copc_semantic_test.py
        --pdal $<TARGET_FILE:pdal>
        --comparator ${PROJECT_SOURCE_DIR}/scripts/pdg/copc_semantic.py)
set_tests_properties(pdg_copc_semantic_comparator_contract
    PROPERTIES LABELS "pdg;conformance;differential;copc;semantic;infrastructure"
        TIMEOUT 120)

add_test(NAME pdg_frozen_release_artifact_contract
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/artifact_manifest_test.py
        --tool ${PROJECT_SOURCE_DIR}/scripts/pdg/artifact_manifest.py)
set_tests_properties(pdg_frozen_release_artifact_contract
    PROPERTIES LABELS "pdg;benchmark;release;artifact;provenance;infrastructure")

add_test(NAME pdg_3dep_preregistration_contract
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/three_dep_study_test.py
        --tool ${PROJECT_SOURCE_DIR}/bench/3dep/study.py
        --preregistration ${PROJECT_SOURCE_DIR}/bench/3dep/preregistration-v1.json)
set_tests_properties(pdg_3dep_preregistration_contract
    PROPERTIES LABELS "pdg;benchmark;3dep;preregistration;statistics;infrastructure")

add_custom_target(pdg_differential_prerequisites
    DEPENDS pdal pdg_cli pdg_frozen_time)

function(pdg_add_native_las_test name fixture)
    add_test(NAME ${name}
        COMMAND ${CMAKE_COMMAND} -E env PDG_REQUIRE_NATIVE=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case ${name}
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            -- translate ${PROJECT_SOURCE_DIR}/test/data/las/${fixture} out.las)
    set_tests_properties(${name}
        PROPERTIES LABELS "differential;native;exact")
endfunction()

pdg_add_native_las_test(pdg_native_translate_format_0 hextest.las)
pdg_add_native_las_test(pdg_native_translate_format_1_nan gps-time-nan.las)
pdg_add_native_las_test(pdg_native_translate_format_3 simple.las)
pdg_add_native_las_test(pdg_native_translate_color 1.2-with-color.las)
pdg_add_native_las_test(pdg_native_translate_nonzero_offset sample_c.las)

function(pdg_add_native_ferry_test name pipeline)
    add_test(NAME ${name}
        COMMAND ${CMAKE_COMMAND} -E env PDG_REQUIRE_NATIVE=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case ${name}
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/${pipeline}
            -- pipeline pipeline.json)
    set_tests_properties(${name}
        PROPERTIES LABELS "differential;native;exact;ferry")
endfunction()

pdg_add_native_ferry_test(pdg_native_ferry_standard ferry-standard.json)
pdg_add_native_ferry_test(pdg_native_ferry_return_summary
    ferry-return-summary.json)

function(pdg_add_native_point_program_test name pipeline)
    add_test(NAME ${name}
        COMMAND ${CMAKE_COMMAND} -E env PDG_REQUIRE_NATIVE=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case ${name}
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/${pipeline}
            -- pipeline pipeline.json)
    set_tests_properties(${name}
        PROPERTIES LABELS "differential;native;exact;point-program")
endfunction()

pdg_add_native_point_program_test(pdg_native_assign_standard
    assign-standard.json)
pdg_add_native_point_program_test(pdg_native_assign_ferry_fused
    assign-ferry-fused.json)
pdg_add_native_point_program_test(pdg_native_assign_math assign-math.json)

add_test(NAME pdg_hybrid_assign_ferry_host_exact
    COMMAND ${CMAKE_COMMAND} -E env
        PDG_REQUIRE_HYBRID=1
        PDG_DISABLE_CUDA_HYBRID=1
        ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_hybrid_assign_ferry_host_exact
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-assign-ferry.json
        -- pipeline pipeline.json)
set_tests_properties(pdg_hybrid_assign_ferry_host_exact
    PROPERTIES LABELS "differential;hybrid;host;exact;point-program")

add_test(NAME pdg_hybrid_coordinate_host_exact
    COMMAND ${CMAKE_COMMAND} -E env
        PDG_REQUIRE_HYBRID=1
        PDG_DISABLE_CUDA_HYBRID=1
        ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_hybrid_coordinate_host_exact
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/sample_c.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-coordinate.json
        -- pipeline pipeline.json)
set_tests_properties(pdg_hybrid_coordinate_host_exact
    PROPERTIES LABELS "differential;hybrid;host;exact;coordinates")

add_test(NAME pdg_hybrid_expression_fused_host_exact
    COMMAND ${CMAKE_COMMAND} -E env
        PDG_REQUIRE_HYBRID=1
        PDG_DISABLE_CUDA_HYBRID=1
        ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_hybrid_expression_fused_host_exact
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-expression-fused.json
        -- pipeline pipeline.json)
set_tests_properties(pdg_hybrid_expression_fused_host_exact
    PROPERTIES LABELS "differential;hybrid;host;exact;expression;compaction")

add_test(NAME pdg_hybrid_range_fused_host_exact
    COMMAND ${CMAKE_COMMAND} -E env
        PDG_REQUIRE_HYBRID=1
        PDG_DISABLE_CUDA_HYBRID=1
        ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_hybrid_range_fused_host_exact
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-range-fused.json
        -- pipeline pipeline.json)
set_tests_properties(pdg_hybrid_range_fused_host_exact
    PROPERTIES LABELS "differential;hybrid;host;exact;range;compaction")

foreach(kind IN ITEMS 2d 3d_outside)
    if(kind STREQUAL "2d")
        set(pipeline hybrid-crop-2d.json)
    else()
        set(pipeline hybrid-crop-3d-outside.json)
    endif()
    add_test(NAME pdg_hybrid_crop_${kind}_host_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_REQUIRE_HYBRID=1
            PDG_DISABLE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_crop_${kind}_host_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/${pipeline}
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_crop_${kind}_host_exact
        PROPERTIES LABELS
            "differential;hybrid;host;exact;crop;compaction")
endforeach()

add_test(NAME pdg_hybrid_crop_same_srs_host_exact
    COMMAND ${CMAKE_COMMAND} -E env
        PDG_REQUIRE_HYBRID=1
        PDG_DISABLE_CUDA_HYBRID=1
        ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_hybrid_crop_same_srs_host_exact
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/interesting.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-crop-2d.json
        -- pipeline pipeline.json)
set_tests_properties(pdg_hybrid_crop_same_srs_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;crop;spatial-reference")

add_test(NAME pdg_hybrid_crop_empty_standard_host_exact
    COMMAND ${CMAKE_COMMAND} -E env
        PDG_REQUIRE_HYBRID=1
        PDG_DISABLE_CUDA_HYBRID=1
        ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_hybrid_crop_empty_standard_host_exact
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/no-points.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-crop-empty-standard.json
        -- pipeline pipeline.json)
set_tests_properties(pdg_hybrid_crop_empty_standard_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;crop;empty;standard-mode")

function(pdg_add_hybrid_ordinal_host_test name pipeline input mode)
    set(mode_environment)
    if(mode STREQUAL "stream")
        list(APPEND mode_environment PDG_REQUIRE_STREAMING_HYBRID=1)
    endif()
    add_test(NAME ${name}
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_DISABLE_CUDA_HYBRID=1
            ${mode_environment}
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case ${name}
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/${input}
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/${pipeline}
            -- pipeline pipeline.json)
    set_tests_properties(${name}
        PROPERTIES LABELS
            "differential;hybrid;host;exact;ordinal;compaction;${mode}-mode")
endfunction()

pdg_add_hybrid_ordinal_host_test(pdg_hybrid_ordinal_stream_host_exact
    hybrid-ordinal-stream.json simple.las stream)
pdg_add_hybrid_ordinal_host_test(
    pdg_hybrid_ordinal_stream_multibatch_host_exact
    hybrid-ordinal-stream.json 4_6.las stream)
pdg_add_hybrid_ordinal_host_test(pdg_hybrid_ordinal_standard_host_exact
    hybrid-ordinal-standard.json simple.las standard)
pdg_add_hybrid_ordinal_host_test(pdg_hybrid_ordinal_split_host_exact
    hybrid-ordinal-split.json simple.las stream)
pdg_add_hybrid_ordinal_host_test(pdg_hybrid_tail_warning_host_exact
    hybrid-tail-warning.json simple.las standard)
pdg_add_hybrid_ordinal_host_test(pdg_hybrid_tail_empty_warning_host_exact
    hybrid-tail-warning.json no-points.las standard)

function(pdg_add_hybrid_locate_host_test name pipeline input)
    add_test(NAME ${name}
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_DISABLE_CUDA_HYBRID=1
            PDG_CUDA_CHUNK_POINTS=17
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case ${name}
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/${input}
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/${pipeline}
            -- pipeline pipeline.json)
    set_tests_properties(${name}
        PROPERTIES LABELS
            "differential;hybrid;host;exact;locate;global-reduction")
endfunction()

pdg_add_hybrid_locate_host_test(pdg_hybrid_locate_max_host_exact
    hybrid-locate-max.json simple.las)
pdg_add_hybrid_locate_host_test(pdg_hybrid_locate_min_host_exact
    hybrid-locate-min.json 1.2-with-color.las)
pdg_add_hybrid_locate_host_test(pdg_hybrid_locate_chain_host_exact
    hybrid-locate-chain.json simple.las)
pdg_add_hybrid_locate_host_test(pdg_hybrid_locate_empty_host_exact
    hybrid-locate-max.json no-points.las)
pdg_add_hybrid_locate_host_test(pdg_hybrid_locate_invalid_kind_host_exact
    hybrid-locate-invalid-kind.json simple.las)

add_test(NAME pdg_ordinal_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/ordinal_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_ordinal_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;ordinal;matrix;fallback")

add_test(NAME pdg_locate_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/locate_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_locate_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;locate;matrix;fallback;global-reduction")

add_test(NAME pdg_transformation_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/transformation_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_transformation_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;transformation;matrix;fallback;stream-mode")

add_test(NAME pdg_robust_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/robust_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_robust_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;robust;iqr;mad;matrix;fallback;global-selection")

add_test(NAME pdg_ordering_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/ordering_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_ordering_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;ordering;sort;matrix;fallback;global-ordering")

add_test(NAME pdg_morton_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/morton_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_morton_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;ordering;morton;matrix;fallback;global-ordering")

add_test(NAME pdg_groupby_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/groupby_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_groupby_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;partition;groupby;matrix;fallback;multi-view")

add_test(NAME pdg_returns_merge_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/returns_merge_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_returns_merge_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;partition;returns;merge;matrix;fallback;multi-view")

add_test(NAME pdg_divider_splitter_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/divider_splitter_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_divider_splitter_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;partition;divider;splitter;matrix;fallback;multi-view")

add_test(NAME pdg_colorinterp_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/colorinterp_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_colorinterp_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;colorinterp;matrix;fallback;multi-view")

add_test(NAME pdg_randomize_composition_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/randomize_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_randomize_composition_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;randomize;ordering;matrix;fallback")

add_test(NAME pdg_stats_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/stats_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_stats_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;stats;metadata;matrix;fallback;multi-view")

add_test(NAME pdg_info_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/info_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_info_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;info;metadata;matrix;fallback;multi-view")

add_test(NAME pdg_r9_polygon_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/r9_polygon_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_r9_polygon_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;fallback;exact;crop;polygon;reference")

add_test(NAME pdg_r7_dsm_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/r7_dsm_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_r7_dsm_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;fallback;exact;returns;gdal;raster;reference")

add_test(NAME pdg_r10_decimation_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/r10_decimation_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_r10_decimation_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;fallback;exact;decimation;voxel;reference")

set(PDG_R11_PINNED_ORACLE_ARGS)
if (PDG_PINNED_ORACLE_EXECUTABLE)
    set(PDG_R11_PINNED_ORACLE_ARGS
        --pinned-oracle ${PDG_PINNED_ORACLE_EXECUTABLE})
endif()
add_test(NAME pdg_r11_classification_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/r11_classification_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        ${PDG_R11_PINNED_ORACLE_ARGS})
set_tests_properties(pdg_r11_classification_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;fallback;exact;classification;smrf;outlier;neighborclassifier;reference")

add_test(NAME pdg_r14_conversion_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/r14_conversion_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_r14_conversion_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;fallback;exact;conversion;compression;copc;reference")

add_test(NAME pdg_expressionstats_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/expressionstats_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_expressionstats_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;expressionstats;metadata;matrix;fallback;multi-view")

add_test(NAME pdg_outlier_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/outlier_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_outlier_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;outlier;neighborhood;spatial-index;matrix;fallback")

add_test(NAME pdg_radialdensity_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/radialdensity_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_radialdensity_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;radialdensity;neighborhood;spatial-index;radius;matrix;fallback")

add_test(NAME pdg_radiusassign_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/radiusassign_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_radiusassign_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;radiusassign;neighborhood;spatial-index;radius;matrix;fallback")

add_test(NAME pdg_label_duplicates_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/label_duplicates_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_label_duplicates_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;label_duplicates;ordering;matrix;fallback")

set(PDG_REPROJECTION_PINNED_ORACLE_ARGS)
if (PDG_PINNED_ORACLE_EXECUTABLE)
    set(PDG_REPROJECTION_PINNED_ORACLE_ARGS
        --pinned-oracle ${PDG_PINNED_ORACLE_EXECUTABLE})
endif()
add_test(NAME pdg_reprojection_process_matrix_pinned_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/reprojection_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        ${PDG_REPROJECTION_PINNED_ORACLE_ARGS})
set_tests_properties(pdg_reprojection_process_matrix_pinned_exact
    PROPERTIES LABELS
        "differential;host;exact;reprojection;pinned;matrix")

add_test(NAME pdg_sample_process_matrix_pinned_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/sample_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        ${PDG_REPROJECTION_PINNED_ORACLE_ARGS})
set_tests_properties(pdg_sample_process_matrix_pinned_exact
    PROPERTIES LABELS
        "differential;host;exact;sample;pinned;matrix")

# D0268: exact parallel LAS record packing (stream batch hook and standard
# blocks) against pinned PDAL, including the deferred-warning and thrown
# serial repeats, discard/where/skip declines, and the disable control.
add_test(NAME pdg_las_writer_pack_matrix_pinned_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/las_writer_pack_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        ${PDG_REPROJECTION_PINNED_ORACLE_ARGS})
set_tests_properties(pdg_las_writer_pack_matrix_pinned_exact
    PROPERTIES LABELS
        "differential;host;exact;las;writer;pinned;matrix;fallback")

# D0269: exact parallel LAS record unpacking (stream batch hook and standard
# per-tile runs) against pinned PDAL, including multi-tile fixtures, extra
# bytes, count/start, downstream skips, small files, and the disable control.
add_test(NAME pdg_las_reader_unpack_matrix_pinned_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/las_reader_unpack_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        ${PDG_REPROJECTION_PINNED_ORACLE_ARGS})
set_tests_properties(pdg_las_reader_unpack_matrix_pinned_exact
    PROPERTIES LABELS
        "differential;host;exact;las;reader;pinned;matrix")

# D0270: exact ordered parallel COPC tile decode (requests=1) against pinned
# PDAL: full/bounds/resolution/polygon/count queries, keep-alive
# backpressure, empty selections, the disable control, both modes.
add_test(NAME pdg_copc_reader_matrix_pinned_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/copc_reader_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        ${PDG_REPROJECTION_PINNED_ORACLE_ARGS})
set_tests_properties(pdg_copc_reader_matrix_pinned_exact
    PROPERTIES LABELS
        "differential;host;exact;copc;reader;pinned;matrix")

# D0273: exact banded parallel raster accumulation in writers.gdal against
# pinned PDAL: dynamic and fixed grids, all statistics, IDW power/radius,
# bin mode, window fill, percentiles (serial), skips, both modes, controls.
add_test(NAME pdg_gdal_writer_matrix_pinned_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/gdal_writer_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        ${PDG_REPROJECTION_PINNED_ORACLE_ARGS})
set_tests_properties(pdg_gdal_writer_matrix_pinned_exact
    PROPERTIES LABELS
        "differential;host;exact;gdal;raster;writer;pinned;matrix")

set(PDG_SMRF_PINNED_ORACLE_ARGS)
if (PDG_PINNED_ORACLE_EXECUTABLE)
    set(PDG_SMRF_PINNED_ORACLE_ARGS
        --pinned-oracle ${PDG_PINNED_ORACLE_EXECUTABLE})
endif()
add_test(NAME pdg_smrf_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/smrf_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        ${PDG_SMRF_PINNED_ORACLE_ARGS})
set_tests_properties(pdg_smrf_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;smrf;grid;matrix;fallback"
    RESOURCE_LOCKS "smrf_matrix")

add_test(NAME pdg_pmf_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/pmf_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_pmf_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;pmf;grid;matrix;fallback"
    RESOURCE_LOCKS "pmf_matrix")

add_test(NAME pdg_csf_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/csf_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_csf_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;csf;grid;matrix;fallback"
    RESOURCE_LOCKS "csf_matrix")

add_test(NAME pdg_skewnessbalancing_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/skewness_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_skewnessbalancing_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;skewnessbalancing;global-ordering;matrix;fallback"
    RESOURCE_LOCKS "skewness_matrix")

add_test(NAME pdg_elm_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/elm_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_elm_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;elm;grid;matrix;fallback"
    RESOURCE_LOCKS "elm_matrix")

add_test(NAME pdg_hag_nn_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/hag_nn_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_hag_nn_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;hag_nn;terrain;neighborhood;spatial-index;matrix;fallback"
    RESOURCE_LOCKS "hag_nn_matrix")

add_test(NAME pdg_hag_delaunay_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/hag_delaunay_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_hag_delaunay_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;hag_delaunay;terrain;neighborhood;spatial-index;matrix;fallback"
    RESOURCE_LOCKS "hag_delaunay_matrix")

add_test(NAME pdg_nndistance_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/nndistance_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_nndistance_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;nndistance;neighborhood;spatial-index;matrix;fallback")

add_test(NAME pdg_normal_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/normal_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_normal_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;normal;neighborhood;spatial-index;eigen;matrix;fallback")

add_test(NAME pdg_approximatecoplanar_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/approximatecoplanar_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_approximatecoplanar_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;approximatecoplanar;neighborhood;spatial-index;eigen;uint8;matrix;fallback")

add_test(NAME pdg_eigenvalues_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/eigenvalues_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_eigenvalues_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;eigenvalues;neighborhood;spatial-index;eigen;matrix;fallback")

add_test(NAME pdg_covariancefeatures_process_matrix_host_exact
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/covariancefeatures_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_covariancefeatures_process_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;covariancefeatures;neighborhood;spatial-index;eigen;matrix;fallback")

add_test(NAME pdg_hybrid_neighborhood_resident_host_exact
    COMMAND ${CMAKE_COMMAND} -E env
        PDG_DISABLE_NATIVE=1
        PDG_REQUIRE_HYBRID=1
        PDG_DISABLE_CUDA_HYBRID=1
        ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_hybrid_neighborhood_resident_host_exact
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-neighborhood-resident.json
        -- pipeline pipeline.json)
set_tests_properties(pdg_hybrid_neighborhood_resident_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;normal;eigenvalues;covariancefeatures;neighborhood;resident;reuse")

add_test(NAME pdg_hybrid_approximatecoplanar_resident_host_exact
    COMMAND ${CMAKE_COMMAND} -E env
        PDG_DISABLE_NATIVE=1
        PDG_REQUIRE_HYBRID=1
        PDG_DISABLE_CUDA_HYBRID=1
        ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_hybrid_approximatecoplanar_resident_host_exact
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-approximatecoplanar-resident.json
        -- pipeline pipeline.json)
set_tests_properties(pdg_hybrid_approximatecoplanar_resident_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;approximatecoplanar;normal;eigenvalues;point-program;neighborhood;resident;uint8")

add_test(NAME pdg_hybrid_nndistance_resident_host_exact
    COMMAND ${CMAKE_COMMAND} -E env
        PDG_DISABLE_NATIVE=1
        PDG_REQUIRE_HYBRID=1
        PDG_DISABLE_CUDA_HYBRID=1
        ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_hybrid_nndistance_resident_host_exact
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-nndistance-resident.json
        -- pipeline pipeline.json)
set_tests_properties(pdg_hybrid_nndistance_resident_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;nndistance;normal;point-program;neighborhood;resident")

add_test(NAME pdg_hybrid_neighborhood_two_bridges_host_exact
    COMMAND ${CMAKE_COMMAND} -E env
        PDG_DISABLE_NATIVE=1
        PDG_REQUIRE_HYBRID=1
        PDG_DISABLE_CUDA_HYBRID=1
        ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_hybrid_neighborhood_two_bridges_host_exact
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-neighborhood-two-bridges.json
        -- pipeline pipeline.json)
set_tests_properties(pdg_hybrid_neighborhood_two_bridges_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;nndistance;normal;eigenvalues;point-program;neighborhood;resident;bridge")

add_test(NAME pdg_hybrid_neighborhood_custom_ferry_layout_fallback_host_exact
    COMMAND ${CMAKE_COMMAND} -E env
        PDG_DISABLE_NATIVE=1
        PDG_EXPERIMENTAL_CUDA_HYBRID=1
        PDG_DISABLE_CUDA_HYBRID=1
        ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_hybrid_neighborhood_custom_ferry_layout_fallback_host_exact
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-neighborhood-custom-ferry-layout.json
        -- pipeline pipeline.json)
set_tests_properties(
    pdg_hybrid_neighborhood_custom_ferry_layout_fallback_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;normal;eigenvalues;point-program;neighborhood;bridge;ferry;custom-dimension;layout-fallback")

add_test(NAME pdg_hybrid_neighborhood_custom_ferry_layout_guard_delegates
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/hybrid_layout_fallback_test.py
        $<TARGET_FILE:pdg_cli>
        ${PROJECT_SOURCE_DIR}/test/data/las/simple.las
        ${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-neighborhood-custom-ferry-layout.json)
set_tests_properties(
    pdg_hybrid_neighborhood_custom_ferry_layout_guard_delegates
    PROPERTIES LABELS
        "differential;hybrid;host;normal;eigenvalues;point-program;neighborhood;bridge;ferry;custom-dimension;layout-fallback;delegation")

add_test(NAME pdg_hybrid_colorinterp_stream_host_exact
    COMMAND ${CMAKE_COMMAND} -E env
        PDG_DISABLE_NATIVE=1
        PDG_REQUIRE_HYBRID=1
        PDG_REQUIRE_STREAMING_HYBRID=1
        PDG_DISABLE_CUDA_HYBRID=1
        ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_hybrid_colorinterp_stream_host_exact
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-colorinterp-stream.json
        -- pipeline pipeline.json)
set_tests_properties(pdg_hybrid_colorinterp_stream_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;colorinterp;stream-mode")

add_test(NAME pdg_hybrid_two_regions_host_exact
    COMMAND ${CMAKE_COMMAND} -E env
        PDG_REQUIRE_HYBRID=1
        PDG_DISABLE_CUDA_HYBRID=1
        ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_hybrid_two_regions_host_exact
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-two-regions.json
        -- pipeline pipeline.json)
set_tests_properties(pdg_hybrid_two_regions_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;mixed-pipeline;compaction")

add_test(NAME pdg_hybrid_reader_matrix_host_exact
    COMMAND ${CMAKE_COMMAND} -E env
        PDG_REQUIRE_HYBRID=1
        PDG_DISABLE_CUDA_HYBRID=1
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/hybrid_reader_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --case-prefix pdg-hybrid-reader-host)
set_tests_properties(pdg_hybrid_reader_matrix_host_exact
    PROPERTIES LABELS "differential;hybrid;host;exact;reader-matrix")

add_test(NAME pdg_hybrid_stream_mode_reader_matrix_host_exact
    COMMAND ${CMAKE_COMMAND} -E env
        PDG_REQUIRE_HYBRID=1
        PDG_REQUIRE_STREAMING_HYBRID=1
        PDG_DISABLE_CUDA_HYBRID=1
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/hybrid_reader_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --fixture-root ${PROJECT_SOURCE_DIR}/test/data
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --case-prefix pdg-hybrid-stream-reader-host
        --original-streamable)
set_tests_properties(pdg_hybrid_stream_mode_reader_matrix_host_exact
    PROPERTIES LABELS
        "differential;hybrid;host;exact;reader-matrix;stream-mode")

if(PDG_ENABLE_CUDA AND PDG_QUALIFY_AUTOMATIC_APPROXIMATECOPLANAR)
    add_test(NAME pdg_automatic_approximatecoplanar_cuda_exact
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/automatic_approximatecoplanar.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --fixture-root ${PROJECT_SOURCE_DIR}/test/data
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
    set_tests_properties(pdg_automatic_approximatecoplanar_cuda_exact
        PROPERTIES
            LABELS
                "differential;hybrid;cuda;exact;automatic;default-mode;approximatecoplanar;metadata;threshold;bounded"
            RUN_SERIAL TRUE
            TIMEOUT 600)
endif()

if(PDG_ENABLE_CUDA)
    # D0277: `gpupdal calibrate` and the local placement profile.
    set(PDG_CALIBRATE_PINNED_ORACLE_ARGS)
    if (PDG_PINNED_ORACLE_EXECUTABLE)
        set(PDG_CALIBRATE_PINNED_ORACLE_ARGS
            --pinned-oracle ${PDG_PINNED_ORACLE_EXECUTABLE})
    endif()
    add_test(NAME pdg_calibrate_local_profile_cuda_exact
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/calibrate_test.py
            --candidate $<TARGET_FILE:pdg_cli>
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --make-shipped ${PROJECT_SOURCE_DIR}/bench/report/make_shipped_profile.py
            ${PDG_CALIBRATE_PINNED_ORACLE_ARGS})
    set_tests_properties(pdg_calibrate_local_profile_cuda_exact
        PROPERTIES
            LABELS "differential;cuda;exact;calibrate;placement;automatic;default-mode"
            RUN_SERIAL TRUE
            TIMEOUT 2400)
endif()

if(PDG_ENABLE_CUDA)
    # D0271: the `--fast` tie-order contract on the CUDA hybrid route.
    add_test(NAME pdg_fast_tie_contract_cuda
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/fast_tie_contract_test.py
            --candidate $<TARGET_FILE:pdg_cli>
            --reference-runner ${PROJECT_SOURCE_DIR}/scripts/pdg/benchmark_reference.py
            --points ${PROJECT_SOURCE_DIR}/test/data/pdg/neighborhood-points.csv
            --pipeline ${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-neighborhood-deterministic-ties.json
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential)
    set_tests_properties(pdg_fast_tie_contract_cuda
        PROPERTIES LABELS
            "differential;hybrid;cuda;fast;tie-repair;neighborhood;deterministic-ties")
endif()

if(PDG_ENABLE_CUDA)
    add_test(NAME pdg_automatic_r2_ground_normalize_cuda_exact
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/automatic_r2_ground_normalize.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --fixture ${PROJECT_SOURCE_DIR}/build/bench-data/reference/ref-1m.laz
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --cuda-toolkit ${CMAKE_CUDA_COMPILER_VERSION})
    set_tests_properties(pdg_automatic_r2_ground_normalize_cuda_exact
        PROPERTIES
            LABELS
                "differential;hybrid;cuda;gpu;exact;automatic;default-mode;r2;smrf;hag-nn;selective-repair;bounded"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            TIMEOUT 600)
endif()

if(PDG_ENABLE_CUDA)
    add_test(NAME pdg_automatic_r4_outlier_cuda_exact
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/automatic_r4_outlier.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --fixture ${PROJECT_SOURCE_DIR}/build/bench-data/reference/ref-1m.laz
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --cuda-toolkit ${CMAKE_CUDA_COMPILER_VERSION})
    set_tests_properties(pdg_automatic_r4_outlier_cuda_exact
        PROPERTIES
            LABELS
                "differential;hybrid;cuda;gpu;exact;automatic;default-mode;outlier;range;sample;bounded"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            TIMEOUT 600)
endif()

if(PDG_ENABLE_CUDA)
    add_test(NAME pdg_automatic_label_nndistance_cuda_exact
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/automatic_label_nndistance.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --cuda-toolkit ${CMAKE_CUDA_COMPILER_VERSION})
    set_tests_properties(pdg_automatic_label_nndistance_cuda_exact
        PROPERTIES
            LABELS
                "differential;hybrid;cuda;gpu;exact;automatic;default-mode;label_duplicates;nndistance;point-program;bounded"
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
            TIMEOUT 900)

    add_test(NAME pdg_label_duplicates_process_matrix_cuda_exact
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/label_duplicates_matrix.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --fixture-root ${PROJECT_SOURCE_DIR}/test/data
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --require-cuda)
    set_tests_properties(pdg_label_duplicates_process_matrix_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;label_duplicates;ordering;matrix")

    add_test(NAME pdg_radiusassign_process_matrix_cuda_exact
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/radiusassign_matrix.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --fixture-root ${PROJECT_SOURCE_DIR}/test/data
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --require-cuda)
    set_tests_properties(pdg_radiusassign_process_matrix_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;radiusassign;neighborhood;spatial-index;radius;matrix")

    add_test(NAME pdg_pmf_process_matrix_cuda_exact
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/pmf_matrix.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --require-cuda)
    set_tests_properties(pdg_pmf_process_matrix_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;pmf;grid;matrix"
        RESOURCE_LOCKS "pmf_matrix")

    add_test(NAME pdg_csf_process_matrix_cuda_exact
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/csf_matrix.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --require-cuda)
    set_tests_properties(pdg_csf_process_matrix_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;csf;grid;matrix"
        RESOURCE_LOCKS "csf_matrix")

    add_test(NAME pdg_skewnessbalancing_process_matrix_cuda_exact
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/skewness_matrix.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --require-cuda)
    set_tests_properties(pdg_skewnessbalancing_process_matrix_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;skewnessbalancing;global-ordering;matrix"
        RESOURCE_LOCKS "skewness_matrix")

    add_test(NAME pdg_elm_process_matrix_cuda_exact
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/elm_matrix.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --require-cuda)
    set_tests_properties(pdg_elm_process_matrix_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;elm;grid;matrix"
        RESOURCE_LOCKS "elm_matrix")

    # B0179: exactness coverage for the shapes generalized automatic selection
    # would newly admit (B0178). Every case runs the resident executor -- the
    # one such a generalization would choose -- and requires byte-identical
    # output against pinned PDAL, so the proof burden is paid whether or not
    # the selection gate is ever widened.
    add_test(NAME pdg_resident_selection_matrix_cuda_exact
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/resident_selection_matrix.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
    set_tests_properties(pdg_resident_selection_matrix_cuda_exact
        PROPERTIES LABELS
            "differential;resident;cuda;exact;selection;neighborhood;matrix"
        SKIP_RETURN_CODE 77
        RESOURCE_LOCKS "resident_selection_matrix")

    add_test(NAME pdg_hag_nn_process_matrix_cuda_exact
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/hag_nn_matrix.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --require-cuda)
    set_tests_properties(pdg_hag_nn_process_matrix_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;hag_nn;terrain;neighborhood;spatial-index;matrix"
        RESOURCE_LOCKS "hag_nn_matrix")

    add_test(NAME pdg_hag_delaunay_process_matrix_cuda_exact
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/hag_delaunay_matrix.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --require-cuda)
    set_tests_properties(pdg_hag_delaunay_process_matrix_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;hag_delaunay;terrain;neighborhood;spatial-index;matrix"
        RESOURCE_LOCKS "hag_delaunay_matrix")

    add_test(NAME pdg_expressionstats_process_matrix_cuda_exact
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/expressionstats_matrix.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --fixture-root ${PROJECT_SOURCE_DIR}/test/data
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --require-cuda)
    set_tests_properties(pdg_expressionstats_process_matrix_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;expressionstats;metadata;matrix;multi-view;stream-mode;histogram")

    add_test(NAME pdg_hybrid_colorinterp_stream_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_STREAMING_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_colorinterp_stream_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-colorinterp-stream.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_colorinterp_stream_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;colorinterp;stream-mode")

    add_test(NAME pdg_hybrid_colorinterp_autorange_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_colorinterp_autorange_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-colorinterp-autorange.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_colorinterp_autorange_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;colorinterp;standard-mode;range")

    add_test(NAME pdg_hybrid_colorinterp_divider_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_colorinterp_divider_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-colorinterp-divider.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_colorinterp_divider_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;colorinterp;partition;multi-view;range")

    add_test(NAME pdg_cuda_assign_ferry_fused
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_REQUIRE_FUSED_CUDA_POINT_PROGRAM=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_cuda_assign_ferry_fused
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/assign-ferry-fused.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_cuda_assign_ferry_fused
        PROPERTIES LABELS
            "differential;cuda;exact;point-program;fusion;resident")

    foreach(kind IN ITEMS affine chain)
        if(kind STREQUAL "affine")
            set(pipeline hybrid-transformation-affine.json)
            set(input simple.las)
        else()
            set(pipeline hybrid-transformation-chain.json)
            set(input 4_6.las)
        endif()
        add_test(NAME pdg_hybrid_transformation_${kind}_cuda_exact
            COMMAND ${CMAKE_COMMAND} -E env
                PDG_DISABLE_NATIVE=1
                PDG_REQUIRE_HYBRID=1
                PDG_REQUIRE_STREAMING_HYBRID=1
                PDG_REQUIRE_CUDA_HYBRID=1
                PDG_CUDA_CHUNK_POINTS=17
                ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
                --oracle $<TARGET_FILE:pdal>
                --candidate $<TARGET_FILE:pdg_cli>
                --case pdg_hybrid_transformation_${kind}_cuda_exact
                --work-dir ${PROJECT_BINARY_DIR}/tests/differential
                --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
                --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/${input}
                --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/${pipeline}
                -- pipeline pipeline.json)
        set_tests_properties(pdg_hybrid_transformation_${kind}_cuda_exact
            PROPERTIES LABELS
                "differential;hybrid;cuda;exact;transformation;point-program;stream-mode")
    endforeach()

    foreach(kind IN ITEMS iqr mad chain)
        if(kind STREQUAL "iqr")
            set(pipeline hybrid-iqr.json)
        elseif(kind STREQUAL "mad")
            set(pipeline hybrid-mad.json)
        else()
            set(pipeline hybrid-robust-chain.json)
        endif()
        add_test(NAME pdg_hybrid_robust_${kind}_cuda_exact
            COMMAND ${CMAKE_COMMAND} -E env
                PDG_DISABLE_NATIVE=1
                PDG_REQUIRE_HYBRID=1
                PDG_REQUIRE_CUDA_HYBRID=1
                ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
                --oracle $<TARGET_FILE:pdal>
                --candidate $<TARGET_FILE:pdg_cli>
                --case pdg_hybrid_robust_${kind}_cuda_exact
                --work-dir ${PROJECT_BINARY_DIR}/tests/differential
                --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
                --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
                --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/${pipeline}
                -- pipeline pipeline.json)
        set_tests_properties(pdg_hybrid_robust_${kind}_cuda_exact
            PROPERTIES LABELS
                "differential;hybrid;cuda;exact;robust;global-selection;${kind}")
    endforeach()

    add_test(NAME pdg_hybrid_outlier_radius_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_outlier_radius_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.csv=${PROJECT_SOURCE_DIR}/test/data/pdg/neighborhood-points.csv
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-outlier-radius.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_outlier_radius_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;outlier;neighborhood;spatial-index;radius")

    add_test(NAME pdg_hybrid_outlier_radius_tiled_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            PDG_SPATIAL_TILE_EDGE=2
            PDG_REQUIRE_SPATIAL_TILING=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_outlier_radius_tiled_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.csv=${PROJECT_SOURCE_DIR}/test/data/pdg/neighborhood-points.csv
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-outlier-radius.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_outlier_radius_tiled_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;outlier;neighborhood;spatial-index;radius;tiling;ghost")

    add_test(NAME pdg_hybrid_radialdensity_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_radialdensity_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.csv=${PROJECT_SOURCE_DIR}/test/data/pdg/neighborhood-points.csv
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-radialdensity.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_radialdensity_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;radialdensity;neighborhood;spatial-index;radius")

    add_test(NAME pdg_hybrid_radialdensity_tiled_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            PDG_SPATIAL_TILE_EDGE=2
            PDG_REQUIRE_SPATIAL_TILING=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_radialdensity_tiled_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.csv=${PROJECT_SOURCE_DIR}/test/data/pdg/neighborhood-points.csv
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-radialdensity.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_radialdensity_tiled_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;radialdensity;neighborhood;spatial-index;radius;tiling;ghost")

    add_test(NAME pdg_hybrid_outlier_statistical_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_outlier_statistical_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-outlier-statistical.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_outlier_statistical_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;outlier;neighborhood;spatial-index;knn;statistical")

    add_test(NAME pdg_hybrid_normal_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_normal_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-normal.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_normal_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;normal;neighborhood;spatial-index;knn;eigen")

    add_test(NAME pdg_hybrid_eigenvalues_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_eigenvalues_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-eigenvalues.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_eigenvalues_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;eigenvalues;neighborhood;spatial-index;knn;eigen")

    add_test(NAME pdg_hybrid_nndistance_resident_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            PDG_REQUIRE_NEIGHBORHOOD_REUSE=1
            PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE=1
            PDG_FORCE_MORTON_BVH=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_nndistance_resident_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-nndistance-resident.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_nndistance_resident_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;nndistance;normal;point-program;neighborhood;spatial-index;knn;resident;reuse;column-reuse")

    add_test(NAME pdg_hybrid_neighborhood_two_bridges_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            PDG_REQUIRE_NEIGHBORHOOD_REUSE=1
            PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE=1
            PDG_FORCE_MORTON_BVH=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_neighborhood_two_bridges_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-neighborhood-two-bridges.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_neighborhood_two_bridges_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;nndistance;normal;eigenvalues;point-program;neighborhood;spatial-index;knn;resident;reuse;column-reuse;bridge")

    add_test(NAME pdg_hybrid_neighborhood_two_bridges_autzen_ties_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            PDG_REQUIRE_NEIGHBORHOOD_REUSE=1
            PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE=1
            PDG_FORCE_MORTON_BVH=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_neighborhood_two_bridges_autzen_ties_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/autzen_trim.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-neighborhood-two-bridges.json
            -- pipeline pipeline.json)
    set_tests_properties(
        pdg_hybrid_neighborhood_two_bridges_autzen_ties_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;nndistance;normal;eigenvalues;point-program;neighborhood;spatial-index;knn;resident;reuse;column-reuse;bridge;tie-repair;real-data")

    add_test(NAME pdg_hybrid_neighborhood_deterministic_ties_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            PDG_REQUIRE_NEIGHBORHOOD_REUSE=1
            PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE=1
            PDG_REQUIRE_NEIGHBORHOOD_TIE_REPAIR=1
            PDG_FORCE_MORTON_BVH=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_neighborhood_deterministic_ties_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.csv=${PROJECT_SOURCE_DIR}/test/data/pdg/neighborhood-points.csv
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-neighborhood-deterministic-ties.json
            -- pipeline pipeline.json)
    set_tests_properties(
        pdg_hybrid_neighborhood_deterministic_ties_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;normal;eigenvalues;point-program;neighborhood;spatial-index;knn;resident;reuse;column-reuse;bridge;tie-repair;deterministic-ties")

    add_test(NAME pdg_hybrid_neighborhood_varying_k_cache_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            PDG_REQUIRE_NEIGHBORHOOD_REUSE=1
            PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE=1
            PDG_REQUIRE_NEIGHBORHOOD_EIGENSYSTEM_REUSE=6
            PDG_FORCE_MORTON_BVH=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_neighborhood_varying_k_cache_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-neighborhood-varying-k-cache.json
            -- pipeline pipeline.json)
    set_tests_properties(
        pdg_hybrid_neighborhood_varying_k_cache_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;normal;eigenvalues;covariancefeatures;point-program;neighborhood;spatial-index;knn;resident;reuse;column-reuse;eigensystem-cache;varying-k;bridge")

    add_test(NAME pdg_hybrid_neighborhood_bridge_fallback_rebuild_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_EXPERIMENTAL_CUDA_HYBRID=1
            PDG_REQUIRE_NEIGHBORHOOD_BRIDGE_REBUILD=1
            PDG_FORCE_MORTON_BVH=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_neighborhood_bridge_fallback_rebuild_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-neighborhood-bridge-fallback-rebuild.json
            -- pipeline pipeline.json)
    set_tests_properties(
        pdg_hybrid_neighborhood_bridge_fallback_rebuild_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;normal;eigenvalues;point-program;neighborhood;spatial-index;knn;resident;bridge;fallback;rebuild;unsupported-expression;stale-state")

    add_test(NAME pdg_hybrid_neighborhood_increasing_k_fallback_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_EXPERIMENTAL_CUDA_HYBRID=1
            PDG_FORCE_MORTON_BVH=1
            PDG_REQUIRE_NONTERMINAL_NEIGHBORHOOD_COLUMN_REUSE=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_neighborhood_increasing_k_fallback_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.csv=${PROJECT_SOURCE_DIR}/test/data/pdg/neighborhood-points.csv
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-neighborhood-increasing-k-fallback.json
            -- pipeline pipeline.json)
    set_tests_properties(
        pdg_hybrid_neighborhood_increasing_k_fallback_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;eigenvalues;point-program;neighborhood;resident;bridge;fallback;stale-column")

    add_test(NAME pdg_hybrid_covariancefeatures_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_covariancefeatures_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-covariancefeatures.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_covariancefeatures_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;covariancefeatures;neighborhood;spatial-index;knn;eigen")

    foreach(mode raw sqrt normalized)
        add_test(NAME pdg_hybrid_covariancefeatures_extended_${mode}_cuda_exact
            COMMAND ${CMAKE_COMMAND} -E env
                PDG_DISABLE_NATIVE=1
                PDG_REQUIRE_HYBRID=1
                PDG_REQUIRE_CUDA_HYBRID=1
                ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
                --oracle $<TARGET_FILE:pdal>
                --candidate $<TARGET_FILE:pdg_cli>
                --case pdg_hybrid_covariancefeatures_extended_${mode}_cuda_exact
                --work-dir ${PROJECT_BINARY_DIR}/tests/differential
                --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
                --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
                --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-covariancefeatures-extended-${mode}.json
                -- pipeline pipeline.json)
        set_tests_properties(
            pdg_hybrid_covariancefeatures_extended_${mode}_cuda_exact
            PROPERTIES LABELS
                "differential;hybrid;cuda;exact;covariancefeatures;neighborhood;spatial-index;knn;eigen;transcendental-bridge")
    endforeach()

    add_test(NAME pdg_hybrid_approximatecoplanar_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            PDG_FORCE_UNIFORM_GRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_approximatecoplanar_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-approximatecoplanar.json
            -- pipeline pipeline.json --metadata=metadata.json)
    set_tests_properties(pdg_hybrid_approximatecoplanar_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;approximatecoplanar;neighborhood;spatial-index;knn;eigen;uint8")

    add_test(NAME pdg_hybrid_approximatecoplanar_max_knn_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            PDG_FORCE_MORTON_BVH=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_approximatecoplanar_max_knn_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-approximatecoplanar-max.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_approximatecoplanar_max_knn_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;approximatecoplanar;neighborhood;spatial-index;knn;max-knn;eigen;uint8")

    add_test(NAME pdg_hybrid_approximatecoplanar_tie_repair_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            PDG_REQUIRE_NEIGHBORHOOD_TIE_REPAIR=1
            PDG_FORCE_MORTON_BVH=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_approximatecoplanar_tie_repair_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.csv=${PROJECT_SOURCE_DIR}/test/data/pdg/neighborhood-points.csv
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-approximatecoplanar-ties.json
            -- pipeline pipeline.json)
    set_tests_properties(
        pdg_hybrid_approximatecoplanar_tie_repair_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;approximatecoplanar;neighborhood;spatial-index;knn;eigen;tie-repair;uint8")

    add_test(NAME pdg_hybrid_approximatecoplanar_solver_failure_fallback_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_EXPERIMENTAL_CUDA_HYBRID=1
            PDG_REQUIRE_APPROXIMATECOPLANAR_HOST_FALLBACK=1
            PDG_FORCE_MORTON_BVH=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_approximatecoplanar_solver_failure_fallback_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.csv=${PROJECT_SOURCE_DIR}/test/data/pdg/approximatecoplanar-solver-failure.csv
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-approximatecoplanar-solver-failure.json
            -- pipeline pipeline.json)
    set_tests_properties(
        pdg_hybrid_approximatecoplanar_solver_failure_fallback_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;approximatecoplanar;neighborhood;spatial-index;knn;eigen;fallback;solver-failure;diagnostic")

    add_test(NAME pdg_hybrid_approximatecoplanar_boundary_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            PDG_FORCE_MORTON_BVH=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_approximatecoplanar_boundary_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.csv=${PROJECT_SOURCE_DIR}/test/data/pdg/approximatecoplanar-boundary.csv
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-approximatecoplanar-boundary.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_approximatecoplanar_boundary_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;approximatecoplanar;neighborhood;spatial-index;knn;eigen;uint8;strict-boundary")

    add_test(NAME pdg_hybrid_approximatecoplanar_zero_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            PDG_FORCE_MORTON_BVH=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_approximatecoplanar_zero_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.csv=${PROJECT_SOURCE_DIR}/test/data/pdg/approximatecoplanar-boundary.csv
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-approximatecoplanar-zero.json
            -- pipeline pipeline.json --verbose=2)
    set_tests_properties(pdg_hybrid_approximatecoplanar_zero_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;approximatecoplanar;neighborhood;spatial-index;knn;eigen;uint8;zero-covariance;diagnostic;preserve")

    add_test(NAME pdg_hybrid_approximatecoplanar_small_view_fallback_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_EXPERIMENTAL_CUDA_HYBRID=1
            PDG_REQUIRE_APPROXIMATECOPLANAR_HOST_FALLBACK=1
            PDG_FORCE_MORTON_BVH=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_approximatecoplanar_small_view_fallback_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.csv=${PROJECT_SOURCE_DIR}/test/data/pdg/neighborhood-points.csv
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-approximatecoplanar-small-view.json
            -- pipeline pipeline.json)
    set_tests_properties(
        pdg_hybrid_approximatecoplanar_small_view_fallback_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;approximatecoplanar;neighborhood;knn;fallback;small-view")

    add_test(NAME pdg_hybrid_approximatecoplanar_resident_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            PDG_REQUIRE_NEIGHBORHOOD_REUSE=1
            PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE=1
            PDG_REQUIRE_NEIGHBORHOOD_COPLANAR_COLUMN_REUSE=1
            PDG_REQUIRE_NEIGHBORHOOD_EIGENSYSTEM_REUSE=8
            PDG_FORCE_MORTON_BVH=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_approximatecoplanar_resident_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-approximatecoplanar-resident.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_approximatecoplanar_resident_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;approximatecoplanar;normal;eigenvalues;point-program;neighborhood;spatial-index;knn;eigen;resident;reuse;column-reuse;eigensystem-cache;uint8")

    add_test(NAME pdg_hybrid_neighborhood_resident_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            PDG_REQUIRE_NEIGHBORHOOD_REUSE=1
            PDG_FORCE_MORTON_BVH=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_neighborhood_resident_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-neighborhood-resident.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_neighborhood_resident_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;normal;eigenvalues;covariancefeatures;neighborhood;spatial-index;knn;eigen;resident;reuse")

    add_test(NAME pdg_hybrid_neighborhood_column_consumer_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            PDG_REQUIRE_NEIGHBORHOOD_REUSE=1
            PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE=1
            PDG_FORCE_MORTON_BVH=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_neighborhood_column_consumer_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-neighborhood-column-consumer.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_neighborhood_column_consumer_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;normal;eigenvalues;covariancefeatures;point-program;neighborhood;spatial-index;knn;eigen;resident;reuse;column-reuse")

    add_test(NAME pdg_hybrid_sort_stable_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_sort_stable_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-sort-stable.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_sort_stable_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;ordering;sort;global-ordering")

    foreach(mode IN ITEMS forward reverse)
        add_test(NAME pdg_hybrid_morton_${mode}_cuda_exact
            COMMAND ${CMAKE_COMMAND} -E env
                PDG_DISABLE_NATIVE=1
                PDG_REQUIRE_HYBRID=1
                PDG_REQUIRE_CUDA_HYBRID=1
                ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
                --oracle $<TARGET_FILE:pdal>
                --candidate $<TARGET_FILE:pdg_cli>
                --case pdg_hybrid_morton_${mode}_cuda_exact
                --work-dir ${PROJECT_BINARY_DIR}/tests/differential
                --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
                --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
                --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-morton-${mode}.json
                -- pipeline pipeline.json)
        set_tests_properties(pdg_hybrid_morton_${mode}_cuda_exact
            PROPERTIES LABELS
                "differential;hybrid;cuda;exact;ordering;morton;global-ordering;${mode}")
    endforeach()

    add_test(NAME pdg_hybrid_groupby_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_groupby_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-groupby-classification.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_groupby_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;partition;groupby;multi-view")

    add_test(NAME pdg_hybrid_returns_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_returns_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-returns-all.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_returns_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;partition;returns;multi-view")

    add_test(NAME pdg_hybrid_divider_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_divider_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-divider-round-robin.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_divider_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;partition;divider;multi-view")

    add_test(NAME pdg_hybrid_splitter_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_DISABLE_NATIVE=1
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_splitter_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-splitter-primary.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_splitter_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;partition;splitter;multi-view")

    foreach(kind IN ITEMS max min chain)
        if(kind STREQUAL "max")
            set(pipeline hybrid-locate-max.json)
            set(input simple.las)
        elseif(kind STREQUAL "min")
            set(pipeline hybrid-locate-min.json)
            set(input 1.2-with-color.las)
        else()
            set(pipeline hybrid-locate-chain.json)
            set(input simple.las)
        endif()
        add_test(NAME pdg_hybrid_locate_${kind}_cuda_exact
            COMMAND ${CMAKE_COMMAND} -E env
                PDG_DISABLE_NATIVE=1
                PDG_REQUIRE_HYBRID=1
                PDG_REQUIRE_CUDA_HYBRID=1
                PDG_CUDA_CHUNK_POINTS=17
                ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
                --oracle $<TARGET_FILE:pdal>
                --candidate $<TARGET_FILE:pdg_cli>
                --case pdg_hybrid_locate_${kind}_cuda_exact
                --work-dir ${PROJECT_BINARY_DIR}/tests/differential
                --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
                --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/${input}
                --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/${pipeline}
                -- pipeline pipeline.json)
        set_tests_properties(pdg_hybrid_locate_${kind}_cuda_exact
            PROPERTIES LABELS
                "differential;hybrid;cuda;exact;locate;global-reduction")
    endforeach()

    foreach(kind IN ITEMS stream standard split)
        if(kind STREQUAL "stream")
            set(pipeline hybrid-ordinal-stream.json)
        elseif(kind STREQUAL "standard")
            set(pipeline hybrid-ordinal-standard.json)
        else()
            set(pipeline hybrid-ordinal-split.json)
        endif()
        add_test(NAME pdg_cuda_ordered_ordinal_${kind}_exact
            COMMAND ${CMAKE_COMMAND} -E env
                PDG_REQUIRE_CUDA_POINT_PROGRAM=1
                PDG_CUDA_CHUNK_POINTS=17
                ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
                --oracle $<TARGET_FILE:pdal>
                --candidate $<TARGET_FILE:pdg_cli>
                --case pdg_cuda_ordered_ordinal_${kind}_exact
                --work-dir ${PROJECT_BINARY_DIR}/tests/differential
                --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
                --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
                --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/${pipeline}
                -- pipeline pipeline.json)
        set_tests_properties(pdg_cuda_ordered_ordinal_${kind}_exact
            PROPERTIES LABELS
                "differential;cuda;exact;point-program;ordinal;compaction;${kind}-mode")
    endforeach()

    foreach(kind IN ITEMS range expression)
        if(kind STREQUAL "range")
            set(pipeline hybrid-range-fused.json)
        else()
            set(pipeline ordered-expression-fused.json)
        endif()
        add_test(NAME pdg_cuda_ordered_${kind}_fused_exact
            COMMAND ${CMAKE_COMMAND} -E env
                PDG_REQUIRE_CUDA_POINT_PROGRAM=1
                PDG_CUDA_CHUNK_POINTS=2
                ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
                --oracle $<TARGET_FILE:pdal>
                --candidate $<TARGET_FILE:pdg_cli>
                --case pdg_cuda_ordered_${kind}_fused_exact
                --work-dir ${PROJECT_BINARY_DIR}/tests/differential
                --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
                --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
                --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/${pipeline}
                -- pipeline pipeline.json)
        set_tests_properties(pdg_cuda_ordered_${kind}_fused_exact
            PROPERTIES LABELS
                "differential;cuda;exact;point-program;compaction;${kind}")
    endforeach()

    add_test(NAME pdg_hybrid_crop_same_srs_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            PDG_CUDA_CHUNK_POINTS=2
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_crop_same_srs_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/interesting.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-crop-2d.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_crop_same_srs_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;crop;spatial-reference")

    foreach(kind IN ITEMS 2d 3d_outside)
        if(kind STREQUAL "2d")
            set(pipeline hybrid-crop-2d.json)
        else()
            set(pipeline hybrid-crop-3d-outside.json)
        endif()
        add_test(NAME pdg_cuda_ordered_crop_${kind}_exact
            COMMAND ${CMAKE_COMMAND} -E env
                PDG_REQUIRE_CUDA_POINT_PROGRAM=1
                PDG_CUDA_CHUNK_POINTS=2
                ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
                --oracle $<TARGET_FILE:pdal>
                --candidate $<TARGET_FILE:pdg_cli>
                --case pdg_cuda_ordered_crop_${kind}_exact
                --work-dir ${PROJECT_BINARY_DIR}/tests/differential
                --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
                --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
                --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/${pipeline}
                -- pipeline pipeline.json)
        set_tests_properties(pdg_cuda_ordered_crop_${kind}_exact
            PROPERTIES LABELS
                "differential;cuda;exact;point-program;compaction;crop")
    endforeach()

    add_test(NAME pdg_hybrid_assign_ferry_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_assign_ferry_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-assign-ferry.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_assign_ferry_cuda_exact
        PROPERTIES LABELS "differential;hybrid;cuda;exact;point-program")

    add_test(NAME pdg_hybrid_coordinate_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_coordinate_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/sample_c.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-coordinate.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_coordinate_cuda_exact
        PROPERTIES LABELS "differential;hybrid;cuda;exact;coordinates")

    add_test(NAME pdg_hybrid_expression_fused_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_expression_fused_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-expression-fused.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_expression_fused_cuda_exact
        PROPERTIES LABELS "differential;hybrid;cuda;exact;expression;compaction")

    add_test(NAME pdg_hybrid_range_fused_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_range_fused_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-range-fused.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_range_fused_cuda_exact
        PROPERTIES LABELS "differential;hybrid;cuda;exact;range;compaction")

    foreach(kind IN ITEMS stream standard split)
        if(kind STREQUAL "stream")
            set(pipeline hybrid-ordinal-stream.json)
            set(mode_environment PDG_REQUIRE_STREAMING_HYBRID=1)
        elseif(kind STREQUAL "standard")
            set(pipeline hybrid-ordinal-standard.json)
            set(mode_environment)
        else()
            set(pipeline hybrid-ordinal-split.json)
            set(mode_environment PDG_REQUIRE_STREAMING_HYBRID=1)
        endif()
        add_test(NAME pdg_hybrid_ordinal_${kind}_cuda_exact
            COMMAND ${CMAKE_COMMAND} -E env
                PDG_DISABLE_NATIVE=1
                PDG_REQUIRE_HYBRID=1
                PDG_REQUIRE_CUDA_HYBRID=1
                PDG_CUDA_CHUNK_POINTS=17
                ${mode_environment}
                ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
                --oracle $<TARGET_FILE:pdal>
                --candidate $<TARGET_FILE:pdg_cli>
                --case pdg_hybrid_ordinal_${kind}_cuda_exact
                --work-dir ${PROJECT_BINARY_DIR}/tests/differential
                --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
                --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
                --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/${pipeline}
                -- pipeline pipeline.json)
        set_tests_properties(pdg_hybrid_ordinal_${kind}_cuda_exact
            PROPERTIES LABELS
                "differential;hybrid;cuda;exact;ordinal;compaction;${kind}-mode")
    endforeach()

    foreach(kind IN ITEMS 2d 3d_outside)
        if(kind STREQUAL "2d")
            set(pipeline hybrid-crop-2d.json)
        else()
            set(pipeline hybrid-crop-3d-outside.json)
        endif()
        add_test(NAME pdg_hybrid_crop_${kind}_cuda_exact
            COMMAND ${CMAKE_COMMAND} -E env
                PDG_REQUIRE_HYBRID=1
                PDG_REQUIRE_CUDA_HYBRID=1
                PDG_CUDA_CHUNK_POINTS=2
                ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
                --oracle $<TARGET_FILE:pdal>
                --candidate $<TARGET_FILE:pdg_cli>
                --case pdg_hybrid_crop_${kind}_cuda_exact
                --work-dir ${PROJECT_BINARY_DIR}/tests/differential
                --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
                --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
                --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/${pipeline}
                -- pipeline pipeline.json)
        set_tests_properties(pdg_hybrid_crop_${kind}_cuda_exact
            PROPERTIES LABELS
                "differential;hybrid;cuda;exact;crop;compaction")
    endforeach()

    add_test(NAME pdg_hybrid_two_regions_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_hybrid_two_regions_cuda_exact
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/hybrid-two-regions.json
            -- pipeline pipeline.json)
    set_tests_properties(pdg_hybrid_two_regions_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;mixed-pipeline;compaction")

    add_test(NAME pdg_hybrid_reader_matrix_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/hybrid_reader_matrix.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --fixture-root ${PROJECT_SOURCE_DIR}/test/data
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --case-prefix pdg-hybrid-reader-cuda)
    set_tests_properties(pdg_hybrid_reader_matrix_cuda_exact
        PROPERTIES LABELS "differential;hybrid;cuda;exact;reader-matrix")

    add_test(NAME pdg_hybrid_stream_mode_reader_matrix_cuda_exact
        COMMAND ${CMAKE_COMMAND} -E env
            PDG_REQUIRE_HYBRID=1
            PDG_REQUIRE_STREAMING_HYBRID=1
            PDG_REQUIRE_CUDA_HYBRID=1
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/hybrid_reader_matrix.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --fixture-root ${PROJECT_SOURCE_DIR}/test/data
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --case-prefix pdg-hybrid-stream-reader-cuda
            --original-streamable)
    set_tests_properties(pdg_hybrid_stream_mode_reader_matrix_cuda_exact
        PROPERTIES LABELS
            "differential;hybrid;cuda;exact;reader-matrix;stream-mode")
endif()

add_test(NAME pdg_decimation_standard_offset_fallback_matches_oracle
    COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_decimation_standard_offset_fallback_matches_oracle
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/decimation-standard-offset-fallback.json
        -- pipeline pipeline.json)
set_tests_properties(pdg_decimation_standard_offset_fallback_matches_oracle
    PROPERTIES LABELS "differential;fallback;exact;ordinal;standard-mode")

foreach(kind IN ITEMS multibounds reproject invalid)
    if(kind STREQUAL "multibounds")
        set(pipeline crop-multibounds-fallback.json)
        set(input simple.las)
    elseif(kind STREQUAL "reproject")
        set(pipeline crop-reproject-fallback.json)
        set(input interesting.las)
    else()
        set(pipeline crop-invalid-fallback.json)
        set(input simple.las)
    endif()
    add_test(NAME pdg_crop_${kind}_fallback_matches_oracle
        COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
            --oracle $<TARGET_FILE:pdal>
            --candidate $<TARGET_FILE:pdg_cli>
            --case pdg_crop_${kind}_fallback_matches_oracle
            --work-dir ${PROJECT_BINARY_DIR}/tests/differential
            --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
            --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/${input}
            --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/${pipeline}
            -- pipeline pipeline.json)
    set_tests_properties(pdg_crop_${kind}_fallback_matches_oracle
        PROPERTIES LABELS "differential;fallback;exact;crop")
endforeach()

add_test(NAME pdg_fallback_assign_coordinate_matches_oracle
    COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_fallback_assign_coordinate_matches_oracle
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/sample_c.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/assign-coordinate-fallback.json
        -- pipeline pipeline.json)
set_tests_properties(pdg_fallback_assign_coordinate_matches_oracle
    PROPERTIES LABELS "differential;fallback;exact;assign")

add_test(NAME pdg_native_ferry_format_0
    COMMAND ${CMAKE_COMMAND} -E env PDG_REQUIRE_NATIVE=1
        ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_native_ferry_format_0
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/hextest.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/ferry-base.json
        -- pipeline pipeline.json)
set_tests_properties(pdg_native_ferry_format_0
    PROPERTIES LABELS "differential;native;exact;ferry")

add_test(NAME pdg_fallback_ferry_custom_matches_oracle
    COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_fallback_ferry_custom_matches_oracle
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/ferry-custom-fallback.json
        -- pipeline pipeline.json)
set_tests_properties(pdg_fallback_ferry_custom_matches_oracle
    PROPERTIES LABELS "differential;fallback;exact;ferry")

add_test(NAME pdg_fallback_ferry_type_widening_matches_oracle
    COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case pdg_fallback_ferry_type_widening_matches_oracle
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file input.las=${PROJECT_SOURCE_DIR}/test/data/las/simple.las
        --seed-file pipeline.json=${PROJECT_SOURCE_DIR}/test/data/pdg/ferry-widen-fallback.json
        -- pipeline pipeline.json)
set_tests_properties(pdg_fallback_ferry_type_widening_matches_oracle
    PROPERTIES LABELS "differential;fallback;exact;ferry")

add_test(NAME pdg_native_translate_generated_matrix
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/native_las_matrix.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --differential ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>)
set_tests_properties(pdg_native_translate_generated_matrix
    PROPERTIES LABELS "differential;native;exact;generated")

add_test(NAME pdg_fallback_translate_vlr_matches_oracle
    COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case fallback-translate-vlr
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        -- translate ${PROJECT_SOURCE_DIR}/test/data/las/test1_4.las out.las)
set_tests_properties(pdg_fallback_translate_vlr_matches_oracle
    PROPERTIES LABELS "differential;fallback;exact")

add_test(NAME pdg_fallback_existing_output_matches_oracle
    COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case fallback-existing-output
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --seed-file out.las=${PROJECT_SOURCE_DIR}/test/data/las/hextest.las
        -- translate ${PROJECT_SOURCE_DIR}/test/data/las/simple.las out.las)
set_tests_properties(pdg_fallback_existing_output_matches_oracle
    PROPERTIES LABELS "differential;fallback;exact")

add_test(NAME pdg_fallback_failure_matches_oracle
    COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case fallback-missing-input
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        -- info definitely-missing-input.las)
set_tests_properties(pdg_fallback_failure_matches_oracle
    PROPERTIES LABELS "differential;fallback")

add_test(NAME pdg_fallback_metadata_matches_oracle
    COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case fallback-info-metadata
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        -- info --metadata ${PROJECT_SOURCE_DIR}/test/data/las/simple.las)
set_tests_properties(pdg_fallback_metadata_matches_oracle
    PROPERTIES LABELS "differential;fallback")

add_test(NAME pdg_fallback_drivers_matches_oracle
    COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case fallback-drivers
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        -- --drivers)
set_tests_properties(pdg_fallback_drivers_matches_oracle
    PROPERTIES LABELS "differential;fallback;catalog")

add_test(NAME pdg_fallback_no_arguments_matches_oracle
    COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case fallback-no-arguments
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        --allow-empty-command --)
set_tests_properties(pdg_fallback_no_arguments_matches_oracle
    PROPERTIES LABELS "differential;fallback")

add_test(NAME pdg_fallback_version_matches_oracle
    COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/pdg/differential.py
        --oracle $<TARGET_FILE:pdal>
        --candidate $<TARGET_FILE:pdg_cli>
        --case fallback-version
        --work-dir ${PROJECT_BINARY_DIR}/tests/differential
        --frozen-time-library $<TARGET_FILE:pdg_frozen_time>
        -- --version)
set_tests_properties(pdg_fallback_version_matches_oracle
    PROPERTIES LABELS "differential;fallback")

if(PDG_DIFFERENTIAL_ASAN_RUNTIME)
    get_property(PDG_DIFFERENTIAL_TESTS DIRECTORY PROPERTY TESTS)
    foreach(PDG_DIFFERENTIAL_TEST IN LISTS PDG_DIFFERENTIAL_TESTS)
        get_test_property(${PDG_DIFFERENTIAL_TEST} LABELS
            PDG_DIFFERENTIAL_TEST_LABELS)
        if("differential" IN_LIST PDG_DIFFERENTIAL_TEST_LABELS)
            # LeakSanitizer diagnostics contain process-specific addresses and
            # cannot be compared byte-for-byte between the oracle and
            # candidate.  Leak-enabled PDG-labelled tests remain a separate
            # sanitizer lane; exact process differentials retain ASan/UBSan.
            # B0270: vendored lazperf's LAZ integer predictors wrap int32
            # arithmetic by design (shared with pinned PDAL, byte-identical
            # output); UBSan's report of it is not application stderr.
            set_property(TEST ${PDG_DIFFERENTIAL_TEST} APPEND PROPERTY
                ENVIRONMENT
                "PDG_DIFFERENTIAL_ASAN_PRELOAD=${PDG_DIFFERENTIAL_ASAN_RUNTIME}"
                "ASAN_OPTIONS=detect_leaks=0:halt_on_error=1:abort_on_error=1"
                "UBSAN_OPTIONS=suppressions=${PROJECT_SOURCE_DIR}/tests/sanitizers/ubsan-lazperf.supp")
        endif()
    endforeach()
endif()
