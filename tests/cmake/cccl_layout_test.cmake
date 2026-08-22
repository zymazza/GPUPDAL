if(NOT DEFINED TEST_SOURCE_DIR OR NOT DEFINED TEST_BINARY_DIR)
    message(FATAL_ERROR "CCCL layout test requires source and binary roots")
endif()

include("${TEST_SOURCE_DIR}/cmake/pdg-cccl.cmake")

set(_pdg_layout_root "${TEST_BINARY_DIR}/cccl-layout-contract")
set(_pdg_cuda12_include "${_pdg_layout_root}/cuda12/include")
file(MAKE_DIRECTORY "${_pdg_cuda12_include}/cub/device")
file(WRITE "${_pdg_cuda12_include}/cub/device/device_select.cuh" "// fixture\n")
set(CUDAToolkit_INCLUDE_DIRS "${_pdg_cuda12_include}")
pdg_find_cccl_include(_pdg_cuda12_result)
file(REAL_PATH "${_pdg_cuda12_result}" _pdg_cuda12_result_real)
file(REAL_PATH "${_pdg_cuda12_include}" _pdg_cuda12_expected_real)
if(NOT _pdg_cuda12_result_real STREQUAL _pdg_cuda12_expected_real)
    message(FATAL_ERROR
        "CUDA 12-style CCCL layout resolved to '${_pdg_cuda12_result}'")
endif()

set(_pdg_cuda13_include "${_pdg_layout_root}/cuda13/include")
file(MAKE_DIRECTORY "${_pdg_cuda13_include}/cccl/cub/device")
file(WRITE "${_pdg_cuda13_include}/cccl/cub/device/device_select.cuh"
    "// fixture\n")
set(CUDAToolkit_INCLUDE_DIRS "${_pdg_cuda13_include}")
pdg_find_cccl_include(_pdg_cuda13_result)
file(REAL_PATH "${_pdg_cuda13_result}" _pdg_cuda13_result_real)
file(REAL_PATH "${_pdg_cuda13_include}/cccl" _pdg_cuda13_expected_real)
if(NOT _pdg_cuda13_result_real STREQUAL _pdg_cuda13_expected_real)
    message(FATAL_ERROR
        "CUDA 13-style CCCL layout resolved to '${_pdg_cuda13_result}'")
endif()
