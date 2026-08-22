function(pdg_find_cccl_include output_variable)
    set(_pdg_cccl_hints)
    foreach(_pdg_cuda_include IN LISTS CUDAToolkit_INCLUDE_DIRS)
        list(APPEND _pdg_cccl_hints
            "${_pdg_cuda_include}"
            "${_pdg_cuda_include}/cccl")
    endforeach()
    find_path(_pdg_detected_cccl_include
        NAMES cub/device/device_select.cuh
        HINTS ${_pdg_cccl_hints}
        NO_CACHE)
    set(${output_variable} "${_pdg_detected_cccl_include}" PARENT_SCOPE)
endfunction()
