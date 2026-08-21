#pragma once

#include <stdexcept>
#include <string>

#ifndef PDG_HAS_CUDA
#define PDG_HAS_CUDA 0
#endif

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace pdg
{

class CudaError : public std::runtime_error
{
public:
    CudaError(int code, std::string message);
    [[nodiscard]] int code() const noexcept;

private:
    int m_code;
};

#if PDG_HAS_CUDA
namespace cuda
{
void check(cudaError_t result, const char* expression, const char* file,
           int line);
void checkNoexcept(cudaError_t result, const char* expression, const char* file,
                   int line) noexcept;
} // namespace cuda
#endif

} // namespace pdg

#if PDG_HAS_CUDA
#define PDG_CUDA_CHECK(expression)                                             \
    ::pdg::cuda::check((expression), #expression, __FILE__, __LINE__)
#define PDG_CUDA_CHECK_NOEXCEPT(expression)                                    \
    ::pdg::cuda::checkNoexcept((expression), #expression, __FILE__, __LINE__)
#endif
