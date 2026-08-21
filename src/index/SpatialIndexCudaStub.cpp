#include <pdg/index/SpatialIndex.hpp>

#include <stdexcept>

namespace pdg::detail
{

void buildUniformGridDevice(SpatialIndex&)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void radiusCountsDevice(const SpatialIndex&, double, std::uint32_t*)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void radiusScaledValuesDevice(const SpatialIndex&, double, double, double*)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void radiusAnyDevice(const SpatialIndex&, double, const std::uint8_t*,
                     const std::uint8_t*, double, double, std::uint8_t*)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void knnGatherDevice(const SpatialIndex&, std::uint32_t, std::uint32_t*,
                     double*, std::uint8_t*)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void knnGatherMaskedDevice(const SpatialIndex&, std::uint32_t, std::uint32_t,
                           const std::uint8_t*, const std::uint8_t*,
                           std::uint32_t*, double*, std::uint8_t*)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void knnMeanDistancesDevice(const SpatialIndex&, std::uint32_t, double*,
                            std::uint8_t*)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void knnDistanceValuesDevice(const SpatialIndex&, std::uint32_t,
                             KnnDistanceMode, double*, std::uint8_t*)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void projectKnnMeanDistancesDevice(const SpatialIndex&, std::uint32_t,
                                   std::uint32_t, const double*, double*)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void projectKnnDistanceValuesDevice(const SpatialIndex&, std::uint32_t,
                                    std::uint32_t, KnnDistanceMode,
                                    const double*, double*)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void repairIncompleteKthDistanceValuesDevice(const SpatialIndex&,
                                             std::uint32_t, double*,
                                             const std::uint32_t*, std::size_t,
                                             bool)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void repairIncompleteMeanDistancesDevice(const SpatialIndex&,
                                         std::uint32_t, double*,
                                         const std::uint32_t*, std::size_t,
                                         bool)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void knnCovariancesDevice(const SpatialIndex&, std::uint32_t, Covariance3d*,
                          std::uint8_t*)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void knnEigenSystemsDevice(const SpatialIndex&, std::uint32_t, EigenSystem3d*,
                           std::uint8_t*)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void knnRankCovariancesDevice(const SpatialIndex&, std::uint32_t, Covariance3d*,
                              std::uint8_t*)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void copyCoordinateColumnsToHost(const SpatialIndex&, double*, double*, double*)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void knnAdjacencyHostDevice(const SpatialIndex&, std::uint32_t, std::uint32_t*,
                            double*, std::uint8_t*)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void knnNeighborVotesDevice(const SpatialIndex&, std::uint32_t,
                            const std::uint8_t*, std::uint8_t*, std::uint8_t*)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

void knnLofValuesDevice(const SpatialIndex&, std::uint32_t, double*, double*,
                        double*, std::uint8_t*, std::uint8_t*, std::uint8_t*)
{
    throw std::logic_error("CUDA spatial index support is not compiled");
}

} // namespace pdg::detail
