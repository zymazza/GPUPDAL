#include <pdg/stages/Neighborhood.hpp>

#include <stdexcept>

namespace pdg
{

void projectNormalColumns(PointBatch&, const EigenSystem3d*,
                          const std::uint8_t*, const NormalProgram&)
{
    throw std::logic_error("CUDA neighborhood projection is not compiled");
}

void projectEigenvalueColumns(PointBatch&, const EigenSystem3d*,
                              const std::uint8_t*, const EigenvaluesProgram&)
{
    throw std::logic_error("CUDA neighborhood projection is not compiled");
}

void projectApproximateCoplanarColumn(PointBatch&, const EigenSystem3d*,
                                      const std::uint8_t*,
                                      const ApproximateCoplanarProgram&)
{
    throw std::logic_error("CUDA neighborhood projection is not compiled");
}

void projectCovarianceFeatureColumns(PointBatch&, const EigenSystem3d*,
                                     std::uint8_t*,
                                     const CovarianceFeaturesProgram&)
{
    throw std::logic_error("CUDA neighborhood projection is not compiled");
}

void projectHagNnColumn(PointBatch&, const std::uint8_t*, const std::uint32_t*,
                        const double*, const std::uint8_t*, std::uint32_t,
                        std::uint32_t, double, bool, double, double, double,
                        double)
{
    throw std::logic_error("CUDA HAG projection is not compiled");
}

void projectHagDelaunayColumn(PointBatch&, const std::uint8_t*,
                              const std::uint32_t*, const std::uint8_t*,
                              std::uint32_t, bool, double, double, double,
                              double)
{
    throw std::logic_error("CUDA HAG Delaunay projection is not compiled");
}

} // namespace pdg
