#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Neighborhood.hpp>

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pdg
{
namespace
{
constexpr unsigned int BlockSize = 256U;

constexpr DimensionId NormalX(StandardDimension::NormalX);
constexpr DimensionId NormalY(StandardDimension::NormalY);
constexpr DimensionId NormalZ(StandardDimension::NormalZ);
constexpr DimensionId Curvature(StandardDimension::Curvature);
constexpr DimensionId Eigenvalue0(StandardDimension::Eigenvalue0);
constexpr DimensionId Eigenvalue1(StandardDimension::Eigenvalue1);
constexpr DimensionId Eigenvalue2(StandardDimension::Eigenvalue2);
constexpr DimensionId Coplanar(StandardDimension::Coplanar);
constexpr DimensionId Linearity(StandardDimension::Linearity);
constexpr DimensionId Planarity(StandardDimension::Planarity);
constexpr DimensionId Scattering(StandardDimension::Scattering);
constexpr DimensionId Verticality(StandardDimension::Verticality);
constexpr DimensionId Anisotropy(StandardDimension::Anisotropy);
constexpr DimensionId EigenvalueSum(StandardDimension::EigenvalueSum);
constexpr DimensionId SurfaceVariation(StandardDimension::SurfaceVariation);
constexpr DimensionId
    DemantkeVerticality(StandardDimension::DemantkeVerticality);
constexpr DimensionId X(StandardDimension::X);
constexpr DimensionId Y(StandardDimension::Y);
constexpr DimensionId Z(StandardDimension::Z);
constexpr DimensionId HeightAboveGround(StandardDimension::HeightAboveGround);

constexpr std::uint32_t SupportedCovarianceFeatures =
    CovarianceLinearity | CovariancePlanarity | CovarianceScattering |
    CovarianceVerticality | CovarianceAnisotropy | CovarianceEigenvalueSum |
    CovarianceSurfaceVariation | CovarianceDemantkeVerticality;

unsigned int launchBlocks(std::size_t size)
{
    const std::size_t blocks = (size + BlockSize - 1U) / BlockSize;
    if (blocks > std::numeric_limits<unsigned int>::max())
        throw std::overflow_error("CUDA neighborhood launch grid overflow");
    return static_cast<unsigned int>(blocks);
}

void validateDeviceProjection(const PointBatch& batch,
                              const EigenSystem3d* systems,
                              const std::uint8_t* status)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "neighborhood columns require a device point batch");
    if (batch.size() && (!systems || !status))
        throw std::invalid_argument(
            "neighborhood columns require device eigensystem and status "
            "buffers");
}

__global__ void normalKernel(std::size_t size, const EigenSystem3d* systems,
                             const std::uint8_t* status, bool alwaysUp,
                             double* normalX, double* normalY, double* normalZ,
                             double* curvature)
{
    for (std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
         point < size; point += gridDim.x * blockDim.x)
    {
        if (status[point] != KnnExact)
            continue;
        const EigenSystem3d& system = systems[point];
        double nx = system.vectors[0];
        double ny = system.vectors[3];
        double nz = system.vectors[6];
        if (alwaysUp && nz < 0.0)
        {
            nx = -nx;
            ny = -ny;
            nz = -nz;
        }
        const double sum =
            system.values[0] + system.values[1] + system.values[2];
        normalX[point] = nx;
        normalY[point] = ny;
        normalZ[point] = nz;
        curvature[point] = sum ? ::fabs(system.values[0] / sum) : 0.0;
    }
}

__global__ void eigenvalueKernel(std::size_t size, const EigenSystem3d* systems,
                                 const std::uint8_t* status, bool normalize,
                                 double* value0, double* value1, double* value2)
{
    for (std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
         point < size; point += gridDim.x * blockDim.x)
    {
        if (status[point] != KnnExact)
            continue;
        double first = systems[point].values[0];
        double second = systems[point].values[1];
        double third = systems[point].values[2];
        if (normalize)
        {
            const double sum = first + second + third;
            first /= sum;
            second /= sum;
            third /= sum;
        }
        value0[point] = first;
        value1[point] = second;
        value2[point] = third;
    }
}

__global__ void approximateCoplanarKernel(std::size_t size,
                                          const EigenSystem3d* systems,
                                          const std::uint8_t* status,
                                          double threshold1, double threshold2,
                                          std::uint8_t* coplanar)
{
    for (std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
         point < size; point += gridDim.x * blockDim.x)
    {
        if (status[point] != KnnExact)
            continue;
        const EigenSystem3d& system = systems[point];
        const bool label =
            system.values[1] > __dmul_rn(threshold1, system.values[0]) &&
            __dmul_rn(threshold2, system.values[1]) > system.values[2];
        coplanar[point] = label ? 1U : 0U;
    }
}

__global__ void hagNnKernel(
    std::size_t size, const double* x, const double* y, const double* z,
    const std::uint8_t* groundMask, const std::uint32_t* nearestPointIds,
    const double* squaredDistances, const std::uint8_t* status,
    std::uint32_t neighbors, std::uint32_t groundCount,
    double maximumDistanceSquared, bool allowExtrapolation, double minimumX,
    double minimumY, double maximumX, double maximumY, double* hag)
{
    for (std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
         point < size; point += gridDim.x * blockDim.x)
    {
        if (groundMask[point] != 0U)
        {
            hag[point] = 0.0;
            continue;
        }
        if (status[point] != KnnExact)
            continue;
        const std::size_t row = point * static_cast<std::size_t>(neighbors);
        const std::uint32_t nearest = nearestPointIds[row];
        if (nearest == 0xffffffffU)
            continue;

        const double x0 = x[point];
        const double y0 = y[point];
        const double z0 = z[point];
        double groundZ = z0;
        if (neighbors == 1U || groundCount == 1U ||
            (x0 == x[nearest] && y0 == y[nearest]))
            groundZ = z[nearest];
        else if (!allowExtrapolation && (x0 < minimumX || x0 > maximumX ||
                                         y0 < minimumY || y0 > maximumY))
            groundZ = z0;
        else
        {
            double weights = 0.0;
            double accumulatedZ = 0.0;
            const std::uint32_t available =
                groundCount < neighbors ? groundCount : neighbors;
            for (std::uint32_t item = 0U; item < available; ++item)
            {
                const double squaredDistance = squaredDistances[row + item];
                if (maximumDistanceSquared > 0.0 &&
                    squaredDistance > maximumDistanceSquared)
                    break;
                const double weight = __ddiv_rn(1.0, squaredDistance);
                weights = __dadd_rn(weights, weight);
                accumulatedZ = __dadd_rn(
                    accumulatedZ,
                    __dmul_rn(weight, z[nearestPointIds[row + item]]));
            }
            if (weights != 0.0)
                groundZ = __ddiv_rn(accumulatedZ, weights);
        }
        hag[point] = __dsub_rn(z0, groundZ);
    }
}

__device__ double hagDelaunayMagnitude2(double x1, double y1, double x2,
                                        double y2)
{
    const double dx = __dsub_rn(x2, x1);
    const double dy = __dsub_rn(y2, y1);
    return __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
}

__device__ bool hagDelaunaySignbit(double value)
{
    return (__double_as_longlong(value) & (1ULL << 63U)) != 0ULL;
}

__device__ double hagDelaunayArea(double ax, double ay, double bx, double by,
                                  double x, double y)
{
    return __dsub_rn(__dmul_rn(__dsub_rn(bx, ax), __dsub_rn(y, ay)),
                     __dmul_rn(__dsub_rn(by, ay), __dsub_rn(x, ax)));
}

__device__ bool hagDelaunayOutsideEdge(double& area, bool totalSign,
                                       double ax, double ay, double bx,
                                       double by, double x, double y)
{
    if (area != 0.0 && hagDelaunaySignbit(area) != totalSign)
    {
        const double magnitude1 = hagDelaunayMagnitude2(ax, ay, bx, by);
        const double magnitude2 = hagDelaunayMagnitude2(ax, ay, x, y);
        const double magnitudeSum = __dadd_rn(magnitude1, magnitude2);
        if (fabs(__ddiv_rn(area, magnitudeSum)) > 1e-14)
            return true;
        area = 0.0;
    }
    return false;
}

__device__ double hagDelaunayBarycentric(
    double x1, double y1, double z1, double x2, double y2, double z2,
    double x3, double y3, double z3, double x, double y)
{
    const double areaTotal = __dsub_rn(
        __dmul_rn(__dsub_rn(x2, x1), __dsub_rn(y3, y2)),
        __dmul_rn(__dsub_rn(y2, y1), __dsub_rn(x3, x2)));
    if (areaTotal == 0.0)
        return (std::numeric_limits<double>::infinity)();
    const bool totalSign = hagDelaunaySignbit(areaTotal);

    double area12 = hagDelaunayArea(x1, y1, x2, y2, x, y);
    if (hagDelaunayOutsideEdge(area12, totalSign, x1, y1, x2, y2, x, y))
        return (std::numeric_limits<double>::infinity)();
    double area23 = hagDelaunayArea(x2, y2, x3, y3, x, y);
    if (hagDelaunayOutsideEdge(area23, totalSign, x3, y3, x2, y2, x, y))
        return (std::numeric_limits<double>::infinity)();
    double area31 = hagDelaunayArea(x3, y3, x1, y1, x, y);
    if (hagDelaunayOutsideEdge(area31, totalSign, x3, y3, x1, y1, x, y))
        return (std::numeric_limits<double>::infinity)();

    const double first = __dmul_rn(area12, z3);
    const double second = __dmul_rn(area23, z1);
    const double third = __dmul_rn(area31, z2);
    return __ddiv_rn(__dadd_rn(__dadd_rn(first, second), third), areaTotal);
}

__device__ bool hagDelaunayTriangleOrder(const double* px, const double* py,
                                         std::uint32_t& i0,
                                         std::uint32_t& i1,
                                         std::uint32_t& i2)
{
    double minimumX = (std::numeric_limits<double>::max)();
    double minimumY = (std::numeric_limits<double>::max)();
    double maximumX = (std::numeric_limits<double>::lowest)();
    double maximumY = (std::numeric_limits<double>::lowest)();
    for (std::uint32_t item = 0U; item < 3U; ++item)
    {
        minimumX = minimumX < px[item] ? minimumX : px[item];
        minimumY = minimumY < py[item] ? minimumY : py[item];
        maximumX = px[item] < maximumX ? maximumX : px[item];
        maximumY = py[item] < maximumY ? maximumY : py[item];
    }
    const double centerX = __ddiv_rn(__dadd_rn(minimumX, maximumX), 2.0);
    const double centerY = __ddiv_rn(__dadd_rn(minimumY, maximumY), 2.0);

    constexpr std::uint32_t Invalid = 0xffffffffU;
    i0 = Invalid;
    double minimumDistance = (std::numeric_limits<double>::max)();
    for (std::uint32_t item = 0U; item < 3U; ++item)
    {
        const double distance =
            hagDelaunayMagnitude2(centerX, centerY, px[item], py[item]);
        if (distance < minimumDistance)
        {
            i0 = item;
            minimumDistance = distance;
        }
    }

    i1 = Invalid;
    minimumDistance = (std::numeric_limits<double>::max)();
    for (std::uint32_t item = 0U; item < 3U; ++item)
    {
        if (item == i0)
            continue;
        const double distance =
            hagDelaunayMagnitude2(px[i0], py[i0], px[item], py[item]);
        if (distance < minimumDistance && distance > 0.0)
        {
            i1 = item;
            minimumDistance = distance;
        }
    }
    if (i0 == Invalid || i1 == Invalid)
        return false;

    i2 = 3U - i0 - i1;
    const double dx = __dsub_rn(px[i1], px[i0]);
    const double dy = __dsub_rn(py[i1], py[i0]);
    const double ex = __dsub_rn(px[i2], px[i0]);
    const double ey = __dsub_rn(py[i2], py[i0]);
    const double bl = __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
    const double cl = __dadd_rn(__dmul_rn(ex, ex), __dmul_rn(ey, ey));
    const double determinant =
        __dsub_rn(__dmul_rn(dx, ey), __dmul_rn(dy, ex));
    if (bl == 0.0 || cl == 0.0 || determinant == 0.0)
        return false;
    const double radiusX = __ddiv_rn(
        __dmul_rn(__dsub_rn(__dmul_rn(ey, bl), __dmul_rn(dy, cl)), 0.5),
        determinant);
    const double radiusY = __ddiv_rn(
        __dmul_rn(__dsub_rn(__dmul_rn(dx, cl), __dmul_rn(ex, bl)), 0.5),
        determinant);
    const double radius = __dadd_rn(__dmul_rn(radiusX, radiusX),
                                    __dmul_rn(radiusY, radiusY));
    if (!(radius < (std::numeric_limits<double>::max)()))
        return false;

    const double windingDistance = __dadd_rn(bl, cl);
    const double relativeDeterminant =
        fabs(__ddiv_rn(windingDistance, determinant));
    const bool counterclockwise =
        relativeDeterminant <= 1e14 && determinant > 0.0;
    if (counterclockwise)
    {
        const std::uint32_t swap = i1;
        i1 = i2;
        i2 = swap;
    }
    return true;
}

__global__ void hagDelaunayKernel(
    std::size_t size, const double* x, const double* y, const double* z,
    const std::uint8_t* groundMask, const std::uint32_t* nearestPointIds,
    const std::uint8_t* status, std::uint32_t groundCount,
    bool allowExtrapolation, double minimumX, double minimumY, double maximumX,
    double maximumY, double* hag)
{
    for (std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
         point < size; point += gridDim.x * blockDim.x)
    {
        if (groundMask[point] != 0U)
        {
            hag[point] = 0.0;
            continue;
        }
        if (status[point] != KnnExact)
            continue;
        const std::size_t row = point * 3U;
        const std::uint32_t nearest = nearestPointIds[row];
        if (nearest == 0xffffffffU)
            continue;
        const double x0 = x[point];
        const double y0 = y[point];
        const double z0 = z[point];
        double groundZ = z0;
        if (groundCount == 1U ||
            (x0 == x[nearest] && y0 == y[nearest]))
            groundZ = z[nearest];
        else if (!allowExtrapolation && (x0 < minimumX || x0 > maximumX ||
                                         y0 < minimumY || y0 > maximumY))
            groundZ = z0;
        else
        {
            double px[3];
            double py[3];
            double pz[3];
            for (std::uint32_t item = 0U; item < 3U; ++item)
            {
                const std::uint32_t id = nearestPointIds[row + item];
                px[item] = x[id];
                py[item] = y[id];
                pz[item] = z[id];
            }
            std::uint32_t i0 = 0U;
            std::uint32_t i1 = 0U;
            std::uint32_t i2 = 0U;
            if (hagDelaunayTriangleOrder(px, py, i0, i1, i2))
            {
                const double interpolated = hagDelaunayBarycentric(
                    px[i0], py[i0], pz[i0], px[i1], py[i1], pz[i1], px[i2],
                    py[i2], pz[i2], x0, y0);
                groundZ =
                    interpolated ==
                            (std::numeric_limits<double>::infinity)()
                        ? z[nearest]
                        : interpolated;
            }
        }
        hag[point] = __dsub_rn(z0, groundZ);
    }
}

struct CovarianceColumns
{
    double* linearity = nullptr;
    double* planarity = nullptr;
    double* scattering = nullptr;
    double* verticality = nullptr;
    double* anisotropy = nullptr;
    double* eigenvalueSum = nullptr;
    double* surfaceVariation = nullptr;
    double* demantkeVerticality = nullptr;
};

__global__ void
covarianceFeatureKernel(std::size_t size, const EigenSystem3d* systems,
                        std::uint8_t* status, EigenvalueMode mode,
                        std::uint32_t features, CovarianceColumns columns)
{
    for (std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
         point < size; point += gridDim.x * blockDim.x)
    {
        if (status[point] != KnnExact)
            continue;
        const EigenSystem3d& system = systems[point];
        double lambda0 = system.values[2] < 0.0 ? 0.0 : system.values[2];
        double lambda1 = system.values[1] < 0.0 ? 0.0 : system.values[1];
        double lambda2 = system.values[0] < 0.0 ? 0.0 : system.values[0];
        const double sum = lambda0 + lambda1 + lambda2;
        if (lambda0 == 0.0)
        {
            status[point] =
                static_cast<std::uint8_t>(status[point] | KnnFeatureInvalid);
            continue;
        }
        if (mode == EigenvalueMode::Sqrt)
        {
            lambda0 = ::sqrt(lambda0);
            lambda1 = ::sqrt(lambda1);
            lambda2 = ::sqrt(lambda2);
        }
        else if (mode == EigenvalueMode::Normalized)
        {
            lambda0 /= sum;
            lambda1 /= sum;
            lambda2 /= sum;
        }

        if ((features & CovarianceLinearity) != 0U)
            columns.linearity[point] = (lambda0 - lambda1) / lambda0;
        if ((features & CovariancePlanarity) != 0U)
            columns.planarity[point] = (lambda1 - lambda2) / lambda0;
        if ((features & CovarianceScattering) != 0U)
            columns.scattering[point] = lambda2 / lambda0;
        if ((features & CovarianceVerticality) != 0U)
        {
            double unary[3];
            double norm = 0.0;
            for (std::size_t axis = 0; axis < 3U; ++axis)
            {
                unary[axis] = lambda0 * ::fabs(system.vectors[axis * 3U + 2U]) +
                              lambda1 * ::fabs(system.vectors[axis * 3U + 1U]) +
                              lambda2 * ::fabs(system.vectors[axis * 3U]);
                norm += unary[axis] * unary[axis];
            }
            norm = ::sqrt(norm);
            columns.verticality[point] = unary[2] / norm;
        }
        if ((features & CovarianceEigenvalueSum) != 0U)
            columns.eigenvalueSum[point] = sum;
        if ((features & CovarianceAnisotropy) != 0U)
            columns.anisotropy[point] = (lambda0 - lambda2) / lambda0;
        if ((features & CovarianceSurfaceVariation) != 0U)
            columns.surfaceVariation[point] = lambda2 / sum;
        if ((features & CovarianceDemantkeVerticality) != 0U)
            columns.demantkeVerticality[point] =
                1.0 - ::fabs(system.vectors[6]);
    }
}

double* materializeDouble(PointBatch& batch, DimensionId dimension)
{
    batch.materialize(dimension, DimensionType::Double);
    return batch.data<double>(dimension);
}

std::uint8_t* materializeUnsigned8(PointBatch& batch, DimensionId dimension)
{
    batch.materialize(dimension, DimensionType::Unsigned8);
    return batch.data<std::uint8_t>(dimension);
}
} // unnamed namespace

void projectNormalColumns(PointBatch& batch, const EigenSystem3d* systems,
                          const std::uint8_t* status,
                          const NormalProgram& program)
{
    validateDeviceProjection(batch, systems, status);
    double* normalX = materializeDouble(batch, NormalX);
    double* normalY = materializeDouble(batch, NormalY);
    double* normalZ = materializeDouble(batch, NormalZ);
    double* curvature = materializeDouble(batch, Curvature);
    if (!batch.size())
        return;
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    normalKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.size(), systems, status, program.alwaysUp, normalX, normalY,
        normalZ, curvature);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void projectEigenvalueColumns(PointBatch& batch, const EigenSystem3d* systems,
                              const std::uint8_t* status,
                              const EigenvaluesProgram& program)
{
    validateDeviceProjection(batch, systems, status);
    double* value0 = materializeDouble(batch, Eigenvalue0);
    double* value1 = materializeDouble(batch, Eigenvalue1);
    double* value2 = materializeDouble(batch, Eigenvalue2);
    if (!batch.size())
        return;
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    eigenvalueKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.size(), systems, status, program.normalize, value0, value1,
        value2);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void projectApproximateCoplanarColumn(PointBatch& batch,
                                      const EigenSystem3d* systems,
                                      const std::uint8_t* status,
                                      const ApproximateCoplanarProgram& program)
{
    validateDeviceProjection(batch, systems, status);
    std::uint8_t* coplanar = materializeUnsigned8(batch, Coplanar);
    if (!batch.size())
        return;
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    approximateCoplanarKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                                stream>>>(batch.size(), systems, status,
                                          program.threshold1,
                                          program.threshold2, coplanar);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void projectCovarianceFeatureColumns(PointBatch& batch,
                                     const EigenSystem3d* systems,
                                     std::uint8_t* status,
                                     const CovarianceFeaturesProgram& program)
{
    validateDeviceProjection(batch, systems, status);
    if ((program.features & ~SupportedCovarianceFeatures) != 0U)
        throw std::invalid_argument(
            "unsupported CUDA covariance feature selection");
    if (program.mode != EigenvalueMode::Raw &&
        program.mode != EigenvalueMode::Sqrt &&
        program.mode != EigenvalueMode::Normalized)
        throw std::invalid_argument("invalid CUDA covariance feature mode");

    CovarianceColumns columns;
    if ((program.features & CovarianceLinearity) != 0U)
        columns.linearity = materializeDouble(batch, Linearity);
    if ((program.features & CovariancePlanarity) != 0U)
        columns.planarity = materializeDouble(batch, Planarity);
    if ((program.features & CovarianceScattering) != 0U)
        columns.scattering = materializeDouble(batch, Scattering);
    if ((program.features & CovarianceVerticality) != 0U)
        columns.verticality = materializeDouble(batch, Verticality);
    if ((program.features & CovarianceAnisotropy) != 0U)
        columns.anisotropy = materializeDouble(batch, Anisotropy);
    if ((program.features & CovarianceEigenvalueSum) != 0U)
        columns.eigenvalueSum = materializeDouble(batch, EigenvalueSum);
    if ((program.features & CovarianceSurfaceVariation) != 0U)
        columns.surfaceVariation = materializeDouble(batch, SurfaceVariation);
    if ((program.features & CovarianceDemantkeVerticality) != 0U)
        columns.demantkeVerticality =
            materializeDouble(batch, DemantkeVerticality);
    if (!batch.size())
        return;

    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    covarianceFeatureKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                              stream>>>(
        batch.size(), systems, status, program.mode, program.features, columns);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void projectHagNnColumn(PointBatch& batch, const std::uint8_t* groundMask,
                        const std::uint32_t* nearestPointIds,
                        const double* squaredDistances,
                        const std::uint8_t* status, std::uint32_t neighbors,
                        std::uint32_t groundCount,
                        double maximumDistanceSquared, bool allowExtrapolation,
                        double minimumX, double minimumY, double maximumX,
                        double maximumY)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "HAG nearest projection requires a device point batch");
    if (!batch.has(X) || !batch.has(Y) || !batch.has(Z) ||
        batch.columnInfo(X).physicalType != DimensionType::Double ||
        batch.columnInfo(Y).physicalType != DimensionType::Double ||
        batch.columnInfo(Z).physicalType != DimensionType::Double)
        throw std::invalid_argument("HAG projection requires binary64 XYZ");
    if (neighbors < 1U || neighbors > 64U || groundCount == 0U ||
        groundCount > batch.size())
        throw std::invalid_argument(
            "HAG projection requires a bounded ground-neighbor request");
    if (batch.size() &&
        (!groundMask || !nearestPointIds || !squaredDistances || !status))
        throw std::invalid_argument(
            "HAG projection requires device masks, ids, distances, and status");
    double* hag = materializeDouble(batch, HeightAboveGround);
    if (!batch.size())
        return;
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    hagNnKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.size(), batch.data<double>(X), batch.data<double>(Y),
        batch.data<double>(Z), groundMask, nearestPointIds, squaredDistances,
        status, neighbors, groundCount, maximumDistanceSquared,
        allowExtrapolation, minimumX, minimumY, maximumX, maximumY, hag);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void projectHagDelaunayColumn(
    PointBatch& batch, const std::uint8_t* groundMask,
    const std::uint32_t* nearestPointIds, const std::uint8_t* status,
    std::uint32_t groundCount, bool allowExtrapolation, double minimumX,
    double minimumY, double maximumX, double maximumY)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "HAG Delaunay projection requires a device point batch");
    if (!batch.has(X) || !batch.has(Y) || !batch.has(Z) ||
        batch.columnInfo(X).physicalType != DimensionType::Double ||
        batch.columnInfo(Y).physicalType != DimensionType::Double ||
        batch.columnInfo(Z).physicalType != DimensionType::Double)
        throw std::invalid_argument(
            "HAG Delaunay projection requires binary64 XYZ");
    if (groundCount < 3U || groundCount > batch.size())
        throw std::invalid_argument(
            "HAG Delaunay projection requires three ground neighbors");
    if (batch.size() && (!groundMask || !nearestPointIds || !status))
        throw std::invalid_argument(
            "HAG Delaunay projection requires device masks, ids, and status");
    double* hag = materializeDouble(batch, HeightAboveGround);
    if (!batch.size())
        return;
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    hagDelaunayKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.size(), batch.data<double>(X), batch.data<double>(Y),
        batch.data<double>(Z), groundMask, nearestPointIds, status, groundCount,
        allowExtrapolation, minimumX, minimumY, maximumX, maximumY, hag);
    PDG_CUDA_CHECK(cudaGetLastError());
}

} // namespace pdg
