#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace pdg
{

class PointBatch;
class RasterGridProduct;

class PmfRasterTieError final : public std::invalid_argument
{
public:
    PmfRasterTieError()
        : std::invalid_argument(
              "PMF nearest-fill tie has distinct source elevations")
    {
    }
};

// The first exact device lane uses one bounded global raster. The planner
// preflights this fixed workspace, but it is not yet reusable across stages.
inline constexpr std::size_t PmfExactDeviceMaximumRasterCells = 4096U;
inline constexpr int PmfExactDeviceMaximumMorphologyRadius = 64;
inline constexpr std::size_t PmfExactDeviceMaximumPasses = 64U;
inline constexpr std::size_t PmfTiledDeviceBytesPerCell = 2U * sizeof(double);
inline constexpr std::size_t PmfTiledDeviceProofBytesPerCell =
    2U * sizeof(double);
inline constexpr std::size_t PmfTiledHostBytesPerCell = 2U * sizeof(double);
inline constexpr std::size_t PmfTiledHostStagingBytesPerPoint =
    3U * sizeof(double) + sizeof(std::uint8_t);
inline constexpr std::size_t PmfTiledDeviceFixedScratchBytes = 65536U;
inline constexpr std::size_t PmfExactDeviceMaximumFixedScratchBytes =
    PmfExactDeviceMaximumRasterCells * 96U + 65536U;
inline constexpr std::size_t PmfExactDeviceScratchBytesPerPoint = 1U;

struct PmfProgram
{
    double cellSize = 1.0;
    bool exponential = true;
    double initialDistance = 0.15;
    double maxDistance = 2.5;
    double maxWindowSize = 33.0;
    double slope = 1.0;
    std::uint8_t groundClass = 2U;
    std::uint8_t otherClass = 1U;
    bool onlyGround = false;
};

struct PmfRasterFrame
{
    double minimumX = 0.0;
    double minimumY = 0.0;
    std::size_t rows = 0U;
    std::size_t columns = 0U;

    [[nodiscard]] std::size_t size() const noexcept
    {
        return rows * columns;
    }
};

struct PmfResult
{
    std::size_t rows = 0U;
    std::size_t columns = 0U;
    std::size_t groundPoints = 0U;
    std::size_t nongroundPoints = 0U;
};

struct PmfTiledExecutionFacts
{
    bool deviceResidentPhases = false;
    bool deviceNativeRaster = false;
    std::size_t rasterHostToDeviceTransfers = 0U;
    std::size_t rasterHostToDeviceBytes = 0U;
    std::size_t rasterDeviceToHostTransfers = 0U;
    std::size_t rasterDeviceToHostBytes = 0U;
};

struct PmfRasterBuildFacts
{
    std::size_t populatedCells = 0U;
    std::size_t sourceSlotsVisited = 0U;
    std::size_t hierarchyNodesVisited = 0U;
    bool usedBoundedSourceScan = false;
    bool usedOccupancyHierarchy = false;
    bool deviceNativeSourceBuild = false;
    bool usedDeviceTieProof = false;
};

[[nodiscard]] bool
pmfProgramWithinExactDeviceEnvelope(const PmfProgram& program) noexcept;

[[nodiscard]] bool pmfSupportsExactDevice(const PointBatch& hostBatch,
                                          const PmfProgram& program) noexcept;
[[nodiscard]] bool
pmfSupportsExactTiledDevice(const PointBatch& hostBatch,
                            const PmfProgram& program) noexcept;
[[nodiscard]] PmfRasterFrame pmfRasterFrame(const PointBatch& hostBatch,
                                            const PmfProgram& program);

// Builds the pinned-oracle sparse minima and nearest-filled surface once into
// the planner-owned host backings. It uses no storage outside the product's
// preflighted two-backings contract and publishes only a complete, tie-safe
// raster generation.
void buildPmfTiledRaster(const PointBatch& hostBatch, const PmfProgram& program,
                         RasterGridProduct& product,
                         PmfRasterBuildFacts* facts = nullptr);

// Constructs and proves the complete source raster in the product's
// provisional device workspace. Success publishes and promotes that exact
// allocation into the two phase backings; failure discards it without a
// visible raster generation.
void buildPmfTiledRasterDevice(PointBatch& deviceBatch,
                               const PmfProgram& program,
                               RasterGridProduct& product,
                               PmfRasterBuildFacts* facts = nullptr);

// Reproduces the pinned PMF global-raster phase order, including its distinct
// initial fractional-cell binning and later point-lookup formulas.
[[nodiscard]] PmfResult classifyPmf(PointBatch& batch,
                                    const PmfProgram& program);
[[nodiscard]] PmfResult
classifyPmfTiledDevice(PointBatch& batch, const PmfProgram& program,
                       RasterGridProduct& product,
                       PmfTiledExecutionFacts* facts = nullptr);

} // namespace pdg
