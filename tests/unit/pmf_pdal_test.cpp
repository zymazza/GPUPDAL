#include <pdg/Cuda.hpp>
#include <pdg/ExecutionStats.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/Plan.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/index/RasterGrid.hpp>
#include <pdg/stages/Pmf.hpp>

#include "src/pdal/PdgResidentContext.hpp"

#include <io/BufferReader.hpp>
#include <pdal/Dimension.hpp>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>

#include <gtest/gtest.h>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
bool cudaDeviceAvailable()
{
    try
    {
        return !pdg::cudaDevices().empty();
    }
    catch (const pdg::CudaError&)
    {
        return false;
    }
}

std::size_t boundaryId(const pdg::Plan& plan, pdg::ResidencyBoundaryKind kind)
{
    const auto position =
        std::find_if(plan.summary().residencyBoundaries.begin(),
                     plan.summary().residencyBoundaries.end(),
                     [&](const pdg::ResidencyBoundary& boundary)
                     { return boundary.kind == kind; });
    EXPECT_NE(position, plan.summary().residencyBoundaries.end());
    return static_cast<std::size_t>(
        std::distance(plan.summary().residencyBoundaries.begin(), position));
}

struct PmfPoint
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    std::uint8_t classification = 0U;
    std::uint8_t returnNumber = 1U;
    std::uint8_t numberOfReturns = 1U;
};

struct PmfSnapshot
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<std::uint8_t> classification;
    std::vector<std::uint8_t> returnNumber;
    std::vector<std::uint8_t> numberOfReturns;
    std::vector<double> heightAboveGround;
    std::string error;
};

std::vector<PmfPoint> makePmfDifferentialFixture()
{
    constexpr std::size_t Width = 81U;
    constexpr double Cell = 0.6;
    constexpr double MinX = -12.25;
    constexpr double MinY = 3.125;
    constexpr double PhaseX = 0.17;
    constexpr double PhaseY = 0.11;

    std::vector<PmfPoint> points;
    points.reserve(Width * Width + 2U);

    for (std::size_t column = 0U; column < Width; ++column)
        for (std::size_t row = 0U; row < Width; ++row)
        {
            if ((column == 0U && row == 0U) || (column == 40U && row == 40U) ||
                (column == 40U && row == 41U) || (column == 41U && row == 40U))
                continue;

            const std::size_t pointHash = column * 131U + row * 17U;
            PmfPoint point;
            point.x = MinX + static_cast<double>(column) * Cell + PhaseX;
            point.y = MinY + static_cast<double>(row) * Cell + PhaseY;
            point.z = -0.0;
            point.classification =
                static_cast<std::uint8_t>(8U + (pointHash & 1U));
            if (point.classification == 9U)
                point.classification = 7U;
            point.classification +=
                static_cast<std::uint8_t>((column + row) % 2U);
            point.numberOfReturns = ((column + row) % 7U == 0U) ? 1U : 2U;
            point.returnNumber =
                (point.numberOfReturns == 1U
                     ? 1U
                     : (static_cast<std::uint8_t>((column + row) % 2U) + 1U));
            points.push_back(point);
        }

    points.push_back({MinX + PhaseX, MinY + PhaseY, -0.0, 7U, 2U, 2U});
    points.push_back(
        {MinX + 0.17 + 0.01, MinY + 0.11 + 0.01, +0.0, 8U, 2U, 2U});
    const auto addElevated = [&](std::size_t column, std::size_t row, double z)
    {
        points.push_back({MinX + static_cast<double>(column) * Cell + PhaseX,
                          MinY + static_cast<double>(row) * Cell + PhaseY, z,
                          8U, 2U, 2U});
    };
    addElevated(3U, 66U, 0.15);
    addElevated(20U, 21U, 0.14999999999999999);
    addElevated(39U, 40U, 1.0);
    addElevated(60U, 61U, 2.5);

    return points;
}

std::vector<PmfPoint> makePmfMorphologyFixture()
{
    constexpr std::size_t Width = 65U;
    std::vector<PmfPoint> points;
    points.reserve(Width * Width);
    for (std::size_t column = 0U; column < Width; ++column)
        for (std::size_t row = 0U; row < Width; ++row)
        {
            const std::size_t hash = column * 13U + row * 7U;
            const double z = static_cast<double>(hash % 23U) * 0.125 +
                             ((column == 32U && row == 32U) ? 8.0 : 0.0);
            points.push_back({-7.0 + static_cast<double>(column),
                              3.5 + static_cast<double>(row), z, 7U, 1U, 1U});
        }
    return points;
}

PmfSnapshot snapshotPmf(const pdal::PointView& view)
{
    using pdal::Dimension::Id;
    PmfSnapshot snapshot;
    const pdal::PointId pointCount = view.size();
    snapshot.x.reserve(static_cast<std::size_t>(pointCount));
    snapshot.y.reserve(static_cast<std::size_t>(pointCount));
    snapshot.z.reserve(static_cast<std::size_t>(pointCount));
    snapshot.classification.reserve(static_cast<std::size_t>(pointCount));
    snapshot.returnNumber.reserve(static_cast<std::size_t>(pointCount));
    snapshot.numberOfReturns.reserve(static_cast<std::size_t>(pointCount));
    const bool hasHag = view.layout()->hasDim(Id::HeightAboveGround);
    if (hasHag)
        snapshot.heightAboveGround.reserve(
            static_cast<std::size_t>(pointCount));
    for (pdal::PointId point = 0U; point < pointCount; ++point)
    {
        snapshot.x.push_back(view.getFieldAs<double>(Id::X, point));
        snapshot.y.push_back(view.getFieldAs<double>(Id::Y, point));
        snapshot.z.push_back(view.getFieldAs<double>(Id::Z, point));
        snapshot.classification.push_back(
            view.getFieldAs<std::uint8_t>(Id::Classification, point));
        snapshot.returnNumber.push_back(
            view.getFieldAs<std::uint8_t>(Id::ReturnNumber, point));
        snapshot.numberOfReturns.push_back(
            view.getFieldAs<std::uint8_t>(Id::NumberOfReturns, point));
        if (hasHag)
            snapshot.heightAboveGround.push_back(
                view.getFieldAs<double>(Id::HeightAboveGround, point));
    }
    return snapshot;
}

pdal::PointViewPtr buildPmfInputView(const std::vector<PmfPoint>& points,
                                     pdal::PointTable& table)
{
    using pdal::Dimension::Id;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                  Id::ReturnNumber, Id::NumberOfReturns});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (pdal::PointId point = 0U; point < points.size(); ++point)
    {
        const PmfPoint& fixture = points[static_cast<std::size_t>(point)];
        view->setField(Id::X, point, fixture.x);
        view->setField(Id::Y, point, fixture.y);
        view->setField(Id::Z, point, fixture.z);
        view->setField(Id::Classification, point, fixture.classification);
        view->setField(Id::ReturnNumber, point, fixture.returnNumber);
        view->setField(Id::NumberOfReturns, point, fixture.numberOfReturns);
    }
    return view;
}

pdal::PointViewPtr buildPmfHagInputView(const std::vector<PmfPoint>& points,
                                        pdal::PointTable& table)
{
    using pdal::Dimension::Id;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                  Id::ReturnNumber, Id::NumberOfReturns,
                                  Id::HeightAboveGround});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (pdal::PointId point = 0U; point < points.size(); ++point)
    {
        const PmfPoint& fixture = points[static_cast<std::size_t>(point)];
        view->setField(Id::X, point, fixture.x);
        view->setField(Id::Y, point, fixture.y);
        view->setField(Id::Z, point, fixture.z);
        view->setField(Id::Classification, point, fixture.classification);
        view->setField(Id::ReturnNumber, point, fixture.returnNumber);
        view->setField(Id::NumberOfReturns, point, fixture.numberOfReturns);
    }
    return view;
}

PmfSnapshot runPmfFilter(std::string stageName,
                         const std::vector<PmfPoint>& points,
                         const pdal::Options& baseOptions)
{
    pdal::PointTable table;
    pdal::PointViewPtr view = buildPmfInputView(points, table);

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage(stageName);
    if (!filter)
        throw std::runtime_error("PMF differential stage is unavailable");

    pdal::Options options = baseOptions;
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);

    PmfSnapshot snapshot;
    try
    {
        static_cast<void>(filter->execute(table));
    }
    catch (const std::exception& error)
    {
        snapshot.error = error.what();
    }
    if (snapshot.error.empty())
        snapshot = snapshotPmf(*view);
    return snapshot;
}

PmfSnapshot runPmfFilterSequence(const std::vector<PmfPoint>& points,
                                 const std::vector<pdal::Options>& stageOptions)
{
    pdal::PointTable table;
    pdal::PointViewPtr view = buildPmfInputView(points, table);
    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* input = &reader;
    PmfSnapshot snapshot;
    try
    {
        for (const pdal::Options& options : stageOptions)
        {
            pdal::Stage* filter = factory.createStage("filters.pmf");
            if (!filter)
                throw std::runtime_error(
                    "PMF differential stage is unavailable");
            filter->setOptions(options);
            filter->setInput(*input);
            input = filter;
        }
        input->prepare(table);
        const pdal::PointViewSet output = input->execute(table);
        if (output.size() != 1U)
            throw std::runtime_error(
                "PMF differential sequence changed view count");
        view = *output.begin();
    }
    catch (const std::exception& error)
    {
        snapshot.error = error.what();
    }
    if (snapshot.error.empty())
        snapshot = snapshotPmf(*view);
    return snapshot;
}

PmfSnapshot runPmfHagPmfSequence(const std::vector<PmfPoint>& points,
                                 const pdal::Options& pmfOptions)
{
    pdal::PointTable table;
    pdal::PointViewPtr view = buildPmfHagInputView(points, table);
    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* first = factory.createStage("filters.pmf");
    pdal::Stage* hag = factory.createStage("filters.hag_nn");
    pdal::Stage* second = factory.createStage("filters.pmf");
    if (!first || !hag || !second)
        throw std::runtime_error(
            "PMF/HAG differential sequence is unavailable");
    first->setOptions(pmfOptions);
    pdal::Options hagOptions;
    hagOptions.add("count", 2U);
    hag->setOptions(hagOptions);
    second->setOptions(pmfOptions);
    first->setInput(reader);
    hag->setInput(*first);
    second->setInput(*hag);

    PmfSnapshot snapshot;
    try
    {
        second->prepare(table);
        const pdal::PointViewSet output = second->execute(table);
        if (output.size() != 1U)
            throw std::runtime_error(
                "PMF/HAG differential sequence changed view count");
        view = *output.begin();
    }
    catch (const std::exception& error)
    {
        snapshot.error = error.what();
    }
    if (snapshot.error.empty())
        snapshot = snapshotPmf(*view);
    return snapshot;
}

PmfSnapshot runPmfHybridResident(const pdg::Plan& plan,
                                 const pdg::DimensionRegistry& dimensions,
                                 const std::vector<PmfPoint>& points,
                                 const pdal::Options& baseOptions,
                                 std::size_t budget, std::size_t region,
                                 std::size_t upload, std::size_t spill,
                                 std::size_t* tileCount = nullptr,
                                 std::size_t* activeLaneCount = nullptr)
{
    pdal::PointTable table;
    pdal::PointViewPtr view = buildPmfInputView(points, table);

    pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions, budget,
                                                   64U);
    const std::array selectedRegions{region};
    scope.preflight(*table.layout(), view->size(), selectedRegions);
    pdal::pdg_detail::ResidentExecutionContext& context = scope.context();

    context.enterBoundary(
        *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage(std::string(pdg::HybridPmfStage));
    if (!filter)
        throw std::runtime_error(
            "resident PMF differential stage is unavailable");
    pdal::Options options = baseOptions;
    options.add("pdg_resident_context", true);
    options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);
    static_cast<void>(filter->execute(table));

    context.enterBoundary(
        *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
        region,
        plan.summary().residencyBoundaries.at(spill).requiresFullPointRecord);

    if (tileCount)
        *tileCount = scope.context().schedule().tileCount;
    if (activeLaneCount)
        *activeLaneCount = scope.context().schedule().activeLaneCount;

    return snapshotPmf(*view);
}

void expectPmfSnapshotsEqual(const PmfSnapshot& lhs, const PmfSnapshot& rhs,
                             const std::string& label)
{
    EXPECT_EQ(lhs.error, rhs.error) << label;
    if (!lhs.error.empty() || !rhs.error.empty())
        return;
    EXPECT_EQ(lhs.x, rhs.x) << label;
    EXPECT_EQ(lhs.y, rhs.y) << label;
    EXPECT_EQ(lhs.z, rhs.z) << label;
    ASSERT_EQ(lhs.classification.size(), rhs.classification.size()) << label;
    for (std::size_t point = 0U; point < lhs.classification.size(); ++point)
        if (lhs.classification[point] != rhs.classification[point])
        {
            ADD_FAILURE()
                << label << ": Classification first differs at " << point
                << " (upstream="
                << static_cast<unsigned int>(lhs.classification[point])
                << ", resident="
                << static_cast<unsigned int>(rhs.classification[point]) << ")";
            break;
        }
    EXPECT_EQ(lhs.returnNumber, rhs.returnNumber) << label;
    EXPECT_EQ(lhs.numberOfReturns, rhs.numberOfReturns) << label;
    EXPECT_EQ(lhs.heightAboveGround, rhs.heightAboveGround) << label;
}

} // unnamed namespace

TEST(PdgPmfFilter, ExecutesThroughThePlannerSelectedStandaloneGridLifecycle)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","max_window_size":3.0,"returns":[]},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    ASSERT_NE(region, pdg::NoResidentRegion);
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                  Id::ReturnNumber, Id::NumberOfReturns});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (std::size_t column = 0U; column < 5U; ++column)
        for (std::size_t row = 0U; row < 5U; ++row)
        {
            const pdal::PointId point =
                static_cast<pdal::PointId>(column * 5U + row);
            view->setField(Id::X, point, static_cast<double>(column));
            view->setField(Id::Y, point, static_cast<double>(row));
            view->setField(Id::Z, point, point == 12U ? 5.0 : 0.0);
            view->setField(Id::Classification, point, std::uint8_t{7U});
            view->setField(Id::ReturnNumber, point, std::uint8_t{1U});
            view->setField(Id::NumberOfReturns, point, std::uint8_t{1U});
        }

    constexpr std::size_t Budget = 64U * 1024U * 1024U;
    pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions, Budget,
                                                   64U);
    const std::array selectedRegions{region};
    scope.preflight(*table.layout(), view->size(), selectedRegions);
    pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
    EXPECT_EQ(context.schedule().tileCount, 1U);
    EXPECT_EQ(context.schedule().configuredLaneCount, 1U);
    EXPECT_GE(context.schedule().peakLaneBytes,
              pdg::PmfTiledDeviceFixedScratchBytes);

    context.enterBoundary(
        *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);
    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage(std::string(pdg::HybridPmfStage));
    ASSERT_NE(filter, nullptr);
    pdal::Options options;
    options.add("max_window_size", 3.0);
    options.add("pdg_resident_context", true);
    options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);
    static_cast<void>(filter->execute(table));
    context.enterBoundary(
        *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
        region,
        plan.summary().residencyBoundaries.at(spill).requiresFullPointRecord);

    for (pdal::PointId point = 0U; point < view->size(); ++point)
        EXPECT_EQ(view->getFieldAs<std::uint8_t>(Id::Classification, point),
                  point == 12U ? 1U : 2U)
            << point;
}

TEST(PdgPmfFilter,
     DeviceRasterProofMatchesLiteralHostBitsForFractionalLargeOriginSources)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t SourceCount = 256U;
    constexpr double OriginX = 1000000000000.25;
    constexpr double OriginY = -1000000000000.75;
    constexpr double Cell = 1.25;
    pdg::DimensionRegistry dimensions;
    auto staging = pdg::makeCudaPinnedMemoryResource();
    auto execution = pdg::makeCudaMemoryResource();
    pdg::CoordinateEncoding encoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::PointBatch host(SourceCount + 1U, encoding, dimensions, *staging);
    const pdg::DimensionId xId(pdg::StandardDimension::X);
    const pdg::DimensionId yId(pdg::StandardDimension::Y);
    const pdg::DimensionId zId(pdg::StandardDimension::Z);
    const pdg::DimensionId classificationId(
        pdg::StandardDimension::Classification);
    for (pdg::DimensionId id : {xId, yId, zId})
        host.materialize(id, pdg::DimensionType::Double);
    host.materialize(classificationId, pdg::DimensionType::Unsigned8);
    host.setSize(SourceCount + 1U);
    auto* x = host.data<double>(xId);
    auto* y = host.data<double>(yId);
    auto* z = host.data<double>(zId);
    std::fill_n(host.data<std::uint8_t>(classificationId), host.size(),
                std::uint8_t{7U});
    for (std::size_t source = 0U; source < SourceCount; ++source)
    {
        const std::size_t desiredCell = source * 3U;
        const double integerDelta =
            std::ceil(static_cast<double>(desiredCell) * Cell);
        x[source] = OriginX;
        y[source] = OriginY + integerDelta + 0.25;
        z[source] =
            source == 0U ? +0.0 : -17.0 + static_cast<double>(source) * 0.125;
    }
    x[SourceCount] = OriginX + 0.125;
    y[SourceCount] = OriginY + 0.125;
    z[SourceCount] = -0.0;

    pdg::PmfProgram program;
    program.cellSize = Cell;
    program.maxWindowSize = 3.0;
    const pdg::PmfRasterFrame pmfFrame = pdg::pmfRasterFrame(host, program);
    ASSERT_EQ(pmfFrame.columns, 1U);
    ASSERT_EQ(pmfFrame.rows, 766U);
    const pdg::RasterGridFrame frame{pmfFrame.minimumX,
                                     pmfFrame.minimumY,
                                     pmfFrame.rows,
                                     pmfFrame.columns,
                                     Cell,
                                     pdg::RasterGridFramePolicy::PmfV1};
    const std::size_t hostBudget =
        1U + frame.size() * (pdg::PmfTiledHostBytesPerCell + sizeof(double));
    const std::size_t deviceBudget =
        1U + frame.size() * pdg::PmfTiledDeviceProofBytesPerCell;

    std::vector<double> expectedRaster(
        frame.size(), std::numeric_limits<double>::quiet_NaN());
    std::vector<std::uint8_t> source(frame.size(), 0U);
    for (std::size_t point = 0U; point < host.size(); ++point)
    {
        const std::size_t column = static_cast<std::size_t>(
            std::floor(x[point] - frame.minimumX) / Cell);
        const std::size_t row = static_cast<std::size_t>(
            std::floor(y[point] - frame.minimumY) / Cell);
        ASSERT_LT(column, frame.columns);
        ASSERT_LT(row, frame.rows);
        const std::size_t cell = column * frame.rows + row;
        if (!source[cell] || z[point] < expectedRaster[cell])
        {
            source[cell] = 1U;
            expectedRaster[cell] = z[point];
        }
    }
    std::vector<std::size_t> sourceCells;
    for (std::size_t cell = 0U; cell < frame.size(); ++cell)
        if (source[cell])
            sourceCells.push_back(cell);
    ASSERT_EQ(sourceCells.size(), SourceCount);
    for (std::size_t cell = 0U; cell < frame.size(); ++cell)
    {
        if (source[cell])
            continue;
        const std::size_t column = cell / frame.rows;
        const std::size_t row = cell % frame.rows;
        bool found = false;
        double bestDistance = 0.0;
        std::uint64_t bestBits = 0U;
        for (const std::size_t sourceCell : sourceCells)
        {
            const std::size_t sourceColumn = sourceCell / frame.rows;
            const std::size_t sourceRow = sourceCell % frame.rows;
            const double centerX =
                frame.minimumX + (static_cast<double>(column) + 0.5) * Cell;
            const double centerY =
                frame.minimumY + (static_cast<double>(row) + 0.5) * Cell;
            const double sourceX =
                frame.minimumX +
                (static_cast<double>(sourceColumn) + 0.5) * Cell;
            const double sourceY =
                frame.minimumY + (static_cast<double>(sourceRow) + 0.5) * Cell;
            const double deltaX = centerX - sourceX;
            const double deltaY = centerY - sourceY;
            const double distance = deltaX * deltaX + deltaY * deltaY;
            const std::uint64_t bits =
                std::bit_cast<std::uint64_t>(expectedRaster[sourceCell]);
            if (!found || distance < bestDistance)
            {
                found = true;
                bestDistance = distance;
                bestBits = bits;
            }
            else if (distance == bestDistance)
                ASSERT_EQ(bits, bestBits) << cell << ':' << sourceCell;
        }
        ASSERT_TRUE(found) << cell;
        expectedRaster[cell] = std::bit_cast<double>(bestBits);
    }
    EXPECT_EQ(std::bit_cast<std::uint64_t>(expectedRaster.front()),
              std::bit_cast<std::uint64_t>(+0.0));

    pdg::PointBatch device(SourceCount + 1U, encoding, dimensions, *execution);
    for (pdg::DimensionId id : {xId, yId, zId})
        device.materialize(id, pdg::DimensionType::Double);
    device.setSize(SourceCount + 1U);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (pdg::DimensionId id : {xId, yId, zId})
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(id), host.rawData(id),
                                       host.size() * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));

    pdg::RasterGridProduct candidate(
        frame,
        {.haloCells = 1U,
         .deviceBytesPerExpandedCell = pdg::PmfTiledDeviceBytesPerCell,
         .deviceBackingCount = 2U,
         .deviceProofBytesPerCell = pdg::PmfTiledDeviceProofBytesPerCell,
         .hostBytesPerCell = pdg::PmfTiledHostBytesPerCell,
         .hostTileBytesPerExpandedCell = sizeof(double),
         .hostBackingCount = 2U,
         .baseDeviceBytes = 1U,
         .baseHostBytes = 1U,
         .deviceMemoryBudgetBytes = deviceBudget,
         .hostMemoryBudgetBytes = hostBudget},
        *staging, *execution);
    pdg::PmfRasterBuildFacts facts;
    pdg::buildPmfTiledRasterDevice(device, program, candidate, &facts);
    ASSERT_TRUE(candidate.deviceRasterBuild());
    ASSERT_TRUE(candidate.hasResidentDeviceBackings());
    EXPECT_TRUE(facts.deviceNativeSourceBuild);
    EXPECT_TRUE(facts.usedDeviceTieProof);
    EXPECT_EQ(facts.populatedCells, SourceCount);

    std::vector<double> actual(frame.size());
    PDG_CUDA_CHECK(cudaMemcpyAsync(
        actual.data(), candidate.currentDeviceBacking(),
        actual.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::size_t cell = 0U; cell < frame.size(); ++cell)
        EXPECT_EQ(std::bit_cast<std::uint64_t>(actual[cell]),
                  std::bit_cast<std::uint64_t>(expectedRaster[cell]))
            << cell;
}

TEST(PdgPmfFilter,
     DeviceRasterProofRejectsNonfiniteCoordinatesBeforePublishingOrMutation)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    pdg::DimensionRegistry dimensions;
    auto staging = pdg::makeCudaPinnedMemoryResource();
    auto execution = pdg::makeCudaMemoryResource();
    pdg::CoordinateEncoding encoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::PointBatch device(1U, encoding, dimensions, *execution);
    const pdg::DimensionId xId(pdg::StandardDimension::X);
    const pdg::DimensionId yId(pdg::StandardDimension::Y);
    const pdg::DimensionId zId(pdg::StandardDimension::Z);
    const pdg::DimensionId classificationId(
        pdg::StandardDimension::Classification);
    for (const auto [id, type] :
         {std::pair{xId, pdg::DimensionType::Double},
          std::pair{yId, pdg::DimensionType::Double},
          std::pair{zId, pdg::DimensionType::Double},
          std::pair{classificationId, pdg::DimensionType::Unsigned8}})
        device.materialize(id, type);
    device.setSize(1U);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    constexpr std::uint8_t Classification = 91U;
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(classificationId),
                                   &Classification, sizeof(Classification),
                                   cudaMemcpyHostToDevice, stream));

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    const std::array<std::array<double, 3U>, 5U> invalid{{
        {nan, 0.0, 0.0},
        {infinity, 0.0, 0.0},
        {0.0, -infinity, 0.0},
        {0.0, 0.0, nan},
        {0.0, 0.0, infinity},
    }};
    pdg::PmfProgram program;
    program.maxWindowSize = 3.0;
    const std::array coordinateIds{xId, yId, zId};
    for (const auto& values : invalid)
    {
        for (std::size_t coordinate = 0U; coordinate < coordinateIds.size();
             ++coordinate)
            PDG_CUDA_CHECK(
                cudaMemcpyAsync(device.rawData(coordinateIds[coordinate]),
                                values.data() + coordinate, sizeof(double),
                                cudaMemcpyHostToDevice, stream));
        pdg::RasterGridProduct product(
            {0.0, 0.0, 1U, 1U, 1.0, pdg::RasterGridFramePolicy::PmfV1},
            {.haloCells = 1U,
             .deviceBytesPerExpandedCell = pdg::PmfTiledDeviceBytesPerCell,
             .deviceBackingCount = 2U,
             .deviceProofBytesPerCell = pdg::PmfTiledDeviceProofBytesPerCell,
             .hostBytesPerCell = pdg::PmfTiledHostBytesPerCell,
             .hostTileBytesPerExpandedCell = sizeof(double),
             .hostBackingCount = 2U,
             .baseDeviceBytes = 1U,
             .baseHostBytes = 1U,
             .deviceMemoryBudgetBytes =
                 1U + pdg::PmfTiledDeviceProofBytesPerCell,
             .hostMemoryBudgetBytes = 1024U},
            *staging, *execution);
        EXPECT_THROW(pdg::buildPmfTiledRasterDevice(device, program, product),
                     std::out_of_range);
        EXPECT_EQ(product.rasterBuildCount(), 0U);
        EXPECT_FALSE(product.hasDeviceProofWorkspace());
        EXPECT_FALSE(product.hasResidentDeviceBackings());
        std::uint8_t actual = 0U;
        PDG_CUDA_CHECK(
            cudaMemcpyAsync(&actual, device.rawData(classificationId),
                            sizeof(actual), cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(actual, Classification);

        const double valid = 0.0;
        for (const pdg::DimensionId id : coordinateIds)
            PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(id), &valid,
                                           sizeof(valid),
                                           cudaMemcpyHostToDevice, stream));
        EXPECT_NO_THROW(
            pdg::buildPmfTiledRasterDevice(device, program, product));
        EXPECT_EQ(product.rasterBuildCount(), 1U);
        EXPECT_TRUE(product.deviceRasterBuild());
    }
}

TEST(PdgPmfFilter, DeviceRasterTieFailureDiscardsWorkspaceAndCanRetry)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    pdg::DimensionRegistry dimensions;
    auto staging = pdg::makeCudaPinnedMemoryResource();
    auto execution = pdg::makeCudaMemoryResource();
    pdg::CoordinateEncoding encoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::PointBatch device(2U, encoding, dimensions, *execution);
    const pdg::DimensionId xId(pdg::StandardDimension::X);
    const pdg::DimensionId yId(pdg::StandardDimension::Y);
    const pdg::DimensionId zId(pdg::StandardDimension::Z);
    for (pdg::DimensionId id : {xId, yId, zId})
        device.materialize(id, pdg::DimensionType::Double);
    device.setSize(2U);
    const std::array<double, 2U> x{0.0, 4.0};
    const std::array<double, 2U> y{0.0, 0.0};
    std::array<double, 2U> z{0.0, 1.0};
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (const auto [id, values] :
         {std::pair{xId, x.data()}, std::pair{yId, y.data()},
          std::pair{zId, static_cast<const double*>(z.data())}})
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(id), values,
                                       2U * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));

    constexpr std::size_t Cells = 5U;
    const std::size_t budget =
        1U + Cells * pdg::PmfTiledDeviceProofBytesPerCell;
    pdg::RasterGridProduct product(
        {0.0, 0.0, 1U, Cells, 1.0, pdg::RasterGridFramePolicy::PmfV1},
        {.haloCells = 1U,
         .deviceBytesPerExpandedCell = pdg::PmfTiledDeviceBytesPerCell,
         .deviceBackingCount = 2U,
         .deviceProofBytesPerCell = pdg::PmfTiledDeviceProofBytesPerCell,
         .hostBytesPerCell = pdg::PmfTiledHostBytesPerCell,
         .hostTileBytesPerExpandedCell = sizeof(double),
         .hostBackingCount = 2U,
         .baseDeviceBytes = 1U,
         .baseHostBytes = 1U,
         .deviceMemoryBudgetBytes = budget,
         .hostMemoryBudgetBytes = 1024U},
        *staging, *execution);
    pdg::PmfProgram program;
    program.maxWindowSize = 3.0;
    EXPECT_THROW(pdg::buildPmfTiledRasterDevice(device, program, product),
                 std::invalid_argument);
    EXPECT_EQ(product.rasterBuildCount(), 0U);
    EXPECT_FALSE(product.hasDeviceProofWorkspace());
    EXPECT_FALSE(product.hasResidentDeviceBackings());

    z[1] = 0.0;
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(zId), z.data(),
                                   z.size() * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    EXPECT_NO_THROW(pdg::buildPmfTiledRasterDevice(device, program, product));
    EXPECT_EQ(product.rasterBuildCount(), 1U);
    EXPECT_TRUE(product.deviceRasterBuild());
    EXPECT_TRUE(product.hasResidentDeviceBackings());
}

TEST(PdgPmfFilter, BoundedDeviceLaneRejectsDistinctRasterTieBeforeMutation)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    pdg::DimensionRegistry dimensions;
    auto execution = pdg::makeCudaMemoryResource();
    pdg::CoordinateEncoding encoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::PointBatch device(2U, encoding, dimensions, *execution);
    const pdg::DimensionId xId(pdg::StandardDimension::X);
    const pdg::DimensionId yId(pdg::StandardDimension::Y);
    const pdg::DimensionId zId(pdg::StandardDimension::Z);
    const pdg::DimensionId classificationId(
        pdg::StandardDimension::Classification);
    for (const auto [id, type] :
         {std::pair{xId, pdg::DimensionType::Double},
          std::pair{yId, pdg::DimensionType::Double},
          std::pair{zId, pdg::DimensionType::Double},
          std::pair{classificationId, pdg::DimensionType::Unsigned8}})
        device.materialize(id, type);
    device.setSize(2U);
    const std::array<double, 2U> x{0.0, 2.0};
    const std::array<double, 2U> y{0.0, 0.0};
    const std::array<double, 2U> z{0.0, 1.0};
    const std::array<std::uint8_t, 2U> classification{7U, 7U};
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (const auto [id, values] :
         {std::pair{xId, x.data()}, std::pair{yId, y.data()},
          std::pair{zId, z.data()}})
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(id), values,
                                       2U * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(classificationId),
                                   classification.data(),
                                   classification.size() * sizeof(std::uint8_t),
                                   cudaMemcpyHostToDevice, stream));

    pdg::PmfProgram program;
    program.maxWindowSize = 3.0;
    EXPECT_THROW(static_cast<void>(pdg::classifyPmf(device, program)),
                 std::invalid_argument);
    std::array<std::uint8_t, 2U> actual{};
    PDG_CUDA_CHECK(
        cudaMemcpyAsync(actual.data(), device.rawData(classificationId),
                        actual.size(), cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(actual, classification);
}

TEST(PdgPmfFilter, ResidentDeviceRasterMatchesHostBeyondTheGlobalCanvasCap)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","cell_size":0.6,"max_window_size":5.0,
       "returns":[]},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry planDimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, planDimensions);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    ASSERT_NE(region, pdg::NoResidentRegion);
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

    using pdal::Dimension::Id;
    constexpr std::size_t PointCount = 25U;
    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                  Id::ReturnNumber, Id::NumberOfReturns});
    pdal::PointViewPtr view(new pdal::PointView(table));

    pdg::DimensionRegistry expectedDimensions;
    pdg::HostMemoryResource expectedMemory;
    pdg::PointBatch expected(
        PointCount, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        expectedDimensions, expectedMemory);
    const pdg::DimensionId xId(pdg::StandardDimension::X);
    const pdg::DimensionId yId(pdg::StandardDimension::Y);
    const pdg::DimensionId zId(pdg::StandardDimension::Z);
    const pdg::DimensionId classificationId(
        pdg::StandardDimension::Classification);
    expected.materialize(xId, pdg::DimensionType::Double);
    expected.materialize(yId, pdg::DimensionType::Double);
    expected.materialize(zId, pdg::DimensionType::Double);
    expected.materialize(classificationId, pdg::DimensionType::Unsigned8);
    expected.setSize(PointCount);
    for (std::size_t point = 0U; point < PointCount; ++point)
    {
        std::size_t column = (point * 17U) % 65U;
        std::size_t row = (point * 29U) % 65U;
        if (point == 0U)
            column = row = 0U;
        else if (point == 1U)
            column = row = 64U;
        else if (point == 2U)
        {
            column = 0U;
            row = 64U;
        }
        else if (point == 3U)
        {
            column = 64U;
            row = 0U;
        }
        const double x = -12.25 + static_cast<double>(column) * 0.6;
        const double y = 3.125 + static_cast<double>(row) * 0.6;
        const double z = -0.0;
        view->setField(Id::X, point, x);
        view->setField(Id::Y, point, y);
        view->setField(Id::Z, point, z);
        view->setField(Id::Classification, point, std::uint8_t{8U});
        view->setField(Id::ReturnNumber, point, std::uint8_t{1U});
        view->setField(Id::NumberOfReturns, point, std::uint8_t{1U});
        expected.data<double>(xId)[point] = x;
        expected.data<double>(yId)[point] = y;
        expected.data<double>(zId)[point] = z;
        expected.data<std::uint8_t>(classificationId)[point] = 8U;
    }
    pdg::PmfProgram program;
    program.cellSize = 0.6;
    program.maxWindowSize = 5.0;
    const pdg::PmfResult expectedResult = pdg::classifyPmf(expected, program);
    ASSERT_EQ(expectedResult.rows, 65U);
    ASSERT_EQ(expectedResult.columns, 65U);
    ASSERT_GT(expectedResult.rows * expectedResult.columns,
              pdg::PmfExactDeviceMaximumRasterCells);

    const std::size_t base =
        plan.estimatedDeviceBytes(PointCount) +
        PointCount * pdg::PmfExactDeviceScratchBytesPerPoint;
    const std::size_t budget =
        base + 65U * 65U * pdg::PmfTiledDeviceBytesPerCell;
    pdal::pdg_detail::ResidentExecutionScope scope(plan, planDimensions, budget,
                                                   64U);
    const std::array selectedRegions{region};
    scope.preflight(*table.layout(), view->size(), selectedRegions);
    pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
    pdg::ExecutionObservationScope observation;
    context.enterBoundary(
        *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage(std::string(pdg::HybridPmfStage));
    ASSERT_NE(filter, nullptr);
    pdal::Options options;
    options.add("cell_size", 0.6);
    options.add("max_window_size", 5.0);
    options.add("pdg_resident_context", true);
    options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);
    static_cast<void>(filter->execute(table));
    context.enterBoundary(
        *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
        region,
        plan.summary().residencyBoundaries.at(spill).requiresFullPointRecord);

    EXPECT_EQ(context.schedule().itemCount, 65U * 65U);
    EXPECT_EQ(context.schedule().tileCount, 1U);
    EXPECT_EQ(context.schedule().activeLaneCount, 1U);
    EXPECT_TRUE(context.schedule().serialDependency);
    EXPECT_EQ(context.schedule().peakLaneBytes, budget);
    const pdg::ExecutionStatsSnapshot stats = observation.snapshot();
    const auto gridBuild = stats.totals.at(
        static_cast<std::size_t>(pdg::ExecutionEventKind::GridBuild));
    const auto rasterBuild = stats.totals.at(
        static_cast<std::size_t>(pdg::ExecutionEventKind::RasterBuild));
    const auto rasterUpload = stats.totals.at(
        static_cast<std::size_t>(pdg::ExecutionEventKind::RasterUpload));
    const auto rasterDownload = stats.totals.at(
        static_cast<std::size_t>(pdg::ExecutionEventKind::RasterDownload));
    EXPECT_EQ(gridBuild.count, 1U);
    EXPECT_EQ(gridBuild.bytes, context.schedule().peakLaneBytes);
    EXPECT_EQ(rasterBuild.count, 1U);
    EXPECT_EQ(rasterBuild.bytes, 65U * 65U * sizeof(double));
    EXPECT_EQ(rasterUpload.count, 0U);
    EXPECT_EQ(rasterUpload.bytes, 0U);
    EXPECT_EQ(rasterDownload.count, 0U);
    EXPECT_EQ(rasterDownload.bytes, 0U);
    for (std::size_t point = 0U; point < PointCount; ++point)
        EXPECT_EQ(view->getFieldAs<std::uint8_t>(Id::Classification, point),
                  expected.data<std::uint8_t>(classificationId)[point])
            << point;
}

TEST(PdgPmfFilter, ExhaustedRasterTileBudgetFailsBeforeClassificationMutation)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","max_window_size":3.0,"returns":[]},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    ASSERT_NE(region, pdg::NoResidentRegion);
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                  Id::ReturnNumber, Id::NumberOfReturns});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (pdal::PointId point = 0U; point < 2U; ++point)
    {
        view->setField(Id::X, point, static_cast<double>(point * 2U));
        view->setField(Id::Y, point, static_cast<double>(point * 2U));
        view->setField(Id::Z, point, static_cast<double>(point));
        view->setField(Id::Classification, point, std::uint8_t{7U});
        view->setField(Id::ReturnNumber, point, std::uint8_t{1U});
        view->setField(Id::NumberOfReturns, point, std::uint8_t{1U});
    }

    const std::size_t base =
        plan.estimatedDeviceBytes(view->size()) +
        view->size() * pdg::PmfExactDeviceScratchBytesPerPoint;
    const std::size_t budget = base + 8U * pdg::PmfTiledDeviceBytesPerCell;
    pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions, budget,
                                                   64U);
    const std::array selectedRegions{region};
    scope.preflight(*table.layout(), view->size(), selectedRegions);
    scope.context().enterBoundary(
        *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage(std::string(pdg::HybridPmfStage));
    ASSERT_NE(filter, nullptr);
    pdal::Options options;
    options.add("max_window_size", 3.0);
    options.add("pdg_resident_context", true);
    options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);
    EXPECT_THROW(static_cast<void>(filter->execute(table)), std::exception);
    EXPECT_NO_THROW(scope.context().enterBoundary(
        *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
        region,
        plan.summary().residencyBoundaries.at(spill).requiresFullPointRecord));
    for (pdal::PointId point = 0U; point < view->size(); ++point)
        EXPECT_EQ(view->getFieldAs<std::uint8_t>(Id::Classification, point),
                  7U);
}

TEST(PdgPmfFilter, AmbiguousNearestFillTieFallsBackToUpstreamBeforeMutation)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","max_window_size":3.0,"returns":[]},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    ASSERT_NE(region, pdg::NoResidentRegion);
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                  Id::ReturnNumber, Id::NumberOfReturns});
    pdal::PointViewPtr view(new pdal::PointView(table));
    const std::array<PmfPoint, 3U> points{{
        {0.0, 0.0, 0.0, 7U, 1U, 1U},
        {2.0, 0.0, 1.0, 7U, 1U, 1U},
        {64.0, 64.0, 0.0, 7U, 1U, 1U},
    }};
    const std::vector<PmfPoint> fixture(points.begin(), points.end());
    pdal::Options upstreamOptions;
    upstreamOptions.add("max_window_size", 3.0);
    const PmfSnapshot expected =
        runPmfFilter("filters.pmf", fixture, upstreamOptions);
    ASSERT_TRUE(expected.error.empty());
    for (pdal::PointId point = 0U; point < points.size(); ++point)
    {
        const PmfPoint& input = points[static_cast<std::size_t>(point)];
        view->setField(Id::X, point, input.x);
        view->setField(Id::Y, point, input.y);
        view->setField(Id::Z, point, input.z);
        view->setField(Id::Classification, point, input.classification);
        view->setField(Id::ReturnNumber, point, input.returnNumber);
        view->setField(Id::NumberOfReturns, point, input.numberOfReturns);
    }

    constexpr std::size_t Budget = 256U * 1024U;
    pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions, Budget,
                                                   64U);
    const std::array selectedRegions{region};
    scope.preflight(*table.layout(), view->size(), selectedRegions);
    pdg::ExecutionObservationScope observation;
    scope.context().enterBoundary(
        *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage(std::string(pdg::HybridPmfStage));
    ASSERT_NE(filter, nullptr);
    pdal::Options options;
    options.add("max_window_size", 3.0);
    options.add("pdg_resident_context", true);
    options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);
    EXPECT_NO_THROW(static_cast<void>(filter->execute(table)));
    EXPECT_NO_THROW(scope.context().enterBoundary(
        *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
        region,
        plan.summary().residencyBoundaries.at(spill).requiresFullPointRecord));
    EXPECT_EQ(scope.context().schedule().tileCount, 1U);
    const pdg::ExecutionStatsSnapshot stats = observation.snapshot();
    EXPECT_EQ(
        stats.totals
            .at(static_cast<std::size_t>(pdg::ExecutionEventKind::RasterBuild))
            .count,
        0U);
    EXPECT_EQ(
        stats.totals
            .at(static_cast<std::size_t>(pdg::ExecutionEventKind::RasterUpload))
            .count,
        0U);
    expectPmfSnapshotsEqual(expected, snapshotPmf(*view),
                            "ambiguous resident upstream fallback");
}

TEST(PdgPmfFilter, RasterProductLifetimeEndsAtItsRegionSpill)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","max_window_size":3.0,"returns":[]},
      {"type":"filters.hag_nn","count":2},
      {"type":"filters.pmf","max_window_size":3.0,"returns":[]},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    const std::size_t firstRegion = plan.stages().at(1U).residentRegion;
    const std::size_t secondRegion = plan.stages().at(3U).residentRegion;
    ASSERT_NE(firstRegion, pdg::NoResidentRegion);
    ASSERT_NE(secondRegion, pdg::NoResidentRegion);
    ASSERT_NE(firstRegion, secondRegion);

    const auto boundaryFor =
        [&](pdg::ResidencyBoundaryKind kind, std::size_t region)
    {
        for (std::size_t id = 0U;
             id < plan.summary().residencyBoundaries.size(); ++id)
        {
            const pdg::ResidencyBoundary& boundary =
                plan.summary().residencyBoundaries[id];
            const std::size_t deviceStage =
                kind == pdg::ResidencyBoundaryKind::Upload ? boundary.consumer
                                                           : boundary.producer;
            if (boundary.kind == kind &&
                plan.stages().at(deviceStage).residentRegion == region)
                return id;
        }
        return pdg::NoResidentRegion;
    };

    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                  Id::ReturnNumber, Id::NumberOfReturns});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (pdal::PointId point = 0U; point < 4U; ++point)
    {
        view->setField(Id::X, point, static_cast<double>(point / 2U));
        view->setField(Id::Y, point, static_cast<double>(point % 2U));
        view->setField(Id::Z, point, static_cast<double>(point));
        view->setField(Id::Classification, point, std::uint8_t{7U});
        view->setField(Id::ReturnNumber, point, std::uint8_t{1U});
        view->setField(Id::NumberOfReturns, point, std::uint8_t{1U});
    }

    constexpr std::size_t Budget = 1024U * 1024U;
    pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions, Budget,
                                                   64U);
    const std::array selectedRegions{firstRegion, secondRegion};
    scope.preflight(*table.layout(), view->size(), selectedRegions);
    pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
    pdg::ExecutionObservationScope observation;

    const auto executeRegion =
        [&](std::size_t region, const pdg::RasterGridFrame& frame)
    {
        const std::size_t upload =
            boundaryFor(pdg::ResidencyBoundaryKind::Upload, region);
        const std::size_t spill =
            boundaryFor(pdg::ResidencyBoundaryKind::Spill, region);
        ASSERT_NE(upload, pdg::NoResidentRegion);
        ASSERT_NE(spill, pdg::NoResidentRegion);
        context.enterBoundary(
            *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
            region,
            plan.summary()
                .residencyBoundaries.at(upload)
                .requiresFullPointRecord);
        context.beginDelegatedRegion(*view, region);
        EXPECT_NO_THROW(static_cast<void>(
            context.acquireRasterGridProduct(*view, region, frame, false)));
        context.endDelegatedRegion(*view, region);
        context.enterBoundary(
            *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
            region,
            plan.summary()
                .residencyBoundaries.at(spill)
                .requiresFullPointRecord);
    };

    executeRegion(firstRegion,
                  {0.0, 0.0, 2U, 2U, 1.0, pdg::RasterGridFramePolicy::PmfV1});
    view->setField(Id::X, 3U, 3.0);
    executeRegion(secondRegion,
                  {0.0, 0.0, 2U, 4U, 1.0, pdg::RasterGridFramePolicy::PmfV1});

    const pdg::ExecutionStatsSnapshot stats = observation.snapshot();
    EXPECT_EQ(
        stats.totals
            .at(static_cast<std::size_t>(pdg::ExecutionEventKind::GridBuild))
            .count,
        2U);
}

TEST(PdgPmfFilter, PmfCountTwoHagNnPmfExecutesEveryResidentBoundaryExactly)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","max_window_size":3.0},
      {"type":"filters.hag_nn","count":2},
      {"type":"filters.pmf","max_window_size":3.0},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    const std::array regions{plan.stages().at(1U).residentRegion,
                             plan.stages().at(2U).residentRegion,
                             plan.stages().at(3U).residentRegion};
    ASSERT_NE(regions[0], pdg::NoResidentRegion);
    ASSERT_NE(regions[1], pdg::NoResidentRegion);
    ASSERT_NE(regions[2], pdg::NoResidentRegion);
    ASSERT_NE(regions[0], regions[1]);
    ASSERT_NE(regions[1], regions[2]);

    std::vector<PmfPoint> points;
    points.reserve(25U);
    for (std::size_t column = 0U; column < 5U; ++column)
        for (std::size_t row = 0U; row < 5U; ++row)
        {
            const double x = static_cast<double>(column) +
                             static_cast<double>(row) * 0.013 +
                             static_cast<double>(column * row) * 0.0007;
            const double y = static_cast<double>(row) +
                             static_cast<double>(column) * 0.021 +
                             static_cast<double>(column * row) * 0.0003;
            const double z =
                (column == 2U && row == 2U) ? 5.0 : x * 0.01 + y * 0.005;
            points.push_back({x, y, z, 7U, 1U, 1U});
        }
    pdal::Options pmfOptions;
    pmfOptions.add("max_window_size", 3.0);
    const PmfSnapshot expected = runPmfHagPmfSequence(points, pmfOptions);
    ASSERT_TRUE(expected.error.empty()) << expected.error;
    ASSERT_EQ(expected.heightAboveGround.size(), points.size());

    pdal::PointTable table;
    pdal::PointViewPtr view = buildPmfHagInputView(points, table);
    constexpr std::size_t Budget = 64U * 1024U * 1024U;
    pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions, Budget,
                                                   64U);
    scope.preflight(*table.layout(), view->size(), regions);
    pdg::ExecutionObservationScope observation;

    const auto boundary = [&](pdg::ResidencyBoundaryKind kind,
                              std::size_t producer, std::size_t consumer)
    {
        const auto position =
            std::find_if(plan.summary().residencyBoundaries.begin(),
                         plan.summary().residencyBoundaries.end(),
                         [&](const pdg::ResidencyBoundary& item)
                         {
                             return item.kind == kind &&
                                    item.producer == producer &&
                                    item.consumer == consumer;
                         });
        EXPECT_NE(position, plan.summary().residencyBoundaries.end());
        return static_cast<std::size_t>(std::distance(
            plan.summary().residencyBoundaries.begin(), position));
    };
    const std::array boundaryIds{
        boundary(pdg::ResidencyBoundaryKind::Upload, 0U, 1U),
        boundary(pdg::ResidencyBoundaryKind::Spill, 1U, 2U),
        boundary(pdg::ResidencyBoundaryKind::Upload, 1U, 2U),
        boundary(pdg::ResidencyBoundaryKind::Spill, 2U, 3U),
        boundary(pdg::ResidencyBoundaryKind::Upload, 2U, 3U),
        boundary(pdg::ResidencyBoundaryKind::Spill, 3U, 4U)};
    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* input = &reader;
    const auto appendStage = [&](std::string_view name, pdal::Options options)
    {
        pdal::Stage* stage = factory.createStage(std::string(name));
        EXPECT_NE(stage, nullptr);
        if (!stage)
            throw std::runtime_error("resident sequence stage is unavailable");
        stage->setOptions(options);
        stage->setInput(*input);
        input = stage;
    };
    const auto appendBoundary =
        [&](std::size_t id, std::string_view kind, std::size_t region)
    {
        pdal::Options options;
        options.add("pdg_boundary_kind", std::string(kind));
        options.add("pdg_boundary_id", static_cast<std::uint64_t>(id));
        options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
        options.add(
            "pdg_requires_full_point_record",
            plan.summary().residencyBoundaries.at(id).requiresFullPointRecord);
        appendStage(pdg::HybridResidentBoundaryStage, options);
    };

    appendBoundary(boundaryIds[0], "upload", regions[0]);
    pdal::Options firstOptions = pmfOptions;
    firstOptions.add("pdg_resident_context", true);
    firstOptions.add("pdg_execution_region",
                     static_cast<std::uint64_t>(regions[0]));
    appendStage(pdg::HybridPmfStage, firstOptions);
    appendBoundary(boundaryIds[1], "spill", regions[0]);

    appendBoundary(boundaryIds[2], "upload", regions[1]);
    pdal::Options hagOptions;
    hagOptions.add("count", 2U);
    hagOptions.add("pdg_region_id",
                   static_cast<std::uint64_t>(regions[1] + 1U));
    hagOptions.add("pdg_region_neighbors", 2U);
    hagOptions.add("pdg_region_dimensions", 2U);
    hagOptions.add("pdg_region_last", true);
    hagOptions.add("pdg_resident_context", true);
    hagOptions.add("pdg_execution_region",
                   static_cast<std::uint64_t>(regions[1]));
    appendStage(pdg::HybridHagNnStage, hagOptions);
    appendBoundary(boundaryIds[3], "spill", regions[1]);

    appendBoundary(boundaryIds[4], "upload", regions[2]);
    pdal::Options secondOptions = pmfOptions;
    secondOptions.add("pdg_resident_context", true);
    secondOptions.add("pdg_execution_region",
                      static_cast<std::uint64_t>(regions[2]));
    appendStage(pdg::HybridPmfStage, secondOptions);
    appendBoundary(boundaryIds[5], "spill", regions[2]);

    input->prepare(table);
    const pdal::PointViewSet output = input->execute(table);
    ASSERT_EQ(output.size(), 1U);
    view = *output.begin();

    expectPmfSnapshotsEqual(expected, snapshotPmf(*view),
                            "PMF-HAG2-PMF resident boundary sequence");
    const pdg::ExecutionStatsSnapshot stats = observation.snapshot();
    const auto count = [&](pdg::ExecutionEventKind kind)
    { return stats.totals.at(static_cast<std::size_t>(kind)).count; };
    EXPECT_EQ(count(pdg::ExecutionEventKind::BoundaryUpload), 3U);
    EXPECT_EQ(count(pdg::ExecutionEventKind::BoundarySpill), 3U);
    EXPECT_EQ(count(pdg::ExecutionEventKind::DeviceRegionBegin), 3U);
    EXPECT_EQ(count(pdg::ExecutionEventKind::DeviceRegionEnd), 3U);
    EXPECT_EQ(count(pdg::ExecutionEventKind::GridBuild), 2U);
    EXPECT_EQ(count(pdg::ExecutionEventKind::IndexBuild), 1U);
}

TEST(PdgPmfFilter, AdjacentPmfStagesReuseOneExactRasterAllocation)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","cell_size":1.0,"max_window_size":3.0,
       "returns":"only"},
      {"type":"filters.pmf","cell_size":1.0,"max_window_size":5.0,
       "returns":"only","only_ground":true,"ground_class":9,
       "other_class":9},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    ASSERT_EQ(region, plan.stages().at(2U).residentRegion);
    ASSERT_NE(region, pdg::NoResidentRegion);
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

    const std::vector<PmfPoint> points = makePmfMorphologyFixture();
    pdal::Options firstOptions;
    firstOptions.add("cell_size", 1.0);
    firstOptions.add("max_window_size", 3.0);
    firstOptions.add("returns", "only");
    pdal::Options secondOptions;
    secondOptions.add("cell_size", 1.0);
    secondOptions.add("max_window_size", 5.0);
    secondOptions.add("returns", "only");
    secondOptions.add("only_ground", true);
    secondOptions.add("ground_class", 9);
    secondOptions.add("other_class", 9);
    const PmfSnapshot expected =
        runPmfFilterSequence(points, {firstOptions, secondOptions});
    ASSERT_TRUE(expected.error.empty()) << expected.error;

    pdal::PointTable table;
    pdal::PointViewPtr view = buildPmfInputView(points, table);
    const std::size_t pointCount = static_cast<std::size_t>(view->size());
    const std::size_t budget =
        plan.estimatedDeviceBytes(pointCount) +
        pointCount * pdg::PmfExactDeviceScratchBytesPerPoint +
        65U * 65U * pdg::PmfTiledDeviceBytesPerCell;
    pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions, budget,
                                                   64U);
    const std::array selectedRegions{region};
    scope.preflight(*table.layout(), view->size(), selectedRegions);
    pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
    pdg::ExecutionObservationScope observation;
    context.enterBoundary(
        *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* first = factory.createStage(std::string(pdg::HybridPmfStage));
    pdal::Stage* second = factory.createStage(std::string(pdg::HybridPmfStage));
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    firstOptions.add("pdg_resident_context", true);
    firstOptions.add("pdg_execution_region",
                     static_cast<std::uint64_t>(region));
    firstOptions.add("pdg_grid_reuse", false);
    firstOptions.add("pdg_grid_region_last", false);
    first->setOptions(firstOptions);
    first->setInput(reader);
    secondOptions.add("pdg_resident_context", true);
    secondOptions.add("pdg_execution_region",
                      static_cast<std::uint64_t>(region));
    secondOptions.add("pdg_grid_reuse", true);
    secondOptions.add("pdg_grid_region_last", true);
    second->setOptions(secondOptions);
    second->setInput(*first);
    second->prepare(table);
    static_cast<void>(second->execute(table));
    context.enterBoundary(
        *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
        region,
        plan.summary().residencyBoundaries.at(spill).requiresFullPointRecord);

    expectPmfSnapshotsEqual(expected, snapshotPmf(*view),
                            "adjacent resident pmf allocation reuse");
    const pdg::ExecutionStatsSnapshot stats = observation.snapshot();
    const auto gridBuild = stats.totals.at(
        static_cast<std::size_t>(pdg::ExecutionEventKind::GridBuild));
    const auto rasterBuild = stats.totals.at(
        static_cast<std::size_t>(pdg::ExecutionEventKind::RasterBuild));
    const auto rasterUpload = stats.totals.at(
        static_cast<std::size_t>(pdg::ExecutionEventKind::RasterUpload));
    const auto rasterDownload = stats.totals.at(
        static_cast<std::size_t>(pdg::ExecutionEventKind::RasterDownload));
    EXPECT_EQ(gridBuild.count, 1U);
    EXPECT_EQ(rasterBuild.count, 2U);
    EXPECT_EQ(rasterUpload.count, 0U);
    EXPECT_EQ(rasterDownload.count, 0U);
}

TEST(PdgPmfFilter, ThreeStagePmfChainKeepsIntermediateReuseExact)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","cell_size":1.0,"max_window_size":3.0,
       "returns":"only"},
      {"type":"filters.pmf","cell_size":1.0,"max_window_size":5.0,
       "returns":"only","only_ground":true,"ground_class":9,
       "other_class":9},
      {"type":"filters.pmf","cell_size":1.0,"max_window_size":7.0,
       "returns":"only","only_ground":true,"ground_class":9,
       "other_class":9},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    ASSERT_EQ(region, plan.stages().at(2U).residentRegion);
    ASSERT_EQ(region, plan.stages().at(3U).residentRegion);
    ASSERT_NE(region, pdg::NoResidentRegion);
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

    const std::vector<PmfPoint> points = makePmfMorphologyFixture();
    pdal::Options firstOptions;
    firstOptions.add("cell_size", 1.0);
    firstOptions.add("max_window_size", 3.0);
    firstOptions.add("returns", "only");
    pdal::Options secondOptions;
    secondOptions.add("cell_size", 1.0);
    secondOptions.add("max_window_size", 5.0);
    secondOptions.add("returns", "only");
    secondOptions.add("only_ground", true);
    secondOptions.add("ground_class", 9);
    secondOptions.add("other_class", 9);
    pdal::Options thirdOptions;
    thirdOptions.add("cell_size", 1.0);
    thirdOptions.add("max_window_size", 7.0);
    thirdOptions.add("returns", "only");
    thirdOptions.add("only_ground", true);
    thirdOptions.add("ground_class", 9);
    thirdOptions.add("other_class", 9);
    const PmfSnapshot expected = runPmfFilterSequence(
        points, {firstOptions, secondOptions, thirdOptions});
    ASSERT_TRUE(expected.error.empty()) << expected.error;

    pdal::PointTable table;
    pdal::PointViewPtr view = buildPmfInputView(points, table);
    const std::size_t pointCount = static_cast<std::size_t>(view->size());
    const std::size_t budget =
        plan.estimatedDeviceBytes(pointCount) +
        pointCount * pdg::PmfExactDeviceScratchBytesPerPoint +
        65U * 65U * pdg::PmfTiledDeviceBytesPerCell;
    pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions, budget,
                                                   64U);
    const std::array selectedRegions{region};
    scope.preflight(*table.layout(), view->size(), selectedRegions);
    pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
    pdg::ExecutionObservationScope observation;
    context.enterBoundary(
        *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* first = factory.createStage(std::string(pdg::HybridPmfStage));
    pdal::Stage* second = factory.createStage(std::string(pdg::HybridPmfStage));
    pdal::Stage* third = factory.createStage(std::string(pdg::HybridPmfStage));
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    ASSERT_NE(third, nullptr);
    firstOptions.add("pdg_resident_context", true);
    firstOptions.add("pdg_execution_region",
                     static_cast<std::uint64_t>(region));
    firstOptions.add("pdg_grid_reuse", false);
    firstOptions.add("pdg_grid_region_last", false);
    first->setOptions(firstOptions);
    first->setInput(reader);
    secondOptions.add("pdg_resident_context", true);
    secondOptions.add("pdg_execution_region",
                      static_cast<std::uint64_t>(region));
    secondOptions.add("pdg_grid_reuse", true);
    secondOptions.add("pdg_grid_region_last", false);
    second->setOptions(secondOptions);
    second->setInput(*first);
    thirdOptions.add("pdg_resident_context", true);
    thirdOptions.add("pdg_execution_region",
                     static_cast<std::uint64_t>(region));
    thirdOptions.add("pdg_grid_reuse", true);
    thirdOptions.add("pdg_grid_region_last", true);
    third->setOptions(thirdOptions);
    third->setInput(*second);
    third->prepare(table);
    static_cast<void>(third->execute(table));
    context.enterBoundary(
        *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
        region,
        plan.summary().residencyBoundaries.at(spill).requiresFullPointRecord);

    expectPmfSnapshotsEqual(expected, snapshotPmf(*view),
                            "three-stage resident pmf allocation reuse");
    const pdg::ExecutionStatsSnapshot stats = observation.snapshot();
    EXPECT_EQ(
        stats.totals
            .at(static_cast<std::size_t>(pdg::ExecutionEventKind::GridBuild))
            .count,
        1U);
    EXPECT_EQ(
        stats.totals
            .at(static_cast<std::size_t>(pdg::ExecutionEventKind::RasterBuild))
            .count,
        3U);
    EXPECT_EQ(
        stats.totals
            .at(static_cast<std::size_t>(pdg::ExecutionEventKind::RasterUpload))
            .count,
        0U);
    EXPECT_EQ(stats.totals
                  .at(static_cast<std::size_t>(
                      pdg::ExecutionEventKind::RasterDownload))
                  .count,
              0U);
    EXPECT_EQ(stats.totals
                  .at(static_cast<std::size_t>(
                      pdg::ExecutionEventKind::DeviceRegionBegin))
                  .count,
              1U);
    EXPECT_EQ(stats.totals
                  .at(static_cast<std::size_t>(
                      pdg::ExecutionEventKind::DeviceRegionEnd))
                  .count,
              1U);
}

TEST(PdgPmfFilter, ThreeStagePmfTieFallbackKeepsRegionOpenUntilFinalFallback)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","max_window_size":3.0,"returns":[]},
      {"type":"filters.pmf","max_window_size":3.0,"returns":[]},
      {"type":"filters.pmf","max_window_size":3.0,"returns":[]},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    ASSERT_EQ(region, plan.stages().at(2U).residentRegion);
    ASSERT_EQ(region, plan.stages().at(3U).residentRegion);
    ASSERT_NE(region, pdg::NoResidentRegion);
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

    const std::vector<PmfPoint> points{
        {0.0, 0.0, 0.0, 7U, 1U, 1U},
        {2.0, 0.0, 1.0, 7U, 1U, 1U},
        {64.0, 64.0, 0.0, 7U, 1U, 1U},
    };
    pdal::Options firstOptions;
    firstOptions.add("max_window_size", 3.0);
    pdal::Options secondOptions;
    secondOptions.add("max_window_size", 3.0);
    pdal::Options thirdOptions;
    thirdOptions.add("max_window_size", 3.0);
    const PmfSnapshot expected = runPmfFilterSequence(
        points, {firstOptions, secondOptions, thirdOptions});
    ASSERT_TRUE(expected.error.empty()) << expected.error;

    pdal::PointTable table;
    pdal::PointViewPtr view = buildPmfInputView(points, table);
    constexpr std::size_t Budget = 256U * 1024U;
    pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions, Budget,
                                                   64U);
    const std::array selectedRegions{region};
    scope.preflight(*table.layout(), view->size(), selectedRegions);
    pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
    pdg::ExecutionObservationScope observation;
    context.enterBoundary(
        *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* first = factory.createStage(std::string(pdg::HybridPmfStage));
    pdal::Stage* second = factory.createStage(std::string(pdg::HybridPmfStage));
    pdal::Stage* third = factory.createStage(std::string(pdg::HybridPmfStage));
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    ASSERT_NE(third, nullptr);
    firstOptions.add("pdg_resident_context", true);
    firstOptions.add("pdg_execution_region",
                     static_cast<std::uint64_t>(region));
    firstOptions.add("pdg_grid_reuse", false);
    firstOptions.add("pdg_grid_region_last", false);
    first->setOptions(firstOptions);
    first->setInput(reader);
    secondOptions.add("pdg_resident_context", true);
    secondOptions.add("pdg_execution_region",
                      static_cast<std::uint64_t>(region));
    secondOptions.add("pdg_grid_reuse", true);
    secondOptions.add("pdg_grid_region_last", false);
    second->setOptions(secondOptions);
    second->setInput(*first);
    thirdOptions.add("pdg_resident_context", true);
    thirdOptions.add("pdg_execution_region",
                     static_cast<std::uint64_t>(region));
    thirdOptions.add("pdg_grid_reuse", true);
    thirdOptions.add("pdg_grid_region_last", true);
    third->setOptions(thirdOptions);
    third->setInput(*second);
    third->prepare(table);
    EXPECT_NO_THROW(static_cast<void>(third->execute(table)));
    EXPECT_NO_THROW(context.enterBoundary(
        *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
        region,
        plan.summary().residencyBoundaries.at(spill).requiresFullPointRecord));

    expectPmfSnapshotsEqual(expected, snapshotPmf(*view),
                            "three-stage resident pmf tie fallback");
    const pdg::ExecutionStatsSnapshot stats = observation.snapshot();
    EXPECT_EQ(
        stats.totals
            .at(static_cast<std::size_t>(pdg::ExecutionEventKind::GridBuild))
            .count,
        1U);
    EXPECT_EQ(
        stats.totals
            .at(static_cast<std::size_t>(pdg::ExecutionEventKind::RasterBuild))
            .count,
        0U);
    EXPECT_EQ(stats.totals
                  .at(static_cast<std::size_t>(
                      pdg::ExecutionEventKind::DeviceRegionBegin))
                  .count,
              1U);
    EXPECT_EQ(stats.totals
                  .at(static_cast<std::size_t>(
                      pdg::ExecutionEventKind::DeviceRegionEnd))
                  .count,
              1U);
}

TEST(PdgPmfFilter, IntermediateReuseExceptionClosesRegionBeforeSpill)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","cell_size":1.0,"max_window_size":3.0,
       "returns":"only"},
      {"type":"filters.pmf","cell_size":1.0,"max_window_size":5.0,
       "returns":"only"},
      {"type":"filters.pmf","cell_size":1.0,"max_window_size":7.0,
       "returns":"only"},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    ASSERT_EQ(region, plan.stages().at(2U).residentRegion);
    ASSERT_EQ(region, plan.stages().at(3U).residentRegion);
    ASSERT_NE(region, pdg::NoResidentRegion);
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

    const std::vector<PmfPoint> points = makePmfMorphologyFixture();
    pdal::Options firstOptions;
    firstOptions.add("cell_size", 1.0);
    firstOptions.add("max_window_size", 3.0);
    firstOptions.add("returns", "only");
    const PmfSnapshot expected = runPmfFilterSequence(points, {firstOptions});
    ASSERT_TRUE(expected.error.empty()) << expected.error;
    pdal::Options secondOptions;
    secondOptions.add("cell_size", 1.0);
    secondOptions.add("max_window_size", 5.0);
    secondOptions.add("returns", "only");
    pdal::Options thirdOptions;
    thirdOptions.add("cell_size", 1.0);
    thirdOptions.add("max_window_size", 7.0);
    thirdOptions.add("returns", "only");

    pdal::PointTable table;
    pdal::PointViewPtr view = buildPmfInputView(points, table);
    const std::size_t pointCount = static_cast<std::size_t>(view->size());
    const std::size_t budget =
        plan.estimatedDeviceBytes(pointCount) +
        pointCount * pdg::PmfExactDeviceScratchBytesPerPoint +
        65U * 65U * pdg::PmfTiledDeviceBytesPerCell;
    pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions, budget,
                                                   64U);
    const std::array selectedRegions{region};
    scope.preflight(*table.layout(), view->size(), selectedRegions);
    pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
    pdg::ExecutionObservationScope observation;
    context.enterBoundary(
        *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* first = factory.createStage(std::string(pdg::HybridPmfStage));
    pdal::Stage* second = factory.createStage(std::string(pdg::HybridPmfStage));
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    firstOptions.add("pdg_resident_context", true);
    firstOptions.add("pdg_execution_region",
                     static_cast<std::uint64_t>(region));
    firstOptions.add("pdg_grid_reuse", false);
    firstOptions.add("pdg_grid_region_last", false);
    first->setOptions(firstOptions);
    first->setInput(reader);
    secondOptions.add("pdg_resident_context", true);
    secondOptions.add("pdg_execution_region",
                      static_cast<std::uint64_t>(region));
    secondOptions.add("pdg_grid_reuse", false);
    secondOptions.add("pdg_grid_region_last", false);
    second->setOptions(secondOptions);
    second->setInput(*first);
    second->prepare(table);
    EXPECT_THROW(static_cast<void>(second->execute(table)), std::logic_error);

    pdal::BufferReader finalReader;
    finalReader.addView(view);
    pdal::Stage* third = factory.createStage(std::string(pdg::HybridPmfStage));
    ASSERT_NE(third, nullptr);
    thirdOptions.add("pdg_resident_context", true);
    thirdOptions.add("pdg_execution_region",
                     static_cast<std::uint64_t>(region));
    thirdOptions.add("pdg_grid_reuse", true);
    thirdOptions.add("pdg_grid_region_last", true);
    third->setOptions(thirdOptions);
    third->setInput(finalReader);
    pdal::PointTable finalTable;
    finalTable.layout()->registerDims(
        {pdal::Dimension::Id::X, pdal::Dimension::Id::Y, pdal::Dimension::Id::Z,
         pdal::Dimension::Id::Classification, pdal::Dimension::Id::ReturnNumber,
         pdal::Dimension::Id::NumberOfReturns});
    third->prepare(finalTable);
    EXPECT_THROW(static_cast<void>(third->execute(finalTable)),
                 std::logic_error);
    EXPECT_NO_THROW(context.enterBoundary(
        *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
        region,
        plan.summary().residencyBoundaries.at(spill).requiresFullPointRecord));

    expectPmfSnapshotsEqual(expected, snapshotPmf(*view),
                            "intermediate resident pmf exception cleanup");
    const pdg::ExecutionStatsSnapshot stats = observation.snapshot();
    EXPECT_EQ(
        stats.totals
            .at(static_cast<std::size_t>(pdg::ExecutionEventKind::GridBuild))
            .count,
        1U);
    EXPECT_EQ(
        stats.totals
            .at(static_cast<std::size_t>(pdg::ExecutionEventKind::RasterBuild))
            .count,
        1U);
    EXPECT_EQ(stats.totals
                  .at(static_cast<std::size_t>(
                      pdg::ExecutionEventKind::DeviceRegionBegin))
                  .count,
              1U);
    EXPECT_EQ(stats.totals
                  .at(static_cast<std::size_t>(
                      pdg::ExecutionEventKind::DeviceRegionEnd))
                  .count,
              1U);
}

TEST(PdgPmfFilter,
     HybridResidentMatchesUpstreamAbove4096CellsAcrossTieAndMorphologyCases)
{
    try
    {
        if (pdg::cudaDevices().empty())
            GTEST_SKIP() << "no CUDA device is available";
    }
    catch (const pdg::CudaError&)
    {
        GTEST_SKIP() << "no CUDA device is available";
    }

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","cell_size":0.6,"max_window_size":6.0,
       "returns":"last","only_ground":true},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    ASSERT_NE(region, pdg::NoResidentRegion);
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

    const std::vector<PmfPoint> points = makePmfDifferentialFixture();
    ASSERT_GT(points.size(), pdg::PmfExactDeviceMaximumRasterCells);
    ASSERT_GT(points.size(), 0U);

    constexpr double CellSize = 0.6;
    constexpr double MaxWindow = 6.0;
    pdal::Options testOptions;
    testOptions.add("cell_size", CellSize);
    testOptions.add("max_window_size", MaxWindow);
    testOptions.add("returns", "last");
    testOptions.add("only_ground", true);

    const PmfSnapshot expected =
        runPmfFilter("filters.pmf", points, testOptions);
    ASSERT_TRUE(expected.error.empty()) << expected.error;
    EXPECT_EQ(points.size(), 6563U);
    ASSERT_GT(81U * 81U, pdg::PmfExactDeviceMaximumRasterCells);

    const std::size_t budget =
        plan.estimatedDeviceBytes(points.size()) +
        points.size() * pdg::PmfExactDeviceScratchBytesPerPoint +
        81U * 81U * pdg::PmfTiledDeviceBytesPerCell;
    std::size_t tileCount = 0U;
    std::size_t activeLaneCount = 0U;
    const PmfSnapshot candidate = runPmfHybridResident(
        plan, dimensions, points, testOptions, budget, region, upload, spill,
        &tileCount, &activeLaneCount);
    EXPECT_EQ(tileCount, 1U);
    EXPECT_GE(activeLaneCount, 1U);

    const auto runResident = [&]()
    {
        return runPmfHybridResident(plan, dimensions, points, testOptions,
                                    budget, region, upload, spill);
    };
    const PmfSnapshot candidateSecond = runResident();
    const PmfSnapshot candidateThird = runResident();
    const PmfSnapshot candidateFourth = runResident();
    const PmfSnapshot candidateFifth = runResident();

    expectPmfSnapshotsEqual(expected, candidate, "first resident run");
    expectPmfSnapshotsEqual(expected, candidateSecond, "resident repeat #2");
    expectPmfSnapshotsEqual(expected, candidateThird, "resident repeat #3");
    expectPmfSnapshotsEqual(expected, candidateFourth, "resident repeat #4");
    expectPmfSnapshotsEqual(expected, candidateFifth, "resident repeat #5");
}

TEST(PdgPmfFilter, TiledMorphologyMatchesUpstreamAcrossNontrivialSeams)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","cell_size":1.0,"max_window_size":5.0,
       "returns":"only"},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    ASSERT_NE(region, pdg::NoResidentRegion);
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

    const std::vector<PmfPoint> points = makePmfMorphologyFixture();
    ASSERT_EQ(points.size(), 65U * 65U);
    pdal::Options options;
    options.add("cell_size", 1.0);
    options.add("max_window_size", 5.0);
    options.add("returns", "only");
    const PmfSnapshot expected = runPmfFilter("filters.pmf", points, options);
    ASSERT_TRUE(expected.error.empty()) << expected.error;

    std::size_t tileCount = 0U;
    constexpr std::size_t Budget = 224U * 1024U;
    const PmfSnapshot candidate =
        runPmfHybridResident(plan, dimensions, points, options, Budget, region,
                             upload, spill, &tileCount);
    EXPECT_GT(tileCount, 1U);
    expectPmfSnapshotsEqual(expected, candidate, "nontrivial morphology seams");

    const std::size_t deviceBudget =
        plan.estimatedDeviceBytes(points.size()) +
        points.size() * pdg::PmfExactDeviceScratchBytesPerPoint +
        65U * 65U * pdg::PmfTiledDeviceBytesPerCell;
    std::size_t deviceTileCount = 0U;
    const PmfSnapshot deviceCandidate =
        runPmfHybridResident(plan, dimensions, points, options, deviceBudget,
                             region, upload, spill, &deviceTileCount);
    EXPECT_EQ(deviceTileCount, 1U);
    expectPmfSnapshotsEqual(expected, deviceCandidate,
                            "device-resident morphology phases");

    std::size_t belowProofTileCount = 0U;
    const PmfSnapshot belowProofCandidate = runPmfHybridResident(
        plan, dimensions, points, options, deviceBudget - 1U, region, upload,
        spill, &belowProofTileCount);
    EXPECT_GT(belowProofTileCount, 1U);
    expectPmfSnapshotsEqual(expected, belowProofCandidate,
                            "one-byte-below device proof host fallback");
}
