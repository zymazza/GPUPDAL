#include <pdg/Hybrid.hpp>

#include "PdgNeighborhood.hpp"
#include "PdgResidentContext.hpp"

#include <io/BufferReader.hpp>
#include <pdal/Filter.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#if PDG_HAS_CUDA
#include <nvtx3/nvToolsExt.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace pdal
{
namespace
{
class ScopedLogLevel
{
public:
    ScopedLogLevel(const LogPtr& log, LogLevel level)
        : m_log(log), m_prior(log->getLevel())
    {
        m_log->setLevel(level);
    }

    ~ScopedLogLevel()
    {
        m_log->setLevel(m_prior);
    }

private:
    LogPtr m_log;
    LogLevel m_prior;
};

#if PDG_HAS_CUDA
class NvtxRange
{
public:
    explicit NvtxRange(const char* name)
    {
        nvtxRangePushA(name);
    }

    ~NvtxRange()
    {
        nvtxRangePop();
    }
};
#endif
} // unnamed namespace

// Exact compatibility wrapper for the pinned filters.hag_delaunay count-three
// implementation. The native lane consumes the planner-owned masked 2D kNN
// product and reproduces Delaunator's one-triangle seed ordering and PDAL's
// barycentric arithmetic. Wider triangulations and every data-dependent query
// ambiguity execute the original upstream stage.
class PdgHagDelaunayFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.hag_delaunay";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("count",
                 "The number of points to fetch to determine the ground "
                 "surface [Default: 10].",
                 m_count, point_count_t{10U});
        args.add("allow_extrapolation",
                 "Allow extrapolation outside local triangulations "
                 "[Default: true].",
                 m_allowExtrapolation, true);
        args.add("class", "Class to use for ground points. [Default: 2]",
                 m_groundClass, ClassLabel::Ground);
        args.add("pdg_region_id", "Internal resident-region identifier",
                 m_region.id, std::uint64_t{0U})
            .setHidden();
        args.add("pdg_region_neighbors",
                 "Internal resident-region neighbor envelope",
                 m_region.maximumNeighbors, std::uint32_t{0U})
            .setHidden();
        args.add("pdg_region_dimensions",
                 "Internal resident-region spatial dimensions",
                 m_region.dimensions, std::uint32_t{2U})
            .setHidden();
        args.add("pdg_region_reuse", "Internal resident-region reuse marker",
                 m_region.reuseExpected, false)
            .setHidden();
        args.add("pdg_region_last", "Internal resident-region final marker",
                 m_region.last, true)
            .setHidden();
        args.add("pdg_region_terminal_sink",
                 "Internal resident-region terminal-sink marker",
                 m_region.terminalSink, false)
            .setHidden();
        args.add("pdg_resident_context",
                 "Internal planner-owned resident execution marker",
                 m_residentContext, false)
            .setHidden();
        args.add("pdg_execution_region",
                 "Internal planner-owned execution region identifier",
                 m_executionRegion, std::uint64_t{0U})
            .setHidden();
    }

    void addDimensions(PointLayoutPtr layout) override
    {
        layout->registerDim(Dimension::Id::HeightAboveGround);
    }

    void prepared(PointTableRef table) override
    {
        if (m_count < 3U)
            throwError("Option 'count' must be at least 3.");
        if (!table.layout()->hasDim(Dimension::Id::Classification))
            throwError("Missing Classification dimension in input PointView.");
    }

    void prerun(const PointViewSet& views) override
    {
        if (!m_residentContext || views.size() != 1U ||
            !(*views.begin())->empty())
            return;
        pdg_detail::ResidentExecutionContext& context =
            pdg_detail::requireResidentExecutionContext();
        PointView& view = **views.begin();
        const std::size_t region = static_cast<std::size_t>(m_executionRegion);
        context.beginDelegatedRegion(view, region);
        if (m_region.last)
            context.endDelegatedRegion(view, region);
    }

    void runUpstreamFallback(PointView& view)
    {
        PointTable table;
        table.layout()->registerDims({Dimension::Id::X, Dimension::Id::Y,
                                      Dimension::Id::Z,
                                      Dimension::Id::Classification});
        PointViewPtr fallbackView(new PointView(table));
        BufferReader reader;
        reader.addView(fallbackView);
        StageFactory factory;
        Stage* filter = factory.createStage("filters.hag_delaunay");
        if (!filter)
            throwError("upstream filters.hag_delaunay fallback is unavailable");
        Options options;
        options.add("count", m_count);
        options.add("allow_extrapolation", m_allowExtrapolation);
        options.add("class", static_cast<unsigned int>(m_groundClass));
        filter->setOptions(options);
        filter->setLog(log());
        filter->setInput(reader);
        stopLogging();
        try
        {
            filter->prepare(table);
            for (PointId point = 0U; point < view.size(); ++point)
            {
                fallbackView->setField(
                    Dimension::Id::X, point,
                    view.getFieldAs<double>(Dimension::Id::X, point));
                fallbackView->setField(
                    Dimension::Id::Y, point,
                    view.getFieldAs<double>(Dimension::Id::Y, point));
                fallbackView->setField(
                    Dimension::Id::Z, point,
                    view.getFieldAs<double>(Dimension::Id::Z, point));
                fallbackView->setField(
                    Dimension::Id::Classification, point,
                    view.getFieldAs<std::uint8_t>(Dimension::Id::Classification,
                                                  point));
            }
            // Stage::execute emits a pipeline-level Debug line that the outer
            // pipeline already emitted. The pinned HAG Delaunay stage has no
            // Debug diagnostics of its own, so suppress only that nested
            // execution line while retaining its Info/Warning/Error output.
            ScopedLogLevel suppressNestedPipelineDebug(log(), LogLevel::Info);
            const PointViewSet output = filter->execute(table);
            if (output.size() != 1U || (*output.begin())->size() != view.size())
                throwError(
                    "upstream filters.hag_delaunay fallback changed view "
                    "topology");
            const PointView& result = **output.begin();
            for (PointId point = 0U; point < view.size(); ++point)
                view.setField(Dimension::Id::HeightAboveGround, point,
                              result.getFieldAs<double>(
                                  Dimension::Id::HeightAboveGround, point));
        }
        catch (...)
        {
            startLogging();
            throw;
        }
        startLogging();
    }

    void filter(PointView& view) override
    {
#if PDG_HAS_CUDA
        NvtxRange range("pdg::filters.hag_delaunay");
#endif
        const bool eligible = m_count == 3U;
        const pdg::HagDelaunayProgram program{
            static_cast<std::uint32_t>(m_count), m_allowExtrapolation,
            m_groundClass};
        if (m_residentContext)
        {
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            const auto executionRegion =
                static_cast<std::size_t>(m_executionRegion);
            context.beginDelegatedRegion(view, executionRegion);
            try
            {
                const bool usedCuda =
                    eligible && !view.empty() &&
                    !std::getenv(
                        "PDG_TEST_AUTOMATIC_HAG_DELAUNAY_DEVICE_DECLINE") &&
                    pdg_detail::tryCudaHagDelaunayColumn(
                        view, program, m_region, /*requireCuda=*/true);
                if (!usedCuda)
                {
                    runUpstreamFallback(view);
                    if (!m_region.last && !view.empty() &&
                        !pdg_detail::retainCudaHagHostColumn(
                            view, program.count, m_region,
                            /*requireCuda=*/true))
                        throwError(
                            "planner-selected HAG Delaunay host repair could "
                            "not retain its resident output column");
                }
                if (m_region.last)
                    context.endDelegatedRegion(view, executionRegion);
            }
            catch (...)
            {
                context.endDelegatedRegion(view, executionRegion);
                throw;
            }
            return;
        }

        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID"));
        const bool usedCuda = requestCuda && eligible &&
                              pdg_detail::tryCudaHagDelaunayColumn(
                                  view, program, m_region, requireCuda);
        if (requireCuda && !usedCuda)
            throwError(
                "required exact CUDA hybrid hag_delaunay path was not used");
        if (!usedCuda)
            runUpstreamFallback(view);
    }

    point_count_t m_count = 10U;
    bool m_allowExtrapolation = true;
    std::uint8_t m_groundClass = ClassLabel::Ground;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0U;
    pdg_detail::CudaNeighborhoodRegion m_region;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridHagDelaunayStage),
    "Internal exact PDG shared-index Delaunay height-above-ground filter", ""};

CREATE_STATIC_STAGE(PdgHagDelaunayFilter, s_info)

} // namespace pdal
