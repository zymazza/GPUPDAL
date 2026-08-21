#include <pdg/Hybrid.hpp>

#include <pdal/Filter.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/Streamable.hpp>

#include <memory>
#include <string>

namespace pdal
{

// Behaviorally derived from the pinned upstream filters/MergeFilter.cpp;
// see NOTICE. The persistent output view is part of the stage contract.
class PdgMergeFilter final : public Filter, public Streamable
{
public:
    std::string getName() const override
    {
        // Preserve upstream log and diagnostic attribution. Registration is
        // still under the private stage name below.
        return "filters.merge";
    }

private:
    void ready(PointTableRef table) override
    {
        SpatialReference srs = getSpatialReference();
        if (srs.empty())
            srs = table.anySpatialReference();
        m_view = std::make_shared<PointView>(table, srs);
    }

    bool processOne(PointRef&) override
    {
        return true;
    }

    PointViewSet run(PointViewPtr input) override
    {
        PointViewSet result;
        if (getSpatialReference().empty() &&
            input->spatialReference() != m_view->spatialReference())
        {
            log()->get(LogLevel::Warning)
                << "filters.merge: merging points with inconsistent spatial "
                   "references."
                << std::endl;
        }
        m_view->append(*input);
        result.insert(m_view);
        return result;
    }

    PointViewPtr m_view;
};

static StaticPluginInfo const s_info{std::string(pdg::HybridMergeStage),
                                     "Internal exact PDG PointView merge", ""};

CREATE_STATIC_STAGE(PdgMergeFilter, s_info)

} // namespace pdal
