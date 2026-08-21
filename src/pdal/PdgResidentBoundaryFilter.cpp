#include "PdgResidentContext.hpp"

#include <pdg/Hybrid.hpp>

#include <pdal/Filter.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace pdal
{
namespace
{

class PdgResidentBoundaryFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return std::string(pdg::HybridResidentBoundaryStage);
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("pdg_boundary_kind", "Internal resident boundary direction",
                 m_kind, std::string{})
            .setHidden();
        args.add("pdg_boundary_id", "Internal planner boundary identifier",
                 m_boundaryId, std::uint64_t{0})
            .setHidden();
        args.add("pdg_execution_region",
                 "Internal planner execution-region identifier", m_region,
                 std::uint64_t{0})
            .setHidden();
        args.add("pdg_requires_full_point_record",
                 "Internal full-record boundary marker", m_fullPointRecord,
                 false)
            .setHidden();
    }

    void prepared(PointTableRef) override
    {
        validate();
    }

    void prerun(const PointViewSet& views) override
    {
        validate();
        if (views.size() == 1U && (*views.begin())->empty())
            pdg_detail::requireResidentExecutionContext().enterBoundary(
                **views.begin(), m_boundaryIdSize, m_direction, m_regionSize,
                m_fullPointRecord);
    }

    PointViewSet run(PointViewPtr view) override
    {
        validate();
        if (!view)
            throwError("resident boundary received a null point view");
        pdg_detail::requireResidentExecutionContext().enterBoundary(
            *view, m_boundaryIdSize, m_direction, m_regionSize,
            m_fullPointRecord);
        PointViewSet result;
        result.insert(view);
        return result;
    }

    void validate()
    {
        if (m_kind == "upload")
            m_direction = pdg_detail::ResidentBoundaryDirection::Upload;
        else if (m_kind == "spill")
            m_direction = pdg_detail::ResidentBoundaryDirection::Spill;
        else
            throwError("resident boundary kind must be 'upload' or 'spill'");
        if (m_boundaryId > (std::numeric_limits<std::size_t>::max)() ||
            m_region > (std::numeric_limits<std::size_t>::max)())
            throwError("resident boundary identifier exceeds size_t");
        m_boundaryIdSize = static_cast<std::size_t>(m_boundaryId);
        m_regionSize = static_cast<std::size_t>(m_region);
    }

    std::string m_kind;
    std::uint64_t m_boundaryId = 0;
    std::uint64_t m_region = 0;
    bool m_fullPointRecord = false;
    pdg_detail::ResidentBoundaryDirection m_direction =
        pdg_detail::ResidentBoundaryDirection::Spill;
    std::size_t m_boundaryIdSize = 0;
    std::size_t m_regionSize = 0;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridResidentBoundaryStage),
    "Internal PDG resident spill/upload boundary", ""};

CREATE_STATIC_STAGE(PdgResidentBoundaryFilter, s_info)

} // unnamed namespace
} // namespace pdal
