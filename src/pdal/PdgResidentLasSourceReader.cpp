#include <pdg/Hybrid.hpp>

#include <pdal/Reader.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#include <string>

namespace pdal
{
namespace
{

// Execution-only source for the bounded direct-LAS resident endpoint. The CLI
// has already mapped and validated the source and the neighborhood product
// hydrates its planner-owned columns from that mapping. This reader creates
// only stable point identity; the caller-supplied sparse table owns the exact
// original byte fields and on-demand XYZ compatibility view.
class PdgResidentLasSourceReader final : public Reader
{
public:
    std::string getName() const override
    {
        return std::string(pdg::HybridResidentLasSourceStage);
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("return_number",
                 "Internal mapped LAS ReturnNumber compatibility field",
                 m_returnNumber, false)
            .setHidden();
    }

    void addDimensions(PointLayoutPtr layout) override
    {
        layout->registerDims(
            {Dimension::Id::X, Dimension::Id::Y, Dimension::Id::Z,
             Dimension::Id::Classification, Dimension::Id::UserData});
        if (m_returnNumber)
            layout->registerDim(Dimension::Id::ReturnNumber);
    }

    point_count_t read(PointViewPtr view, point_count_t count) override
    {
        if (!view)
            throwError("direct resident LAS source received a null view");
        for (PointId point = 0; point < count; ++point)
            static_cast<void>(view->point(point));
        return count;
    }

    bool m_returnNumber = false;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridResidentLasSourceStage),
    "Internal mapped LAS source for a planner-owned resident region", ""};

CREATE_STATIC_STAGE(PdgResidentLasSourceReader, s_info)

} // unnamed namespace
} // namespace pdal
