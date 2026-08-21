#include <pdg/Compaction.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace
{

TEST(Compaction, HostSelectionIsStableAcrossPhysicalTypes)
{
    constexpr std::size_t Count = 257;
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId signedValue =
        dimensions.registerCustom("SignedValue", pdg::DimensionType::Signed32)
            .id;
    const pdg::DimensionId doubleValue =
        dimensions.registerCustom("DoubleValue", pdg::DimensionType::Double).id;
    const std::vector<pdg::DimensionId> columns = {signedValue, doubleValue};
    pdg::HostMemoryResource memory;
    pdg::PointBatch source(
        Count, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    pdg::PointBatch destination(
        Count, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    for (pdg::PointBatch* batch : {&source, &destination})
        for (pdg::DimensionId id : columns)
            batch->materialize(id);
    source.setSize(Count);

    std::vector<std::uint8_t> keep(Count);
    std::vector<std::size_t> selected;
    for (std::size_t index = 0; index < Count; ++index)
    {
        source.data<std::int32_t>(signedValue)[index] =
            static_cast<std::int32_t>(index) - 100;
        source.data<double>(doubleValue)[index] =
            static_cast<double>(index) * 0.125;
        keep[index] =
            static_cast<std::uint8_t>(index % 7U == 1U || index % 11U == 3U);
        if (keep[index])
            selected.push_back(index);
    }

    ASSERT_EQ(pdg::compactPointBatch(source, destination, columns, keep.data()),
              selected.size());
    ASSERT_EQ(destination.size(), selected.size());
    for (std::size_t index = 0; index < selected.size(); ++index)
    {
        EXPECT_EQ(destination.data<std::int32_t>(signedValue)[index],
                  source.data<std::int32_t>(signedValue)[selected[index]]);
        EXPECT_EQ(destination.data<double>(doubleValue)[index],
                  source.data<double>(doubleValue)[selected[index]]);
    }
    EXPECT_THROW(static_cast<void>(pdg::compactPointBatch(
                     source, source, columns, keep.data())),
                 std::invalid_argument);
}

} // unnamed namespace
