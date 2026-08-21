#include <pdg/Dimension.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>

TEST(DimensionRegistry, MirrorsGeneratedStandardCatalog)
{
    pdg::DimensionRegistry registry;
    const auto generatedCount =
        static_cast<std::uint16_t>(pdg::StandardDimension::Count) - 1U;
    EXPECT_EQ(registry.standardCount(), generatedCount);
    EXPECT_EQ(registry.customCount(), 0U);

    const auto& x = registry.require("x");
    EXPECT_EQ(x.id, pdg::DimensionId(pdg::StandardDimension::X));
    EXPECT_EQ(x.type, pdg::DimensionType::Double);
    EXPECT_TRUE(x.standard);

    EXPECT_EQ(registry.require("class").id,
              pdg::DimensionId(pdg::StandardDimension::Classification));
    EXPECT_EQ(registry.require("CLS").id,
              pdg::DimensionId(pdg::StandardDimension::Classification));
    EXPECT_EQ(registry.require("user_data").id,
              pdg::DimensionId(pdg::StandardDimension::UserData));
}

TEST(DimensionRegistry, RegistersStableTypedCustomDimensions)
{
    pdg::DimensionRegistry registry;
    const auto& custom =
        registry.registerCustom("CanopyScore", pdg::DimensionType::Float);
    EXPECT_EQ(custom.id.value(), pdg::CustomDimensionBase);
    EXPECT_EQ(custom.type, pdg::DimensionType::Float);
    EXPECT_FALSE(custom.standard);
    EXPECT_EQ(registry.require("canopyscore").id, custom.id);
    EXPECT_EQ(registry.require(custom.id).name, "CanopyScore");
    EXPECT_EQ(registry.customCount(), 1U);

    EXPECT_THROW(
        registry.registerCustom("CanopyScore", pdg::DimensionType::Float),
        std::invalid_argument);
    EXPECT_THROW(registry.registerCustom("9invalid", pdg::DimensionType::Float),
                 std::invalid_argument);
    EXPECT_THROW(registry.registerCustom("NoType", pdg::DimensionType::None),
                 std::invalid_argument);
}

TEST(DimensionType, EncodesByteWidths)
{
    EXPECT_EQ(pdg::dimensionTypeSize(pdg::DimensionType::Unsigned8), 1U);
    EXPECT_EQ(pdg::dimensionTypeSize(pdg::DimensionType::Signed32), 4U);
    EXPECT_EQ(pdg::dimensionTypeSize(pdg::DimensionType::Double), 8U);
    EXPECT_TRUE(pdg::isFloating(pdg::DimensionType::Float));
    EXPECT_FALSE(pdg::isFloating(pdg::DimensionType::Unsigned32));
}
