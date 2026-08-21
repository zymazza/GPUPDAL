// D0277: local placement profiles, the calibration override, the shared
// calibration fit, and the synthetic calibration cloud.

#include <pdg/Dimension.hpp>
#include <pdg/LocalProfile.hpp>
#include <pdg/Memory.hpp>
#include <pdg/Placement.hpp>
#include <pdg/PlacementCalibration.hpp>
#include <pdg/SyntheticCloud.hpp>
#include <pdg/Version.hpp>

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace
{
class ScopedEnvironment
{
public:
    ScopedEnvironment(const char* name, const char* value) : m_name(name)
    {
        if (const char* prior = std::getenv(name))
            m_prior = prior;
        if (value)
            ::setenv(name, value, 1);
        else
            ::unsetenv(name);
    }
    ~ScopedEnvironment()
    {
        if (m_prior)
            ::setenv(m_name.c_str(), m_prior->c_str(), 1);
        else
            ::unsetenv(m_name.c_str());
    }

private:
    std::string m_name;
    std::optional<std::string> m_prior;
};

pdg::LocalPlacementProfile sampleProfile()
{
    pdg::LocalPlacementProfile profile;
    profile.id = "local-test";
    profile.createdUtc = "2026-08-16T00:00:00Z";
    profile.machine = {.deviceName = "Test GPU",
                       .computeCapability = "9.0",
                       .driverVersion = "1.2.3",
                       .cudaToolkitVersion = "13.3",
                       .cpuModel = "Test CPU",
                       .logicalCpus = 8,
                       .pdgVersion = std::string(pdg::Version)};
    profile.coefficients = {.cudaStartupNanoseconds = 1.0e8,
                            .hostToDeviceNanosecondsPerByte = 0.05,
                            .deviceToHostNanosecondsPerByte = 0.06,
                            .packingNanosecondsPerByte = 0.003,
                            .indexBuildNanosecondsPerByte = 0.008,
                            .synchronizationNanoseconds = 4000.0};
    profile.modelNames = {"lof", "normal-covariancefeatures-compose"};
    profile.stageModels = {
        {"lof",
         {.hostFixedNanoseconds = 0.0,
          .deviceFixedNanoseconds = 1.2e8,
          .hostNanosecondsPerPoint = 4700.0,
          .deviceNanosecondsPerPoint = 0.0,
          .minimumDevicePointCount = 100000U,
          .maximumDevicePointCount = 250000U,
          .calibrated = true}},
        {"normal-covariancefeatures-compose",
         {.hostFixedNanoseconds = 0.0,
          .deviceFixedNanoseconds = 6.0e7,
          .hostNanosecondsPerPoint = 2600.0,
          .deviceNanosecondsPerPoint = 0.0,
          .minimumDevicePointCount = 100000U,
          .maximumDevicePointCount = 250000U,
          .calibrated = true}}};
    for (std::size_t index = 0; index < profile.stageModels.size(); ++index)
        profile.stageModels[index].name = profile.modelNames[index];
    profile.summaries.push_back({.name = "lof",
                                 .minimumDevicePoints = 100000U,
                                 .maximumDevicePoints = 250000U,
                                 .hostSecondsAtMaximum = 1.2,
                                 .deviceSecondsAtMaximum = 0.23,
                                 .byteExact = true,
                                 .deviceWinsSomewhere = true});
    return profile;
}

std::filesystem::path temporaryFile(const char* name)
{
    return std::filesystem::temp_directory_path() /
           (std::string("pdg-local-profile-test-") + name + "-" +
            std::to_string(::getpid()));
}
} // namespace

TEST(LocalProfile, SerializeParseRoundTrip)
{
    const pdg::LocalPlacementProfile original = sampleProfile();
    const std::string text = pdg::serializeLocalProfile(
        original, R"({"note":"evidence is carried but not parsed"})");
    std::string error;
    const std::optional<pdg::LocalPlacementProfile> parsed =
        pdg::parseLocalProfile(text, error);
    ASSERT_TRUE(parsed) << error;
    EXPECT_EQ(parsed->id, original.id);
    EXPECT_EQ(parsed->createdUtc, original.createdUtc);
    EXPECT_EQ(parsed->machine, original.machine);
    EXPECT_EQ(parsed->coefficients.cudaStartupNanoseconds, 1.0e8);
    EXPECT_EQ(parsed->coefficients.synchronizationNanoseconds, 4000.0);
    ASSERT_EQ(parsed->stageModels.size(), 2U);
    const pdg::PlacementCalibrationProfile view = parsed->view();
    EXPECT_EQ(view.id, "local-test");
    EXPECT_EQ(view.device.name, "Test GPU");
    EXPECT_EQ(view.device.driverVersion, "1.2.3");
    const pdg::StagePlacementCost* lof =
        pdg::placementStageCalibration(view, "lof");
    ASSERT_NE(lof, nullptr);
    EXPECT_TRUE(lof->calibrated);
    EXPECT_EQ(lof->deviceFixedNanoseconds, 1.2e8);
    EXPECT_EQ(lof->hostNanosecondsPerPoint, 4700.0);
    EXPECT_EQ(lof->minimumDevicePointCount, 100000U);
    EXPECT_EQ(lof->maximumDevicePointCount, 250000U);
    EXPECT_EQ(pdg::placementStageCalibration(view, "missing"), nullptr);
    ASSERT_EQ(parsed->summaries.size(), 1U);
    EXPECT_EQ(parsed->summaries.front().name, "lof");
    EXPECT_TRUE(parsed->summaries.front().deviceWinsSomewhere);
}

TEST(LocalProfile, RejectsMalformedDocuments)
{
    const pdg::LocalPlacementProfile original = sampleProfile();
    const std::string good = pdg::serializeLocalProfile(original);
    std::string error;
    const auto rejects = [&](const std::string& text, const char* why)
    {
        std::string message;
        EXPECT_FALSE(pdg::parseLocalProfile(text, message)) << why;
        EXPECT_FALSE(message.empty()) << why;
    };
    rejects("not json", "not JSON");
    rejects("[]", "not an object");
    {
        std::string text = good;
        const std::size_t at = text.find("pdg-local-placement-profile-v1");
        ASSERT_NE(at, std::string::npos);
        text.replace(at, std::strlen("pdg-local-placement-profile-v1"),
                     "pdg-local-placement-profile-v0");
        rejects(text, "wrong schema");
    }
    {
        std::string text = good;
        const std::size_t at = text.find("\"cpu_model\"");
        ASSERT_NE(at, std::string::npos);
        text.replace(at, std::strlen("\"cpu_model\""), "\"cpu_modle\"");
        rejects(text, "missing machine field");
    }
    {
        std::string text = good;
        const std::size_t at = text.find("\"synchronization_ns\": 4000.0");
        ASSERT_NE(at, std::string::npos);
        text.replace(at, std::strlen("\"synchronization_ns\": 4000.0"),
                     "\"synchronization_ns\": -1.0");
        rejects(text, "negative coefficient");
    }
    {
        pdg::LocalPlacementProfile inverted = original;
        inverted.stageModels.front().cost.minimumDevicePointCount = 300000U;
        rejects(pdg::serializeLocalProfile(inverted), "min above max");
    }
    EXPECT_TRUE(pdg::parseLocalProfile(good, error)) << error;
}

TEST(LocalProfile, LookupStatusesFollowEnvironmentAndFile)
{
    const std::filesystem::path path = temporaryFile("lookup");
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    ScopedEnvironment disable(
        std::string(pdg::LocalProfileDisableEnvironment).c_str(), nullptr);
    ScopedEnvironment location(
        std::string(pdg::LocalProfilePathEnvironment).c_str(),
        path.string().c_str());
    EXPECT_EQ(pdg::defaultLocalProfilePath(), path);
    if (!pdg::cudaBackendCompiled())
    {
        EXPECT_EQ(pdg::lookupLocalProfile().status,
                  pdg::LocalProfileStatus::NotAttempted);
        return;
    }
    EXPECT_EQ(pdg::lookupLocalProfile().status,
              pdg::LocalProfileStatus::NotFound);
    {
        std::ofstream out(path);
        out << "{ this is not json";
    }
    EXPECT_EQ(pdg::lookupLocalProfile().status,
              pdg::LocalProfileStatus::Malformed);
    {
        std::ofstream out(path);
        out << pdg::serializeLocalProfile(sampleProfile());
    }
    const pdg::LocalProfileLookup lookup = pdg::lookupLocalProfile();
    // The sample profile is keyed to a fictitious machine: never applied.
    bool cudaUsable = true;
    try
    {
        static_cast<void>(pdg::currentLocalProfileMachineKey());
    }
    catch (const std::exception&)
    {
        cudaUsable = false;
    }
    if (cudaUsable)
    {
        EXPECT_EQ(lookup.status, pdg::LocalProfileStatus::MachineMismatch);
        EXPECT_NE(lookup.detail.find("device_name"), std::string::npos);
        ASSERT_TRUE(lookup.profile);
        EXPECT_EQ(lookup.profile->id, "local-test");
    }
    else
        EXPECT_EQ(lookup.status, pdg::LocalProfileStatus::NotAttempted);
    {
        ScopedEnvironment disabled(
            std::string(pdg::LocalProfileDisableEnvironment).c_str(), "1");
        EXPECT_EQ(pdg::lookupLocalProfile().status,
                  pdg::LocalProfileStatus::Disabled);
    }
    std::filesystem::remove(path, ignored);
}

TEST(LocalProfile, ReferenceKeyStillMapsToEmbeddedProfile)
{
    const pdg::PlacementDeviceKey reference{
        .name = "NVIDIA GeForce RTX 4090",
        .computeCapability = "8.9",
        .driverVersion = "610.43.03",
        .cudaToolkitVersion = "13.3"};
    ScopedEnvironment ignore(
        std::string(pdg::IgnoreBuiltinProfileTestEnvironment).c_str(),
        nullptr);
    ScopedEnvironment force(
        std::string(pdg::CalibrationForceDeviceEnvironment).c_str(), nullptr);
    const pdg::PlacementCalibrationProfile* profile =
        pdg::placementCalibrationFor(reference);
    ASSERT_NE(profile, nullptr);
    EXPECT_EQ(profile->id, "sm89-2026-08-17-r6-large-layouts");
    // An unknown device below every shipped/generic bound fails closed
    // (D0279: the generic fallback applies from compute capability 8.0 and
    // only inside its memory bound; a 7.0 key is outside).
    const pdg::PlacementDeviceKey other{.name = "Some Other GPU",
                                        .computeCapability = "7.0",
                                        .driverVersion = "610.43.03",
                                        .cudaToolkitVersion = "13.3"};
    if (pdg::loadedLocalProfile().status != pdg::LocalProfileStatus::Applied)
        EXPECT_EQ(pdg::placementCalibrationFor(other), nullptr);
    // With the shipped tiers disabled, nothing but the embedded profile and
    // a local profile can answer.
    ScopedEnvironment disabled(
        std::string(pdg::DisableShippedProfilesEnvironment).c_str(), "1");
    const pdg::PlacementDeviceKey modern{.name = "Some Other GPU",
                                         .computeCapability = "9.0",
                                         .driverVersion = "610.43.03",
                                         .cudaToolkitVersion = "13.3"};
    if (pdg::loadedLocalProfile().status != pdg::LocalProfileStatus::Applied)
        EXPECT_EQ(pdg::placementCalibrationFor(modern), nullptr);
}

TEST(LocalProfile, CalibrationOverrideOnlyUnderItsEnvironment)
{
    const pdg::PlacementDeviceKey key{.name = "Any GPU",
                                      .computeCapability = "8.6",
                                      .driverVersion = "0.0",
                                      .cudaToolkitVersion = "13.3"};
    {
        ScopedEnvironment off(
            std::string(pdg::CalibrationForceDeviceEnvironment).c_str(),
            nullptr);
        EXPECT_EQ(pdg::calibrationForcedDevicePlacementProfile(key), nullptr);
    }
    ScopedEnvironment on(
        std::string(pdg::CalibrationForceDeviceEnvironment).c_str(), "1");
    const pdg::PlacementCalibrationProfile* forced =
        pdg::calibrationForcedDevicePlacementProfile(key);
    ASSERT_NE(forced, nullptr);
    EXPECT_EQ(forced->id, "calibration-forced-device");
    // Every embedded model. B0280 folded D0279's formerly calibrate-only
    // normal-covariancefeatures-compose-extradims model into the refreshed
    // reference profile.
    std::vector<std::string_view> names = pdg::embeddedPlacementModelNames();
    EXPECT_EQ(forced->stageModels.size(), names.size());
    for (std::string_view name : names)
    {
        const pdg::StagePlacementCost* cost =
            pdg::placementStageCalibration(*forced, name);
        ASSERT_NE(cost, nullptr) << name;
        EXPECT_TRUE(cost->calibrated);
        EXPECT_GT(cost->hostNanosecondsPerPoint, cost->deviceNanosecondsPerPoint);
    }
    // A different key than the first one seen is not served (the override
    // is bound to one device per process).
    const pdg::PlacementDeviceKey another{.name = "Another GPU",
                                          .computeCapability = "8.6",
                                          .driverVersion = "0.0",
                                          .cudaToolkitVersion = "13.3"};
    EXPECT_EQ(pdg::calibrationForcedDevicePlacementProfile(another), nullptr);
}

TEST(DimensionRegistry, RedefineTypeKeepsIdAndName)
{
    pdg::DimensionRegistry registry;
    const pdg::DimensionDefinition* deviation = registry.find("Deviation");
    ASSERT_NE(deviation, nullptr);
    const pdg::DimensionId id = deviation->id;
    const pdg::DimensionType original = deviation->type;
    EXPECT_NE(original, pdg::DimensionType::Unsigned16);
    const pdg::DimensionDefinition& redefined =
        registry.redefineType(id, pdg::DimensionType::Unsigned16);
    EXPECT_EQ(redefined.id, id);
    EXPECT_EQ(redefined.name, "Deviation");
    EXPECT_EQ(redefined.type, pdg::DimensionType::Unsigned16);
    EXPECT_EQ(registry.find("Deviation")->type, pdg::DimensionType::Unsigned16);
    EXPECT_EQ(registry.find(id)->type, pdg::DimensionType::Unsigned16);
    // Another registry is untouched: the redefinition is per instance.
    pdg::DimensionRegistry other;
    EXPECT_EQ(other.find("Deviation")->type, original);
    EXPECT_THROW(static_cast<void>(registry.redefineType(
                     id, pdg::DimensionType::None)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(registry.redefineType(
                     pdg::DimensionId(65000U), pdg::DimensionType::Double)),
                 std::invalid_argument);
}

TEST(PlacementCalibration, ResidualFitSplitsBySign)
{
    // residual = -1e8 + 3000 * points  ->  host fixed 1e8, device 3000/pt
    const std::vector<pdg::PlacementResidualSample> samples{
        {250000.0, -1.0e8 + 3000.0 * 250000.0},
        {1000000.0, -1.0e8 + 3000.0 * 1000000.0},
        {4000000.0, -1.0e8 + 3000.0 * 4000000.0}};
    const pdg::StagePlacementCost fit = pdg::fitPlacementResidualModel(samples);
    EXPECT_TRUE(fit.calibrated);
    EXPECT_NEAR(fit.hostFixedNanoseconds, 1.0e8, 1.0);
    EXPECT_EQ(fit.deviceFixedNanoseconds, 0.0);
    EXPECT_EQ(fit.hostNanosecondsPerPoint, 0.0);
    EXPECT_NEAR(fit.deviceNanosecondsPerPoint, 3000.0, 1.0e-6);

    const std::vector<pdg::PlacementResidualSample> negative{
        {250000.0, 5.0e7 - 2000.0 * 250000.0},
        {1000000.0, 5.0e7 - 2000.0 * 1000000.0}};
    const pdg::StagePlacementCost host = pdg::fitPlacementResidualModel(negative);
    EXPECT_NEAR(host.deviceFixedNanoseconds, 5.0e7, 1.0);
    EXPECT_NEAR(host.hostNanosecondsPerPoint, 2000.0, 1.0e-6);

    const std::vector<pdg::PlacementResidualSample> single{{1000.0, -2000.0}};
    const pdg::StagePlacementCost slopeOnly =
        pdg::fitPlacementResidualModel(single);
    EXPECT_EQ(slopeOnly.hostFixedNanoseconds, 0.0);
    EXPECT_EQ(slopeOnly.hostNanosecondsPerPoint, 2.0);

    const std::vector<pdg::PlacementResidualSample> duplicate{{1000.0, 1.0},
                                                              {1000.0, 2.0}};
    EXPECT_THROW(static_cast<void>(pdg::fitPlacementResidualModel(duplicate)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(pdg::fitPlacementResidualModel({})),
                 std::invalid_argument);
}

TEST(SyntheticCloud, DeterministicBoundedAndWritable)
{
    const pdg::SyntheticCloudGenerator a({.points = 20000U});
    const pdg::SyntheticCloudGenerator b({.points = 20000U});
    EXPECT_GT(a.sideMetres(), 0.0);
    std::size_t ground = 0;
    for (std::size_t index = 0; index < 20000U; index += 97U)
    {
        const pdg::SyntheticPoint p = a.point(index);
        const pdg::SyntheticPoint q = b.point(index);
        EXPECT_EQ(p.x, q.x);
        EXPECT_EQ(p.y, q.y);
        EXPECT_EQ(p.z, q.z);
        EXPECT_GE(p.x, 0.0);
        EXPECT_LE(p.x, a.sideMetres());
        EXPECT_GE(p.y, 0.0);
        EXPECT_LE(p.y, a.sideMetres());
        EXPECT_GE(p.returnNumber, 1);
        EXPECT_LE(p.returnNumber, p.numberOfReturns);
        if (p.classification == 2)
            ++ground;
    }
    EXPECT_GT(ground, 0U);
    // A different seed changes the sample.
    const pdg::SyntheticCloudGenerator c({.points = 20000U, .seed = 7});
    EXPECT_NE(a.point(5).x, c.point(5).x);

    const std::filesystem::path path = temporaryFile("cloud.las");
    EXPECT_EQ(pdg::writeSyntheticLas(path, {.points = 12345U}), 12345U);
    std::ifstream in(path, std::ios::binary);
    std::vector<unsigned char> header(375);
    in.read(reinterpret_cast<char*>(header.data()), 375);
    ASSERT_EQ(in.gcount(), 375);
    EXPECT_EQ(std::memcmp(header.data(), "LASF", 4), 0);
    EXPECT_EQ(header[24], 1);
    EXPECT_EQ(header[25], 4);
    EXPECT_EQ(header[104], 7);
    EXPECT_EQ(header[105] | (header[106] << 8), 36);
    std::uint64_t count = 0;
    for (int i = 7; i >= 0; --i)
        count = (count << 8) | header[247 + static_cast<std::size_t>(i)];
    EXPECT_EQ(count, 12345U);
    std::error_code ignored;
    EXPECT_EQ(std::filesystem::file_size(path, ignored), 375U + 12345U * 36U);
    std::filesystem::remove(path, ignored);
}
