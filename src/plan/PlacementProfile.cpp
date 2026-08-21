#include <pdg/Placement.hpp>

#include <pdg/LocalProfile.hpp>
#include <pdg/Plan.hpp>

#include <array>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace pdg
{
namespace
{
constexpr std::array<PlacementStageCalibration, 42> Sm89StageModels{{
    {"fused-point-program",
     {.hostFixedNanoseconds = 3860.000000477,
      .hostNanosecondsPerPoint = 21.972376347771,
      .calibrated = true}},
    // Fitted by pdg_placement_audit --suggest-models over the six clean
    // forced ordered-CUDA rows ordered-cal-{1m..22m}-e48e67ad6; complete
    // pinned-PDAL process versus the ordered decode/predicate/pack sink,
    // valid only from the smallest measured cardinality.
    {"ordered-point-program",
     {.hostFixedNanoseconds = 9733181.294258117676,
      .hostNanosecondsPerPoint = 366.606724345247,
      .minimumDevicePointCount = 1000000U,
      .calibrated = true}},
    {"simple-ferry",
     {.deviceNanosecondsPerPoint = 1.015529713932, .calibrated = true}},
    {"colorinterp-explicit",
     {.deviceFixedNanoseconds = 3095544.799632192,
      .deviceNanosecondsPerPoint = 58.056861445934,
      .calibrated = true}},
    {"colorinterp-auto",
     {.deviceFixedNanoseconds = 34520948.77750188,
      .deviceNanosecondsPerPoint = 23.583364917390,
      .calibrated = true}},
    {"transformation",
     {.deviceFixedNanoseconds = 9690019.501705825,
      .hostNanosecondsPerPoint = 15.615418991432,
      .calibrated = true}},
    {"iqr",
     {.deviceFixedNanoseconds = 14814790.828565225,
      .deviceNanosecondsPerPoint = 0.273015050315,
      .calibrated = true}},
    {"mad",
     {.deviceFixedNanoseconds = 18709769.73106096,
      .hostNanosecondsPerPoint = 0.503965236615,
      .calibrated = true}},
    {"sort",
     {.hostFixedNanoseconds = 72811793.5560519,
      .hostNanosecondsPerPoint = 11.446822573828,
      .calibrated = true}},
    // B0232 exact direct-LAS global-order ladder. The host slope is the
    // smallest observed pinned-PDAL ns/point among the measured winning rows;
    // the device slope is the largest current direct-route residual after
    // subtracting the planner-owned cold start, 8 B/point upload, 8 B/point
    // permutation download, and two synchronizations. This deliberately
    // overpredicts every measured device row and never overpredicts host work.
    // RuntimePlacement admits it only for the literal sort(Z,ASC,NORMAL),
    // extra_dims=all, uncompressed format-7/36-byte route with exact scratch.
    // The 550K public-path direction is unresolved; the clear 600K win is the
    // measured floor, and 16M is the largest calibrated exact row.
    {"sort-direct-compose",
     {.hostFixedNanoseconds = 0.0,
      .deviceFixedNanoseconds = 0.0,
      .hostNanosecondsPerPoint = 793.782365,
      .deviceNanosecondsPerPoint = 58.74206627340533,
      .minimumDevicePointCount = 600000U,
      .maximumDevicePointCount = 16000000U,
      .calibrated = true}},
    // B0234 exact option-free public-path skewness ladder. Host is the smallest
    // observed pinned-PDAL ns/point among the measured winning rows, while
    // device is the largest automatic-route residual after subtracting the
    // planner-owned cold start, two 8 B/point permutation boundaries, and two
    // synchronizations. The 400K public row wins by only 6.51% and fails
    // D0190's predeclared 10% gate; 450K is the measured automatic floor and
    // 16M the measured explicit-route ceiling.
    {"skewness-direct-compose",
     {.hostFixedNanoseconds = 0.0,
      .deviceFixedNanoseconds = 0.0,
      .hostNanosecondsPerPoint = 953.993317777778,
      .deviceNanosecondsPerPoint = 290.879214148791,
      .minimumDevicePointCount = 450000U,
      .maximumDevicePointCount = 16000000U,
      .calibrated = true}},
    // B0235 exact mapped-source/direct-extra-double HAG-NN(count=1) ladder.
    // Host is the smallest pinned-PDAL ns/point among the measured winning
    // rows. Device is the largest current direct-route residual after
    // subtracting the planner-owned cold start, two synchronizations,
    // 25 B/point upload, 8 B/point spill, zero packing, and the full
    // 112 B/point shared-index build. Both are conservative at every admitted
    // row. The 400,002 row wins by only 10.43%; the harder 450K floor keeps a
    // safer 23.14% measured margin. The 16,000,002 row is the largest measured
    // exact input and therefore the hard ceiling.
    {"hag-nn-count1-direct-compose",
     {.hostFixedNanoseconds = 0.0,
      .deviceFixedNanoseconds = 0.0,
      .hostNanosecondsPerPoint = 923.5390448076193,
      .deviceNanosecondsPerPoint = 229.06845240043793,
      .minimumDevicePointCount = 450000U,
      .maximumDevicePointCount = 16000002U,
      .calibrated = true}},
    // B0236 exact mapped-source/direct-extra-double HAG-Delaunay(count=3)
    // ladder. Host is the smallest pinned-PDAL ns/point among the measured
    // winning rows. Device is the largest current direct-route residual after
    // subtracting cold start, two synchronizations, 25 B/point upload,
    // 8 B/point spill, zero packing, and the 112 B/point shared-index build.
    // The final proof-bearing public 450K row wins by only 9.34% and misses
    // the predeclared 10% gate. The exact 500,001-point row is therefore the
    // hard automatic floor. The 16,000,002 row is the largest measured exact
    // input and remains the ceiling.
    {"hag-delaunay-count3-direct-compose",
     {.hostFixedNanoseconds = 0.0,
      .deviceFixedNanoseconds = 0.0,
      .hostNanosecondsPerPoint = 823.250610498779,
      .deviceNanosecondsPerPoint = 185.91405462266016,
      .minimumDevicePointCount = 500001U,
      .maximumDevicePointCount = 16000002U,
      .calibrated = true}},
    {"morton",
     {.hostFixedNanoseconds = 5486827.052324653,
      .hostNanosecondsPerPoint = 233.723750703360,
      .calibrated = true}},
    {"groupby",
     {.deviceFixedNanoseconds = 70215040.87074247,
      .deviceNanosecondsPerPoint = 7.687851327383,
      .calibrated = true}},
    {"returns-merge",
     {.deviceFixedNanoseconds = 13003910.340386555,
      .hostNanosecondsPerPoint = 7.612194578857,
      .calibrated = true}},
    {"divider",
     {.deviceFixedNanoseconds = 31050475.306348428,
      .deviceNanosecondsPerPoint = 1.686108156052,
      .calibrated = true}},
    {"splitter",
     {.deviceFixedNanoseconds = 39175859.04089631,
      .deviceNanosecondsPerPoint = 3.220492080382,
      .calibrated = true}},
    {"stats",
     {.deviceFixedNanoseconds = 112386157.84642076,
      .deviceNanosecondsPerPoint = 279.967931785780,
      .calibrated = true}},
    {"info",
     {.deviceFixedNanoseconds = 4438911.466160834,
      .deviceNanosecondsPerPoint = 27.882203133235,
      .calibrated = true}},
    {"expressionstats-1",
     {.hostFixedNanoseconds = 51773318.62461388,
      .hostNanosecondsPerPoint = 83.171651907949,
      .calibrated = true}},
    {"expressionstats-2",
     {.hostFixedNanoseconds = 34643877.12461382,
      .hostNanosecondsPerPoint = 190.728398657949,
      .calibrated = true}},
    {"expressionstats-3",
     {.hostFixedNanoseconds = 33971180.62461439,
      .hostNanosecondsPerPoint = 278.358706657949,
      .calibrated = true}},
    {"approximatecoplanar",
     {.deviceFixedNanoseconds = 173014060.03622097,
      .hostNanosecondsPerPoint = 4256.005557780218,
      .deviceNanosecondsPerPoint = 1019.5148811130455,
      .minimumDevicePointCount = 131072U,
      .calibrated = true}},
    // B0096 exact direct-output composition ladder. The residual is fitted
    // to the six paired current-binary device-minus-host medians after the
    // planner-owned 239.214 ms cold start, 25 B/point upload, 2 B/point
    // spill, 72 B/point packing, 112 B/point shared-index build, and two
    // synchronizations. RuntimePlacement admits it only for the measured
    // knn=8 -> Coplanar=>UserData shape under required direct output.
    {"approximatecoplanar-direct-compose",
     {.hostFixedNanoseconds = 87387779.74090123,
      .hostNanosecondsPerPoint = 3385.876023673574,
      .minimumDevicePointCount = 250000U,
      .maximumDevicePointCount = 16000000U,
      .calibrated = true}},
    {"lof",
     {.deviceFixedNanoseconds = 121343390.77881624,
      .hostNanosecondsPerPoint = 9526.749844570424,
      .deviceNanosecondsPerPoint = 1078.1727489064672,
      .minimumDevicePointCount = 250000U,
      .calibrated = true}},
    {"normal",
     {.deviceFixedNanoseconds = 174287321.01746765,
      .hostNanosecondsPerPoint = 4539.955309098323,
      .deviceNanosecondsPerPoint = 1083.9846298206462,
      .minimumDevicePointCount = 250000U,
      .calibrated = true}},
    {"eigenvalues",
     {.hostFixedNanoseconds = 32201923.27978566,
      .deviceFixedNanoseconds = 180968776.82526013,
      .hostNanosecondsPerPoint = 4516.448544010281,
      .deviceNanosecondsPerPoint = 1060.40514681755,
      .minimumDevicePointCount = 250000U,
      .calibrated = true}},
    {"covariancefeatures",
     {.deviceFixedNanoseconds = 154275342.5721225,
      .hostNanosecondsPerPoint = 4524.570230601391,
      .deviceNanosecondsPerPoint = 1086.5715638581526,
      .minimumDevicePointCount = 250000U,
      .calibrated = true}},
    // B0074 exact explicit-resident ladder. RuntimePlacement admits this
    // residual only for the measured same-k three-consumer and point-program
    // shape; unrelated combinations retain mixed-model rejection.
    {"eigen-family-compose",
     {.hostFixedNanoseconds = 120004600.90127563,
      .deviceFixedNanoseconds = 155357777.57434177,
      .hostNanosecondsPerPoint = 14659.538986578957,
      .deviceNanosecondsPerPoint = 1234.7451449457265,
      .minimumDevicePointCount = 250000U,
      .maximumDevicePointCount = 16000000U,
      .calibrated = true}},
    {"nndistance",
     {.hostFixedNanoseconds = 2330246.210784518,
      .hostNanosecondsPerPoint = 4396.264753367529,
      .deviceNanosecondsPerPoint = 990.5971252981246,
      .minimumDevicePointCount = 250000U,
      .calibrated = true}},
    // B0092 exact direct-LAS composition ladder. RuntimePlacement admits this
    // residual only for the measured statistical outlier defaults immediately
    // followed by kth NNDistance(k=10), with one planner-proved max-k gather,
    // the 36-byte mapped source/output boundary, and one whole-view lane.
    {"outlier-nndistance-direct-compose",
     {.deviceFixedNanoseconds = 75276867.54198201,
      .hostNanosecondsPerPoint = 8034.195945434299,
      .minimumDevicePointCount = 50000U,
      .maximumDevicePointCount = 16000000U,
      .calibrated = true}},
    // B0127 exact direct-LAS same-radius composition ladder. The residual is
    // fitted over the current-binary 250K, 1M, and 4M paired medians after
    // the planner-owned cold start, 24 B/point upload, 10 B/point logical
    // spill, 28 B/point shared radius index, and two synchronizations. The
    // 50K direction row is deliberately below the conservative envelope.
    {"radius-outlier-radialdensity-direct-compose",
     {.deviceFixedNanoseconds = 266587419.5420545,
      .hostNanosecondsPerPoint = 6228.969755310317,
      .minimumDevicePointCount = 250000U,
      .maximumDevicePointCount = 4000000U,
      .calibrated = true}},
    {"estimaterank",
     {.hostFixedNanoseconds = 27127612.124431267,
      .deviceFixedNanoseconds = 193217335.34935838,
      .hostNanosecondsPerPoint = 4372.373230856109,
      .deviceNanosecondsPerPoint = 993.5218764129231,
      .minimumDevicePointCount = 250000U,
      .calibrated = true}},
    {"optimalneighborhood",
     {.hostFixedNanoseconds = 0.0,
      .deviceFixedNanoseconds = 152741159.5954907,
      .hostNanosecondsPerPoint = 6431.156586069055,
      .deviceNanosecondsPerPoint = 1167.2887542536657,
      .minimumDevicePointCount = 250000U,
      .calibrated = true}},
    // B0079 exact explicit-resident ladder. RuntimePlacement admits this
    // residual only for the measured estimate-rank/optimal-neighborhood and
    // three-assignment shape; the individual stage models remain unchanged.
    {"rank-optimal-compose",
     {.hostFixedNanoseconds = 133650155.04406099,
      .deviceFixedNanoseconds = 240029160.187007,
      .hostNanosecondsPerPoint = 11448.442351895543,
      .deviceNanosecondsPerPoint = 1186.5275421400943,
      .minimumDevicePointCount = 250000U,
      .maximumDevicePointCount = 16000000U,
      .calibrated = true}},
    // B0187 exact resident ladder over 50K/250K/1M/2M/4M, every lane
    // byte-exact with stats-proven `planner_resident_shared_index` execution.
    // The host term is the complete pinned-PDAL process curve; the device term
    // is the residual after the planner-owned cold start, upload, spill,
    // packing, shared-index build, and synchronizations are subtracted, so
    // evaluating the plan reconstructs the measured absolute curve.
    //
    // Both intercepts are fitted through the origin rather than free. The
    // unconstrained fit produced negative fixed terms (-143.9 ms host,
    // -31.7 ms device), which are physically meaningless and would understate
    // device cost near the envelope floor. Constraining them costs at most
    // 3.63% prediction error and never inverts the choice: host residuals
    // +1.15/+3.63/+2.20/+0.13/-0.18%, device +2.86/-2.44/+2.58/+2.93/-0.91%.
    //
    // The 50K floor is measured, not inherited. The sibling neighborhood
    // models stop at 250K, but forcing this pair onto device at 50K measures
    // 0.2857 s against 0.4166 s pinned host — device wins by 1.458x — and the
    // model predicts that correctly, so declining there would have discarded a
    // real win. Device is faster at all five rows (1.458x to 7.260x).
    //
    // RuntimePlacement admits this residual only for the measured knn=8
    // normal -> covariancefeatures pair; the individual stage models are
    // unchanged and every other combination keeps mixed-model rejection.
    // B0280 independently qualified the reference machine through its real
    // 47,478,228-point AHN4 row. Keep the original exact-machine coefficients
    // and floor, and stop at that observed ceiling rather than copying the
    // shipped class profile's rounded 48M bound.
    {"normal-covariancefeatures-compose",
     {.hostFixedNanoseconds = 0.0,
      .deviceFixedNanoseconds = 0.0,
      .hostNanosecondsPerPoint = 8428.236852266,
      .deviceNanosecondsPerPoint = 1087.699620198,
      .minimumDevicePointCount = 50000U,
      .maximumDevicePointCount = 47478228U,
      .calibrated = true}},
    // B0280 proves both the previously shadowed 35,976,465-point format-6 row
    // and the reference machine's 47,478,228-point format-8 AHN4 row. Boundary
    // packing, transfer, output stride, and memory remain derived from runtime
    // facts, so the compute residual is the same conservative exact-machine
    // curve. The runtime matcher separately bounds the qualified layout family.
    {"normal-covariancefeatures-compose-extradims",
     {.hostFixedNanoseconds = 0.0,
      .deviceFixedNanoseconds = 0.0,
      .hostNanosecondsPerPoint = 8428.236852266,
      .deviceNanosecondsPerPoint = 1087.699620198,
      .minimumDevicePointCount = 250000U,
      .maximumDevicePointCount = 47478228U,
      .calibrated = true}},
    {"neighborclassifier",
     {.hostFixedNanoseconds = 0.0,
      .deviceFixedNanoseconds = 126610942.40784068,
      .hostNanosecondsPerPoint = 3721.0713612740547,
      .deviceNanosecondsPerPoint = 954.6708793776946,
      .minimumDevicePointCount = 250000U,
      .calibrated = true}},
    // B0231 exact mapped-source/direct-Classification ladder. Host is the
    // through-origin fit of the six 250K--16M pinned-PDAL rows. Device is the
    // nonnegative through-origin residual after subtracting the planner-owned
    // 239.214 ms cold start, two synchronizations, 25 B/point upload,
    // 1 B/point spill, zero packing, and 112 B/point shared-index build.
    // The linear residual overpredicts the 250K row by 15.79% and is within
    // 15.56% at every other measured row; that conservative error cannot
    // invert any winner. The separately measured 50K row is a 0.685x loss
    // and defines the hard floor rather than being projected into this model.
    {"neighborclassifier-direct-compose",
     {.hostFixedNanoseconds = 0.0,
      .deviceFixedNanoseconds = 0.0,
      .hostNanosecondsPerPoint = 3689.146693032069,
      .deviceNanosecondsPerPoint = 790.7876247929496,
      .minimumDevicePointCount = 250000U,
      .maximumDevicePointCount = 16000000U,
      .calibrated = true}},
    // D0081 audit residual over seven exact resident-executor process pairs.
    // Planner-owned startup/transfer/index terms carry the device curve; this
    // stage residual fits the measured device-minus-host delta, not either
    // executable's absolute wall time. The unconstrained 206.68 ms intercept
    // is capped at 170 ms so the fit preserves every measured winner,
    // including the 250k envelope floor.
    {"radiusassign",
     {.hostFixedNanoseconds = 0.0,
      .deviceFixedNanoseconds = 170000000.0,
      .hostNanosecondsPerPoint = 1664.610423445786,
      .deviceNanosecondsPerPoint = 0.0,
      .minimumDevicePointCount = 250000U,
      .maximumDevicePointCount = 21970934U,
      .calibrated = true}},
    // B0083 exact direct-LAS radiusassign ladder. The raw fit is the complete
    // cold-process wall curve. These residual coefficients subtract the
    // planner-owned cold start, two boundary synchronizations, 25 B/point
    // upload, 1 B/point spill, and 28 B/point shared-index terms so evaluating
    // the plan reconstructs that absolute curve exactly. Ordinary
    // radiusassign remains the independent B0081 shared-index model above.
    {"radiusassign-direct",
     {.hostFixedNanoseconds = 0.0,
      .deviceFixedNanoseconds = 66189962.889582455,
      .hostNanosecondsPerPoint = 1985.1180564323188,
      .deviceNanosecondsPerPoint = 67.0868094224532,
      .minimumDevicePointCount = 250000U,
      .maximumDevicePointCount = 16000000U,
      .calibrated = true}},
}};

constexpr PlacementCalibrationProfile Sm89Profile{
    .id = "sm89-2026-08-17-r6-large-layouts",
    .device = {.name = "NVIDIA GeForce RTX 4090",
               .computeCapability = "8.9",
               .driverVersion = "610.43.03",
               .cudaToolkitVersion = "13.3"},
    .coefficients = {.cudaStartupNanoseconds = 239214115.124614,
                     .hostToDeviceNanosecondsPerByte = 0.038850571092396,
                     .deviceToHostNanosecondsPerByte = 0.045576629155188,
                     .packingNanosecondsPerByte = 0.002759959539676,
                     .indexBuildNanosecondsPerByte = 0.008297172727,
                     .synchronizationNanoseconds = 1930.0},
    .stageModels = Sm89StageModels};
} // unnamed namespace

std::vector<std::string_view> embeddedPlacementModelNames()
{
    std::vector<std::string_view> names;
    names.reserve(Sm89StageModels.size());
    for (const PlacementStageCalibration& model : Sm89StageModels)
        names.push_back(model.name);
    return names;
}

const PlacementCalibrationProfile*
placementCalibrationFor(const PlacementDeviceKey& device) noexcept
{
    // 1. The embedded, physically measured reference profile (D0049 lineage).
    //    A test hook hides it so the local-profile path below can be
    //    exercised on the reference machine.
    const PlacementDeviceKey& expected = Sm89Profile.device;
    if (device.name == expected.name &&
        device.computeCapability == expected.computeCapability &&
        device.driverVersion == expected.driverVersion &&
        device.cudaToolkitVersion == expected.cudaToolkitVersion &&
        !std::getenv(IgnoreBuiltinProfileTestEnvironment.data()))
        return &Sm89Profile;
    // 2. The calibrate command's forced-device override (explicit `resident`
    //    runs only; the automatic route declines while it is set).
    if (const PlacementCalibrationProfile* forced =
            calibrationForcedDevicePlacementProfile(device))
        return forced;
    // 3. A locally calibrated profile keyed to this exact machine (D0277).
    const LocalProfileLookup& local = loadedLocalProfile();
    if (local.status == LocalProfileStatus::Applied && local.profile)
    {
        static const PlacementCalibrationProfile localView =
            local.profile->view();
        if (localView.device.name == device.name &&
            localView.device.computeCapability == device.computeCapability &&
            localView.device.driverVersion == device.driverVersion &&
            localView.device.cudaToolkitVersion == device.cudaToolkitVersion)
            return &localView;
    }
    // 4. A shipped GPU-class profile measured on a rented host of this GPU
    //    model / compute capability / toolkit (D0279): admits only the models
    //    whose device win cleared the shipping margin at every measured size.
    if (const PlacementCalibrationProfile* shipped =
            shippedPlacementProfileFor(device))
        return shipped;
    // 5. The generic fallback for any other CUDA device inside its measured
    //    compute-capability / memory bounds (D0279): the big-margin family
    //    only. Everything else fails closed to the host path.
    if (const PlacementCalibrationProfile* generic =
            genericPlacementProfileFor(device))
        return generic;
    return nullptr;
}

const StagePlacementCost*
placementStageCalibration(const PlacementCalibrationProfile& profile,
                          std::string_view name) noexcept
{
    for (const PlacementStageCalibration& model : profile.stageModels)
        if (model.name == name)
            return &model.cost;
    return nullptr;
}

bool applyPlacementRegionCalibrations(
    const Plan& plan, const PlacementCalibrationProfile& profile,
    std::span<const PlacementRegionCalibration> calibrations,
    PlacementRequest& request)
{
    std::vector<StagePlacementCost> stageCosts(plan.stages().size());
    if (calibrations.size() != plan.summary().residentRegions)
    {
        request.stageCosts = std::move(stageCosts);
        return false;
    }

    std::vector<bool> seen(plan.summary().residentRegions, false);
    for (const PlacementRegionCalibration& calibration : calibrations)
    {
        if (calibration.residentRegion >= seen.size() ||
            seen[calibration.residentRegion])
        {
            request.stageCosts.assign(plan.stages().size(), {});
            return false;
        }
        const StagePlacementCost* model =
            placementStageCalibration(profile, calibration.model);
        if (!model)
        {
            request.stageCosts.assign(plan.stages().size(), {});
            return false;
        }

        bool first = true;
        for (const PlannedStage& stage : plan.stages())
        {
            if (!stage.native ||
                stage.preferredResidency != MemoryKind::Device ||
                stage.residentRegion != calibration.residentRegion)
                continue;
            stageCosts[stage.id] =
                first ? *model : StagePlacementCost{.calibrated = true};
            first = false;
        }
        if (first)
        {
            request.stageCosts.assign(plan.stages().size(), {});
            return false;
        }
        seen[calibration.residentRegion] = true;
    }
    request.stageCosts = std::move(stageCosts);
    return true;
}

} // namespace pdg
