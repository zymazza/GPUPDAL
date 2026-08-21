#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace pdg
{

inline constexpr std::string_view HybridPointProgramStage =
    "filters.pdg_pointprogram";
inline constexpr std::string_view HybridResidentBoundaryStage =
    "filters.pdg_resident_boundary";
inline constexpr std::string_view HybridResidentLasSourceStage =
    "readers.pdg_resident_las_source";
inline constexpr std::string_view HybridLocateStage = "filters.pdg_locate";
inline constexpr std::string_view HybridRobustStage = "filters.pdg_robust";
inline constexpr std::string_view HybridOrderStage = "filters.pdg_order";
inline constexpr std::string_view HybridMortonOrderStage =
    "filters.pdg_mortonorder";
inline constexpr std::string_view HybridGroupByStage = "filters.pdg_groupby";
inline constexpr std::string_view HybridReturnsStage = "filters.pdg_returns";
inline constexpr std::string_view HybridMergeStage = "filters.pdg_merge";
inline constexpr std::string_view HybridDividerStage = "filters.pdg_divider";
inline constexpr std::string_view HybridSplitterStage = "filters.pdg_splitter";
inline constexpr std::string_view HybridColorinterpStage =
    "filters.pdg_colorinterp";
inline constexpr std::string_view HybridStatsStage = "filters.pdg_stats";
inline constexpr std::string_view HybridInfoStage = "filters.pdg_info";
inline constexpr std::string_view HybridExpressionStatsStage =
    "filters.pdg_expressionstats";
inline constexpr std::string_view HybridOutlierStage = "filters.pdg_outlier";
inline constexpr std::string_view HybridRadialDensityStage =
    "filters.pdg_radialdensity";
inline constexpr std::string_view HybridRadiusAssignStage =
    "filters.pdg_radiusassign";
inline constexpr std::string_view HybridLabelDuplicatesStage =
    "filters.pdg_label_duplicates";
inline constexpr std::string_view HybridSmrfStage = "filters.pdg_smrf";
inline constexpr std::string_view HybridPmfStage = "filters.pdg_pmf";
inline constexpr std::string_view HybridCsfStage = "filters.pdg_csf";
inline constexpr std::string_view HybridElmStage = "filters.pdg_elm";
inline constexpr std::string_view HybridSkewnessStage =
    "filters.pdg_skewnessbalancing";
inline constexpr std::string_view HybridNnDistanceStage =
    "filters.pdg_nndistance";
inline constexpr std::string_view HybridHagNnStage = "filters.pdg_hag_nn";
inline constexpr std::string_view HybridHagDelaunayStage =
    "filters.pdg_hag_delaunay";
inline constexpr std::string_view HybridNormalStage = "filters.pdg_normal";
inline constexpr std::string_view HybridEigenvaluesStage =
    "filters.pdg_eigenvalues";
inline constexpr std::string_view HybridCovarianceFeaturesStage =
    "filters.pdg_covariancefeatures";
inline constexpr std::string_view HybridApproximateCoplanarStage =
    "filters.pdg_approximatecoplanar";
inline constexpr std::string_view HybridLofStage = "filters.pdg_lof";
inline constexpr std::string_view HybridOptimalNeighborhoodStage =
    "filters.pdg_optimalneighborhood";
inline constexpr std::string_view HybridNeighborClassifierStage =
    "filters.pdg_neighborclassifier";
inline constexpr std::string_view HybridEstimateRankStage =
    "filters.pdg_estimaterank";
inline constexpr std::uint64_t AutomaticApproximateCoplanarCudaMinimumPoints =
    262144U;
inline constexpr std::uint64_t AutomaticLabelNnDistanceCudaMinimumPoints =
    250000U;
inline constexpr std::uint64_t AutomaticLabelNnDistanceCudaMaximumPoints =
    16000000U;

struct HybridPipelineRewrite
{
    std::string json;
    std::size_t replacementRegions = 0;
    std::size_t pointProgramRegions = 0;
    std::size_t fusedStages = 0;
    bool linearPipeline = false;
    bool hasUnstableInputOrderRegion = false;
    bool standardModeRewriteIsExact = false;
    // writers.las serializes every non-point-format dimension in prepared
    // PointLayout order when extra_dims=all.  A rewrite must retain that
    // order even when all point values themselves are exact.
    bool preparedLayoutOrderObservable = false;
    bool standardModeRequiresPointCountValidation = false;
    bool hasPointCountDependentCudaCandidate = false;
    bool automaticApproximateCoplanarCuda = false;
    bool automaticR4OutlierCuda = false;
    bool automaticLabelNnDistanceCuda = false;
    std::string automaticPointCountFilename;
    std::uint64_t automaticPointCountLimit =
        (std::numeric_limits<std::uint64_t>::max)();
};

// Automatic selection is a performance promise in addition to an exactness
// promise.  Keep the qualified device list narrow until a physical exactness
// lane and a clean break-even curve have passed for each additional profile.
[[nodiscard]] bool automaticApproximateCoplanarCudaDeviceQualified() noexcept;

// B0227 is intentionally a reference-workflow qualification, not a general
// statistical-outlier model.  Keep its physical-device promise separate from
// the stage's functional CUDA availability.
[[nodiscard]] bool automaticR4OutlierCudaDeviceQualified() noexcept;

// B0239 qualifies the complete r2 SMRF/HAG-NN hybrid only on the same exact
// physical profile as the measured r4 selector.
[[nodiscard]] bool automaticR2GroundNormalizeDeviceQualified() noexcept;

// B0233 fits this complete hybrid composition independently of the rejected
// resident prototype.  The count envelope and device identity are both part
// of the performance promise, not merely functional CUDA eligibility.
[[nodiscard]] bool automaticLabelNnDistanceCudaDeviceQualified() noexcept;
[[nodiscard]] bool
preferAutomaticLabelNnDistanceCuda(std::uint64_t pointCount) noexcept;
[[nodiscard]] bool
automaticLabelNnDistanceHybridCandidate(std::string_view pipelineJson) noexcept;

// Replaces conservative, linear point-program regions and exact global
// reductions/selections with internal stages. Every other stage and option
// remains in the upstream PDAL pipeline and is therefore executed by its
// original implementation. Invalid JSON is reported to the caller so the exact
// CLI can delegate it to the pinned oracle without changing diagnostics.
[[nodiscard]] HybridPipelineRewrite rewriteHybridPipeline(
    std::string_view pipelineJson, bool preserveStageBoundaries = false,
    bool enableExperimentalReplacements = false,
    std::uint64_t automaticPointCount = 0, bool measuredR4Input = false,
    bool measuredLabelNnDistanceInput = false);

} // namespace pdg
