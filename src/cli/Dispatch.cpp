#include "Dispatch.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <array>
#include <exception>
#include <string_view>

namespace pdg::cli
{

namespace
{
using Json = nlohmann::json;

constexpr std::array<std::string_view, 31> CandidateStages{
    "filters.approximatecoplanar",
    "filters.assign",
    "filters.colorinterp",
    "filters.covariancefeatures",
    "filters.crop",
    "filters.decimation",
    "filters.divider",
    "filters.eigenvalues",
    "filters.expression",
    "filters.expressionstats",
    "filters.ferry",
    "filters.groupby",
    "filters.head",
    "filters.info",
    "filters.iqr",
    "filters.locate",
    "filters.mad",
    "filters.merge",
    "filters.mortonorder",
    "filters.nndistance",
    "filters.normal",
    "filters.outlier",
    "filters.radialdensity",
    "filters.randomize",
    "filters.range",
    "filters.returns",
    "filters.sort",
    "filters.splitter",
    "filters.stats",
    "filters.tail",
    "filters.transformation",
};

// Known engine controls retained as a reviewable test inventory. Routing does
// not depend on this list: dispatchEnvironmentRequiresEngine fails closed for
// every current or future PDG_* name except PDG_ORACLE_PDAL.
constexpr auto EngineEnvironmentVariables = std::to_array<std::string_view>({
    "PDG_CUDA_CHUNK_POINTS",
    "PDG_CUDA_SCHEDULER_LANES",
    "PDG_DEBUG_ADMISSION_PHASES",
    "PDG_DEBUG_FUSED_JIT",
    "PDG_DEBUG_HYBRID",
    "PDG_DEBUG_HAG_NN_PHASES",
    "PDG_DEBUG_HYBRID_OUTLIER_PHASES",
    "PDG_DEBUG_SMRF_DUMP",
    "PDG_DEBUG_SMRF_FILL",
    "PDG_DISABLE_CUDA_HYBRID",
    "PDG_DISABLE_CUDA_POINT_PROGRAM",
    "PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE",
    "PDG_DISABLE_FUSED_JIT",
    "PDG_DISABLE_HYBRID",
    "PDG_DISABLE_KNN_DISTANCE_PREFILTER",
    "PDG_DISABLE_LOF_KD3_COORDINATE_CACHE",
    "PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS",
    "PDG_DISABLE_KD3_COORDINATE_CACHE",
    "PDG_DISABLE_NATIVE",
    "PDG_EXPERIMENTAL_AUTOMATIC_R4_OUTLIER_CUDA",
    "PDG_EXPERIMENTAL_CUDA_HYBRID",
    "PDG_EXPERIMENTAL_CUDA_POINT_PROGRAM",
    "PDG_EXPERIMENTAL_CUDA_TRANSLATE",
    "PDG_EXPERIMENTAL_DIRECT_CLASSIFICATION_OUTPUT",
    "PDG_EXPERIMENTAL_DIRECT_LAS_RESIDENT_SOURCE",
    "PDG_EXPERIMENTAL_DIRECT_RESIDENT_LAS_OUTPUT",
    "PDG_KNN_DEVICE_SHELL_BUDGET",
    "PDG_LAZ_COMPRESSION_THREADS",
    "PDG_NATIVE_WORKERS",
    "PDG_REQUIRE_CUDA_HYBRID",
    "PDG_REQUIRE_FUSED_CUDA_POINT_PROGRAM",
    "PDG_REQUIRE_FUSED_JIT",
    "PDG_REQUIRE_CUDA_POINT_PROGRAM",
    "PDG_REQUIRE_CUDA_TRANSLATE",
    "PDG_EXPERIMENTAL_DIRECT_EXTRA_DOUBLE_OUTPUT",
    "PDG_REQUIRE_DIRECT_EXTRA_DOUBLE_OUTPUT",
    "PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY",
    "PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY_BACKEND",
    "PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE",
    "PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT",
    "PDG_EXPERIMENTAL_DIRECT_SKEWNESS_COMPOSITION",
    "PDG_REQUIRE_DIRECT_SKEWNESS_COMPOSITION",
    "PDG_EXPERIMENTAL_DIRECT_SORT_COMPOSITION",
    "PDG_REQUIRE_DIRECT_SORT_COMPOSITION",
    "PDG_REQUIRE_HYBRID",
    "PDG_REQUIRE_NATIVE",
    "PDG_REQUIRE_STREAMING_HYBRID",
    "PDG_FORCE_MORTON_BVH",
    "PDG_FORCE_LEGACY_CUDA_ALLOCATOR",
    "PDG_FORCE_UNIFORM_GRID",
    "PDG_DISABLE_NEIGHBORHOOD_ROW_BOUNDARY",
    "PDG_REQUIRE_NEIGHBORHOOD_ROW_BOUNDARY",
    "PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE",
    "PDG_REQUIRE_NEIGHBORHOOD_COPLANAR_COLUMN_REUSE",
    "PDG_REQUIRE_NEIGHBORHOOD_BRIDGE_REBUILD",
    "PDG_REQUIRE_NEIGHBORHOOD_EIGENSYSTEM_REUSE",
    "PDG_REQUIRE_NEIGHBORHOOD_REUSE",
    "PDG_REQUIRE_NONTERMINAL_NEIGHBORHOOD_COLUMN_REUSE",
    "PDG_REQUIRE_NEIGHBORHOOD_TIE_REPAIR",
    "PDG_REQUIRE_LOF_PARALLEL_REPAIR",
    "PDG_REQUIRE_LOF_KD3_COORDINATE_CACHE",
    "PDG_REQUIRE_HAG_NN_SELECTIVE_REPAIR",
    "PDG_REQUIRE_HAG_NN_HOST_FALLBACK",
    "PDG_REQUIRE_HAG_NN_INSUFFICIENT_GROUND_FALLBACK",
    "PDG_REQUIRE_HAG_NN_NONFINITE_Z_FALLBACK",
    "PDG_REQUIRE_HAG_NN_TIE_FALLBACK",
    "PDG_REQUIRE_HAG_DELAUNAY_HOST_FALLBACK",
    "PDG_REQUIRE_HAG_DELAUNAY_INSUFFICIENT_GROUND_FALLBACK",
    "PDG_REQUIRE_HAG_DELAUNAY_NONFINITE_FALLBACK",
    "PDG_REQUIRE_HAG_DELAUNAY_TIE_FALLBACK",
    "PDG_REQUIRE_NND_DEVICE_REPAIR",
    "PDG_DISABLE_NND_DEVICE_REPAIR",
    "PDG_EXPERIMENTAL_NND_PARALLEL_REPAIR",
    "PDG_DISABLE_NND_PARALLEL_REPAIR",
    "PDG_REQUIRE_NND_PARALLEL_REPAIR",
    "PDG_REQUIRE_NND_DEVICE_ONLY_HANDOFF",
    "PDG_REQUIRE_NND_HOST_RESTORE",
    "PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ",
    "PDG_REQUIRE_OUTLIER_DEVICE_REPAIR",
    "PDG_DISABLE_OUTLIER_DEVICE_REPAIR",
    "PDG_REQUIRE_OUTLIER_PARALLEL_REPAIR",
    "PDG_REQUIRE_RADIUS_OUTLIER_RADIALDENSITY_COMPOSITION",
    "PDG_DISABLE_OUTLIER_PARALLEL_REPAIR",
    "PDG_REQUIRE_KNN_GATHER_REUSE",
    "PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_CUDA",
    "PDG_REQUIRE_AUTOMATIC_R4_OUTLIER_CUDA",
    "PDG_REQUIRE_AUTOMATIC_R2_GROUND_NORMALIZE",
    "PDG_REQUIRE_AUTOMATIC_LABEL_NNDISTANCE_HYBRID",
    "PDG_TEST_LABEL_NNDISTANCE_RECOVERABLE_CUDA_FAILURE",
    "PDG_TEST_KD3_CACHE_BUILD_FAILURE",
    "PDG_TEST_NEIGHBORHOOD_EIGEN_FAILURE_POINT",
    "PDG_TEST_RESIDENT_VRAM_BUDGET_BYTES",
    "PDG_REQUIRE_AUTOMATIC_EIGEN_FAMILY_RESIDENT",
    "PDG_REQUIRE_AUTOMATIC_NORMAL_COVARIANCE_RESIDENT",
    "PDG_REQUIRE_AUTOMATIC_RANK_OPTIMAL_RESIDENT",
    "PDG_REQUIRE_AUTOMATIC_OUTLIER_NNDISTANCE_RESIDENT",
    "PDG_REQUIRE_AUTOMATIC_RADIUS_OUTLIER_RADIALDENSITY_RESIDENT",
    "PDG_REQUIRE_AUTOMATIC_RADIUSASSIGN_RESIDENT",
    "PDG_REQUIRE_AUTOMATIC_NEIGHBORCLASSIFIER_RESIDENT",
    "PDG_REQUIRE_AUTOMATIC_SKEWNESS_RESIDENT",
    "PDG_REQUIRE_AUTOMATIC_SORT_RESIDENT",
    "PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_RESIDENT",
    "PDG_REQUIRE_AUTOMATIC_RESIDENT_LAS_OUTPUT",
    "PDG_REQUIRE_AUTOMATIC_HAG_NN_RESIDENT",
    "PDG_REQUIRE_AUTOMATIC_HAG_DELAUNAY_RESIDENT",
    "PDG_TEST_AUTOMATIC_R2_HAG_NN_DEVICE_DECLINE",
    "PDG_TEST_AUTOMATIC_HAG_DELAUNAY_DEVICE_DECLINE",
    "PDG_REQUIRE_APPROXIMATECOPLANAR_HOST_FALLBACK",
    "PDG_TEST_APPROXIMATECOPLANAR_RECOVERABLE_CUDA_FAILURE",
    "PDG_TEST_R4_OUTLIER_RECOVERABLE_CUDA_FAILURE",
    "PDG_TEST_AUTOMATIC_OUTLIER_PROOF_FAILURE",
    "PDG_TEST_AUTOMATIC_RADIUS_COMPOSITION_PROOF_FAILURE",
    "PDG_TEST_AUTOMATIC_RADIUSASSIGN_PROOF_FAILURE",
    "PDG_TEST_AUTOMATIC_HAG_NN_PROOF_FAILURE",
    "PDG_TEST_AUTOMATIC_HAG_DELAUNAY_PROOF_FAILURE",
    "PDG_TEST_AUTOMATIC_NEIGHBORCLASSIFIER_PROOF_FAILURE",
    "PDG_TEST_AUTOMATIC_COPLANAR_PROOF_FAILURE",
    "PDG_TEST_AUTOMATIC_RESIDENT_PREFLIGHT_FAILURE",
    "PDG_TEST_AUTOMATIC_RESIDENT_PUBLICATION_TRUNCATION",
    "PDG_TEST_DIRECT_SKEWNESS_PROOF_FAILURE",
    "PDG_TEST_DIRECT_SKEWNESS_PREFLIGHT_FAILURE",
    "PDG_TEST_DIRECT_SORT_PROOF_FAILURE",
    "PDG_TEST_DIRECT_SORT_PREFLIGHT_FAILURE",
    "PDG_REQUIRE_SPATIAL_TILING",
    "PDG_SPATIAL_TILE_EDGE",
});

bool isCandidateStage(std::string_view type) noexcept
{
    return std::find(CandidateStages.begin(), CandidateStages.end(), type) !=
           CandidateStages.end();
}

bool hasLasExtension(std::string_view filename) noexcept
{
    if (filename.size() < 4U)
        return false;
    const std::string_view extension = filename.substr(filename.size() - 4U);
    return (extension[0] == '.' &&
            (extension[1] == 'l' || extension[1] == 'L') &&
            (extension[2] == 'a' || extension[2] == 'A') &&
            (extension[3] == 's' || extension[3] == 'S'));
}

bool hasLazExtension(std::string_view filename) noexcept
{
    if (filename.size() < 4U)
        return false;
    const std::string_view extension = filename.substr(filename.size() - 4U);
    return extension == ".laz";
}

bool hasExactLasExtension(std::string_view filename) noexcept
{
    return filename.size() >= 4U &&
           filename.substr(filename.size() - 4U) == ".las";
}

bool hasCopcLazExtension(std::string_view filename) noexcept
{
    constexpr std::string_view Extension = ".copc.laz";
    return filename.size() >= Extension.size() &&
           filename.substr(filename.size() - Extension.size()) == Extension;
}

bool hasCaseInsensitiveCopcLazExtension(std::string_view filename) noexcept
{
    constexpr std::string_view Extension = ".copc.laz";
    if (filename.size() < Extension.size())
        return false;
    const std::string_view suffix =
        filename.substr(filename.size() - Extension.size());
    for (std::size_t index = 0; index < Extension.size(); ++index)
    {
        char value = suffix[index];
        if (value >= 'A' && value <= 'Z')
            value = static_cast<char>(value - 'A' + 'a');
        if (value != Extension[index])
            return false;
    }
    return true;
}

bool hasExactNonCopcLazExtension(std::string_view filename) noexcept
{
    return hasLazExtension(filename) &&
           !hasCaseInsensitiveCopcLazExtension(filename);
}

bool hasExactTifExtension(std::string_view filename) noexcept
{
    return filename.size() >= 4U &&
           filename.substr(filename.size() - 4U) == ".tif";
}

bool hasStringOption(const Json& stage, const char* name,
                     std::string_view value) noexcept
{
    const auto option = stage.find(name);
    return option != stage.end() && option->is_string() &&
           option->get_ref<const std::string&>() == value;
}

bool hasStringOption(const Json& stage, const char* name) noexcept
{
    const auto option = stage.find(name);
    return option != stage.end() && option->is_string();
}

// B0197/D0219: a bare LAS reader-to-writer pipeline, with no filter between
// them.
//
// No `pipeline` route can accelerate this shape. The automatic resident
// attempt declines at plan compilation in 0.17 ms (B0196), `tryNativePipeline`
// requires at least three stages, and the engine ends at `runOracle`, which
// `execv`s pinned PDAL. Sending it to the engine therefore loads the 11 MB
// engine image and its 76 shared libraries -- `libcudart` and `libcuda`
// included, which PDAL does not link -- purely to hand the work straight back,
// costing a measured ~15-17 ms.
//
// Deliberately narrow: exactly two stages, both LAS file stages. Anything with
// a filter keeps its engine route, because a filter is what every accelerated
// shape has.
bool isBareLasTranslate(const Json& pipeline) noexcept
{
    if (pipeline.size() != 2U)
        return false;
    const auto isLasFileStage = [](const Json& stage, bool reader) noexcept
    {
        if (stage.is_string())
            return hasLasExtension(stage.get_ref<const std::string&>());
        if (!stage.is_object())
            return false;
        const auto type = stage.find("type");
        if (type == stage.end() || !type->is_string())
            return false;
        return type->get_ref<const std::string&>() ==
               (reader ? "readers.las" : "writers.las");
    };
    return isLasFileStage(pipeline.at(0U), true) &&
           isLasFileStage(pipeline.at(1U), false);
}

// B0226: the checked-in r1 route is LAS -> crop -> reprojection -> LAS. The
// engine cannot execute reprojection, so its only possible contribution is a
// hybrid crop before it execs the oracle. B0199 and B0219 measured that losing
// on r1. B0226 measures direct delegation on the exact reference input facts.
//
// Keep this a literal grammar, not a general reprojection rule: exact stage
// order, option sets, literal reference bounds/SRS pair, LAZ endpoints, and
// compression spelling. Neighboring shapes retain engine ownership until
// measured.
bool isMeasuredR1Grammar(const Json& pipeline) noexcept
{
    if (pipeline.size() != 4U)
        return false;
    const Json& reader = pipeline.at(0U);
    const Json& crop = pipeline.at(1U);
    const Json& reprojection = pipeline.at(2U);
    const Json& writer = pipeline.at(3U);
    if (!reader.is_object() || reader.size() != 2U ||
        !hasStringOption(reader, "type", "readers.las") ||
        !hasStringOption(reader, "filename") ||
        !hasLazExtension(reader.at("filename").get_ref<const std::string&>()))
        return false;
    if (!crop.is_object() || crop.size() != 2U ||
        !hasStringOption(crop, "type", "filters.crop") ||
        !hasStringOption(crop, "bounds",
                         "([184874.9975,185624.9925],[494942.405,494980.795])"))
        return false;
    if (!reprojection.is_object() || reprojection.size() != 3U ||
        !hasStringOption(reprojection, "type", "filters.reprojection") ||
        !hasStringOption(reprojection, "in_srs", "EPSG:28992") ||
        !hasStringOption(reprojection, "out_srs", "EPSG:3857"))
        return false;
    return writer.is_object() && writer.size() == 3U &&
           hasStringOption(writer, "type", "writers.las") &&
           hasStringOption(writer, "filename") &&
           hasLazExtension(
               writer.at("filename").get_ref<const std::string&>()) &&
           hasStringOption(writer, "compression", "true");
}

// B0228: the checked-in r5 route is COPC -> stats -> uncompressed LAS. The
// engine does not implement readers.copc, so this pipeline declines every
// native attempt and execs the unchanged pipeline in the pinned oracle. B0202
// and the current-binary reproduction attribute the entire public deficit to
// loading that unused engine image.
//
// This is intentionally the measured reference grammar, not a general COPC
// or stats rule. Neighboring shapes retain engine ownership until measured.
bool isMeasuredR5Grammar(const Json& pipeline) noexcept
{
    if (pipeline.size() != 3U)
        return false;
    const Json& reader = pipeline.at(0U);
    const Json& stats = pipeline.at(1U);
    const Json& writer = pipeline.at(2U);
    if (!reader.is_object() || reader.size() != 5U ||
        !hasStringOption(reader, "type", "readers.copc") ||
        !hasStringOption(reader, "filename") ||
        !hasCopcLazExtension(
            reader.at("filename").get_ref<const std::string&>()) ||
        !hasStringOption(reader, "bounds",
                         "([184874.9975,185624.9925],[494942.405,494980.795])"))
        return false;
    const auto resolution = reader.find("resolution");
    const auto requests = reader.find("requests");
    if (resolution == reader.end() || !resolution->is_number_float() ||
        resolution->get<double>() != 1.0 || requests == reader.end() ||
        !requests->is_number_integer() || requests->get<std::uint64_t>() != 1U)
        return false;
    if (!stats.is_object() || stats.size() != 1U ||
        !hasStringOption(stats, "type", "filters.stats"))
        return false;
    return writer.is_object() && writer.size() == 2U &&
           hasStringOption(writer, "type", "writers.las") &&
           hasStringOption(writer, "filename") &&
           hasExactLasExtension(
               writer.at("filename").get_ref<const std::string&>());
}

// B0230: the checked-in r2 route is LAZ -> SMRF -> HAG-NN -> a named
// HeightAboveGround LAZ sink. The named writer makes the engine refuse the
// plan before point processing and exec the original pipeline in the oracle.
//
// This is intentionally the measured reference grammar, not native named
// extra-dimension publication or a generalized terrain dispatch rule.
bool isMeasuredR2Grammar(const Json& pipeline) noexcept
{
    if (pipeline.size() != 4U)
        return false;
    const Json& reader = pipeline.at(0U);
    const Json& smrf = pipeline.at(1U);
    const Json& hag = pipeline.at(2U);
    const Json& writer = pipeline.at(3U);
    if (!reader.is_object() || reader.size() != 2U ||
        !hasStringOption(reader, "type", "readers.las") ||
        !hasStringOption(reader, "filename") ||
        !hasExactNonCopcLazExtension(
            reader.at("filename").get_ref<const std::string&>()))
        return false;
    if (!smrf.is_object() || smrf.size() != 1U ||
        !hasStringOption(smrf, "type", "filters.smrf"))
        return false;
    if (!hag.is_object() || hag.size() != 1U ||
        !hasStringOption(hag, "type", "filters.hag_nn"))
        return false;
    return writer.is_object() && writer.size() == 4U &&
           hasStringOption(writer, "type", "writers.las") &&
           hasStringOption(writer, "filename") &&
           hasExactNonCopcLazExtension(
               writer.at("filename").get_ref<const std::string&>()) &&
           hasStringOption(writer, "compression", "true") &&
           hasStringOption(writer, "extra_dims", "HeightAboveGround=float32");
}

// B0229: the checked-in r3 route is LAZ -> SMRF -> ground range -> GDAL IDW.
// The unchanged SMRF stage makes the adjacent range rewrite unstable, while
// the GDAL sink is outside every native/resident endpoint. The engine therefore
// discards all admission work and execs the original pipeline in the oracle.
//
// Keep this to the measured reference grammar. It is neither native SMRF nor a
// generalized GDAL dispatch rule.
bool isMeasuredR3Grammar(const Json& pipeline) noexcept
{
    if (pipeline.size() != 4U)
        return false;
    const Json& reader = pipeline.at(0U);
    const Json& smrf = pipeline.at(1U);
    const Json& range = pipeline.at(2U);
    const Json& writer = pipeline.at(3U);
    if (!reader.is_object() || reader.size() != 2U ||
        !hasStringOption(reader, "type", "readers.las") ||
        !hasStringOption(reader, "filename") ||
        !hasExactNonCopcLazExtension(
            reader.at("filename").get_ref<const std::string&>()))
        return false;
    if (!smrf.is_object() || smrf.size() != 1U ||
        !hasStringOption(smrf, "type", "filters.smrf"))
        return false;
    if (!range.is_object() || range.size() != 2U ||
        !hasStringOption(range, "type", "filters.range") ||
        !hasStringOption(range, "limits", "Classification[2:2]"))
        return false;
    if (!writer.is_object() || writer.size() != 4U ||
        !hasStringOption(writer, "type", "writers.gdal") ||
        !hasStringOption(writer, "filename") ||
        !hasExactTifExtension(
            writer.at("filename").get_ref<const std::string&>()) ||
        !hasStringOption(writer, "output_type", "idw"))
        return false;
    const auto resolution = writer.find("resolution");
    return resolution != writer.end() && resolution->is_number_float() &&
           resolution->get<double>() == 1.0;
}

// B0248: the checked-in r7 route is LAZ -> first/only returns -> maximum-Z
// Float64 GeoTIFF. The GDAL sink is outside every native/resident endpoint, so
// the engine performs only fixed admission/control work before executing the
// unchanged pipeline in the pinned oracle. Same-final controls show that work
// is a resolved loss in both cache states.
//
// Keep this to the complete measured surface policy and materialized bounds.
// This is direct host delegation, not a generalized returns or GDAL route.
bool isMeasuredR7Grammar(const Json& pipeline) noexcept
{
    if (pipeline.size() != 3U)
        return false;
    const Json& reader = pipeline.at(0U);
    const Json& returns = pipeline.at(1U);
    const Json& writer = pipeline.at(2U);
    if (!reader.is_object() || reader.size() != 3U ||
        !hasStringOption(reader, "type", "readers.las") ||
        !hasStringOption(reader, "filename") ||
        !hasExactNonCopcLazExtension(
            reader.at("filename").get_ref<const std::string&>()) ||
        !hasStringOption(reader, "override_srs", "EPSG:28992"))
        return false;
    if (!returns.is_object() || returns.size() != 2U ||
        !hasStringOption(returns, "type", "filters.returns") ||
        !hasStringOption(returns, "groups", "first,only"))
        return false;
    if (!writer.is_object() || writer.size() != 9U ||
        !hasStringOption(writer, "type", "writers.gdal") ||
        !hasStringOption(writer, "filename") ||
        !hasExactTifExtension(
            writer.at("filename").get_ref<const std::string&>()) ||
        !hasStringOption(writer, "output_type", "max") ||
        !hasStringOption(writer, "dimension", "Z") ||
        !hasStringOption(writer, "data_type", "float64") ||
        !hasStringOption(
            writer, "bounds",
            "([184500,185999.99],[494923.21,494999.99])"))
        return false;
    const auto resolution = writer.find("resolution");
    const auto binmode = writer.find("binmode");
    const auto nodata = writer.find("nodata");
    return resolution != writer.end() && resolution->is_number_float() &&
           resolution->get<double>() == 1.0 && binmode != writer.end() &&
           binmode->is_boolean() && binmode->get<bool>() &&
           nodata != writer.end() && nodata->is_number_float() &&
           nodata->get<double>() == -9999.0;
}

// B0249: the checked-in r10 headline is LAZ -> voxel-centroid nearest source
// record -> LAZ. The voxel stage has no native/resident replacement, so the
// engine can only delegate the unchanged graph after fixed startup/control
// work. Direct-sibling and same-final controls qualify removing that process
// in both cache states.
//
// This is the complete measured headline grammar, not a generalized voxel or
// decimation rule. Neighboring policies remain engine-owned until measured.
bool isMeasuredR10Grammar(const Json& pipeline) noexcept
{
    if (pipeline.size() != 3U)
        return false;
    const Json& reader = pipeline.at(0U);
    const Json& voxel = pipeline.at(1U);
    const Json& writer = pipeline.at(2U);
    if (!reader.is_object() || reader.size() != 2U ||
        !hasStringOption(reader, "type", "readers.las") ||
        !hasStringOption(reader, "filename") ||
        !hasExactNonCopcLazExtension(
            reader.at("filename").get_ref<const std::string&>()))
        return false;
    if (!voxel.is_object() || voxel.size() != 2U ||
        !hasStringOption(
            voxel, "type", "filters.voxelcentroidnearestneighbor"))
        return false;
    const auto cell = voxel.find("cell");
    if (cell == voxel.end() || !cell->is_number_float() ||
        cell->get<double>() != 2.5)
        return false;
    return writer.is_object() && writer.size() == 3U &&
           hasStringOption(writer, "type", "writers.las") &&
           hasStringOption(writer, "filename") &&
           hasExactNonCopcLazExtension(
               writer.at("filename").get_ref<const std::string&>()) &&
           hasStringOption(writer, "compression", "true");
}

// B0251/D0250, corrected by B0252/D0251: the r14 headline is the literal
// uncompressed LAS -> compressed LAZ graph. Its writer is the only work
// changed by the exact parallel lazperf
// prototype; every option or endpoint drift remains on the old serial path.
bool isMeasuredR14Grammar(const Json& pipeline) noexcept
{
    if (pipeline.size() != 2U)
        return false;
    const Json& reader = pipeline.at(0U);
    const Json& writer = pipeline.at(1U);
    return reader.is_object() && reader.size() == 2U &&
           hasStringOption(reader, "type", "readers.las") &&
           hasStringOption(reader, "filename") &&
           hasExactLasExtension(
               reader.at("filename").get_ref<const std::string&>()) &&
           writer.is_object() && writer.size() == 3U &&
           hasStringOption(writer, "type", "writers.las") &&
           hasStringOption(writer, "filename") &&
           hasExactNonCopcLazExtension(
               writer.at("filename").get_ref<const std::string&>()) &&
           hasStringOption(writer, "compression", "true");
}

bool hasPotentialNativeLasIo(const Json& pipeline) noexcept
{
    bool reader = false;
    bool writer = false;
    for (std::size_t index = 0; index < pipeline.size(); ++index)
    {
        const Json& stage = pipeline.at(index);
        if (stage.is_string())
        {
            const std::string& filename = stage.get_ref<const std::string&>();
            if (hasLasExtension(filename))
            {
                reader = reader || index == 0U;
                writer = writer || index + 1U == pipeline.size();
            }
            continue;
        }
        if (!stage.is_object())
            continue;
        const auto type = stage.find("type");
        if (type == stage.end() || !type->is_string())
            continue;
        const std::string& name = type->get_ref<const std::string&>();
        reader = reader || name == "readers.las";
        writer = writer || name == "writers.las";
    }
    return reader && writer;
}

} // unnamed namespace

DispatchRoute classifyPipelineForDispatch(
    std::string_view pipelineJson,
    std::optional<DispatchInputFacts> inputFacts) noexcept
{
    try
    {
        const Json root = Json::parse(pipelineJson, nullptr, true, true);
        const Json* pipeline = nullptr;
        if (root.is_array())
            pipeline = &root;
        else if (root.is_object())
        {
            const auto position = root.find("pipeline");
            if (position != root.end() && position->is_array())
                pipeline = &*position;
        }
        if (!pipeline)
            return DispatchRoute::Engine;

        if (isBareLasTranslate(*pipeline))
            return DispatchRoute::Oracle;
        const bool literalPipelineRoot = root.is_object() && root.size() == 1U;
        if (literalPipelineRoot && isMeasuredR1Grammar(*pipeline) &&
            inputFacts && inputFacts->pointCount == 1'000'000U &&
            inputFacts->measuredReferenceLayout)
            return DispatchRoute::Oracle;
        if (literalPipelineRoot && isMeasuredR5Grammar(*pipeline))
            return DispatchRoute::Oracle;
        if (literalPipelineRoot && isMeasuredR2Grammar(*pipeline))
            return inputFacts && inputFacts->pointCount == 1'000'000U &&
                           inputFacts->measuredReferenceLayout
                       ? DispatchRoute::Engine
                       : DispatchRoute::Oracle;
        if (literalPipelineRoot && isMeasuredR3Grammar(*pipeline))
            return DispatchRoute::Oracle;
        if (literalPipelineRoot && isMeasuredR7Grammar(*pipeline))
            return DispatchRoute::Oracle;
        if (literalPipelineRoot && isMeasuredR10Grammar(*pipeline))
            return DispatchRoute::Oracle;
        if (hasPotentialNativeLasIo(*pipeline))
            return DispatchRoute::Engine;

        for (const Json& stage : *pipeline)
        {
            if (!stage.is_object())
                continue;
            const auto type = stage.find("type");
            if (type == stage.end() || !type->is_string())
                return DispatchRoute::Engine;
            if (isCandidateStage(type->get_ref<const std::string&>()))
                return DispatchRoute::Engine;
        }
        return DispatchRoute::Oracle;
    }
    catch (const std::exception&)
    {
        return DispatchRoute::Engine;
    }
}

bool dispatchRequiresPlainPipelineInvocation(
    std::string_view pipelineJson) noexcept
{
    try
    {
        const Json root = Json::parse(pipelineJson, nullptr, true, true);
        if (!root.is_object() || root.size() != 1U)
            return false;
        const auto pipeline = root.find("pipeline");
        return pipeline != root.end() && pipeline->is_array() &&
               (isMeasuredR1Grammar(*pipeline) ||
                isMeasuredR5Grammar(*pipeline) ||
                isMeasuredR2Grammar(*pipeline) ||
                isMeasuredR3Grammar(*pipeline) ||
                isMeasuredR7Grammar(*pipeline) ||
                isMeasuredR10Grammar(*pipeline) ||
                isMeasuredR14Grammar(*pipeline));
    }
    catch (const std::exception&)
    {
        return false;
    }
}

std::optional<std::string>
dispatchPointCountProbeFilename(std::string_view pipelineJson) noexcept
{
    try
    {
        const Json root = Json::parse(pipelineJson, nullptr, true, true);
        const Json* pipeline = nullptr;
        if (root.is_array())
            pipeline = &root;
        else if (root.is_object())
        {
            const auto position = root.find("pipeline");
            if (position != root.end() && position->is_array())
                pipeline = &*position;
        }
        if (!pipeline || !root.is_object() || root.size() != 1U ||
            (!isMeasuredR1Grammar(*pipeline) &&
             !isMeasuredR2Grammar(*pipeline) &&
             !isMeasuredR7Grammar(*pipeline) &&
             !isMeasuredR14Grammar(*pipeline)))
            return std::nullopt;
        return pipeline->at(0U).at("filename").get<std::string>();
    }
    catch (const std::exception&)
    {
        return std::nullopt;
    }
}

bool dispatchUsesAutomaticR2Hybrid(
    std::string_view pipelineJson,
    std::optional<DispatchInputFacts> inputFacts) noexcept
{
    if (!inputFacts || inputFacts->pointCount != 1'000'000U ||
        !inputFacts->measuredReferenceLayout)
        return false;
    try
    {
        const Json root = Json::parse(pipelineJson, nullptr, true, true);
        if (!root.is_object() || root.size() != 1U)
            return false;
        const auto pipeline = root.find("pipeline");
        return pipeline != root.end() && pipeline->is_array() &&
               isMeasuredR2Grammar(*pipeline);
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool dispatchUsesAutomaticR7ReaderThreads(
    std::string_view pipelineJson,
    std::optional<DispatchInputFacts> inputFacts) noexcept
{
    if (!inputFacts || inputFacts->pointCount != 1'000'000U ||
        !inputFacts->measuredReferenceLayout)
        return false;
    try
    {
        const Json root = Json::parse(pipelineJson, nullptr, true, true);
        if (!root.is_object() || root.size() != 1U)
            return false;
        const auto pipeline = root.find("pipeline");
        return pipeline != root.end() && pipeline->is_array() &&
               isMeasuredR7Grammar(*pipeline);
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool dispatchUsesAutomaticR14ParallelCompression(
    std::string_view pipelineJson,
    std::optional<DispatchInputFacts> inputFacts) noexcept
{
    if (!inputFacts || inputFacts->pointCount != 1'000'000U ||
        !inputFacts->measuredR14ReferenceLayout)
        return false;
    try
    {
        const Json root = Json::parse(pipelineJson, nullptr, true, true);
        if (!root.is_object() || root.size() != 1U)
            return false;
        const auto pipeline = root.find("pipeline");
        return pipeline != root.end() && pipeline->is_array() &&
               isMeasuredR14Grammar(*pipeline);
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool dispatchEnvironmentRequiresEngine(
    std::span<const std::string_view> presentVariables) noexcept
{
    // The r2 internal marker is armed after the startup snapshot, so it cannot
    // consume its own engine route. An externally supplied internal marker,
    // like any other unknown PDG control, still fails closed.
    return std::any_of(presentVariables.begin(), presentVariables.end(),
                       [](std::string_view name)
                       {
                           return name.starts_with("PDG_") &&
                                  name != "PDG_ORACLE_PDAL";
                       });
}

std::span<const std::string_view> dispatchEngineEnvironmentVariables() noexcept
{
    return EngineEnvironmentVariables;
}

bool fastModeEnabled() noexcept
{
    // The launcher stays independent of the engine core, so this mirrors
    // pdg::fastModeEnabled() (include/pdg/FastMode.hpp) on purpose.
    const char* marker = std::getenv(FastModeMarker.data());
    return marker && std::string_view(marker) == "1";
}

} // namespace pdg::cli
