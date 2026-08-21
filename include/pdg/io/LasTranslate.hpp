#pragma once

#include <pdg/io/Las.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace pdg::las
{

struct DefaultTranslationMetadata
{
    std::uint16_t creationDayOfYear = 1;
    std::uint16_t creationYear = 1900;
    std::string softwareId;
};

struct ExtraDoubleDimension
{
    std::string_view name;
    std::string_view description;
};

[[nodiscard]] std::string oracleSoftwareId();
[[nodiscard]] bool supportsDefaultTranslation(const FileView& input) noexcept;
[[nodiscard]] std::size_t defaultTranslationSize(const FileView& input);
void translateDefaultInto(const FileView& input,
                          const DefaultTranslationMetadata& metadata,
                          std::span<std::byte> output,
                          std::size_t maximumWorkers = 0);
// Overlays one final standard dimension onto the canonical LAS 1.4 / format 7
// image produced by translateDefaultInto. The narrow first endpoint envelope
// supports only explicitly declared byte overlays; callers must retain pinned
// PDAL for any other writer option, record layout, or cardinality.
void overlayDefaultUserData(std::span<std::byte> canonicalOutput,
                            std::span<const std::uint8_t> values);
void overlayDefaultClassification(std::span<std::byte> canonicalOutput,
                                  std::span<const std::uint8_t> values);
// Publishes the same canonical image as translateDefaultInto, but writes
// complete records in caller-proved source order. The source order must be a
// complete permutation; this keeps exact global-order filters from silently
// duplicating records or changing cardinality.
void translateDefaultPermutedInto(const FileView& input,
                                  const DefaultTranslationMetadata& metadata,
                                  std::span<const std::uint64_t> sourceOrder,
                                  std::span<std::byte> output,
                                  std::size_t maximumWorkers = 0);
// The classification variant applies one destination-ordered byte overlay
// after the complete records have been permuted.
void translateDefaultPermutedClassificationInto(
    const FileView& input, const DefaultTranslationMetadata& metadata,
    std::span<const std::uint64_t> sourceOrder,
    std::span<const std::uint8_t> classification, std::span<std::byte> output,
    std::size_t maximumWorkers = 0);
[[nodiscard]] std::vector<std::byte>
translateDefault(const FileView& input,
                 const DefaultTranslationMetadata& metadata,
                 std::size_t maximumWorkers = 0);

// Separate, deliberately narrow writer envelope for retaining one existing
// scalar Extra Bytes field and appending one standard binary64 result. Values
// are raw IEEE-754 bits so publication cannot canonicalize signed zero, NaNs,
// or infinities. Unsupported LAS/VLR layouts remain on pinned PDAL.
[[nodiscard]] bool
supportsExtraDoubleTranslation(const FileView& input) noexcept;
[[nodiscard]] std::size_t extraDoubleTranslationSize(const FileView& input);
void translateExtraDoubleInto(const FileView& input,
                              const DefaultTranslationMetadata& metadata,
                              ExtraDoubleDimension dimension,
                              std::span<const std::uint64_t> valueBits,
                              std::span<std::byte> output,
                              std::size_t maximumWorkers = 0);
[[nodiscard]] std::vector<std::byte> translateExtraDouble(
    const FileView& input, const DefaultTranslationMetadata& metadata,
    ExtraDoubleDimension dimension, std::span<const std::uint64_t> valueBits,
    std::size_t maximumWorkers = 0);

} // namespace pdg::las
