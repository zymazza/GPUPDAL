#pragma once

// B0150/D0212: LAZ decode for the engine's own LAS reader.
//
// The reference workflows all read and write LAZ, and `FileView::pointRecord`
// throws "compressed LAZ records require the chunk decoder" for compressed
// files, so runtime placement never sees valid facts and no acceleration is
// reachable. This decodes a compressed file's point records into the
// uncompressed layout the rest of the engine already understands, using the
// lazperf already vendored and built by the fork.
//
// This first slice is deliberately sequential and correctness-first. lazperf's
// `reader::chunk_decompressor` exposes per-chunk decode, which is the intended
// route to the chunk-parallel design P4 calls for; that is a separate slice and
// must be measured, not assumed.

#include <cstddef>
#include <filesystem>
#include <vector>

namespace pdg::las
{

class FileView;

// Decodes every point record of a compressed file into one contiguous buffer
// of `header().pointRecordLength`-sized records, in file order.
//
// Throws `Error` if the view is not compressed, if lazperf rejects the file, or
// if the decoded size does not match the header's declared point count and
// record length — a short or long decode is a corrupt read, never a partial
// success.
std::vector<std::byte> decodeCompressedPointRecords(const FileView& view);

// Validates the named LAZ publication without decoding the point cloud a
// second time. The LASzip chunk table must parse, describe the declared point
// count, and cover exactly the compressed point-payload extent. This is
// bounded in the number of chunks rather than the number of points.
//
// Throws when the file/header counts differ, the chunk table is missing or
// truncated, or its point/byte extents disagree with the public header.
void validateCompressedPointRecords(const std::filesystem::path& path,
                                    const FileView& view);

} // namespace pdg::las
