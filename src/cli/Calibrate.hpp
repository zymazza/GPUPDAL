#pragma once

namespace pdg::cli
{

// `gpupdal calibrate` (D0277): measures the placement calibration cases on this
// machine and writes a local placement profile, or reports the profile
// status.  Explicit only; nothing runs it implicitly.
[[nodiscard]] int runCalibrate(int argc, char** argv);

} // namespace pdg::cli
