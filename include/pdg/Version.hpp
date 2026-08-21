#pragma once

#include <string_view>

#ifndef PDG_ORACLE_COMMIT
#define PDG_ORACLE_COMMIT "unknown"
#endif
#ifndef PDG_ORACLE_VERSION
#define PDG_ORACLE_VERSION "unknown"
#endif
#ifndef GPUPAL_VERSION
#define GPUPAL_VERSION "0.1.0-dev"
#endif

namespace pdg
{
inline constexpr std::string_view Version = GPUPAL_VERSION;
inline constexpr std::string_view OracleCommit = PDG_ORACLE_COMMIT;
inline constexpr std::string_view OracleVersion = PDG_ORACLE_VERSION;
} // namespace pdg
