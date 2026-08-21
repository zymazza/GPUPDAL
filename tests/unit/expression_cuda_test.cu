#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Assign.hpp>
#include <pdg/stages/Crop.hpp>
#include <pdg/stages/Expression.hpp>
#include <pdg/stages/Range.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{
pdg::CoordinateEncoding cudaAssignCoordinates()
{
    return {{0.001, 0.001, 0.001}, {500000.0, 4800000.0, 100.0}};
}

bool cudaDeviceAvailable()
{
    try
    {
        return !pdg::cudaDevices().empty();
    }
    catch (const pdg::CudaError&)
    {
        return false;
    }
}

void copyToDevice(pdg::PointBatch& device, const pdg::PointBatch& host,
                  pdg::DimensionId id)
{
    const pdg::ColumnInfo& column = host.columnInfo(id);
    const std::size_t bytes =
        host.size() * pdg::dimensionTypeSize(column.physicalType);
    const auto stream = static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(id), host.rawData(id), bytes,
                                   cudaMemcpyHostToDevice, stream));
}

void expectDeviceColumnEqualsHost(const pdg::PointBatch& device,
                                  const pdg::PointBatch& host,
                                  pdg::DimensionId id)
{
    const pdg::ColumnInfo& column = host.columnInfo(id);
    const std::size_t bytes =
        host.size() * pdg::dimensionTypeSize(column.physicalType);
    std::vector<std::byte> actual(bytes);
    const auto stream = static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), device.rawData(id), bytes,
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto* expected = static_cast<const std::byte*>(host.rawData(id));
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected);
    EXPECT_EQ(mismatch.first, actual.end())
        << "dimension " << id.value() << ", first differing byte "
        << static_cast<std::size_t>(mismatch.first - actual.begin());
}
} // unnamed namespace

TEST(CudaAssign, OrderedArithmeticAndLogicalProgramMatchesHostBytes)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 65537;
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId source =
        dimensions.registerCustom("Source", pdg::DimensionType::Double).id;
    const pdg::DimensionId denominator =
        dimensions.registerCustom("Denominator", pdg::DimensionType::Double).id;
    const pdg::DimensionId middle =
        dimensions.registerCustom("Middle", pdg::DimensionType::Double).id;
    const pdg::DimensionId destination =
        dimensions.registerCustom("Destination", pdg::DimensionType::Signed32)
            .id;
    const pdg::DimensionId divide =
        dimensions.registerCustom("Divide", pdg::DimensionType::Double).id;
    const pdg::DimensionId flag =
        dimensions.registerCustom("AssignFlag", pdg::DimensionType::Unsigned8)
            .id;
    const std::vector<std::string> specifications = {
        "Middle = Source * 7 - 3", "Destination = Middle / 2",
        "AssignFlag = 1 WHERE Destination >= -100 && Destination <= 100",
        "Divide = Source / Denominator", "AssignFlag = 2 WHERE isnan(Divide)"};
    const pdg::AssignProgram program =
        pdg::compileAssignments(specifications, dimensions);
    ASSERT_TRUE(pdg::assignSupportsExactDevice(program));

    pdg::HostMemoryResource hostMemory;
    auto deviceMemory = pdg::makeCudaMemoryResource(32U * 1024U * 1024U);
    pdg::PointBatch host(Count, cudaAssignCoordinates(), dimensions,
                         hostMemory);
    pdg::PointBatch device(Count, cudaAssignCoordinates(), dimensions,
                           *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(source);
        batch->materialize(denominator);
        batch->setSize(Count);
    }
    for (std::size_t index = 0; index < Count; ++index)
    {
        const std::int64_t centered =
            static_cast<std::int64_t>((index * 7919U) % 10000U) - 5000;
        host.data<double>(source)[index] = static_cast<double>(centered) / 8.0;
        host.data<double>(denominator)[index] =
            index % 17U == 0 ? 0.0 : static_cast<double>(index % 13U + 1U);
    }
    copyToDevice(device, host, source);
    copyToDevice(device, host, denominator);

    pdg::executeAssign(host, program, 1);
    pdg::executeAssign(device, program);
    for (pdg::DimensionId id : {middle, destination, divide, flag})
        expectDeviceColumnEqualsHost(device, host, id);
}

TEST(CudaAssign, PreservesProgramOrderAcrossLaunchChunks)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 32769;
    constexpr std::size_t AssignmentCount = 19;
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId source =
        dimensions.registerCustom("Source", pdg::DimensionType::Signed32).id;
    std::array<pdg::DimensionId, AssignmentCount> chain;
    std::vector<std::string> specifications;
    specifications.reserve(AssignmentCount);
    for (std::size_t index = 0; index < AssignmentCount; ++index)
    {
        const std::string name = "Chain" + std::to_string(index);
        chain[index] =
            dimensions.registerCustom(name, pdg::DimensionType::Signed32).id;
        specifications.push_back(name + " = " +
                                 (index == 0
                                      ? std::string("Source")
                                      : "Chain" + std::to_string(index - 1U)) +
                                 " + 1");
    }
    const pdg::AssignProgram program =
        pdg::compileAssignments(specifications, dimensions);
    ASSERT_TRUE(pdg::assignSupportsExactDevice(program));

    pdg::HostMemoryResource hostMemory;
    auto deviceMemory = pdg::makeCudaMemoryResource(16U * 1024U * 1024U);
    pdg::PointBatch host(Count, cudaAssignCoordinates(), dimensions,
                         hostMemory);
    pdg::PointBatch device(Count, cudaAssignCoordinates(), dimensions,
                           *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(source);
        batch->setSize(Count);
    }
    for (std::size_t index = 0; index < Count; ++index)
        host.data<std::int32_t>(source)[index] =
            static_cast<std::int32_t>(index % 10000U) - 5000;
    copyToDevice(device, host, source);

    pdg::executeAssign(host, program, 1);
    pdg::executeAssign(device, program);
    expectDeviceColumnEqualsHost(device, host, chain.back());
}

TEST(CudaAssign, SpecialDoubleValuesMatchHostBitForBit)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 4097;
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId source =
        dimensions.registerCustom("Source", pdg::DimensionType::Double).id;
    const std::array<std::string, 5> destinationNames = {
        "Copied", "Negated", "Added", "Multiplied", "Divided"};
    std::array<pdg::DimensionId, destinationNames.size()> destinations;
    for (std::size_t index = 0; index < destinationNames.size(); ++index)
        destinations[index] = dimensions
                                  .registerCustom(destinationNames[index],
                                                  pdg::DimensionType::Double)
                                  .id;
    const pdg::DimensionId flag =
        dimensions.registerCustom("AssignFlag", pdg::DimensionType::Unsigned8)
            .id;
    const std::vector<std::string> specifications = {
        "Copied = Source",
        "Negated = -Source",
        "Added = Source + 1",
        "Multiplied = Source * -2",
        "Divided = Source / 3",
        "AssignFlag = 1 WHERE isnan(Source) || Source == highest() || Source "
        "== lowest()"};
    const pdg::AssignProgram program =
        pdg::compileAssignments(specifications, dimensions);
    ASSERT_TRUE(pdg::assignSupportsExactDevice(program));

    pdg::HostMemoryResource hostMemory;
    auto deviceMemory = pdg::makeCudaMemoryResource(8U * 1024U * 1024U);
    pdg::PointBatch host(Count, cudaAssignCoordinates(), dimensions,
                         hostMemory);
    pdg::PointBatch device(Count, cudaAssignCoordinates(), dimensions,
                           *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(source);
        batch->setSize(Count);
    }
    constexpr std::array<std::uint64_t, 11> ValueBits = {
        0x0000000000000000ULL, 0x8000000000000000ULL, 0x0000000000000001ULL,
        0x8000000000000001ULL, 0x7fefffffffffffffULL, 0xffefffffffffffffULL,
        0x7ff0000000000000ULL, 0xfff0000000000000ULL, 0x7ff8000000001234ULL,
        0xfff8000000005678ULL, 0x7ff0000000001234ULL};
    for (std::size_t index = 0; index < Count; ++index)
        host.data<double>(source)[index] =
            std::bit_cast<double>(ValueBits[index % ValueBits.size()]);
    copyToDevice(device, host, source);

    pdg::executeAssign(host, program, 1);
    pdg::executeAssign(device, program);
    for (pdg::DimensionId destination : destinations)
        expectDeviceColumnEqualsHost(device, host, destination);
    expectDeviceColumnEqualsHost(device, host, flag);
}

TEST(CudaAssign, GenericDoubleCoordinatesMatchHostBitForBit)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 65537;
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId y(pdg::StandardDimension::Y);
    const pdg::DimensionId z(pdg::StandardDimension::Z);
    const pdg::DimensionId classification(
        pdg::StandardDimension::Classification);
    const std::vector<std::string> specifications = {
        "Z = X * 0.5 + Y", "Classification = 9 WHERE Z >= 1000"};
    const pdg::AssignProgram program =
        pdg::compileAssignments(specifications, dimensions);
    EXPECT_FALSE(pdg::assignSupportsExactDevice(program));

    pdg::HostMemoryResource hostMemory;
    auto deviceMemory = pdg::makeCudaMemoryResource(16U * 1024U * 1024U);
    pdg::PointBatch host(Count, cudaAssignCoordinates(), dimensions,
                         hostMemory);
    pdg::PointBatch device(Count, cudaAssignCoordinates(), dimensions,
                           *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(x, pdg::DimensionType::Double);
        batch->materialize(y, pdg::DimensionType::Double);
        batch->materialize(z, pdg::DimensionType::Double);
        batch->materialize(classification);
        batch->setSize(Count);
    }
    ASSERT_TRUE(pdg::assignSupportsExactDevice(device, program));
    for (std::size_t index = 0; index < Count; ++index)
    {
        host.data<double>(x)[index] =
            500000.0 + static_cast<double>(index % 1009U) * 0.001;
        host.data<double>(y)[index] =
            700.0 + static_cast<double>(index % 997U) * 0.125;
        host.data<double>(z)[index] = -1.0;
        host.data<std::uint8_t>(classification)[index] = 2;
    }
    for (pdg::DimensionId id : {x, y, z, classification})
        copyToDevice(device, host, id);

    pdg::executeAssign(host, program, 1);
    pdg::executeAssign(device, program);
    expectDeviceColumnEqualsHost(device, host, z);
    expectDeviceColumnEqualsHost(device, host, classification);
}

TEST(CudaAssign, RejectsOperationsOutsideExactDeviceEnvelope)
{
    pdg::DimensionRegistry dimensions;
    dimensions.registerCustom("Source", pdg::DimensionType::Double);
    const std::vector<std::string> specifications = {"Result = sqrt(Source)"};
    const pdg::AssignProgram program =
        pdg::compileAssignments(specifications, dimensions);
    EXPECT_FALSE(pdg::assignSupportsExactDevice(program));

    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    auto memory = pdg::makeCudaMemoryResource();
    pdg::PointBatch batch(0, cudaAssignCoordinates(), dimensions, *memory);
    EXPECT_THROW(pdg::executeAssign(batch, program), std::invalid_argument);
}

TEST(CudaPredicate, GenericCoordinatesMatchHostMaskBitForBit)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 65537;
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId intensity(pdg::StandardDimension::Intensity);
    const pdg::PredicateProgram predicate = pdg::compilePredicate(
        "(X >= 500000.25 && X < 500000.75) || Intensity == 17", dimensions);
    EXPECT_FALSE(pdg::predicateSupportsExactDevice(predicate));

    pdg::HostMemoryResource hostMemory;
    auto deviceMemory = pdg::makeCudaMemoryResource(8U * 1024U * 1024U);
    pdg::PointBatch host(Count, cudaAssignCoordinates(), dimensions,
                         hostMemory);
    pdg::PointBatch device(Count, cudaAssignCoordinates(), dimensions,
                           *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(x, pdg::DimensionType::Double);
        batch->materialize(intensity);
        batch->setSize(Count);
    }
    ASSERT_TRUE(pdg::predicateSupportsExactDevice(device, predicate));
    for (std::size_t index = 0; index < Count; ++index)
    {
        host.data<double>(x)[index] =
            500000.0 + static_cast<double>(index % 1001U) * 0.001;
        host.data<std::uint16_t>(intensity)[index] =
            static_cast<std::uint16_t>(index % 257U);
    }
    copyToDevice(device, host, x);
    copyToDevice(device, host, intensity);

    std::vector<std::uint8_t> expected(Count);
    pdg::evaluatePredicate(host, predicate, expected.data(), 1);
    std::unique_ptr<pdg::Allocation> deviceKeep =
        deviceMemory->allocate(Count, alignof(std::uint8_t));
    pdg::evaluatePredicate(device, predicate,
                           static_cast<std::uint8_t*>(deviceKeep->data()));
    std::vector<std::uint8_t> actual(Count);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), deviceKeep->data(), Count,
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(actual, expected);
}

TEST(CudaPredicate, CanonicalEncodedCoordinatesMatchHostMaskBitForBit)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 65537;
    const pdg::CoordinateEncoding coordinates({0.01, 0.01, 0.01},
                                              {0.0, 0.0, 0.0});
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId intensity(pdg::StandardDimension::Intensity);
    const pdg::PredicateProgram predicate = pdg::compilePredicate(
        "(X >= -1234.5 && X < 2345.67) || Intensity == 17", dimensions);
    EXPECT_FALSE(pdg::predicateSupportsExactDevice(predicate));

    pdg::HostMemoryResource hostMemory;
    auto deviceMemory = pdg::makeCudaMemoryResource(8U * 1024U * 1024U);
    pdg::PointBatch host(Count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(Count, coordinates, dimensions, *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(x, pdg::DimensionType::Signed32);
        batch->materialize(intensity);
        batch->setSize(Count);
    }
    ASSERT_TRUE(pdg::predicateSupportsExactDevice(device, predicate));
    for (std::size_t index = 0; index < Count; ++index)
    {
        host.data<std::int32_t>(x)[index] =
            static_cast<std::int32_t>((index * 7919U) % 800001U) - 400000;
        host.data<std::uint16_t>(intensity)[index] =
            static_cast<std::uint16_t>(index % 257U);
    }
    copyToDevice(device, host, x);
    copyToDevice(device, host, intensity);

    std::vector<std::uint8_t> expected(Count);
    pdg::evaluatePredicate(host, predicate, expected.data(), 1);
    std::unique_ptr<pdg::Allocation> deviceKeep =
        deviceMemory->allocate(Count, alignof(std::uint8_t));
    pdg::evaluatePredicate(device, predicate,
                           static_cast<std::uint8_t*>(deviceKeep->data()));
    std::vector<std::uint8_t> actual(Count);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), deviceKeep->data(), Count,
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(actual, expected);
}

TEST(CudaPredicate, RangeBoundsNegationAndNanMatchHostBitForBit)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 65537;
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId source =
        dimensions.registerCustom("Source", pdg::DimensionType::Double).id;
    const pdg::DimensionId classification(
        pdg::StandardDimension::Classification);
    const std::vector<std::string> limits = {"Source(-4.5:2], Source![7:9)",
                                             "Classification[1:4]"};
    const pdg::PredicateProgram predicate =
        pdg::compileRangePredicate(limits, dimensions);

    pdg::HostMemoryResource hostMemory;
    auto deviceMemory = pdg::makeCudaMemoryResource(8U * 1024U * 1024U);
    pdg::PointBatch host(Count, cudaAssignCoordinates(), dimensions,
                         hostMemory);
    pdg::PointBatch device(Count, cudaAssignCoordinates(), dimensions,
                           *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(source);
        batch->materialize(classification);
        batch->setSize(Count);
    }
    ASSERT_TRUE(pdg::predicateSupportsExactDevice(device, predicate));
    for (std::size_t index = 0; index < Count; ++index)
    {
        host.data<double>(source)[index] =
            index % 257U == 0U
                ? std::numeric_limits<double>::quiet_NaN()
                : static_cast<double>(static_cast<std::int32_t>(index % 401U) -
                                      200) /
                      16.0;
        host.data<std::uint8_t>(classification)[index] =
            static_cast<std::uint8_t>(index % 8U);
    }
    copyToDevice(device, host, source);
    copyToDevice(device, host, classification);

    std::vector<std::uint8_t> expected(Count);
    pdg::evaluatePredicate(host, predicate, expected.data(), 1);
    std::unique_ptr<pdg::Allocation> deviceKeep =
        deviceMemory->allocate(Count, alignof(std::uint8_t));
    pdg::evaluatePredicate(device, predicate,
                           static_cast<std::uint8_t*>(deviceKeep->data()));
    std::vector<std::uint8_t> actual(Count);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), deviceKeep->data(), Count,
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(actual, expected);
}

TEST(CudaPredicate, CropBoundsOutsideAndNanMatchHostBitForBit)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 65537;
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId y(pdg::StandardDimension::Y);
    const pdg::DimensionId z(pdg::StandardDimension::Z);
    const pdg::PredicateProgram predicate = pdg::compileCropPredicate(
        "([-10,10],[-20,20],[-30,30])", true, dimensions);

    pdg::HostMemoryResource hostMemory;
    auto deviceMemory = pdg::makeCudaMemoryResource(8U * 1024U * 1024U);
    pdg::PointBatch host(Count, cudaAssignCoordinates(), dimensions,
                         hostMemory);
    pdg::PointBatch device(Count, cudaAssignCoordinates(), dimensions,
                           *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        for (pdg::DimensionId id : {x, y, z})
            batch->materialize(id, pdg::DimensionType::Double);
        batch->setSize(Count);
    }
    ASSERT_TRUE(pdg::predicateSupportsExactDevice(device, predicate));
    for (std::size_t index = 0; index < Count; ++index)
    {
        host.data<double>(x)[index] =
            static_cast<double>(static_cast<std::int32_t>(index % 45U) - 22);
        host.data<double>(y)[index] =
            static_cast<double>(static_cast<std::int32_t>(index % 65U) - 32);
        host.data<double>(z)[index] =
            static_cast<double>(static_cast<std::int32_t>(index % 85U) - 42);
    }
    host.data<double>(x)[0] = -10.0;
    host.data<double>(y)[0] = -20.0;
    host.data<double>(z)[0] = -30.0;
    host.data<double>(x)[1] = 10.0;
    host.data<double>(y)[1] = 20.0;
    host.data<double>(z)[1] = 30.0;
    host.data<double>(x)[2] = std::numeric_limits<double>::quiet_NaN();
    host.data<double>(y)[3] = std::numeric_limits<double>::quiet_NaN();
    host.data<double>(z)[4] = std::numeric_limits<double>::quiet_NaN();
    for (pdg::DimensionId id : {x, y, z})
        copyToDevice(device, host, id);

    std::vector<std::uint8_t> expected(Count);
    pdg::evaluatePredicate(host, predicate, expected.data(), 1);
    std::unique_ptr<pdg::Allocation> deviceKeep =
        deviceMemory->allocate(Count, alignof(std::uint8_t));
    pdg::evaluatePredicate(device, predicate,
                           static_cast<std::uint8_t*>(deviceKeep->data()));
    std::vector<std::uint8_t> actual(Count);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), deviceKeep->data(), Count,
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(actual[0], 0U);
    EXPECT_EQ(actual[1], 0U);
    EXPECT_EQ(actual[2], 1U);
    EXPECT_EQ(actual[3], 1U);
    EXPECT_EQ(actual[4], 1U);
}
