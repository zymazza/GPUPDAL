#include <pdg/PointBatch.hpp>
#include <pdg/stages/Transformation.hpp>

#include <cstddef>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pdg
{

namespace
{
void requireDoubleCoordinates(const PointBatch& batch)
{
    for (const StandardDimension dimension :
         {StandardDimension::X, StandardDimension::Y, StandardDimension::Z})
    {
        const DimensionId id(dimension);
        if (!batch.has(id) ||
            batch.columnInfo(id).physicalType != DimensionType::Double)
            throw std::invalid_argument(
                "transformation requires materialized double XYZ columns");
    }
}

void executeTransformationHost(PointBatch& batch,
                               const TransformationProgram& program)
{
    requireDoubleCoordinates(batch);
    double* xValues = batch.data<double>(DimensionId(StandardDimension::X));
    double* yValues = batch.data<double>(DimensionId(StandardDimension::Y));
    double* zValues = batch.data<double>(DimensionId(StandardDimension::Z));
    const auto& matrix = program.matrix;
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        const double x = xValues[point];
        const double y = yValues[point];
        const double z = zValues[point];
        double scale = x * matrix[12];
        scale = scale + y * matrix[13];
        scale = scale + z * matrix[14];
        scale = scale + matrix[15];

        double outputX = x * matrix[0];
        outputX = outputX + y * matrix[1];
        outputX = outputX + z * matrix[2];
        outputX = outputX + matrix[3];
        outputX = outputX / scale;

        double outputY = x * matrix[4];
        outputY = outputY + y * matrix[5];
        outputY = outputY + z * matrix[6];
        outputY = outputY + matrix[7];
        outputY = outputY / scale;

        double outputZ = x * matrix[8];
        outputZ = outputZ + y * matrix[9];
        outputZ = outputZ + z * matrix[10];
        outputZ = outputZ + matrix[11];
        outputZ = outputZ / scale;

        xValues[point] = outputX;
        yValues[point] = outputY;
        zValues[point] = outputZ;
    }
}
} // unnamed namespace

void executeTransformationDevice(PointBatch& batch,
                                 const TransformationProgram& program);

TransformationProgram compileTransformation(std::string_view specification)
{
    std::istringstream input{std::string(specification)};
    input.imbue(std::locale::classic());
    TransformationProgram program;
    std::size_t count = 0;
    double value = 0.0;
    while (input >> value)
    {
        if (count == program.matrix.size())
            throw std::invalid_argument(
                "too many entries in transformation matrix; expected 16");
        program.matrix[count++] = value;
    }
    if (count != program.matrix.size())
        throw std::invalid_argument(
            "too few entries in transformation matrix; expected 16");
    return program;
}

bool transformationSupportsExactDevice(
    const TransformationProgram& program) noexcept
{
    return program.matrix[12] == 0.0 && program.matrix[13] == 0.0 &&
           program.matrix[14] == 0.0 && program.matrix[15] == 1.0;
}

bool transformationSupportsExactDevice(
    const PointBatch& batch, const TransformationProgram& program) noexcept
{
    if (!transformationSupportsExactDevice(program))
        return false;
    for (const StandardDimension dimension :
         {StandardDimension::X, StandardDimension::Y, StandardDimension::Z})
    {
        const DimensionId id(dimension);
        if (!batch.has(id) ||
            batch.columnInfo(id).physicalType != DimensionType::Double)
            return false;
    }
    return true;
}

void executeTransformation(PointBatch& batch,
                           const TransformationProgram& program)
{
    if (batch.memoryKind() == MemoryKind::Device)
    {
        executeTransformationDevice(batch, program);
        return;
    }
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported transformation memory kind");
    executeTransformationHost(batch, program);
}

} // namespace pdg
