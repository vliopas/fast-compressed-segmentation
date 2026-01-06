#include "fileIO.hpp"
#include "cnpy.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <zlib.h>

using namespace Encoding;

constexpr bool useChecksum = false; // whether to use CRC32 checksums for bricks

void saveDatasetToFile(const CompressedDataset& dataset, const std::string& filename)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file for writing");

    // --- Header ---
    uint32_t magicNumber = 0x43534244; // 'CSBD' for CSV/Compressed Brick Dataset
    uint32_t numBricks   = static_cast<uint32_t>(dataset.bricks.size());
    uint32_t brickSize   = 0; // set appropriately if needed

    out.write(reinterpret_cast<const char*>(&magicNumber), sizeof(magicNumber));
    out.write(reinterpret_cast<const char*>(&numBricks), sizeof(numBricks));
    out.write(reinterpret_cast<const char*>(&brickSize), sizeof(brickSize));

    // --- RANS models ---
    auto writeRansModel = [&](const RansModel& model) {
        out.write(reinterpret_cast<const char*>(&model.totalFreq), sizeof(model.totalFreq));
        out.write(reinterpret_cast<const char*>(model.freq.data()), model.freq.size() * sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(model.cumulativeFreq.data()), model.cumulativeFreq.size() * sizeof(uint32_t));
    };

    writeRansModel(dataset.interiorModel);
    writeRansModel(dataset.leafModel);

    // --- Bricks ---
    for (const auto& brick : dataset.bricks)
    {
        uint32_t paletteSize = static_cast<uint32_t>(brick.palette.size());
        uint32_t dataSize    = static_cast<uint32_t>(brick.encodedData.size());

        out.write(reinterpret_cast<const char*>(&paletteSize), sizeof(paletteSize));
        out.write(reinterpret_cast<const char*>(brick.palette.data()), paletteSize * sizeof(LabelType));

        out.write(reinterpret_cast<const char*>(&dataSize), sizeof(dataSize));
        out.write(reinterpret_cast<const char*>(brick.encodedData.data()), dataSize);

        if constexpr (useChecksum)
        {
            uint32_t crc = crc32(
                0L,                                                // initial CRC
                reinterpret_cast<const Bytef*>(brick.encodedData.data()),
                static_cast<uInt>(brick.encodedData.size())
            );

           out.write(reinterpret_cast<const char*>(&crc), sizeof(crc));

        }
    }

    out.close();
}

NpyArray loadNpy(const std::string& filename)
{
    auto path = std::filesystem::current_path().parent_path().parent_path() / "dataset" / filename;
    cnpy::NpyArray arr = cnpy::npy_load(path.string());

    size_t total = 1;
    std::vector<size_t> shape(arr.shape.begin(), arr.shape.end());
    for (auto d : shape) total *= d;

    std::vector<uint64_t> data(total);

    if (arr.word_size == 4)
    {
        uint32_t* ptr = arr.data<uint32_t>();
        for (size_t i = 0; i < total; ++i)
            data[i] = static_cast<uint64_t>(ptr[i]);
    }
    else if (arr.word_size == 8)
    {
        uint64_t* ptr = arr.data<uint64_t>();
        std::copy(ptr, ptr + total, data.begin());
    }
    else
    {
        throw std::runtime_error("Unsupported .npy element size: " + std::to_string(arr.word_size));
    }

    return { std::move(data), std::move(shape) };
}