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
void saveDatasetToFile(const CompressedDataset& dataset,
                       const std::string& filename)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out)
        throw std::runtime_error("Failed to open file for writing");

    // ============================================================
    // FILE HEADER
    // ============================================================
    uint32_t magicNumber = 0x43534244; // 'CSBD'
    uint32_t version = 1;
    uint32_t numBricks = static_cast<uint32_t>(dataset.bricks.size());
    uint32_t brickSize = 0;

    out.write(reinterpret_cast<const char*>(&magicNumber), sizeof(magicNumber));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&numBricks), sizeof(numBricks));
    out.write(reinterpret_cast<const char*>(&brickSize), sizeof(brickSize));

    // ============================================================
    // RANS MODEL SERIALIZATION
    // ============================================================
    auto writeRansModel = [&](const RansModel& model)
        {
            uint32_t symbolCount = static_cast<uint32_t>(model.freq.size());

            if (model.cumulativeFreq.size() != symbolCount + 1)
                throw std::runtime_error("Invalid RANS model");

            out.write(reinterpret_cast<const char*>(&symbolCount), sizeof(symbolCount));
            out.write(reinterpret_cast<const char*>(&model.totalFreq), sizeof(model.totalFreq));
            out.write(reinterpret_cast<const char*>(model.freq.data()), symbolCount * sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(model.cumulativeFreq.data()), symbolCount * sizeof(uint32_t));
        };

    writeRansModel(dataset.interiorModel);
    writeRansModel(dataset.leafModel);

    // ============================================================
    // BRICKS
    // ============================================================
    for (const auto& brick : dataset.bricks)
    {
        // ------------------------------
        // Write the brick ID (uint64_t)
        // ------------------------------
        out.write(reinterpret_cast<const char*>(&brick.ID), sizeof(brick.ID));

        // ------------------------------
        // Palette
        // ------------------------------
        uint32_t paletteSize = static_cast<uint32_t>(brick.palette.size());
        out.write(reinterpret_cast<const char*>(&paletteSize), sizeof(paletteSize));
        out.write(reinterpret_cast<const char*>(brick.palette.data()), paletteSize * sizeof(LabelType));

        // ------------------------------
        // Encoded data
        // ------------------------------
        uint32_t encodedSize = static_cast<uint32_t>(brick.encodedData.size());
        out.write(reinterpret_cast<const char*>(&encodedSize), sizeof(encodedSize));
        out.write(reinterpret_cast<const char*>(brick.encodedData.data()), encodedSize);

        // ------------------------------
        // Leaf flags
        // ------------------------------
        uint32_t leafSize = static_cast<uint32_t>(brick.isLeaf.size());
        out.write(reinterpret_cast<const char*>(&leafSize), sizeof(leafSize));
        out.write(reinterpret_cast<const char*>(brick.isLeaf.data()), leafSize);

        // ------------------------------
        // Optional checksum
        // ------------------------------
        if constexpr (useChecksum)
        {
            uint32_t crc = crc32(
                0L,
                reinterpret_cast<const Bytef*>(brick.encodedData.data()),
                static_cast<uInt>(brick.encodedData.size())
            );
            out.write(reinterpret_cast<const char*>(&crc), sizeof(crc));
        }
    }

    out.close();
}

CompressedDataset loadDatasetFromFile(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        throw std::runtime_error("Failed to open file");

    CompressedDataset dataset;

    // -----------------------------
    // HEADER
    // -----------------------------
    uint32_t magicNumber, version, numBricks, brickSize;
    in.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&numBricks), sizeof(numBricks));
    in.read(reinterpret_cast<char*>(&brickSize), sizeof(brickSize));

    if (magicNumber != 0x43534244)
        throw std::runtime_error("Invalid file format");

    // -----------------------------
    // RANS MODELS
    // -----------------------------
    auto readRansModel = [&](RansModel& model)
        {
            uint32_t symbolCount;
            in.read(reinterpret_cast<char*>(&symbolCount), sizeof(symbolCount));
            in.read(reinterpret_cast<char*>(&model.totalFreq), sizeof(model.totalFreq));

            model.freq.resize(symbolCount);
            model.cumulativeFreq.resize(symbolCount);

            in.read(reinterpret_cast<char*>(model.freq.data()), symbolCount * sizeof(uint32_t));
            in.read(reinterpret_cast<char*>(model.cumulativeFreq.data()), symbolCount * sizeof(uint32_t));
        };

    readRansModel(dataset.interiorModel);
    readRansModel(dataset.leafModel);

    // -----------------------------
    // BRICKS
    // -----------------------------
    dataset.bricks.clear();
    dataset.bricks.reserve(numBricks);

    for (uint32_t i = 0; i < numBricks; ++i)
    {
        CompressedBrick brick;

        // --- brickID ---
        in.read(reinterpret_cast<char*>(&brick.ID), sizeof(brick.ID));

        // --- palette ---
        uint32_t paletteSize;
        in.read(reinterpret_cast<char*>(&paletteSize), sizeof(paletteSize));
        brick.palette.resize(paletteSize);
        in.read(reinterpret_cast<char*>(brick.palette.data()), paletteSize * sizeof(LabelType));

        // --- encodedData ---
        uint32_t encodedSize;
        in.read(reinterpret_cast<char*>(&encodedSize), sizeof(encodedSize));
        brick.encodedData.resize(encodedSize);
        in.read(reinterpret_cast<char*>(brick.encodedData.data()), encodedSize);

        // --- isLeaf ---
        uint32_t leafSize;
        in.read(reinterpret_cast<char*>(&leafSize), sizeof(leafSize));
        brick.isLeaf.resize(leafSize);
        in.read(reinterpret_cast<char*>(brick.isLeaf.data()), leafSize);

        // --- optional CRC (skip or verify) ---
        if constexpr (useChecksum)
        {
            uint32_t crc;
            in.read(reinterpret_cast<char*>(&crc), sizeof(crc));
            // Optionally verify crc32(brick.encodedData)
        }

        dataset.bricks.push_back(std::move(brick));
    }

    return dataset;
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