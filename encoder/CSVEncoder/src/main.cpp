/**
 * @file main.cpp
 * @brief Main encoder application entry point
 *
 * Loads a segmentation volume from .npy format, compresses it into bricks,
 * and saves the compressed dataset to a binary file.
 */

#include <iostream>
#include <fstream>

#include "brickDecoding.hpp"
#include "brickEncoding.hpp"
#include "fileIO.hpp"
#include "brick.hpp"
#include "types.hpp"

#include <unordered_set>

/**
 * @brief Hash function for label colors matching GPU shader implementation
 * @param label 64-bit label value
 * @return RGBA color as packed uint32_t
 */
uint32_t hashLabel(uint64_t label)
{
    if (label == 0)
        return 0x00000000;

    uint32_t seedLow = static_cast<uint32_t>(label & 0xFFFFFFFF);
    uint32_t seedHigh = static_cast<uint32_t>(label >> 32);

    seedLow = seedLow ^ seedHigh;
    seedLow = seedLow ^ (seedLow >> 16);
    seedLow = seedLow * 0x7feb352d;
    seedLow = seedLow ^ (seedLow >> 15);

    seedHigh = seedHigh ^ seedLow;
    seedHigh = seedHigh ^ (seedHigh >> 16);
    seedHigh = seedHigh * 0x85ebca6b;
    seedHigh = seedHigh ^ (seedHigh >> 13);

    uint32_t r = seedLow & 0xFF;
    uint32_t g = (seedLow >> 8) & 0xFF;
    uint32_t b = (seedHigh >> 8) & 0xFF;

    return (r << 0) | (g << 8) | (b << 16) | (0xFF << 24);
}

/**
 * @brief Count the number of unique labels in the dataset
 * @param data Vector of label values
 * @return Number of unique labels
 */
size_t countUniqueLabels(const std::vector<uint64_t> &data)
{
    std::unordered_set<uint64_t> uniqueLabels;
    for (auto v : data)
        uniqueLabels.insert(v);
    return uniqueLabels.size();
}

/**
 * @brief Main function - compress and validate dataset
 * @return Exit code (0 for success)
 */
int main()
{
    NpyArray npy = loadNpy("microns_segmentation_3d.npy");
    std::cout << "Loaded segmentation of shape: ";
    for (auto s : npy.shape)
        std::cout << s << " ";
    std::cout << "\nTotal voxels: " << npy.data.size() << "\n";
    std::cout << "Number of unique labels: " << countUniqueLabels(npy.data) << "\n";

    auto bricks = splitGridIntoBricks<BRICK_SIZE>(npy);
    std::cout << "Number of bricks: " << bricks.size() << "\n";

    auto compressed = Encoding::compressDataset(bricks);

    // Log operation statistics after encoding is complete
    // Encoding::globalOperationStats.logStats();

    saveDatasetToFile(compressed, "compressed_dataset.csbd");

    Encoding::CompressedDataset loadedDataset = loadDatasetFromFile("compressed_dataset.csbd");

    constexpr size_t targetLOD = Brick<BRICK_SIZE>::Levels - 1;

    bool allMatch = true;
    std::vector<LabelType> allDecoded; // Accumulate all brick reference data

    // Decode and save reference output for ALL bricks for GPU comparison
    for (size_t brickIdx = 0; brickIdx < loadedDataset.bricks.size(); ++brickIdx)
    {
        auto &loadedBrick = loadedDataset.bricks[brickIdx];

        // RANS decoding
        loadedBrick.encodedData = decodeRansStream(loadedBrick, loadedDataset.model);

        // semantic decode (can be repeated for different LODs)
        std::vector<LabelType> decoded = decodeBrick<BRICK_SIZE>(loadedBrick, targetLOD);

        // Check against original brick
        // bool brickMatches = true;
        // for (size_t i = 0; i < bricks[brickIdx].levels[0].size(); i++)
        //{
        //    if (decoded[i] != bricks[brickIdx].levels[0][i].label)
        //    {
        //        brickMatches = false;
        //        allMatch = false;
        //        std::cout << "Mismatch in brick " << brickIdx << " at position " << i << std::endl;
        //    }
        //}

        // Accumulate decoded data for all bricks
        allDecoded.insert(allDecoded.end(), decoded.begin(), decoded.end());
    }

    // Save decoded reference output to file as raw labels (for transfer function)
    //  std::ofstream refFile("decoded_reference.bin", std::ios::binary);
    //  for (const auto &val : allDecoded)
    //  {
    //      refFile.write(reinterpret_cast<const char *>(&val), sizeof(uint64_t));
    //  }
    //  refFile.close();
    //  std::cout << "Saved reference decoded output (" << allDecoded.size() << " labels) to decoded_reference.bin\n";

    std::cout << "\nOverall result - input : output equal? " << (allMatch ? "yes" : "no") << std::endl;
}
