#include <iostream>
#include <fstream>

#include "brickDecoding.hpp"
#include "brickEncoding.hpp"
#include "fileIO.hpp"
#include "brick.hpp"
#include "types.hpp"

#include <unordered_set>

// Hash function matching decode.wgsl hashLabel()
uint32_t hashLabel(uint64_t label)
{
    if (label == 0) return 0x00000000;

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

// Utility function to count unique labels in the dataset
size_t countUniqueLabels(const std::vector<uint64_t>& data)
{
    std::unordered_set<uint64_t> uniqueLabels;
    for (auto v : data)
        uniqueLabels.insert(v);
    return uniqueLabels.size();
}

int main() {
    NpyArray npy = loadNpy("microns_segmentation_3d.npy");
    std::cout << "Loaded segmentation of shape: ";
    for (auto s : npy.shape) std::cout << s << " ";
    std::cout << "\nTotal voxels: " << npy.data.size() << "\n";
    std::cout << "Number of unique labels: " << countUniqueLabels(npy.data) << "\n";

    auto bricks = splitGridIntoBricks<BRICK_SIZE>(npy);
    std::cout << "Number of bricks: " << bricks.size() << "\n";

    // auto& firstBrick = bricks[0];
    // auto& L0 = firstBrick.levels[0];
    // std::cout << "First voxel label in first brick: " << static_cast<int>(L0[0].label) << "\n";

    auto compressed = Encoding::compressDataset(bricks);
    
    // Log operation statistics after encoding is complete
    Encoding::globalOperationStats.logStats();
    
    saveDatasetToFile(compressed, "compressed_dataset.csbd");

    Encoding::CompressedDataset loadedDataset = loadDatasetFromFile("compressed_dataset.csbd");

    constexpr size_t targetLOD = Brick<BRICK_SIZE>::Levels - 1;

    bool allMatch = true;
    std::vector<LabelType> decoded;
    
        // Save first brick decoded output for GPU comparison
        const size_t bricksToTest = std::min(size_t(1), loadedDataset.bricks.size());
    
        for (size_t brickIdx = 0; brickIdx < bricksToTest; ++brickIdx)
    {
        auto& loadedBrick = loadedDataset.bricks[brickIdx];

        // RANS decoding
        loadedBrick.encodedData = decodeRansStream(loadedBrick, loadedDataset.model);

        // semantic decode (can be repeated for different LODs)
        decoded = decodeBrick<BRICK_SIZE>(loadedBrick, targetLOD);

        // Check against original brick
        bool brickMatches = true;
        for (size_t i = 0; i < bricks[brickIdx].levels[0].size(); i++)
        {
            if (decoded[i] != bricks[brickIdx].levels[0][i].label)
            {
                brickMatches = false;
                allMatch = false;
                std::cout << "Mismatch in brick " << brickIdx << " at position " << i << std::endl;
            }
        }
        
            // Save decoded reference output to file as raw labels (for transfer function)
            std::ofstream refFile("decoded_reference.bin", std::ios::binary);
            for (const auto& val : decoded)
            {
                refFile.write(reinterpret_cast<const char*>(&val), sizeof(uint64_t));
            }
            refFile.close();
            std::cout << "Saved reference decoded output (" << decoded.size() << " labels) to decoded_reference.bin\n";
    }
    
        std::cout << "\nOverall result - input : output equal? " << (allMatch ? "yes" : "no") << std::endl;
}
