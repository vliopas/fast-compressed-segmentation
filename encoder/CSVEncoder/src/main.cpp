#include <iostream>

#include "brickDecoding.hpp"
#include "brickEncoding.hpp"
#include "fileIO.hpp"
#include "brick.hpp"
#include "types.hpp"

#include <unordered_set>

constexpr size_t brickSize = 8; // Define brick size here

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

    auto bricks = splitGridIntoBricks<brickSize>(npy);
    std::cout << "Number of bricks: " << bricks.size() << "\n";

    // auto& firstBrick = bricks[0];
    // auto& L0 = firstBrick.levels[0];
    // std::cout << "First voxel label in first brick: " << static_cast<int>(L0[0].label) << "\n";

    auto compressed = Encoding::compressDataset(bricks);
    saveDatasetToFile(compressed, "compressed_dataset.csbd");

    Encoding::CompressedDataset loadedDataset = loadDatasetFromFile("compressed_dataset.csbd");

    constexpr size_t targetLOD = Brick<brickSize>::Levels - 1;

    std::vector<LabelType> decoded;
    for (size_t i = 0; i < loadedDataset.bricks.size(); ++i)
    {
        auto& brick = loadedDataset.bricks[i];

        // entropy decode ONCE
        brick.encodedData = decodeRansStream(brick, loadedDataset.interiorModel, loadedDataset.leafModel);

        // 2. semantic decode (can be repeated for different LODs)
        decoded = decodeBrick<brickSize>( brick, targetLOD);

        std::cout << "Decoded brick " << i
            << ", voxels = " << decoded.size() << "\n";
    }

    bool flag = true;
    for (int i = 0; i < bricks[0].levels[0].size(); i++)
    {
        if (decoded[i] != bricks[0].levels[0][i].label)
        {
            flag = false;
            std::cout << "Mismatched element pos: " << i << std::endl;
        }
    }
    std::cout << "input : output equal? " << (flag ? "yes" : "no") << std::endl;
}
