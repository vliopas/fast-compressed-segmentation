#pragma once
#include <string>
#include "types.hpp"

// saves the compressed dataset to a binary file
// format of the file as follows:
// |---------------------------------------------------
// | Global Header                                    |
// |---------------------------------------------------
// | - Magic number / file version                    |
// | - Number of bricks                               |
// | - Brick size (b)                                 |
// | - Other metadata (optional: timestamp, author)   |
// |---------------------------------------------------
// | Global RANS Models                               |
// |---------------------------------------------------
// | Interior Model                                   |
// | - totalFreq (uint32_t)                           |
// | - freq array (per-symbol frequency)              |
// | - cumulativeFreq array                           |
// | Leaf Model                                       |
// | - totalFreq                                      |
// | - freq array                                     |
// | - cumulativeFreq array                           |
// ----------------------------------------------------
// | Brick #0                                         |
// |---------------------------------------------------
// | Palette size (uint32_t)                          |
// | Palette data (LabelType x palette size)          |
// | Compressed operation stream size (uint32_t)      |
// | Compressed operation stream (bytes)              |
// | CRC32 checksum (uint32_t) (optional)             |
// ----------------------------------------------------
// | Brick #1                                         |
// | ...                                              |
// ----------------------------------------------------
// | Brick #N                                         |
// | ...                                              |
// ----------------------------------------------------
void saveDatasetToFile(const Encoding::CompressedDataset& dataset, const std::string& filename);

NpyArray loadNpy(const std::string& filename);