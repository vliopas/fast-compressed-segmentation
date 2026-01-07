/**
 * @file fileIO.hpp
 * @brief File I/O functions for compressed dataset and numpy arrays
 *
 * Provides functions to save/load compressed datasets in binary format
 * and load numpy .npy files.
 */

#pragma once
#include <string>
#include "types.hpp"

/**
 * @brief Save compressed dataset to binary file
 * @param dataset Compressed dataset to save
 * @param filename Output file path
 *
 * File format:
 * - Magic number / file version
 * - Number of bricks
 * - Brick size (b)
 * - Global RANS models (interior and leaf)
 * - Per-brick: palette, compressed operation stream
 */
void saveDatasetToFile(const Encoding::CompressedDataset &dataset, const std::string &filename);

/**
 * @brief Load compressed dataset from binary file
 * @param filename Input file path
 * @return Loaded compressed dataset
 */
Encoding::CompressedDataset loadDatasetFromFile(const std::string &filename);

/**
 * @brief Load numpy .npy file containing segmentation data
 * @param filename Path to .npy file
 * @return NpyArray structure with data and shape
 */
NpyArray loadNpy(const std::string &filename);