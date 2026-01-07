/**
 * @file types.hpp
 * @brief Core type definitions for the compression system
 *
 * Defines fundamental data structures including dimensional vectors,
 * nodes, operations, and the compression model used throughout the encoder.
 */

#pragma once

#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>

using LabelType = uint64_t;                                                     ///< Type for segmentation labels
static constexpr LabelType EMPTY_VALUE = std::numeric_limits<LabelType>::max(); ///< Sentinel value for empty voxels

static constexpr size_t BRICK_SIZE = 64; ///< Default brick size (64^3 voxels)

// rANS encoder constants
static constexpr uint32_t RANS_STATE_BITS = 32;             ///< Bits in the rANS state
static constexpr uint32_t RANS_LIMIT = 1u << 23;            ///< Minimum state before renormalization
static constexpr uint32_t LOG_TOTAL_FREQ = 12;              ///< Log2 of total frequency
static constexpr uint32_t RANS_TOTAL = 1 << LOG_TOTAL_FREQ; ///< Total frequency range (4096)
static constexpr uint32_t RANS_INIT = RANS_LIMIT;           ///< Initial rANS state

/**
 * @brief Generic N-dimensional vector template
 * @tparam N Number of dimensions
 * @tparam T Element type (default: int)
 */
template <std::size_t N, typename T = int>
struct dim
{
    T data[N] = {};

    constexpr dim() = default;

    // Constructor from N arguments
    template <typename... Args>
        requires(sizeof...(Args) == N)
    constexpr dim(Args... args) : data{static_cast<T>(args)...}
    {
    }

    constexpr T &operator[](std::size_t i) { return data[i]; }
    constexpr const T &operator[](std::size_t i) const { return data[i]; }
};

/**
 * @brief 2D vector specialization
 * @tparam T Element type
 */
template <typename T>
struct dim<2, T>
{
    union
    {
        struct
        {
            T x, y;
        };
        T data[2];
    };

    constexpr dim() = default;

    constexpr dim(T a, T b) : x(a), y(b) {}

    constexpr T &operator[](std::size_t i) { return data[i]; }
    constexpr const T &operator[](std::size_t i) const { return data[i]; }
};

/**
 * @brief 3D vector specialization
 * @tparam T Element type
 */
template <typename T>
struct dim<3, T>
{
    union
    {
        struct
        {
            T x, y, z;
        };
        T data[3];
    };

    constexpr dim() = default;

    constexpr dim(T a, T b, T c) : x(a), y(b), z(c) {}

    constexpr T &operator[](std::size_t i) { return data[i]; }
    constexpr const T &operator[](std::size_t i) const { return data[i]; }
};

/**
 * @brief 4D vector specialization
 * @tparam T Element type
 */
template <typename T>
struct dim<4, T>
{
    union
    {
        struct
        {
            T x, y, z, w;
        };
        T data[4];
    };

    constexpr dim() = default;

    constexpr dim(T a, T b, T c, T d) : x(a), y(b), z(c), w(d) {}

    constexpr T &operator[](std::size_t i) { return data[i]; }
    constexpr const T &operator[](std::size_t i) const { return data[i]; }
};

using dim2 = dim<2>; ///< Alias for 2D vector
using dim3 = dim<3>; ///< Alias for 3D vector
using dim4 = dim<4>; ///< Alias for 4D vector

/**
 * @brief Check if a number is a power of two at compile time
 * @tparam N Number to check
 */
template <size_t N>
inline constexpr bool is_power_of_two_v = (N > 0) && ((N & (N - 1)) == 0);

/**
 * @brief Node in the brick pyramid
 */
struct Node
{
    LabelType label;       ///< Label value for this node
    bool constantChildren; ///< True if all children have the same label
};

/**
 * @brief Structure for storing data from .npy file
 */
struct NpyArray
{
    std::vector<uint64_t> data; ///< Flat array of voxel labels
    std::vector<size_t> shape;  ///< Dimensions of the array
};

/**
 * @brief Encoding-related types and structures
 */
namespace Encoding
{
    /**
     * @brief Operation types for encoding
     */
    enum class OpType : uint8_t
    {
        ParentReuse,    ///< Rp - Reuse parent label
        NeighborX,      ///< Rx - Use X-axis neighbor
        NeighborY,      ///< Ry - Use Y-axis neighbor
        NeighborZ,      ///< Rz - Use Z-axis neighbor
        PaletteAdvance, ///< Pa - Advance to next palette entry
        PaletteBack0,   ///< P0 - Use first palette entry
        PaletteBackD,   ///< Pδ - Use palette entry at delta offset
    };

    /**
     * @brief Single operation entry
     */
    struct OpEntry
    {
        OpType op;         ///< Operation type
        uint8_t delta = 0; ///< Delta value (used only for Pδ)

        uint8_t stopBit = 0; ///< Stop bit for encoding
    };

    /**
     * @brief 4-bit nibble encoding for operations
     */
    enum Nibble : uint8_t
    {
        N_Rp = 0x0, ///< ParentReuse
        N_Rx = 0x1, ///< NeighborX
        N_Ry = 0x2, ///< NeighborY
        N_Rz = 0x3, ///< NeighborZ
        N_Pa = 0x4, ///< PaletteAdvance
        N_P0 = 0x5, ///< PaletteBack0
        N_Pd = 0x6, ///< PaletteBackD (delta encoded separately)
    };

    /**
     * @brief Compressed brick data
     */
    struct CompressedBrick
    {
        std::vector<LabelType> palette;   ///< Palette of unique labels
        std::vector<uint8_t> encodedData; ///< Encoded operation stream
        uint32_t nSymbols;                ///< Number of encoded symbols
        uint32_t ID;                      ///< Brick identifier
    };

    using FrequencyTable = std::array<size_t, 16>; ///< Frequency table for 16 nibbles

    extern FrequencyTable interiorFreqTable; ///< Frequency table for interior nodes
    extern FrequencyTable leafFreqTable;     ///< Frequency table for leaf nodes

    /**
     * @brief rANS probability model
     */
    struct RansModel
    {
        std::vector<uint32_t> cumulativeFreq; ///< Cumulative frequency distribution
        std::vector<uint32_t> freq;           ///< Symbol frequencies
        uint32_t totalFreq;                   ///< Sum of all frequencies
    };

    /**
     * @brief Complete compressed dataset
     */
    struct CompressedDataset
    {
        RansModel model;                     ///< Probability model for decoding
        std::vector<CompressedBrick> bricks; ///< All compressed bricks
    };

} // namespace Encoding