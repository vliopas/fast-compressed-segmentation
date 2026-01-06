#pragma once

#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>

using LabelType = uint64_t;
static constexpr LabelType EMPTY_VALUE = std::numeric_limits<LabelType>::max();


static constexpr uint32_t RANS_STATE_BITS = 32; // the current ANS coder state — large integer that evolves as we encode symbols
static constexpr uint32_t RANS_LIMIT = 1u << 23; // minimum allowed value of the state before encoding a new symbol
//If the internal state drops below this threshold, you must renormalize (shift out bytes)
constexpr uint32_t LOG_TOTAL_FREQ = 12;
static constexpr uint32_t RANS_TOTAL = 1 << LOG_TOTAL_FREQ;   // or 8192/16384
static constexpr uint32_t RANS_INIT = RANS_LIMIT;     // initial state


template<std::size_t N, typename T = int>
struct dim {
    T data[N] = {};

    constexpr dim() = default;

    // Constructor from N arguments
    template<typename... Args>
    requires (sizeof...(Args) == N)
    constexpr dim(Args... args) : data{static_cast<T>(args)...} {}

    constexpr T& operator[](std::size_t i) { return data[i]; }
    constexpr const T& operator[](std::size_t i) const { return data[i]; }
};

// Specialization for N = 2
template<typename T>
struct dim<2, T>
{
    union
    {
        struct { T x, y; };
        T data[2];
    };

    constexpr dim() = default;

    constexpr dim(T a, T b) : x(a), y(b) {}

    constexpr T& operator[](std::size_t i) { return data[i]; }
    constexpr const T& operator[](std::size_t i) const { return data[i]; }
};

// Specialization for N = 3
template<typename T>
struct dim<3, T>
{
    union
    {
        struct { T x, y, z; };
        T data[3];
    };

    constexpr dim() = default;

    constexpr dim(T a, T b, T c) : x(a), y(b), z(c) {}

    constexpr T& operator[](std::size_t i) { return data[i]; }
    constexpr const T& operator[](std::size_t i) const { return data[i]; }
};

// Specialization for N = 4
template<typename T>
struct dim<4, T>
{
    union
    {
        struct { T x, y, z, w; };
        T data[4];
    };

    constexpr dim() = default;

    constexpr dim(T a, T b, T c, T d) : x(a), y(b), z(c), w(d) {}

    constexpr T& operator[](std::size_t i) { return data[i]; }
    constexpr const T& operator[](std::size_t i) const { return data[i]; }
};

// Aliases for convenience
using dim2 = dim<2>;
using dim3 = dim<3>;
using dim4 = dim<4>;

// Check if a number is a power of two at compile time
template <size_t N>
inline constexpr bool is_power_of_two_v = (N > 0) && ((N & (N - 1)) == 0);

// Node in the pyramid
struct Node
{
    LabelType label;        // Label for this node
    bool constantChildren; // All the children have the same label - optimization flag
};

// structure for storing data read from the .npy file of the uncompressed dataset
struct NpyArray {
    std::vector<uint64_t> data; // you can template this later
    std::vector<size_t> shape;
};

namespace Encoding
{   
    // Operation types for encoding
    enum class OpType : uint8_t
    {
        ParentReuse,    // Rp
        NeighborX,      // Rx
        NeighborY,      // Ry
        NeighborZ,      // Rz
        PaletteAdvance, // Pa
        PaletteBack0,   // P0
        PaletteBackD,   // Pδ
    };

    struct OpEntry
    {
        OpType op;     // Operation type
        uint8_t delta = 0; // Only used for Pδ - ignored otherwise - in paper it is 4 bits MIGHT(?) change this later

        uint8_t stopBit = 0;
        uint8_t isLeaf = 0;
    };

    // Pack operations into 4-bit nibbles (see section 3.3)
    // We get a tighter symbol stream of operation BEFORE entropy encoding
    enum Nibble : uint8_t
    {
        N_Rp  = 0x0,   // ParentReuse
        N_Rx  = 0x1,
        N_Ry  = 0x2,
        N_Rz  = 0x3,
        N_Pa  = 0x4,   // PaletteAdvance
        N_P0  = 0x5,   // PaletteBack0
        N_Pd  = 0x6,   // PaletteBackD (delta 0–15 encoded separately)
        // 0x7 or unused
    };

    struct CompressedBrick
    {
        std::vector<LabelType> palette; // Palette of labels used in the brick
        std::vector<uint8_t> encodedData; // Encoded operation data as byte stream
        std::vector<uint8_t> isLeaf; // Is node of brick inner or leaf node?
        uint32_t ID;
    };

    using FrequencyTable = std::array<size_t, 16>; // Frequency table for 16 possible nibbles

    extern FrequencyTable interiorFreqTable;
    extern FrequencyTable leafFreqTable;

    // std::vector<uint8_t> ransDecompress(
    //     const std::vector<uint8_t>& compressed,
    //     const FrequencyTable& table);
    
    struct RansModel
    {
        std::vector<uint32_t> cumulativeFreq; // start of each symbol range
        std::vector<uint32_t> freq;           // frequency of each symbol
        uint32_t totalFreq;                    // sum of all frequencies
    };

    struct CompressedDataset
    {
        RansModel interiorModel; 
        RansModel leafModel;     
        std::vector<CompressedBrick> bricks;
    };

} // namespace Encoding