#include "brickDecoding.hpp"
#include "brickEncoding.hpp"

#include <ranges>

// Returns Morton index in output array for a coarse node
size_t coarseNodeToOutputIndex(size_t mortonL, size_t l, size_t t)
{
	dim3 p_l = decodeMorton3D(mortonL);        // coarse-level coordinates
	uint32_t scale = 1u << (t - l);            // size of block in target LOD

	dim3 p_t;
	p_t.x = p_l.x * scale;
	p_t.y = p_l.y * scale;
	p_t.z = p_l.z * scale;

	return morton3D(p_t.x, p_t.y, p_t.z);
}

uint8_t readNibble( const std::vector<uint8_t>& encodedData, size_t& bytePos, bool& highNibbleNext)
{
    if (bytePos >= encodedData.size())
        throw std::runtime_error("End of compressed stream");

    uint8_t byte = encodedData[bytePos];
    uint8_t nib;

    if (highNibbleNext)
    {
        nib = byte >> 4;          // upper 4 bits
        highNibbleNext = false;   // next time read the low nibble
    }
    else
    {
        nib = byte & 0x0F;        // lower 4 bits
        highNibbleNext = true;
        bytePos++;                // we consumed the whole byte now
    }

    return nib;
}

std::tuple<OpType, uint8_t, uint8_t> readNextOperationAndStopBit( const std::vector<uint8_t>& encodedData, size_t& bytePos, bool& highNibbleNext)
{
    // 1️⃣ Read primary nibble (OpCode + stopBit)
    uint8_t nib = readNibble(encodedData, bytePos, highNibbleNext);

    uint8_t opCode = nib >> 1;   // 3-bit opcode
    uint8_t stop = nib & 1;    // LSB = stop flag

    OpType op = static_cast<OpType>(opCode);

    // 2️⃣ Optional: read delta nibble for Pδ
    uint8_t delta = 0;
    if (op == OpType::PaletteBackD)
        delta = readNibble(encodedData, bytePos, highNibbleNext);

    return { op, stop, delta };
}

void fillSubBlock(size_t nodeMortonIdx, size_t level, size_t targetLOD, LabelType label, std::vector<LabelType>& out)
{
    size_t l = level;        // current node level
    size_t t = targetLOD;    // output level

    // Base case: if we are at target LOD, just write the voxel
    if (l == t)
    {
        size_t outIdx = coarseNodeToOutputIndex(nodeMortonIdx, l, t);
        out[outIdx] = label;
        return;
    }

    // Compute children Morton indices
    auto children = computeChildMortonIndices(nodeMortonIdx);

    // Recursively fill each child
    for (size_t c = 0; c < 8; ++c)
    {
        fillSubBlock(children[c], l + 1, t, label, out);
    }
}

LabelType getNeighborValue(const std::vector<LabelType>& out, size_t level, size_t childMortonIdx, OpType op, size_t targetLOD)
{
    // Decode Morton index -> local (x,y,z)
    dim3 childCoords = Utils::decodeMorton3D(childMortonIdx);

    // Select axis (Rx/Ry/Rz)
    int* coord = nullptr;
    if (op == OpType::NeighborX) coord = &childCoords.x;
    else if (op == OpType::NeighborY) coord = &childCoords.y;
    else if (op == OpType::NeighborZ) coord = &childCoords.z;

    // Implicit direction: outside 2×2×2 sibling block
    // even → -1, odd → +1
    *coord += ((*coord & 1) == 0) ? -1 : +1;

    // Neighbor Morton index
    size_t neighborMorton = Utils::morton3D(childCoords.x, childCoords.y, childCoords.z);

    // Already decoded → use neighbor
    if (neighborMorton < childMortonIdx)
        return out[coarseNodeToOutputIndex(neighborMorton, level + 1, targetLOD)];

    // Later Z → fallback to neighbor’s parent
    uint32_t parentMorton = Utils::morton3D(childCoords.x >> 1, childCoords.y >> 1, childCoords.z >> 1);
    return out[coarseNodeToOutputIndex(parentMorton, level, targetLOD)];
}

inline uint32_t findSymbol(uint32_t x, const RansModel& m)
{
    uint32_t s = 0;
    while (s + 1 < m.cumulativeFreq.size() &&
           x >= m.cumulativeFreq[s + 1])
    {
        ++s;
    }
    return s;
}

uint8_t ransDecodeSymbol(uint32_t& state, const std::vector<uint8_t>& in, size_t& inPos, const RansModel& model)
{
    uint32_t x = state % model.totalFreq;

    uint32_t s = findSymbol(x, model); // cumulative lookup
    uint32_t freq = model.freq[s];
    uint32_t start = model.cumulativeFreq[s];

    state = freq * (state / model.totalFreq) + (x - start);

    while (state < (RANS_LIMIT) && inPos > 0)
        state = (state << 8) | in[--inPos];

    return s;
}

std::vector<uint8_t> decodeRansStream(const CompressedBrick& brick, const RansModel& interiorModel, const RansModel& leafModel)
{
    size_t n = brick.isLeaf.size();
    std::vector<uint8_t> repacked;

    uint32_t state = 0;
    size_t inPos = brick.encodedData.size();
    size_t outIdx = 0;

    assert(inPos >= 4);

    // read little-endian state from the END
    uint32_t b0 = brick.encodedData[--inPos];
    uint32_t b1 = brick.encodedData[--inPos];
    uint32_t b2 = brick.encodedData[--inPos];
    uint32_t b3 = brick.encodedData[--inPos];

    state = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;

    // Use reverse iterator over isLeaf
    auto it = brick.isLeaf.begin();

    while (it != brick.isLeaf.end())
    {
        // First decoded symbol is HIGH nibble
        bool hiIsLeaf = *it++;
        uint8_t hi = ransDecodeSymbol( state, brick.encodedData, inPos,  interiorModel ) & 0x0F;

        uint8_t lo = 0;
        if (it != brick.isLeaf.end())
        {
            bool loIsLeaf = *it++;
            lo = ransDecodeSymbol( state, brick.encodedData, inPos,  interiorModel ) & 0x0F;
        }

        repacked.push_back((hi << 4) | lo);
    }


    return repacked;
}