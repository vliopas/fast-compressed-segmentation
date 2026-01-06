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
    // Read primary nibble (OpCode + stopBit)
    uint8_t nib = readNibble(encodedData, bytePos, highNibbleNext);

    uint8_t opCode = nib >> 1;   // 3-bit opcode
    uint8_t stop = nib & 1;    // LSB = stop flag

    OpType op = static_cast<OpType>(opCode);

    // Optional: read delta nibble for Pδ
    uint8_t delta = 0;
    if (op == OpType::PaletteBackD)
        delta = readNibble(encodedData, bytePos, highNibbleNext);

    return { op, stop, delta };
}

void fillSubBlock(size_t nodeMortonIdx, size_t level, size_t targetLOD, LabelType label, std::vector<LabelType>& out)
{
    size_t l = level;
    size_t t = targetLOD;

    // Compute start index in output array
    size_t startIdx = coarseNodeToOutputIndex(nodeMortonIdx, l, t);

    // Compute number of voxels in the subtree
    size_t blockSize = 1ULL << (3 * (t - l)); // 8^(t-l) = 2^(3*(t-l))

    // Fill entire block
    for (size_t i = 0; i < blockSize; ++i)
        out[startIdx + i] = label;
}

LabelType getNeighborValue(const std::vector<LabelType>& out, size_t level, size_t childMortonIdx, OpType op, size_t targetLOD)
{
    // Decode Morton index -> 3D coordinates
    dim3 childCoords = Utils::decodeMorton3D(childMortonIdx);

    // Extract LOCAL position in the 2×2×2 sibling block
    int lx = childCoords.x & 1;
    int ly = childCoords.y & 1;
    int lz = childCoords.z & 1;

    // Select which axis to move along based on operation type and update coordinates
    int nx = childCoords.x;
    int ny = childCoords.y;
    int nz = childCoords.z;
    
    if (op == OpType::NeighborX)
        nx = (lx == 0) ? childCoords.x - 1 : childCoords.x + 1;
    else if (op == OpType::NeighborY)
        ny = (ly == 0) ? childCoords.y - 1 : childCoords.y + 1;
    else if (op == OpType::NeighborZ)
        nz = (lz == 0) ? childCoords.z - 1 : childCoords.z + 1;

    // Check bounds - if neighbor is outside brick, use parent of current node
    size_t maxCoord = (1 << (level + 1)) - 1;
    if (nx < 0 || ny < 0 || nz < 0 || nx > (int)maxCoord || ny > (int)maxCoord || nz > (int)maxCoord)
    {
        uint32_t parentMorton = Utils::morton3D(childCoords.x >> 1, childCoords.y >> 1, childCoords.z >> 1);
        size_t parentOutIdx = coarseNodeToOutputIndex(parentMorton, level, targetLOD);
        return out[parentOutIdx];
    }

    // Neighbor Morton index with adjusted coordinates
    size_t neighborMorton = Utils::morton3D(nx, ny, nz);

    // Already decoded → use neighbor
    if (neighborMorton < childMortonIdx)
    {
        size_t outIdx = coarseNodeToOutputIndex(neighborMorton, level + 1, targetLOD);
        LabelType val = out[outIdx];
        return val;
    }

    // Later Z → fallback to neighbor's parent
    uint32_t parentMorton = Utils::morton3D(nx >> 1, ny >> 1, nz >> 1);
    size_t parentOutIdx = coarseNodeToOutputIndex(parentMorton, level, targetLOD);
    LabelType parentVal = out[parentOutIdx];
    return parentVal;
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

std::vector<uint8_t> decodeRansStream(const CompressedBrick& brick, const RansModel& model)
{
    std::vector<uint8_t> repacked;
    repacked.reserve((brick.encodedData.size() - 4 + 1) / 2); // rough estimate

    uint32_t state = 0;
    size_t inPos = brick.encodedData.size();

    assert(inPos >= 4);

    // read little-endian state from the END
    uint32_t b0 = brick.encodedData[--inPos];
    uint32_t b1 = brick.encodedData[--inPos];
    uint32_t b2 = brick.encodedData[--inPos];
    uint32_t b3 = brick.encodedData[--inPos];
    state = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;

    // Decode nibbles until we exhaust encodedData
    size_t symbolsDecoded = 0;
    while (symbolsDecoded < brick.nSymbols)
    {
        uint8_t hi = ransDecodeSymbol(state, brick.encodedData, inPos, model) & 0x0F;
        symbolsDecoded++;

        uint8_t lo = 0;
        if (symbolsDecoded < brick.nSymbols)
        {
            lo = ransDecodeSymbol(state, brick.encodedData, inPos, model) & 0x0F;
            symbolsDecoded++;
        }

        repacked.push_back((hi << 4) | lo);
    }

    return repacked;
}