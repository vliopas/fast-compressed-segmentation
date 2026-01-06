#include "brickDecoding.hpp"
#include "brickEncoding.hpp"

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
