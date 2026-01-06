#include "types.hpp"
#include "brick.hpp"
#include "utils.hpp"

#include <tuple>
#include <vector>

using namespace Encoding;
using namespace Utils;

/**
 * @brief Map a coarse octree node to its corresponding index in the flat output array.
 *
 * Converts a Morton (Z-order) index of a node at pyramid level @p l into the
 * linear index of the voxel (or voxel block) it represents in the final output
 * array at target level-of-detail @p t.
 *
 * This mapping ensures that nodes processed in Morton order are written to the
 * correct spatial locations in the flat output buffer.
 *
 * @param mortonL  Morton (Z-order) index of the node at level @p l.
 * @param l        Pyramid level of the node.
 * @param t        Target level-of-detail (finest output level).
 *
 * @return size_t  Linear index into the output array corresponding to this node.
 */
size_t coarseNodeToOutputIndex(size_t mortonL, size_t l, size_t t);

/**
 * @brief Read a single 4-bit nibble from the encoded operation stream.
 *
 * Extracts the next 4-bit value (nibble) from the packed byte stream produced
 * by the encoder. Two nibbles are stored per byte; this function maintains the
 * current read position and whether the next nibble is the high or low nibble
 * of the current byte.
 *
 * This function is the lowest-level primitive used for decoding operations.
 *
 * @param encodedData      Packed byte stream containing encoded operations.
 * @param bytePos          Current byte position in the stream (updated on read).
 * @param highNibbleNext   Indicates whether the next nibble to read is the
 *                         high nibble (true) or low nibble (false) of the byte.
 *
 * @return uint8_t         The 4-bit value of the next nibble.
 */
uint8_t readNibble(
    const std::vector<uint8_t>& encodedData,
    size_t& bytePos,
    bool& highNibbleNext);

/**
 * @brief Decode the next operation and stop-bit from the encoded stream.
 *
 * Reads one operation entry from the encoded data stream, consisting of:
 *  - a primary nibble encoding the operation type and stop-bit, and
 *  - an optional delta nibble for PaletteBackD operations.
 *
 * The stop-bit indicates whether the entire sub-tree under the current node
 * should be filled immediately with the decoded label.
 *
 * The function advances the stream read position accordingly and returns all
 * decoded components as a tuple.
 *
 * @param encodedData      Packed byte stream containing encoded operations.
 * @param bytePos          Current byte position in the stream (updated on read).
 * @param highNibbleNext   Tracks whether the next nibble is read from the high
 *                         or low half of the current byte.
 *
 * @return std::tuple<OpType, uint8_t, uint8_t>
 *         - OpType   : Decoded operation type.
 *         - uint8_t  : Delta value (only valid for PaletteBackD; undefined otherwise).
 *         - uint8_t  : Stop-bit (1 = fill sub-tree, 0 = continue decoding).
 */
std::tuple<OpType, uint8_t, uint8_t> readNextOperationAndStopBit(
    const std::vector<uint8_t>& encodedData,
    size_t& bytePos,
    bool& highNibbleNext);

/**
 * @brief Fill all voxels in a sub-tree with a single label value.
 *
 * When the stop-bit of an operation is set, this function recursively fills
 * the entire sub-tree rooted at the specified node with the same label, without
 * consuming additional operations from the encoded stream.
 *
 * This allows large homogeneous regions to be represented efficiently.
 *
 * @param nodeMortonIdx  Morton (Z-order) index of the node whose sub-tree is filled.
 * @param level          Pyramid level of the node.
 * @param targetLOD      Target level-of-detail (finest output level).
 * @param label          Label value to assign to all voxels in the sub-tree.
 * @param out            Flat output array storing decoded voxel labels.
 */
void fillSubBlock(
    size_t nodeMortonIdx,
    size_t level,
    size_t targetLOD,
    LabelType label,
    std::vector<LabelType>& out);

/**
 * @brief Resolve a Neighbor Reuse operation (Rx, Ry, Rz) for a child node.
 *
 * Implements the neighbor reuse rule described in Piochowiak et al.:
 *  - The operation specifies only the AXIS (X/Y/Z); the direction is implicit.
 *  - The neighbor is always chosen OUTSIDE the current 2×2×2 sibling block.
 *  - If the referenced neighbor has a later Morton (Z-order) index and is not
 *    yet decoded, the operation is reinterpreted as a reference to the
 *    neighbor’s PARENT node, whose label is guaranteed to be known.
 *
 * @param out           Output label buffer containing already decoded nodes.
 * @param level         Pyramid level of the child node (parent is at level-1).
 * @param childMortonIdx Morton (Z-order) index of the child node at this level.
 * @param op            Neighbor reuse operation type (Rx, Ry, or Rz).
 * @param targetLOD     Target level-of-detail of the decoded output.
 *
 * @return LabelType    Label to assign to the child node according to the
 *                      neighbor reuse operation.
 */
LabelType getNeighborValue(
    const std::vector<LabelType>& out,
    size_t level,
    size_t childMortonIdx,
    OpType op,
    size_t targetLOD);

uint8_t ransDecodeSymbol(
    uint32_t& state,
    const std::vector<uint8_t>& in,
    size_t& inPos,
    const RansModel& model);

std::vector<uint8_t> decodeRansStream(
    const CompressedBrick& brick,
    const RansModel& interiorModel,
    const RansModel& leafModel);

/**
 * @brief Decode a compressed brick into a flat voxel label array.
 *
 * Implements the BFS decoding algorithm described in Piochowiak et al.:
 *  - Traverses the octree level-by-level (coarse-to-fine) in Morton (Z-order).
 *  - Applies the operation stream (ParentReuse, NeighborReuse, PaletteAdvance, PaletteBack0/Pδ)
 *    to assign labels to child nodes.
 *  - Handles stop-bits by filling the entire sub-tree under a node with the decoded label.
 *  - Resolves neighbor reuse operations (Rx/Ry/Rz) according to implicit axis rules.
 *  - Uses a palette index to track current position for palette-based operations.
 *
 * @tparam b  Brick resolution (edge length of the brick).
 *
 * @param brick      Compressed brick containing the palette and packed operations.
 * @param targetLOD  Target level-of-detail (finest output level) of the decoded output.
 *
 * @return std::vector<LabelType>
 *         Flat output array containing decoded voxel labels at the targetLOD.
 *
 * @note
 *  - The root node is always assigned the first palette entry.
 *  - Children are processed in Morton order.
 *  - Neighbor reuse operations reference only already-decoded nodes;
 *    if a neighbor has a later Morton index, fallback to the neighbor’s parent.
 *  - Stop-bit = 1 triggers `fillSubBlock()` to fill all descendant voxels
 *    without reading further operations from the stream.
 *  - PaletteBackD references the palette entry at ip - δ (0 <= δ <= 15).
 */
template<size_t b>
std::vector<LabelType> decodeBrick(const CompressedBrick& brick, size_t targetLOD)
{
	constexpr size_t levels = Brick<b>::Levels;
	const size_t outCount = Utils::sizeForLevel<b>(targetLOD);
	std::vector<LabelType> out(outCount, EMPTY_VALUE);

    size_t ip = 0; // palette read index
	size_t bytePos = 0;

    out[0] = brick.palette[ip];

	bool highNibbleNext = false;

	for (size_t l = 0; l < levels - 1; ++l)
	{
		const size_t nodeCount = sizeForLevel<b>(l);
		for (size_t mortonIdx = 0; mortonIdx < nodeCount; ++mortonIdx)
		{
			size_t i = coarseNodeToOutputIndex(mortonIdx, l, targetLOD);
			// i points to first voxel in out[] corresponding to this coarse node

			// compute children indices of current node
			auto childrenMortonIndices = computeChildMortonIndices(mortonIdx);

            auto lastChildOfI = coarseNodeToOutputIndex(childrenMortonIndices[7], l + 1, targetLOD);
            if (out[lastChildOfI] != EMPTY_VALUE) continue;

            LabelType parent = out[i]; // store parent label

			for (size_t childMortonIdx : childrenMortonIndices)
			{
				size_t j = coarseNodeToOutputIndex(childMortonIdx, l + 1, targetLOD);
				auto [op, stop, delta] = readNextOperationAndStopBit(brick.encodedData, bytePos, highNibbleNext);

                LabelType val = EMPTY_VALUE;

                switch (op)
                {
                case OpType::ParentReuse:
                    val = parent;
                    break;
                case OpType::NeighborX:
                case OpType::NeighborY:
                case OpType::NeighborZ:
                    val = getNeighborValue(out, l, childMortonIdx, op, targetLOD);
                    break;
                case OpType::PaletteAdvance:
                    val = brick.palette[++ip];
                    break;
                case OpType::PaletteBack0:
                    val = brick.palette[ip];
                    break;
                case OpType::PaletteBackD:
                    val = brick.palette[ip - delta];
                    break;
                }

                out[j] = val;

                if (stop)
                    fillSubBlock(childMortonIdx, l + 1, targetLOD, val, out);
			}
		}
	}

    return out;
}