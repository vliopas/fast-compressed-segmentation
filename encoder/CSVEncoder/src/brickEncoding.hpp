#include "types.hpp"
#include "brick.hpp"

#include <functional>
#include <utility>
#include <ranges>

using OpEntry = Encoding::OpEntry;
using OpType = Encoding::OpType;
using RansModel = Encoding::RansModel;

static constexpr uint32_t RANS_STATE_BITS = 32; // the current ANS coder state — large integer that evolves as we encode symbols

static constexpr uint32_t RANS_LIMIT = 1u << 23; // minimum allowed value of the state before encoding a new symbol
//If the internal state drops below this threshold, you must renormalize (shift out bytes)

static constexpr uint32_t RANS_TOTAL = 4096;   // or 8192/16384
static constexpr uint32_t RANS_INIT = RANS_LIMIT;     // initial state

namespace Encoding
{
    /**
     * @brief Pack a sequence of octree operations into a compact nibble stream.
     *
     * Converts a vector of OpEntry operations into a byte stream where each
     * byte contains two 4-bit nibbles. Each nibble encodes an operation
     * (3-bit op code + 1-bit stop flag) or a delta value for PaletteBackD.
     * A parallel array records whether each nibble corresponds to a leaf or
     * interior node.
     *
     * The packing is performed in **Morton-order-consistent BFS**, preserving
     * the original operation sequence.
     *
     * @param operations  Vector of operations to pack (OpEntry structs).
     *
     * @return std::pair<std::vector<uint8_t>, std::vector<uint8_t>>
     *         - first : Packed byte stream (2 nibbles per byte)
     *         - second: Parallel leaf/interior flags (1 per nibble)
     *
     * @note
     *  - PaletteBackD operations generate two nibbles: primary opcode + delta nibble.
     *  - If an odd number of nibbles exists, the last byte is padded with 0 in
     *    its low nibble.
     *  - This function prepares data for frequency analysis and later decoding.
     */
    std::pair<std::vector<uint8_t>, std::vector<uint8_t>> packOperationsToNibbles(const std::vector<OpEntry>& operations);

    /**
     * @brief Update frequency tables for leaf and interior operations from a packed stream.
     *
     * Traverses a packed nibble stream (from `packOperationsToNibbles`) and
     * counts how many times each opcode/delta occurs separately for leaf and
     * interior nodes. This is typically used to prepare Huffman coding or
     * other entropy-based compression.
     *
     * @param bytes     Packed byte stream (2 nibbles per byte).
     * @param isLeaf    Parallel array indicating if each nibble corresponds
     *                  to a leaf node (1) or interior node (0).
     *
     * @pre
     *  - isLeaf.size() == bytes.size()*2 or bytes.size()*2 - 1 (padding allowed)
     *
     * @note
     *  - High nibble is read first, then low nibble for each byte.
     *  - Updates internal `leafFreqTable` and `interiorFreqTable` arrays.
     */
    void updateFrequencyTables(const CompressedBrick& compressedBrick);

    /**
     * @brief Determine the most suitable operation for encoding a node.
     *
     * Implements the logic described in Sec. 3.2 of the paper:
     *  - Tries ParentReuse first
     *  - Then attempts NeighborReuse (Rx, Ry, Rz) along axes outside the 2×2×2 sibling block
     *  - Falls back to palette-based operations (PaletteBack0, PaletteBackD, PaletteAdvance)
     *
     * @tparam b  Brick resolution (edge length of the brick)
     *
     * @param brick       The brick containing pyramid levels and nodes
     * @param node        Node to encode
     * @param parent      Parent node (coarser level)
     * @param levelSize   Dimension of the current level in one axis
     * @param nodeCoords  3D coordinates of the node within the level
     * @param state       Encoder state containing the palette and any bookkeeping
     *
     * @return OpEntry    The chosen operation (OpType) with any associated delta
     *
     * @note
     *  - NeighborReuse uses implicit direction: chooses outside sibling block based on node local position
     *  - If neighbor is later in Morton order, fallback to neighbor's parent label
     *  - PaletteBackD references entries up to 16 positions back
     *  - PaletteAdvance is returned for a new label not found by any reuse or palette reference
     */
    template<size_t b>
    OpEntry bestOperation(
        const Brick<b>& brick,
        const Node& node,
        const Node& parent,
        size_t levelSize,
        dim3 nodeCoords,
        CompressedBrick& compressedBrick)
    {
        auto L = node.label;

        // Parent reuse
        // ------------------------------
        if (parent.label == L)
            return OpEntry{ OpType::ParentReuse };
        // ------------------------------
        // 
        // Neighbor reuse (Rx, Ry, Rz)
        // ------------------------------
        auto currentMorton = Utils::morton3D(nodeCoords.x, nodeCoords.y, nodeCoords.z);
        size_t levelIndex = std::countr_zero(levelSize); // levelIndex = log_2(levelSize)

        auto tryNeighbor = [&](dim3 nodeCoords, OpType opType) -> std::optional<OpType>
        {
            // ---------------------------------------
            // Local position in the 2×2×2 sibling block
            int lx = nodeCoords.x & 1;
            int ly = nodeCoords.y & 1;
            int lz = nodeCoords.z & 1;

            int nx = nodeCoords.x;
            int ny = nodeCoords.y;
            int nz = nodeCoords.z;

            // ---------------------------------------
            // Move outside the 2×2×2 block depending on opType
            switch (opType)
            {
            case OpType::NeighborX:
                // neighbor must be OUTSIDE sibling block along X
                nx = (lx == 0) ? nodeCoords.x - 1 : nodeCoords.x + 1;
                break;

            case OpType::NeighborY:
                // neighbor must be OUTSIDE sibling block along Y
                ny = (ly == 0) ? nodeCoords.y - 1 : nodeCoords.y + 1;
                break;

            case OpType::NeighborZ:
                // neighbor must be OUTSIDE sibling block along Z
                nz = (lz == 0) ? nodeCoords.z - 1 : nodeCoords.z + 1;
                break;
            }

            // ---------------------------------------
            // Bounds check
            if (nx < 0 || ny < 0 || nz < 0 ||
                nx >= levelSize || ny >= levelSize || nz >= levelSize)
                return std::nullopt;

            // ---------------------------------------
            // Get neighbor
            uint32_t neighborMorton = Utils::morton3D(nx, ny, nz);
            const Node* neighbor = &brick.coarser(levelIndex)[neighborMorton];
            if (!neighbor) return std::nullopt;

            // ---------------------------------------
            if (neighborMorton < currentMorton)
            {
                // EARLIER in morton order -> allow reuse
                return (neighbor->label == L) ? std::optional(opType)
                    : std::nullopt;
            }

            // ---------------------------------------
            // fallback to parent
            if (levelIndex > 0) // must NOT be in the FIRST level
            {
                uint32_t parentMorton = Utils::morton3D(nx >> 1, ny >> 1, nz >> 1);
                const Node& parent = brick.coarser(levelIndex - 1)[parentMorton];

                if (parent.label == L)
                    return opType;
            }

            return std::nullopt;
        };


        if (auto op = tryNeighbor(nodeCoords, OpType::NeighborX)) return OpEntry{ *op };
        if (auto op = tryNeighbor(nodeCoords, OpType::NeighborY)) return OpEntry{ *op };
        if (auto op = tryNeighbor(nodeCoords, OpType::NeighborZ)) return OpEntry{ *op };
        // ------------------------------

        // Palette Lookup
        // ---------------------------------------------
        // Look for L in the palette
        auto it = std::find(compressedBrick.palette.begin(), compressedBrick.palette.end(), L);

        if (it != compressedBrick.palette.end())
        {
            size_t idx = std::distance(compressedBrick.palette.begin(), it);
            size_t last = compressedBrick.palette.size() - 1;

            // ---- P0: last used palette entry ----
            if (idx == last)
                return { OpType::PaletteBack0 };     // P0 in the paper

            // ---- Pδ: back-reference up to 16 entries ----
            // idx == last-1 -> δ = 0
            // idx == last-2 -> δ = 1
            // ...
            size_t dist = last - idx - 1;

            if (dist <= 15)
                return { OpType::PaletteBackD, static_cast<uint8_t>(dist) };
        }

        // Palette advance (new label)
        // ------------------------------
        //state.palette.push_back(L);
        return { OpType::PaletteAdvance };
    }

    /**
     * @brief Encode a brick into a palette and sequence of operations (Algorithm 2, Sec. 3.2).
     *
     * Performs BFS encoding over the pyramid:
     *  - Initializes the palette with the root label
     *  - Traverses each level from coarsest to the second-to-finest
     *  - For each child node:
     *      - Computes 3D coordinates and Morton index
     *      - Determines the best operation via `bestOperation()`
     *      - Updates the palette if necessary (PaletteAdvance)
     *      - Stores stop-bit and leaf status in OpEntry
     *  - Packs the sequence of operations into 4-bit nibbles for compact storage
     *
     * @tparam b  Brick resolution (edge length)
     *
     * @param brick       Input brick to encode
     * @param updateFunc  Optional callback to handle packed operation stream and leaf flags
     *
     * @return EncoderState
     *         Contains the final palette and packed operation stream
     *
     * @note
     *  - Root node is always added to the palette first
     *  - Child nodes are processed in Morton/Z-order
     *  - Stop-bit indicates whether a node's sub-tree has constant children
     *  - Leaf nodes are determined at the finest level of the pyramid
     *  - Packing produces a byte stream (2 nibbles per byte) for efficient storage
     */
    template<size_t b>
    CompressedBrick encodeBrick(const Brick<b>& brick,
                             std::function<void(const CompressedBrick&)> updateFunc = nullptr)
    {
        CompressedBrick compressedBrick;
        compressedBrick.palette.clear(); // Line 2: initialize empty palette

        constexpr size_t levels = Brick<b>::Levels;

        std::vector<OpEntry> operations; // store sequence of operations (op, stop)

        // entry for the root node first
        const Node& root = brick.coarser(0)[0];  // single root node
        compressedBrick.palette.push_back(root.label);      // Add root label to palette

        // Encode an operation for the root node
        OpEntry rootEntry;
        rootEntry.op = OpType::PaletteAdvance;    // root is always in palette
        rootEntry.stopBit = root.constantChildren ? 0x01 : 0x00;
        rootEntry.isLeaf = (levels == 1);        // leaf if only one level
        operations.push_back(rootEntry);

        // --- Loop over levels from coarsest (0) to the second-to-finest (N-1) ---
        // Corresponds to pseudocode line 3: "for l ∈ [N .. 1]"
        for (size_t l = 0; l < levels - 1; ++l)
        {
            const auto& levelNodes = brick.coarser(l); // current level nodes
            const size_t nodesInLevel = levelNodes.size();
            const size_t childLevelSize = b >> (levels - 2 - l); // size in one dimension

            // Traverse nodes in Morton/Z-order
            // Pseudocode line 4: "for all nodes on level l (spacing 2^l) in Z-order do"
            for (size_t mortonIdx = 0; mortonIdx < nodesInLevel; ++mortonIdx)
            {
                const Node& parentNode = levelNodes[mortonIdx];

                // Skip nodes with constant children (interior levels)
                // Pseudocode line 6: "if pyramid[i].constantChildren then continue"
                if (parentNode.constantChildren) continue;

                const auto& childrenLevel = brick.coarser(l + 1); // next finer level

                // Decode Morton index to get 3D coordinates of the parent
                dim3 parentIdx = Utils::decodeMorton3D(mortonIdx);

                // --- Loop over 8 child nodes in Z-order ---
                // Pseudocode lines 8-9: "for all child nodes (spacing 2^(l-1)) in Z-order do"
                for (uint32_t childIdxLocal = 0; childIdxLocal < 8; ++childIdxLocal)
                {
                    // Decode child local offsets in 3D (bitmask)
                    uint32_t dx = childIdxLocal & 1;          // bit 0 → x offset
                    uint32_t dy = (childIdxLocal >> 1) & 1;   // bit 1 → y offset
                    uint32_t dz = (childIdxLocal >> 2) & 1;   // bit 2 → z offset

                    // Compute 3D coordinates of child in the grid
                    // Pseudocode line 9: "j ← index of current child node"
                    uint32_t childX = parentIdx.x * 2 + dx;
                    uint32_t childY = parentIdx.y * 2 + dy;
                    uint32_t childZ = parentIdx.z * 2 + dz;

                    // Convert child coordinates to Morton index to access array
                    size_t mortonChildIdx = Utils::morton3D(childX, childY, childZ);
                    const Node& childNode = childrenLevel[mortonChildIdx];

                    LabelType L = childNode.label;               // Pseudocode line 10: "L ← pyramid[j].label"
                    bool stop = childNode.constantChildren;    // Pseudocode line 11: "stop ← pyramid[j].constantChildren"

                    // Determine the best operation for this child
                    // Pseudocode line 12: "op ← bestOperation(parent, pyramid, palette, L)"
                    OpEntry entry = bestOperation(brick, childNode, parentNode, childLevelSize, parentIdx, compressedBrick);

                    // Update palette if operation requires
                    // Pseudocode line 13: "if op = Pa then palette.push(L)"
                    if (entry.op == OpType::PaletteAdvance)
                        compressedBrick.palette.push_back(L);

                    // Set stopBit and leaf status
                    // Pseudocode line 15: "output (op, stop)"
                    entry.stopBit = stop ? 0x01 : 0x00;
                    entry.isLeaf = (l + 1 == levels - 1); // leaf if finest level

                    operations.push_back(entry); // append operation to stream
                }
            }
        }

        // Pack the operations to nibbles (4-bit) for output
        auto [bytes, isLeaf] = packOperationsToNibbles(operations);
        compressedBrick.encodedData = std::move(bytes);
        compressedBrick.isLeaf = std::move(isLeaf);

        // Optional callback to update external stream
        if (updateFunc)
            updateFunc(compressedBrick);

        return compressedBrick;
    }

    RansModel buildRansModel(const FrequencyTable& rawTable);
    void ransEncodeSymbol(uint32_t& state, std::vector<uint8_t>& out, uint8_t symbol, const RansModel& model);

    template<typename T>
    inline void ransFlush(T& state, std::vector<uint8_t>& out)
    {
        // Determine how many bytes we need to fully represent the state
        size_t byteCount = sizeof(T);

        // Push out least-significant byte first (little-endian)
        for (size_t i = 0; i < byteCount; ++i)
        {
            out.push_back(static_cast<uint8_t>(state & 0xFF));
            state >>= 8;
        }
    }

    template<size_t b>
    CompressedDataset compressDataset(std::vector<Brick<b>>& bricks)
    {
        // Encode to nibbles
        //std::vector<EncoderState> brickStates;
        std::vector<CompressedBrick> nibbleBricks;
        nibbleBricks.reserve(bricks.size());

        for (size_t i = 0; i < bricks.size(); ++i)
        {
            auto& brick = bricks[i];

            // Build pyramid
            brick.build();

            // Encode operations
            CompressedBrick compressedBrick = encodeBrick(brick,
                                             (i % 512 == 0) ? updateFrequencyTables : nullptr);
            //EncoderState compressedBrick = encodeBrick(brick);

            // Construct compressed brick directly

            // Save compressed brick
            nibbleBricks.push_back(compressedBrick);

            //brickStates.push_back(state);
        }

        // build RANS models
        RansModel interiorModel = buildRansModel(interiorFreqTable);
        RansModel leafModel     = buildRansModel(leafFreqTable);


        // rANS compress each brick
        //for(auto& brick : brickStates)
        //{
        //    // Flatten the nibble stream with isLeaf/interior info
        //    struct SymbolEntry { uint8_t nibble; bool isInterior; };
        //    std::vector<SymbolEntry> symbols;

        //    size_t nibIndex = 0;
        //    for (uint8_t byte : brick.operations)
        //    {
        //        uint8_t hi = byte >> 4;
        //        uint8_t lo = byte & 0x0F;

        //        // High nibble
        //        if (nibIndex < brick.isLeaf.size())
        //            symbols.push_back({ hi, !brick.isLeaf[nibIndex++] }); // interior = !isLeaf

        //        // Low nibble
        //        if (nibIndex < brick.isLeaf.size())
        //            symbols.push_back({ lo, !brick.isLeaf[nibIndex++] });
        //    }

        //    // Encode symbols backwards
        //    uint32_t state = RANS_INIT;
        //    std::vector<uint8_t> compressedStream;
        //    compressedStream.reserve(brick.operations.size()); // rough reserve

        //    for (const auto& entry : std::views::reverse(symbols))
        //    {
        //        const RansModel& model = entry.isInterior ? interiorModel : leafModel;
        //        ransEncodeSymbol(state, compressedStream, entry.nibble, model);
        //    }

        //    // Final flush
        //    ransFlush(state, compressedStream);

        //    // The output is now forward-order byte stream for decoder
        //    brick.operations = std::move(compressedStream);

        //    CompressedBrick nibbleBrick{
        //        .palette = std::move(brick.palette),
        //        .encodedData = std::move(brick.operations)
        //    };

        //    // Save compressed brick
        //    nibbleBricks.push_back(std::move(nibbleBrick));
        //}
        
        return CompressedDataset{
            .interiorModel = interiorModel,
            .leafModel     = leafModel,
            .bricks        = std::move(nibbleBricks)
        };
    }

}