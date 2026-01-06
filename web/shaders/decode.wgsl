// ==================== Data Structures ====================

struct StaticBrickInfo {
    encodedSize   : u32, // number of nibbles to read
    paletteSize   : u32, // number of palette entries
    streamOffset  : u32, // u32 index into nibbleStream
    paletteOffset : u32, // u32 index into paletteBuffer
    flags         : u32  // bit 0 = isEmpty (1 = empty, 0 = non-empty)
};

struct DynamicBrickInfo {
    outputOffset : u32, // u32 index into outVoxels
    targetLOD    : u32, // final resolution (varies per brick)
    lodScale     : f32  // precomputed: lodSize / brickSize (for ray marching optimization)
};

// ==================== Constants ====================

/// Palette prediction mode constants used in decoding operations.
/// These define different strategies for predicting voxel values based on
/// previously decoded data and palette entries.
const ParentReuse    : u32 = 0u;  // Rp - Reuse palette index from parent pixel
const NeighborX      : u32 = 1u;  // Rx - Predict from left neighbor
const NeighborY      : u32 = 2u;  // Ry - Predict from top neighbor
const NeighborZ      : u32 = 3u;  // Rz - Predict from diagonal neighbor
const PaletteAdvance : u32 = 4u;  // Pa - Use next palette entry in sequence
const PaletteBack0   : u32 = 5u;  // P0 - Reuse most recent palette entry
const PaletteBackD   : u32 = 6u;  // Pδ - Reuse palette entry from delta positions back

alias LabelType = u32;

const EMPTY_VALUE : u32 = 0xFFFFFFFFu; // Transparent RGBA
const WORKGROUP_SIZE : u32 = 64u;      // Keep in sync with JS dispatch and @workgroup_size

// ==================== GPU Storage Bindings ====================

@group(0) @binding(0)
var<storage, read> staticBricks : array<StaticBrickInfo>;

@group(0) @binding(1)
var<storage, read> dynamicBricks : array<DynamicBrickInfo>;

@group(0) @binding(2)
var<storage, read> nibbleStream : array<u32>;

@group(0) @binding(3)
var<storage, read> paletteBuffer : array<vec2<u32>>;

@group(0) @binding(4)
var<storage, read_write> outVoxels : array<u32>;

@group(0) @binding(5)
var<storage, read> workQueue : array<u32>;

@group(0) @binding(6)
var<uniform> workCount : u32;

@group(0) @binding(7)
var<storage, read> labelKeys : array<vec2<u32>>;  // Transfer function: u64 labels (low, high)

@group(0) @binding(8)
var<storage, read> labelColors : array<u32>;  // Transfer function: corresponding RGBA8 values

// ==================== Nibble Reading ====================

struct NibbleReader {
    bytePos        : u32, // current byte position in nibbleStream (as u32 index)
    highNibbleNext : u32  // 1 if next nibble is high, 0 if low
};

/// Reads a single nibble (4-bit value) from the packed nibble stream.
/// Handles low nibble first, then high nibble of next byte (consistent with C++ readNibble).
fn readNibble(reader : ptr<function, NibbleReader>) -> u32 {
    let streamIdx = (*reader).bytePos;
    // Big-endian packing: byte 0 at bits 24-31, byte 3 at bits 0-7
    // So we need to invert: streamIdx 0 → shift 24, streamIdx 3 → shift 0
    let byteVal = (nibbleStream[streamIdx >> 2u] >> (24u - ((streamIdx & 3u) << 3u))) & 0xFFu;

    var nib: u32;
    if ((*reader).highNibbleNext == 1u) {
        nib = (byteVal >> 4u) & 0xFu;           // upper 4 bits
        (*reader).highNibbleNext = 0u;          // next time read low nibble
    } else {
        nib = byteVal & 0x0Fu;                  // lower 4 bits
        (*reader).highNibbleNext = 1u;
        (*reader).bytePos += 1u;                // we consumed the whole byte now
    }

    return nib;
}

// ==================== Operation Decoding ====================

struct Op {
    code  : u32,
    stop  : u32,
    delta : u32
};

/// Reads and decodes the next operation from the nibble stream.
/// Parses operation code, stop bit, and optional delta value for PaletteBackD.
fn readNextOperationAndStopBit(reader : ptr<function, NibbleReader>) -> Op {
    let nib = readNibble(reader);
    let code = nib >> 1u;
    let stop = nib & 1u;

    var delta = 0u;
    if (code == PaletteBackD) {
        delta = readNibble(reader);
    }

    return Op(code, stop, delta);
}

// ==================== Label to RGBA Conversion ====================

/// Transfer function: Map u64 label to RGBA using lookup table, fallback to hash
/// Label 0 = empty/background -> fully transparent
/// Labels in lookup table -> use defined color and opacity
/// Other labels -> pseudo-random color with semi-transparent alpha
fn hashLabel(label_low: u32, label_high: u32) -> u32 {
    // Check if zero (both parts must be zero)
    if (label_low == 0u && label_high == 0u) {
        return 0x00000000u; // Transparent
    }
    
    // Transfer function lookup: search for label in labelKeys array
    let numEntries = arrayLength(&labelKeys);
    for (var i: u32 = 0u; i < numEntries; i++) {
        let key = labelKeys[i];
        if (key.x == label_low && key.y == label_high) {
            return labelColors[i];  // Found in transfer function
        }
    }
    
    // Fallback: hash-based color for labels not in transfer function
    var seed_low = label_low;
    var seed_high = label_high;
    
    // Mix seeds together
    seed_low = seed_low ^ seed_high;
    seed_low = seed_low ^ (seed_low >> 16u);
    seed_low = seed_low * 0x7feb352du;
    seed_low = seed_low ^ (seed_low >> 15u);
    
    seed_high = seed_high ^ seed_low;
    seed_high = seed_high ^ (seed_high >> 16u);
    seed_high = seed_high * 0x85ebca6bu;
    seed_high = seed_high ^ (seed_high >> 13u);
    
    // Generate RGB from mixed seeds
    let r = seed_low & 0xFFu;
    let g = (seed_low >> 8u) & 0xFFu;
    let b = (seed_high >> 8u) & 0xFFu;
    
    // Pack as RGBA8 with semi-transparent alpha for volumetric compositing
    return (r << 0u) | (g << 8u) | (b << 16u) | (0x40u << 24u);  // Alpha = 64 (~25% opacity)
}

// ==================== Morton Code Utilities ====================

/// Encodes 3D coordinates into a Morton code (Z-order curve).
/// Interleaves bits to preserve spatial locality (supports up to 1024 per axis).
fn morton3D(x: u32, y: u32, z: u32) -> u32 {
    // Interleave bits of x, y, z coordinates into Morton code
    var answer: u32 = 0u;
    for (var i: u32 = 0u; i < 10u; i++) {
        answer |= ((x >> i) & 1u) << (3u * i);
        answer |= ((y >> i) & 1u) << (3u * i + 1u);
        answer |= ((z >> i) & 1u) << (3u * i + 2u);
    }
    return answer;
}

/// Extracts every third bit from input to decode one axis from Morton code.
/// Used to reverse the bit interleaving of Morton encoding.
fn compact1by2(n0: u32) -> u32 {
    // Extract every third bit to reverse Morton encoding
    var n = n0;
    n &= 0x09249249u;
    n = (n ^ (n >> 2u)) & 0x030c30c3u;
    n = (n ^ (n >> 4u)) & 0x0300f00fu;
    n = (n ^ (n >> 8u)) & 0x030000ffu;
    n = (n ^ (n >> 16u)) & 0x000003ffu;
    return n;
}

/// Decodes a Morton code back into separate x, y, z coordinates.
/// Reverses the bit interleaving to extract original 3D position.
fn decodeMorton3D(code: u32) -> vec3<u32> {
    // Decode Morton code into x, y, z coordinates
    let x = compact1by2(code);
    let y = compact1by2(code >> 1u);
    let z = compact1by2(code >> 2u);
    return vec3<u32>(x, y, z);
}

// ==================== Octree Navigation ====================

/// Computes Morton indices of all 8 children of a parent voxel.
/// Each child is at (parent_coord * 2 + offset) where offset is 0 or 1 per dimension.
fn computeChildMortonIndices(parentMorton: u32) -> array<u32, 8> {
    // Compute Morton indices of all 8 children of a parent voxel
    var outIdx: array<u32, 8>;

    let p: vec3<u32> = decodeMorton3D(parentMorton);

    // Child offsets in Morton order
    let dx: array<u32, 8> = array<u32, 8>(0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u);
    let dy: array<u32, 8> = array<u32, 8>(0u, 0u, 1u, 1u, 0u, 0u, 1u, 1u);
    let dz: array<u32, 8> = array<u32, 8>(0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u);

    for (var c: u32 = 0u; c < 8u; c++) {
        let cx: u32 = p.x * 2u + dx[c];
        let cy: u32 = p.y * 2u + dy[c];
        let cz: u32 = p.z * 2u + dz[c];
        outIdx[c] = morton3D(cx, cy, cz);
    }

    return outIdx;
}

/// Converts a node's Morton code at a given level to an output buffer index.
/// Scales coordinates to match the target LOD resolution.
fn coarseNodeToOutputIndex(mortonL: u32, level: u32, targetLOD: u32) -> u32 {
    // Convert coarse node Morton code at given level to output index for target LOD
    let scale: u32 = 1u << (targetLOD - level);
    let p: vec3<u32> = decodeMorton3D(mortonL);
    let p_t = vec3<u32>(p.x * scale, p.y * scale, p.z * scale);
    return morton3D(p_t.x, p_t.y, p_t.z);
}

fn getNeighborValue(
    outBuf: ptr<storage, array<u32>, read_write>,
    level: u32,
    childMortonIdx: u32,
    op: u32,
    targetLOD: u32,
    baseOut: u32
) -> u32 {
    /// Retrieves neighbor value in Morton-encoded octree.
    /// Implements neighbor reuse rule: neighbor outside 2×2×2 sibling block.
    /// If neighbor has later Morton index, falls back to neighbor's parent.
    
    let childCoords: vec3<u32> = decodeMorton3D(childMortonIdx);

    // Extract LOCAL position in the 2×2×2 sibling block
    let lx = i32(childCoords.x & 1u);
    let ly = i32(childCoords.y & 1u);
    let lz = i32(childCoords.z & 1u);

    // Compute neighbor coordinates using signed integers to match C++ logic
    var nxi: i32 = i32(childCoords.x);
    var nyi: i32 = i32(childCoords.y);
    var nzi: i32 = i32(childCoords.z);
    
    if (op == NeighborX) {
        nxi = select(nxi + 1, nxi - 1, lx == 0);
    } else if (op == NeighborY) {
        nyi = select(nyi + 1, nyi - 1, ly == 0);
    } else if (op == NeighborZ) {
        nzi = select(nzi + 1, nzi - 1, lz == 0);
    }

    // Check bounds - if neighbor is outside brick, use parent of current node
    // Explicit signed integer comparison matches C++ implementation
    let maxCoord: i32 = i32((1u << (level + 1u)) - 1u);
    if (nxi < 0 || nyi < 0 || nzi < 0 || nxi > maxCoord || nyi > maxCoord || nzi > maxCoord) {
        let parentMorton = morton3D(childCoords.x >> 1u, childCoords.y >> 1u, childCoords.z >> 1u);
        let parentOutIdx = coarseNodeToOutputIndex(parentMorton, level, targetLOD) + baseOut;
        return (*outBuf)[parentOutIdx];
    }

    // Convert back to unsigned for Morton encoding
    let nx: u32 = u32(nxi);
    let ny: u32 = u32(nyi);
    let nz: u32 = u32(nzi);

    // Neighbor Morton index with adjusted coordinates
    let neighborMorton: u32 = morton3D(nx, ny, nz);

    // Already decoded → use neighbor
    if (neighborMorton < childMortonIdx) {
        let outIdx: u32 = coarseNodeToOutputIndex(neighborMorton, level + 1u, targetLOD) + baseOut;
        return (*outBuf)[outIdx];
    }

    // Later Morton → fallback to neighbor's parent
    let parentMorton: u32 = morton3D(nx >> 1u, ny >> 1u, nz >> 1u);
    let parentOutIdx: u32 = coarseNodeToOutputIndex(parentMorton, level, targetLOD) + baseOut;
    return (*outBuf)[parentOutIdx];
}

// ==================== Decoding Helpers ====================

// Shared state for cooperative sub-block fills and brick metadata broadcast
var<workgroup> wgFillActive    : u32;
var<workgroup> wgFillStart     : u32;
var<workgroup> wgFillBlockSize : u32;
var<workgroup> wgFillValue     : u32;

/// Cooperatively fills an entire sub-block with a single RGBA value.
/// Leader (localId == 0) must pre-populate wgFill* and wgFillActive before invoking.
fn fillSubBlock(localId: u32) {
    // Ensure all threads see wgFill* before starting
    workgroupBarrier();

    if (wgFillActive == 1u) {
        let startIdx = wgFillStart;
        let blockSize = wgFillBlockSize;
        let rgba = wgFillValue;

        var i: u32 = localId;
        while (i < blockSize) {
            outVoxels[startIdx + i] = rgba;
            i += WORKGROUP_SIZE;
        }
    }

    // Keep control flow in lockstep for all threads
    workgroupBarrier();

    if (localId == 0u) {
        wgFillActive = 0u; // clear for next use
    }
}

// ==================== Main Decode Kernel ====================

@compute @workgroup_size(64)
fn decodeBrick(
    @builtin(workgroup_id) wg_id : vec3<u32>,
    @builtin(local_invocation_id) lid3 : vec3<u32>
) {
    let localId = lid3.x;
    let workIdx = wg_id.x;

    // Bounds check work queue
    if (workIdx >= workCount) { return; }

    // Get brick index from work queue (one brick per workgroup)
    let brickIdx = workQueue[workIdx];
    if (brickIdx >= arrayLength(&staticBricks)) { return; }

    let isLeader = localId == 0u;

    // Leader pulls brick metadata and initializes decode state
    var reader: NibbleReader = NibbleReader(0u, 0u);
    var ip: u32 = 0u;
    let staticInfo = staticBricks[brickIdx];
    let dynamicInfo = dynamicBricks[brickIdx];

    let baseOut = dynamicInfo.outputOffset;
    let targetLOD = dynamicInfo.targetLOD;
    let paletteOffset = staticInfo.paletteOffset;

    if (isLeader) {
        reader = NibbleReader(
            staticInfo.streamOffset, // bytePos (u32 index into nibbleStream)
            0u                       // highNibbleNext - start with LOW nibble to match CPU path
        );

        // Pseudocode line 1: i_p ← 0 (palette read index)
        ip = 0u;

        // Pseudocode line 3: out[0] = palette[i_p]
        let rootLabel = paletteBuffer[paletteOffset + ip];
        outVoxels[baseOut] = hashLabel(rootLabel.x, rootLabel.y);
    }

    // Ensure root write visible before traversal
    workgroupBarrier();

    // Pseudocode line 4: for l ∈ [N .. t+1]
    for (var l: u32 = 0u; l < targetLOD; l++) {
        // Pseudocode line 5: for all decoded nodes at level l
        let nodeCount = 1u << (3u * l); // number of nodes at this level (Z-order)

        for (var mortonIdx: u32 = 0u; mortonIdx < nodeCount; mortonIdx++) {
            // Pseudocode line 6: i ← index of node in out
            let i = coarseNodeToOutputIndex(mortonIdx, l, targetLOD) + baseOut;

            // Pseudocode line 7: if last child already filled then skip decode work (but keep control uniform)
            let childrenMortonIndices = computeChildMortonIndices(mortonIdx);
            let lastChildIdx = coarseNodeToOutputIndex(childrenMortonIndices[7], l + 1u, targetLOD) + baseOut;
            let alreadyFilled = outVoxels[lastChildIdx] != EMPTY_VALUE;

            // Pseudocode line 8: parent ← out[i]
            let parent = outVoxels[i];

            // Pseudocode line 9: for all child nodes
            for (var c: u32 = 0u; c < 8u; c++) {
                let childMortonIdx = childrenMortonIndices[c];

                // Pseudocode line 10: j ← index of child node
                let j = coarseNodeToOutputIndex(childMortonIdx, l + 1u, targetLOD) + baseOut;

                if (isLeader && !alreadyFilled) {
                    wgFillActive = 0u; // default: no fill

                    // Pseudocode line 11: (op, stop) ← readNextOperationAndStopBit()
                    let op = readNextOperationAndStopBit(&reader);

                    // Pseudocode lines 12–17: switch(op)
                    var val: u32 = EMPTY_VALUE;
                    switch (op.code) { // !! NO FALLTHROUGH IN WGSL !!
                        case ParentReuse: {
                            val = parent;
                        }
                        case NeighborX: {
                            val = getNeighborValue(&outVoxels, l, childMortonIdx, op.code, targetLOD, baseOut);
                        }
                        case NeighborY: {
                            val = getNeighborValue(&outVoxels, l, childMortonIdx, op.code, targetLOD, baseOut);
                        }
                        case NeighborZ: {
                            val = getNeighborValue(&outVoxels, l, childMortonIdx, op.code, targetLOD, baseOut);
                        }
                        case PaletteAdvance: {
                            ip += 1u;
                            let label = paletteBuffer[paletteOffset + ip];
                            val = hashLabel(label.x, label.y);
                        }
                        case PaletteBack0: {
                            let label = paletteBuffer[paletteOffset + ip];
                            val = hashLabel(label.x, label.y);
                        }
                        case PaletteBackD: {
                            let paletteIdx: u32 = ip - op.delta - 1u;
                            let label = paletteBuffer[paletteOffset + paletteIdx];
                            val = hashLabel(label.x, label.y);
                        }
                        default: {
                            val = 0x00000000u;
                        }
                    }

                    outVoxels[j] = val;

                    // Pseudocode lines 18–19: if stop then fill sub-block cooperatively
                    if (op.stop == 1u) {
                        wgFillActive    = 1u;
                        wgFillStart     = j;
                        wgFillBlockSize = 1u << (3u * (targetLOD - (l + 1u)));
                        wgFillValue     = val;
                    }
                }

                // All threads participate so barriers in fillSubBlock stay balanced
                fillSubBlock(localId);
            }
        }
    }
}