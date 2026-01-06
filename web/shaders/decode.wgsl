struct BrickInfo {
    encodedSize  : u32;   // max number of nibbles to read
    paletteSize  : u32;    // palette size
    level        : u32;   // brick LOD
    targetLOD    : u32;   // final resolution
};

const EMPTY_VALUE : u32 = 0xFFFFFFFFu; // equivalent to max uint32

@group(0) @binding(0)
var<storage, read> bricks : array<BrickInfo>;

@group(0) @binding(1)
var<storage, read> nibbleStream : array<u32>; // 4-bit nibbles packed in bytes

@group(0) @binding(2)
var<storage, read> paletteBuffer : array<u32>;

@group(0) @binding(3)
var<storage, read_write> outVoxels : array<u32>;

struct NibbleReader {
    bytePos : u32;
    high    : bool;
    count   : u32;
};

fn readNibble(reader : ptr<function, NibbleReader>) -> u32 {
    if ((*reader).count == 0u) {
        return 0u; // safety fallback
    }

    let byte = nibbleStream[(*reader).bytePos] & 0xFFu;
    var nib : u32;

    if ((*reader).high) {
        nib = byte >> 4u;
        (*reader).high = false;
    } else {
        nib = byte & 0xFu;
        (*reader).high = true;
        (*reader).bytePos += 1u;
    }

    (*reader).count -= 1u;
    return nib;
}

struct Op {
    code  : u32;
    stop  : u32;
    delta : u32;
};

fn readOp(reader : ptr<function, NibbleReader>) -> Op {
    let nib = readNibble(reader);

    let code = nib >> 1u;
    let stop = nib & 1u;

    var delta = 0u;
    if (code == 6u) { // PaletteBackD
        delta = readNibble(reader);
    }

    return Op(code, stop, delta);
}

fn morton3D(x: u32, y: u32, z: u32) -> u32 {
    // implement 3D Morton encode (interleave bits)
    var answer: u32 = 0u;
    for (var i: u32 = 0u; i < 10u; i++) { // assuming max 1024 per axis
        answer |= ((x >> i) & 1u) << (3u*i);
        answer |= ((y >> i) & 1u) << (3u*i + 1u);
        answer |= ((z >> i) & 1u) << (3u*i + 2u);
    }
    return answer;
}

fn coarseNodeToOutputIndex(mortonL: u32, level: u32, targetLOD: u32) -> u32 {
    let scale: u32 = 1u << (targetLOD - level);
    let p: vec3<u32> = decodeMorton3D(mortonL); // your GPU decodeMorton3D
    let p_t = vec3<u32>(p.x * scale, p.y * scale, p.z * scale);
    return morton3D(p_t.x, p_t.y, p_t.z);
}

fn fillSubBlock(nodeMortonIdx: u32, level: u32, targetLOD: u32, label: u32, baseOut: u32) {
    let startIdx = coarseNodeToOutputIndex(nodeMortonIdx, level, targetLOD) + baseOut;
    let blockSize = 1u << (3u * (targetLOD - level));

    for (var i: u32 = 0u; i < blockSize; i++) {
        outVoxels[startIdx + i] = label;
    }
}

@compute @workgroup_size(1)
fn decodeBrick(@builtin(global_invocation_id) gid : vec3<u32>) {
    let brickIdx = gid.x;
    if (brickIdx >= arrayLength(&bricks)) { return; }

    let brick = bricks[brickIdx];
    let baseOut = brick.outputOffset; // base index in outVoxels

    // -------------------------------
    // Pseudocode line 1: i_p ← 0
    // -------------------------------
    var ip: u32 = 0u; // palette read index

    // -------------------------------
    // Pseudocode line 2: out[...] ← ∅
    // -------------------------------
    // Already handled by host/GPU buffer initialization or can clear here if needed

    // -------------------------------
    // Pseudocode line 3: out[0] = palette[i_p]
    // -------------------------------
    outVoxels[baseOut] = paletteBuffer[ip];

    // Initialize nibble reader
    var reader = NibbleReader(
        0u,               // bytePos in nibbleStream
        true,             // start with high nibble
        brick.encodedSize // total number of nibbles
    );

    // -------------------------------
    // Pseudocode line 4: for l ∈ [N .. t+1]
    // -------------------------------
    for (var l: u32 = 0u; l < brick.level; l++) {

        // -------------------------------
        // Pseudocode line 5: for all decoded nodes at level l
        // -------------------------------
        let nodeCount = 1u << (3u * l); // number of nodes at this level (Z-order)
        for (var mortonIdx: u32 = 0u; mortonIdx < nodeCount; mortonIdx++) {

            // -------------------------------
            // Pseudocode line 6: i ← index of node in out
            // -------------------------------
            let i = coarseNodeToOutputIndex(mortonIdx, l, brick.targetLOD) + baseOut;

            // -------------------------------
            // Pseudocode line 7: if last child already filled then continue
            // -------------------------------
            let childrenMortonIndices = computeChildMortonIndices(mortonIdx); // GPU function
            let lastChildIdx = coarseNodeToOutputIndex(childrenMortonIndices[7], l + 1, brick.targetLOD) + baseOut;
            if (outVoxels[lastChildIdx] != 0u) { continue; }

            // -------------------------------
            // Pseudocode line 8: parent ← out[i]
            // -------------------------------
            let parent = outVoxels[i];

            // -------------------------------
            // Pseudocode line 9: for all child nodes
            // -------------------------------
            for (var c: u32 = 0u; c < 8u; c++) {
                let childMortonIdx = childrenMortonIndices[c];

                // -------------------------------
                // Pseudocode line 10: j ← index of child node
                // -------------------------------
                let j = coarseNodeToOutputIndex(childMortonIdx, l + 1, brick.targetLOD) + baseOut;

                // -------------------------------
                // Pseudocode line 11: (op, stop) ← readNextOperationAndStopBit()
                // -------------------------------
                let op = readOp(&reader);

                var val: u32 = 0u;

                // -------------------------------
                // Pseudocode lines 12–17: switch(op)
                // -------------------------------
                var val: u32 = 0u;

                switch (op.code) 
                {
                    case 0u: { // ParentReuse
                        val = parent;
                        break;
                    }
                    case 1u: { // NeighborX
                        val = getNeighborValue(l, childMortonIdx, op.code, brick.targetLOD, baseOut); 
                        break;
                    }
                    case 2u: { // NeighborY
                        val = getNeighborValue(l, childMortonIdx, op.code, brick.targetLOD, baseOut);
                        break; 
                    }
                    case 3u: { // NeighborZ
                        val = getNeighborValue(l, childMortonIdx, op.code, brick.targetLOD, baseOut);
                        break;
                    }
                    case 4u: { // PaletteAdvance
                        ip += 1u;
                        val = paletteBuffer[ip]; 
                    }
                    case 5u: { // PaletteBack0
                        val = paletteBuffer[ip]; 
                    }
                    case 6u: { // PaletteBackD
                        val = paletteBuffer[ip - op.delta]; 
                    }
                }

                outVoxels[j] = val;

                // -------------------------------
                // Pseudocode lines 18–19: if stop then fill sub-block
                // -------------------------------
                if (op.stop == 1u) {
                    fillSubBlock(childMortonIdx, l + 1, brick.targetLOD, val, baseOut);
                }
            }
        }
    }
}