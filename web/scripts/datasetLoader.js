// ==============================
// Binary Dataset Loader (Browser)
// Port from C++ to JavaScript
// see: encoder/src/fileIO.hpp for reference
// ==============================

// Decode Morton code (Z-order curve) to 3D coordinates
// Inverse of morton3D: extract x, y, z from a 30-bit Morton code
function decodeMorton3D(code) {
    const compact1by2 = (n) => {
        // inverse of "spread bits" (part1by2)
        n &= 0x9249249;
        n = (n ^ (n >> 2)) & 0x30c30c3;
        n = (n ^ (n >> 4)) & 0x300f00f;
        n = (n ^ (n >> 8)) & 0x30000ff;
        n = (n ^ (n >> 16)) & 0x3ff;
        return n;
    };

    const x = compact1by2(code);
    const y = compact1by2(code >> 1);
    const z = compact1by2(code >> 2);

    return { x, y, z };
}

export function loadDatasetFromArrayBuffer(arrayBuffer) {
    const view = new DataView(arrayBuffer);
    let offset = 0;

    const readUInt32 = () => {
        const v = view.getUint32(offset, true);
        offset += 4;
        return v;
    };

    const readUInt64 = () => {
        const low = view.getUint32(offset, true);
        const high = view.getUint32(offset + 4, true);
        offset += 8;
        return (BigInt(high) << 32n) | BigInt(low);
    };

    const readBytes = (size) => {
        const bytes = new Uint8Array(arrayBuffer, offset, size);
        offset += size;
        return bytes;
    };

    const dataset = {
        header: {},
        interiorModel: {},
        bricks: []
    };

    // -----------------------------
    // HEADER
    // -----------------------------
    const magicNumber = readUInt32();
    const version = readUInt32();
    const numBricks = readUInt32();
    const brickSize = readUInt32();

    if (magicNumber !== 0x43534244) {
        throw new Error("Invalid file format");
    }

    dataset.header = { magicNumber, version, numBricks, brickSize };

    // -----------------------------
    // RANS MODELS
    // -----------------------------
    function readRansModel() {
        const symbolCount = readUInt32();
        const totalFreq = readUInt32();

        const freq = new Array(symbolCount);
        const cumulativeFreq = new Array(symbolCount);

        for (let i = 0; i < symbolCount; i++) {
            freq[i] = readUInt32();
        }

        for (let i = 0; i < symbolCount; i++) {
            cumulativeFreq[i] = readUInt32();
        }

        return { symbolCount, totalFreq, freq, cumulativeFreq };
    }

    dataset.interiorModel = readRansModel();

    // -----------------------------
    // BRICKS
    // -----------------------------
    for (let i = 0; i < numBricks; i++) {
        const brick = {};

        // Preserve on-disk order for validation reordering later
        brick.originalIndex = i;

        // --- brickID ---
        brick.ID = readUInt32();

        // Decode Morton code to get 3D position
        const mortCoords = decodeMorton3D(brick.ID);
        brick.position = {
            x: mortCoords.x * brickSize,
            y: mortCoords.y * brickSize,
            z: mortCoords.z * brickSize
        };

        // --- palette (uint64_t) ---
        brick.paletteSize = readUInt32();
        brick.palette = [];

        for (let p = 0; p < brick.paletteSize; p++) {
            brick.palette.push(readUInt64()); // BigInt
        }

        // --- encodedData ---
        brick.encodedSize = readUInt32();
        brick.encodedData = readBytes(brick.encodedSize);

        brick.nSymbols = readUInt32();

        dataset.bricks.push(brick);
    }

    // Ensure deterministic Morton order for all downstream buffers
    dataset.bricks.sort((a, b) => a.ID - b.ID);

    return dataset;
}