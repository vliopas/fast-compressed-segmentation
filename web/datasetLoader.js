// ==============================
// Binary Dataset Loader (Browser)
// Port from C++ to JavaScript
// see: encoder/src/fileIO.hpp for reference
// ==============================

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

        // --- brickID ---
        brick.ID = readUInt32();

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

    return dataset;
}