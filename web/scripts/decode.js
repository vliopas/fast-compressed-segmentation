/**
 * @file decode.js
 * @brief CPU-side rANS decoder for compressed bricks
 * 
 * Implements range Asymmetric Numeral Systems (rANS) decoding
 * for decompressing brick operation streams.
 */

const RANS_LIMIT = 1 << 23; ///< Minimum state before renormalization

/**
 * Decode all bricks using rANS (CPU)
 * @async
 * @param {Object} dataset - Dataset with encoded bricks
 * @return {Promise<Array<Uint8Array>>} Array of decoded brick data
 */
async function decodeAllRansBricks(dataset) {
    const interiorModel = dataset.interiorModel;
    const leafModel = dataset.leafModel;

    const decodedBricks = await Promise.all(
        dataset.bricks.map(async (brick) => decodeBrickRans(brick, interiorModel, leafModel))
    );

    return decodedBricks; // array of Uint8Array, one per brick
}

/**
 * Decode one brick using rANS
 * @param {Object} brick - Brick with encoded data
 * @param {Object} interiorModel - rANS model for interior nodes
 * @param {Object} leafModel - rANS model for leaf nodes
 * @return {Uint8Array} Decoded nibble stream
 * @private
 */
function decodeBrickRans(brick, interiorModel, leafModel) {
    const encoded = brick.encodedData;
    const isLeaf = brick.isLeaf;
    const nSymbols = brick.nSymbols;

    if (encoded.length < 4) throw new Error("Encoded data too short");

    // Initialize state from last 4 bytes (little-endian)
    let state =
        (encoded[encoded.length - 1] << 24) |
        (encoded[encoded.length - 2] << 16) |
        (encoded[encoded.length - 3] << 8) |
        encoded[encoded.length - 4];

    // Start context
    let context = { state, pos: encoded.length - 4 };

    const output = [];
    let i = 0;
    while (i < nSymbols) {
        // High nibble
        const hi = ransDecodeSymbol(context, encoded, interiorModel) & 0x0F;
        let lo = 0;
        i++;

        if (i < nSymbols) {
            // Low nibble
            lo = ransDecodeSymbol(context, encoded, interiorModel) & 0x0F;
            i++;
        }

        output.push((hi << 4) | lo);
    }

    return new Uint8Array(output);
}

/**
 * Decode one symbol using cumulative frequency table
 * @param {Object} context - Decoder context with state and position
 * @param {Uint8Array} input - Input byte stream
 * @param {Object} model - rANS probability model
 * @return {number} Decoded symbol
 * @private
 */
function ransDecodeSymbol(context, input, model) {
    const totalFreq = model.totalFreq;
    const x = context.state % totalFreq;

    // Find symbol s
    let s = 0;
    while (s + 1 < model.cumulativeFreq.length && x >= model.cumulativeFreq[s + 1]) {
        s++;
    }

    const freq = model.freq[s];
    const start = model.cumulativeFreq[s];

    // Update state
    context.state = freq * Math.floor(context.state / totalFreq) + (x - start);

    // Refill state if below RANS_LIMIT
    while (context.state < RANS_LIMIT && context.pos > 0) {
        context.pos--;
        context.state = (context.state << 8) | input[context.pos];
    }

    return s;
}


// Exports
export { decodeAllRansBricks, decodeBrickRans, ransDecodeSymbol };