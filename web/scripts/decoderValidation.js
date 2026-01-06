/**
 * Validation module for comparing CPU and GPU decoded brick outputs
 */

export class DecoderValidator {
    constructor() {
        this.cpuReference = null;
        this.gpuOutput = null;
        this.transferFunctionMap = null;  // Map from label (BigInt) to RGBA (u32)
    }

    static loggingEnabled = true;

    static setLoggingEnabled(enabled) {
        DecoderValidator.loggingEnabled = !!enabled;
    }

    log(message, ...args) {
        if (!DecoderValidator.loggingEnabled) return;
        console.log(message, ...args);
    }

    warn(message, ...args) {
        if (!DecoderValidator.loggingEnabled) return;
        console.warn(message, ...args);
    }

    /**
     * Set transfer function mapping for label->RGBA conversion
     * @param {Map<BigInt, number>} transferFunctionMap - Label to RGBA mapping
     */
    setTransferFunction(transferFunctionMap) {
        this.transferFunctionMap = transferFunctionMap;
        this.log(`✓ Set transfer function with ${transferFunctionMap.size} label mappings`);
    }

    /**
     * Convert label to RGBA using transfer function (matches GPU logic)
     * @param {BigInt} label - 64-bit label value
     * @returns {number} RGBA u32 value
     */
    labelToRGBA(label) {
        // Check for empty label
        if (label === 0n) {
            return 0x00000000;
        }

        // Check transfer function first (matches GPU lookup)
        if (this.transferFunctionMap && this.transferFunctionMap.has(label)) {
            return this.transferFunctionMap.get(label);
        }

        // Fallback to hash (matches GPU hashLabel fallback)
        const low = Number(label & 0xFFFFFFFFn) >>> 0;
        const high = Number(label >> 32n) >>> 0;
        return this.hashLabelParts(low, high);
    }

    // Hash function matches decode.wgsl hashLabel(low, high) fallback
    hashLabelParts(low, high) {
        let seedLow = low >>> 0;
        let seedHigh = high >>> 0;

        seedLow = (seedLow ^ seedHigh) >>> 0;
        seedLow = (seedLow ^ (seedLow >>> 16)) >>> 0;
        seedLow = Math.imul(seedLow, 0x7feb352d) >>> 0;
        seedLow = (seedLow ^ (seedLow >>> 15)) >>> 0;

        seedHigh = (seedHigh ^ seedLow) >>> 0;
        seedHigh = (seedHigh ^ (seedHigh >>> 16)) >>> 0;
        seedHigh = Math.imul(seedHigh, 0x85ebca6b) >>> 0;
        seedHigh = (seedHigh ^ (seedHigh >>> 13)) >>> 0;

        const r = seedLow & 0xFF;
        const g = (seedLow >>> 8) & 0xFF;
        const b = (seedHigh >>> 8) & 0xFF;

        // Match GPU fallback: 25% opacity (0x40)
        return (0x40 << 24) | (b << 16) | (g << 8) | r;
    }

    hashLabel64(label) {
        // label is BigInt
        const low = Number(label & 0xFFFFFFFFn) >>> 0;
        const high = Number(label >> 32n) >>> 0;
        if (low === 0 && high === 0) return 0x00000000;
        return this.hashLabelParts(low, high);
    }

    // Split bits to interleave for a Morton code (sufficient for small brick grids)
    part1By2(n) {
        let x = n & 0x3FF; // 10 bits are plenty for 2x2x2 bricks
        x = (x | (x << 16)) & 0x30000FF;
        x = (x | (x << 8)) & 0x300F00F;
        x = (x | (x << 4)) & 0x30C30C3;
        x = (x | (x << 2)) & 0x9249249;
        return x >>> 0;
    }

    // Encode 3D coords into a Morton index (Z-order)
    encodeMorton3D(x, y, z) {
        return (
            this.part1By2(x) |
            (this.part1By2(y) << 1) |
            (this.part1By2(z) << 2)
        ) >>> 0;
    }

    // Bricks are stored in Morton order in the bin; build traversal that follows it
    mortonBrickTraversal(brickCount) {
        if (brickCount <= 0) return [];

        const dim = Math.round(Math.cbrt(brickCount));
        if (dim ** 3 !== brickCount) {
            // Fallback: assume storage index already matches logical index
            return Array.from({ length: brickCount }, (_, i) => ({
                storageIndex: i,
                mortonIndex: i,
                coords: { x: null, y: null, z: null }
            }));
        }

        // Compute Morton code for each brick coordinate, then sort by Morton to match bin order
        const bricks = [];
        for (let z = 0; z < dim; z++) {
            for (let y = 0; y < dim; y++) {
                for (let x = 0; x < dim; x++) {
                    bricks.push({
                        mortonIndex: this.encodeMorton3D(x, y, z),
                        coords: { x, y, z }
                    });
                }
            }
        }

        bricks.sort((a, b) => a.mortonIndex - b.mortonIndex);

        return bricks.map((b, storageIndex) => ({
            storageIndex,
            mortonIndex: b.mortonIndex,
            coords: b.coords
        }));
    }

    /**
     * Load CPU reference data from binary file
     * @param {string} filePath - Path to decoded_reference.bin
     * @returns {Promise<Uint32Array>} Reference RGBA values
     */
    async loadCPUReference(filePath = '../encoder/CSVEncoder/decoded_reference.bin', reorderMap = null) {
        try {
            const response = await fetch(filePath);
            if (!response.ok) {
                console.error(`Failed to load reference: ${response.status}`);
                return null;
            }
            const arrayBuffer = await response.arrayBuffer();

            // File now contains raw u64 labels (8 bytes each) from C++
            let labelData = new BigUint64Array(arrayBuffer);

            // Reorder to match the browser's sorted-by-ID brick order (file is original scan order)
            if (Array.isArray(reorderMap) && reorderMap.length > 0) {
                const voxelsPerBrick = labelData.length / reorderMap.length;
                if (!Number.isInteger(voxelsPerBrick)) {
                    this.warn('CPU reference reorder skipped: voxel count not divisible by brick count');
                } else {
                    const reordered = new BigUint64Array(labelData.length);
                    for (let sortedIdx = 0; sortedIdx < reorderMap.length; sortedIdx++) {
                        const srcBrick = reorderMap[sortedIdx];
                        const srcStart = srcBrick * voxelsPerBrick;
                        const dstStart = sortedIdx * voxelsPerBrick;
                        reordered.set(labelData.subarray(srcStart, srcStart + voxelsPerBrick), dstStart);
                    }
                    labelData = reordered;
                    this.log(`✓ Reordered CPU reference using original brick order mapping (${reorderMap.length} bricks)`);
                }
            }

            // Convert labels to RGBA using transfer function (same as GPU)
            this.cpuReference = new Uint32Array(labelData.length);
            for (let i = 0; i < labelData.length; i++) {
                this.cpuReference[i] = this.labelToRGBA(labelData[i]);
            }

            this.log(`✓ Loaded ${labelData.length} labels from CPU reference, converted to RGBA`);
            return this.cpuReference;
        } catch (error) {
            console.error('Error loading CPU reference:', error);
            return null;
        }
    }

    /**
     * Set GPU output data for comparison
     * @param {Uint32Array} gpuData - GPU decoded output
     */
    setGPUOutput(gpuData) {
        this.gpuOutput = new Uint32Array(gpuData);
        this.log(`✓ Set GPU output: ${this.gpuOutput.length} labels`);
    }

    /**
     * Compare CPU and GPU outputs with per-brick statistics
     * @returns {Object} Comparison results with statistics
     */
    compare() {
        if (!this.cpuReference) {
            // console.error('CPU reference not loaded'); // Disabled: CPU reference is optional
            return null;
        }
        if (!this.gpuOutput) {
            console.error('GPU output not set');
            return null;
        }

        if (this.cpuReference.length !== this.gpuOutput.length) {
            console.error(
                `Size mismatch: CPU=${this.cpuReference.length}, GPU=${this.gpuOutput.length}`
            );
            return null;
        }

        const brickSize = 262144; // voxels per LOD 6 brick
        const totalCount = this.cpuReference.length;
        const brickCount = Math.floor(totalCount / brickSize);

        const traversal = this.mortonBrickTraversal(brickCount);

        let matches = 0;
        let mismatches = 0;
        const firstMismatches = [];
        const maxMismatchesToShow = 10;
        const brickStats = [];

        for (const brick of traversal) {
            const brickStart = brick.storageIndex * brickSize;
            const brickEnd = Math.min(brickStart + brickSize, totalCount);
            let brickMatches = 0;

            for (let i = brickStart; i < brickEnd; i++) {
                const cpu = this.cpuReference[i];
                const gpu = this.gpuOutput[i];

                if (cpu === gpu) {
                    matches++;
                    brickMatches++;
                } else {
                    mismatches++;
                    if (firstMismatches.length < maxMismatchesToShow) {
                        firstMismatches.push({
                            index: i,
                            brick: brick.mortonIndex,
                            storageIndex: brick.storageIndex,
                            brickOffset: i - brickStart,
                            cpu,
                            gpu
                        });
                    }
                }
            }

            const brickTotal = brickEnd - brickStart;
            brickStats.push({
                brick: brick.mortonIndex,
                storageIndex: brick.storageIndex,
                coords: brick.coords,
                matches: brickMatches,
                total: brickTotal,
                percentage: ((brickMatches / brickTotal) * 100).toFixed(2)
            });
        }

        const matchPercentage = ((matches / totalCount) * 100).toFixed(2);

        return {
            matches,
            mismatches,
            totalCount,
            matchPercentage,
            firstMismatches,
            brickStats
        };
    }

    /**
     * Print comparison results to console
     */
    printResults() {
        const result = this.compare();
        if (!result) return;

        this.log('\n' + '='.repeat(60));
        this.log('DECODER VALIDATION RESULTS');
        this.log('='.repeat(60));
        this.log(`✓ Matches:     ${result.matches} / ${result.totalCount}`);
        this.log(`✗ Mismatches:  ${result.mismatches}`);
        this.log(`Match Rate:    ${result.matchPercentage}%`);
        this.log('='.repeat(60));
        // Print per-brick statistics
        this.log('\nPER-BRICK STATISTICS (Morton order as stored in bin):');
        for (const stat of result.brickStats) {
            const coordStr = stat.coords.x !== null ? ` xyz(${stat.coords.x},${stat.coords.y},${stat.coords.z})` : '';
            this.log(
                `  Brick ${stat.brick} [storage ${stat.storageIndex}${coordStr}]: ` +
                `${stat.matches}/${stat.total} (${stat.percentage}%)`
            );
        }
        if (result.mismatches > 0) {
            this.log('\nFirst mismatches:');
            result.firstMismatches.forEach(m => {
                this.log(
                    `  [${m.index}] brick ${m.brick} (storage ${m.storageIndex}, offset ${m.brickOffset}): ` +
                    `CPU: 0x${m.cpu.toString(16).padStart(8, '0')} ` +
                    `GPU: 0x${m.gpu.toString(16).padStart(8, '0')}`
                );
            });
        } else {
            this.log('\n✓✓✓ ALL LABELS MATCH! GPU decoder is correct! ✓✓✓');
        }
        this.log('='.repeat(60) + '\n');

        return result;
    }
}

export default DecoderValidator;
