/**
 * Validation module for comparing CPU and GPU decoded brick outputs
 */

export class DecoderValidator {
    constructor() {
        this.cpuReference = null;
        this.gpuOutput = null;
        this.transferFunctionMap = null;  // Map from label (BigInt) to RGBA (u32)
    }

    /**
     * Set transfer function mapping for label->RGBA conversion
     * @param {Map<BigInt, number>} transferFunctionMap - Label to RGBA mapping
     */
    setTransferFunction(transferFunctionMap) {
        this.transferFunctionMap = transferFunctionMap;
        console.log(`✓ Set transfer function with ${transferFunctionMap.size} label mappings`);
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

    /**
     * Load CPU reference data from binary file
     * @param {string} filePath - Path to decoded_reference.bin
     * @returns {Promise<Uint32Array>} Reference RGBA values
     */
    async loadCPUReference(filePath = '../encoder/CSVEncoder/decoded_reference.bin') {
        try {
            const response = await fetch(filePath);
            if (!response.ok) {
                console.error(`Failed to load reference: ${response.status}`);
                return null;
            }
            const arrayBuffer = await response.arrayBuffer();
            
            // File now contains raw u64 labels (8 bytes each) from C++
            const labelData = new BigUint64Array(arrayBuffer);
            
            // Convert labels to RGBA using transfer function (same as GPU)
            this.cpuReference = new Uint32Array(labelData.length);
            for (let i = 0; i < labelData.length; i++) {
                this.cpuReference[i] = this.labelToRGBA(labelData[i]);
            }

            console.log(`✓ Loaded ${labelData.length} labels from CPU reference, converted to RGBA`);
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
        console.log(`✓ Set GPU output: ${this.gpuOutput.length} labels`);
    }

    /**
     * Compare CPU and GPU outputs
     * @returns {Object} Comparison results with statistics
     */
    compare() {
        if (!this.cpuReference) {
            console.error('CPU reference not loaded');
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

        let matches = 0;
        let mismatches = 0;
        const firstMismatches = [];
        const maxMismatchesToShow = 10;

        for (let i = 0; i < this.cpuReference.length; i++) {
            if (this.cpuReference[i] === this.gpuOutput[i]) {
                matches++;
            } else {
                mismatches++;
                if (firstMismatches.length < maxMismatchesToShow) {
                    firstMismatches.push({
                        index: i,
                        cpu: this.cpuReference[i],
                        gpu: this.gpuOutput[i]
                    });
                }
            }
        }

        const totalCount = this.cpuReference.length;
        const matchPercentage = ((matches / totalCount) * 100).toFixed(2);

        const result = {
            matches,
            mismatches,
            totalCount,
            matchPercentage,
            firstMismatches
        };

        return result;
    }

    /**
     * Print comparison results to console
     */
    printResults() {
        const result = this.compare();
        if (!result) return;

        console.log('\n' + '='.repeat(60));
        console.log('DECODER VALIDATION RESULTS');
        console.log('='.repeat(60));
        console.log(`✓ Matches:     ${result.matches} / ${result.totalCount}`);
        console.log(`✗ Mismatches:  ${result.mismatches}`);
        console.log(`Match Rate:    ${result.matchPercentage}%`);
        console.log('='.repeat(60));

        if (result.mismatches > 0) {
            console.log('\nFirst mismatches:');
            result.firstMismatches.forEach(m => {
                console.log(
                    `  [${m.index}] CPU: 0x${m.cpu.toString(16).padStart(8, '0')} ` +
                    `GPU: 0x${m.gpu.toString(16).padStart(8, '0')}`
                );
            });
        } else {
            console.log('\n✓✓✓ ALL LABELS MATCH! GPU decoder is correct! ✓✓✓');
        }
        console.log('='.repeat(60) + '\n');

        return result;
    }
}

export default DecoderValidator;
