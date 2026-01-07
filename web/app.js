/**
 * @file app.js
 * @brief Main application entry point for WebGPU volume renderer
 * 
 * Initializes the dataset loader, WebGPU renderer, and user interface.
 * Handles dataset loading, decoding, and rendering setup.
 */

import { loadDatasetFromArrayBuffer } from "./scripts/datasetLoader.js";
import { decodeAllRansBricks } from './scripts/decode.js';
import { initWebGPU, updateLightingOptions, applyLabelVisibility, applyLabelHover, getDatasetLabels } from './scripts/webgpuSetup.js';
import { initRenderLoop, forceFullBrickRefresh, reDecodeCachedBricks } from './scripts/renderLoop.js';

let currentLightAngle = 25;

// ==================== Logger Setup ====================
import { createLogger, setLogConfig } from './scripts/logger.js';
import DecoderValidator from './scripts/decoderValidation.js';
const logger = createLogger('App');
setLogConfig({ namespaces: [], level: 'off', persist: true });
DecoderValidator.setLoggingEnabled(false);

logger.log('Logger initialized');

// ==================== Morton Code Utilities ====================

/**
 * Decode Morton code to 3D coordinates (for debugging)
 * @param {number} code - Morton-encoded index
 * @return {{x: number, y: number, z: number}} 3D coordinates
 */
function decodeMorton3D(code) {
    const compact1by2 = (n) => {
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

// ==================== Dataset Loading ====================

/**
 * Load the compressed dataset from the server
 * @async
 * @return {Promise<Object>} Loaded dataset object
 * @throws {Error} If dataset fails to load
 */
async function loadFixedDataset() {
    const response = await fetch("../encoder/CSVEncoder/compressed_dataset.csbd");

    if (!response.ok) {
        throw new Error("Failed to load dataset.bin");
    }

    const arrayBuffer = await response.arrayBuffer();
    const loadedDataset = loadDatasetFromArrayBuffer(arrayBuffer);

    window.dataset = loadedDataset;
    return loadedDataset;
}

/**
 * Decode all brick data using rANS decoding
 * @async
 * @param {Object} loadedDataset - Dataset with encoded bricks
 * @return {Promise<Object>} Dataset with decoded bricks
 */
async function decodeBricksData(loadedDataset) {
    const decodedBricks = await decodeAllRansBricks(loadedDataset);
    decodedBricks.forEach((decoded, i) => {
        loadedDataset.bricks[i].encodedData = decoded;
        loadedDataset.bricks[i].encodedSize = decoded.length;
    });
    return loadedDataset;
}

// ==================== Application Initialization ====================

/**
 * Initialize the application
 * @async
 */
async function initApp() {
    try {
        // Load and decode dataset
        let dataset = await loadFixedDataset();
        dataset = await decodeBricksData(dataset);

        // Initialize WebGPU
        const gpuState = await initWebGPU(dataset);

        // Start render loop
        initRenderLoop(gpuState, dataset);

        setupLightingUI();
        setupLabelVisibilityUI(dataset);

    } catch (err) {
        console.error("Error during initialization:", err);
    }
}

// Start application
initApp();

// =============== UI Wiring ===============

/**
 * Setup lighting UI controls and apply defaults
 */
function setupLightingUI() {
    // Use default lighting values without UI controls
    currentLightAngle = 25;

    updateLightingOptions({
        lightAngle: 25,
        diffuseStrength: 0.8,
        ambient: 0.5
    });
}

/**
 * Setup label visibility UI controls
 * @param {Object} dataset - Dataset containing labels
 */
function setupLabelVisibilityUI(dataset) {
    const listEl = document.getElementById('label-visibility-list');
    const applyBtn = document.getElementById('label-visibility-apply');
    const statusEl = document.getElementById('label-visibility-status');

    if (!listEl || !applyBtn || !statusEl) return;

    const labels = getDatasetLabels();
    if (!labels || labels.length === 0) {
        statusEl.textContent = 'No labels detected';
        statusEl.classList.add('error');
        applyBtn.disabled = true;
        return;
    }

    const hiddenLabels = dataset.hiddenLabels || new Set();
    dataset.hiddenLabels = hiddenLabels;


    listEl.innerHTML = '';
    labels.forEach((label) => {
        const item = document.createElement('div');
        item.className = 'label-item';

        const chip = document.createElement('span');
        chip.className = 'label-chip';
        chip.textContent = `Label ${label.toString()}`;
        chip.dataset.label = label.toString();
        if (hiddenLabels.has(label)) {
            chip.classList.add('hidden');
        }

        // Toggle visibility UI on click (crosses out the label)
        chip.addEventListener('click', () => {
            if (chip.classList.contains('hidden')) {
                chip.classList.remove('hidden');
            } else {
                chip.classList.add('hidden');
            }
        });

        // Highlight on hover when currently visible (not hidden in UI)
        item.addEventListener('mouseenter', () => {
            if (!chip.classList.contains('hidden')) {
                applyLabelHover(label);
                reDecodeCachedBricks();
            }
        });
        item.addEventListener('mouseleave', () => {
            applyLabelHover(null);
            reDecodeCachedBricks();
        });

        item.appendChild(chip);
        listEl.appendChild(item);
    });


    // Gather hidden labels from chips with .hidden class
    const gatherHidden = () => {
        const set = new Set();
        const chips = listEl.querySelectorAll('.label-chip');
        chips.forEach((chip) => {
            const labelVal = chip.dataset.label;
            if (!labelVal) return;
            if (chip.classList.contains('hidden')) {
                set.add(BigInt(labelVal));
            }
        });
        return set;
    };

    applyBtn.addEventListener('click', () => {
        const nextHidden = gatherHidden();
        dataset.hiddenLabels = nextHidden;

        const result = applyLabelVisibility(Array.from(nextHidden));
        if (result?.applied) {
            statusEl.textContent = `Applied: ${result.hiddenCount}/${result.total} hidden`;
            statusEl.classList.remove('error');
            forceFullBrickRefresh();
            applyLabelHover(null);
        } else {
            statusEl.textContent = result?.reason || 'Failed to apply';
            statusEl.classList.add('error');
        }
    });

    // Clear highlight if mouse leaves the list entirely
    listEl.addEventListener('mouseleave', () => applyLabelHover(null));

    statusEl.textContent = `Loaded ${labels.length} labels`;
}

// compass removed
