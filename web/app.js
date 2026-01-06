import { loadDatasetFromArrayBuffer } from "./scripts/datasetLoader.js";
import { decodeAllRansBricks } from './scripts/decode.js';
import { initWebGPU, updateLightingOptions } from './scripts/webgpuSetup.js';
import { initRenderLoop, requestValidationOnce } from './scripts/renderLoop.js';

let currentLightAngle = 25;

// ==================== Dataset Loading ====================

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

async function decodeBricksData(loadedDataset) {
    const decodedBricks = await decodeAllRansBricks(loadedDataset);
    decodedBricks.forEach((decoded, i) => {
        loadedDataset.bricks[i].encodedData = decoded;
        loadedDataset.bricks[i].encodedSize = decoded.length;
    });
    return loadedDataset;
}

// ==================== Application Initialization ====================

async function initApp() {
    try {
        // Load and decode dataset
        let dataset = await loadFixedDataset();
        dataset = await decodeBricksData(dataset);
        
        console.log("Number of bricks:", dataset.bricks.length);
        console.log("Dataset loaded and decoded:", dataset);

        // Initialize WebGPU
        const gpuState = await initWebGPU(dataset);

        // Start render loop
        initRenderLoop(gpuState, dataset);

        setupLightingUI();
        setupCompass(gpuState);

        // Trigger a one-time GPU vs CPU validation readback
        requestValidationOnce();

    } catch (err) {
        console.error("Error during initialization:", err);
    }
}

// Start application
initApp();

// =============== UI Wiring ===============

function setupLightingUI() {
    const lightAngleSlider = document.getElementById('light-angle-slider');
    const gradientToggle = document.getElementById('gradient-toggle');
    const diffuseSlider = document.getElementById('diffuse-slider');
    const ambientSlider = document.getElementById('ambient-slider');
    const aoStrengthSlider = document.getElementById('ao-strength-slider');
    const shadowThresholdSlider = document.getElementById('shadow-threshold-slider');
    const aoBlendSlider = document.getElementById('ao-blend-slider');

    if (!lightAngleSlider || !gradientToggle || !diffuseSlider || !ambientSlider || !aoStrengthSlider || !shadowThresholdSlider || !aoBlendSlider) return;

    const lightAngleValue = document.getElementById('light-angle-value');
    const diffuseValue = document.getElementById('diffuse-value');
    const ambientValue = document.getElementById('ambient-value');
    const aoStrengthValue = document.getElementById('ao-strength-value');
    const shadowThresholdValue = document.getElementById('shadow-threshold-value');
    const aoBlendValue = document.getElementById('ao-blend-value');

    const apply = () => {
        lightAngleValue.textContent = `${Math.round(lightAngleSlider.value)}°`;
        diffuseValue.textContent = Number(diffuseSlider.value).toFixed(2);
        ambientValue.textContent = Number(ambientSlider.value).toFixed(2);
        aoStrengthValue.textContent = Number(aoStrengthSlider.value).toFixed(2);
        shadowThresholdValue.textContent = Number(shadowThresholdSlider.value).toFixed(2);
        aoBlendValue.textContent = Number(aoBlendSlider.value).toFixed(2);

        currentLightAngle = parseInt(lightAngleSlider.value, 10);

        updateLightingOptions({
            lightAngle: currentLightAngle,
            gradientShadingEnabled: gradientToggle.checked,
            diffuseStrength: parseFloat(diffuseSlider.value),
            ambient: parseFloat(ambientSlider.value),
            aoStrength: parseFloat(aoStrengthSlider.value),
            shadowAlphaThreshold: parseFloat(shadowThresholdSlider.value),
            aoBlend: parseFloat(aoBlendSlider.value)
        });
    };

    [lightAngleSlider, gradientToggle, diffuseSlider, ambientSlider, aoStrengthSlider, shadowThresholdSlider, aoBlendSlider].forEach(control => {
        control.addEventListener('input', apply);
        if (control.type === 'checkbox') {
            control.addEventListener('change', apply);
        }
    });

    apply();
}

function headingFromVector(x, z) {
    const deg = Math.atan2(x, -z) * 180 / Math.PI;
    return (deg + 360) % 360;
}

function headingLabel(deg) {
    const labels = ['North', 'North-East', 'East', 'South-East', 'South', 'South-West', 'West', 'North-West'];
    const idx = Math.round(deg / 45) % 8;
    return labels[idx];
}

function setupCompass(gpuState) {
    const compass = document.getElementById('compass');
    const camNeedle = document.getElementById('compass-camera-needle');
    const lightNeedle = document.getElementById('compass-light-needle');
    const camText = document.getElementById('compass-camera-text');
    const lightText = document.getElementById('compass-light-text');

    if (!compass || !camNeedle || !lightNeedle || !camText || !lightText) return;

    const tick = () => {
        // Camera heading from controller (yaw) or direction vector fallback
        const yaw = gpuState?.cameraController?.yaw;
        if (Number.isFinite(yaw)) {
            const headingDeg = headingFromVector(-Math.sin(yaw), -Math.cos(yaw));
            camNeedle.style.transform = `translate(-50%, -50%) rotate(${headingDeg}deg)`;
            camText.textContent = headingLabel(headingDeg);
        } else if (gpuState?.camera?.direction) {
            const dir = gpuState.camera.direction;
            const headingDeg = headingFromVector(dir.x, dir.z);
            camNeedle.style.transform = `translate(-50%, -50%) rotate(${headingDeg}deg)`;
            camText.textContent = headingLabel(headingDeg);
        }

        // Light heading from current slider angle
        const angleRad = (currentLightAngle * Math.PI) / 180;
        const lx = Math.sin(angleRad);
        const lz = -Math.cos(angleRad);
        const lightDeg = headingFromVector(lx, lz);
        lightNeedle.style.transform = `translate(-50%, -50%) rotate(${lightDeg}deg)`;
        lightText.textContent = `${headingLabel(lightDeg)} (${Math.round(currentLightAngle)}°)`;

        requestAnimationFrame(tick);
    };

    tick();
}
