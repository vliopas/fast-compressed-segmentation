/**
 * GPU Render Passes
 * Handles all WebGPU compute and render passes
 */

export const GPU_PASS_CONFIG = {
    THREADS_PER_GROUP: 8  // 8x8 threads per group for ray tracing
};

/**
 * Run compute pass to decode bricks
 * @param {GPUCommandEncoder} encoder - GPU command encoder
 * @param {Object} gpuState - GPU state object
 */
export function runComputePass(encoder, gpuState) {
    // Skip if no bricks need decompression
    const workCount = gpuState.workCount >>> 0;
    if (!workCount) { return; }

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(gpuState.computePipeline);
    computePass.setBindGroup(0, gpuState.computeBindGroup);
    // Dispatch one workgroup per brick (workgroup cooperatively fills sub-blocks)
    computePass.dispatchWorkgroups(workCount);
    computePass.end();

    // After dispatching, clear CPU-side counter so we don't re-dispatch the same work next frame.
    // Leave GPU buffer untouched; it will be overwritten next time bricks are queued.
    gpuState.workCount = 0;
}

/**
 * Run ray tracing pass
 * @param {GPUCommandEncoder} encoder - GPU command encoder
 * @param {Object} gpuState - GPU state object
 */
export function runRayTracePass(encoder, gpuState) {
    if (!gpuState.rayTracePipeline) return; // Skip if not initialized

    const rayTracePass = encoder.beginComputePass();
    rayTracePass.setPipeline(gpuState.rayTracePipeline);
    rayTracePass.setBindGroup(0, gpuState.rayTraceBindGroup);

    // Dispatch with workgroups based on output resolution
    const threadsPerGroup = GPU_PASS_CONFIG.THREADS_PER_GROUP;
    const groupsX = Math.ceil(gpuState.rayTraceOutputSize.width / threadsPerGroup) >>> 0;
    const groupsY = Math.ceil(gpuState.rayTraceOutputSize.height / threadsPerGroup) >>> 0;

    rayTracePass.dispatchWorkgroups(groupsX, groupsY);
    rayTracePass.end();
}

/**
 * Run clear ray output pass to zero out the ray trace output buffer
 * This prevents stale pixel data from previous frames
 * @param {GPUCommandEncoder} encoder - GPU command encoder
 * @param {Object} gpuState - GPU state object
 */
export function runClearRayOutputPass(encoder, gpuState) {
    if (!gpuState.clearRayOutputPipeline) {
        console.error('Clear ray output pipeline not available - OUTPUT WILL HAVE ARTIFACTS!');
        return; // Skip if not initialized
    }

    const clearPass = encoder.beginComputePass();
    clearPass.setPipeline(gpuState.clearRayOutputPipeline);
    clearPass.setBindGroup(0, gpuState.clearRayOutputBindGroup);

    // Dispatch with same workgroup layout as ray tracing
    const threadsPerGroup = GPU_PASS_CONFIG.THREADS_PER_GROUP;
    const groupsX = Math.ceil(gpuState.rayTraceOutputSize.width / threadsPerGroup) >>> 0;
    const groupsY = Math.ceil(gpuState.rayTraceOutputSize.height / threadsPerGroup) >>> 0;

    clearPass.dispatchWorkgroups(groupsX, groupsY);
    clearPass.end();
}

/**
 * Run display pass to render to canvas
 * @param {GPUCommandEncoder} encoder - GPU command encoder
 * @param {Object} gpuState - GPU state object
 */
export function runDisplayPass(encoder, gpuState) {
    if (!gpuState.displayPipeline || !gpuState.context) return; // Skip if not initialized

    const renderPass = encoder.beginRenderPass({
        colorAttachments: [{
            view: gpuState.context.getCurrentTexture().createView(),
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
        }],
    });

    renderPass.setPipeline(gpuState.displayPipeline);
    renderPass.setBindGroup(0, gpuState.displayBindGroup);
    renderPass.draw(3); // Draw fullscreen triangle
    renderPass.end();
}
