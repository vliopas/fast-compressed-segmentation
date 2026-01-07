/**
 * @file camera.js
 * @brief Camera class and LOD calculation system
 * 
 * Provides camera management and brick level-of-detail (LOD) calculation
 * based on distance and screen projection.
 */

/**
 * @class Camera
 * @brief Camera system for the volume renderer
 */
export class Camera {
    /**
     * Create a new camera
     * @param {{x: number, y: number, z: number}} position - Camera position
     * @param {number} fov - Field of view in radians
     */
    constructor(position = { x: 32, y: 32, z: -50 }, fov = Math.PI / 3) {
        this.position = position;
        this.fov = fov;  // Field of view in radians
        this.direction = { x: 0, y: 0, z: 1 };  // Looking in +Z direction
        this._needsUpload = true;  // Mark for initial upload
    }

    /**
     * Update camera position and optionally direction
     * @param {{x: number, y: number, z: number}} position - New position
     * @param {{x: number, y: number, z: number}|null} direction - New direction (optional)
     */
    update(position, direction = null) {
        this.position = position;
        if (direction) {
            this.direction = direction;
        }
        this._needsUpload = true;
    }

    /**
     * Point camera at a target position
     * @param {{x: number, y: number, z: number}} target - Target position
     */
    lookAt(target) {
        const dir = normalize({
            x: target.x - this.position.x,
            y: target.y - this.position.y,
            z: target.z - this.position.z
        });
        this.direction = dir;
        this._needsUpload = true;
    }

    needsUpload() {
        return this._needsUpload;
    }

    markUploaded() {
        this._needsUpload = false;
    }

    getUniformData() {
        // Build an orthonormal basis on CPU to avoid per-pixel work in the shader
        const forward = normalize(this.direction);
        // Pick an up vector that is not parallel to forward
        const worldUp = Math.abs(forward.y) > 0.999 ? { x: 1, y: 0, z: 0 } : { x: 0, y: 1, z: 0 };
        const right = normalize(cross(forward, worldUp));
        const up = normalize(cross(right, forward));
        const tanHalfFov = Math.tan(this.fov * 0.5);

        // Std140-like packing: four vec4s = 64 bytes
        return new Float32Array([
            this.position.x, this.position.y, this.position.z, tanHalfFov,
            forward.x, forward.y, forward.z, 0,
            right.x, right.y, right.z, 0,
            up.x, up.y, up.z, 0
        ]);
    }
}

function normalize(v) {
    const len = Math.hypot(v.x, v.y, v.z) || 1;
    return { x: v.x / len, y: v.y / len, z: v.z / len };
}

function cross(a, b) {
    return {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x
    };
}

/**
 * Calculate which LOD a brick should be decoded at based on distance and screen projection
 * @param {{x: number, y: number, z: number}} brickCenter - Brick center position
 * @param {number} brickSize - Size of the brick
 * @param {{x: number, y: number, z: number}} cameraPos - Camera position
 * @param {number} fov - Field of view in radians
 * @param {number} screenWidth - Screen width in pixels
 * @param {number} maxLOD - Maximum LOD level
 * @return {number} LOD level (0 = finest, maxLOD = coarsest)
 */
export function calculateBrickLOD(brickCenter, brickSize, cameraPos, fov, screenWidth, maxLOD) {
    // Distance from camera to brick center
    const dx = brickCenter.x - cameraPos.x;
    const dy = brickCenter.y - cameraPos.y;
    const dz = brickCenter.z - cameraPos.z;
    const distToBrick = Math.sqrt(dx * dx + dy * dy + dz * dz);
    // Clamp distance to avoid division by zero and extreme close-ups
    const d = Math.max(distToBrick, 0.1);

    // Focal length based on FOV and screen width
    // focal_length = (screen_width / 2) / tan(fov / 2)
    const focalLength = (screenWidth / 2) / Math.tan(fov / 2);

    // Find the finest LOD where voxels still project to >= 1 pixel
    // We want voxels as small as possible while still being >= 1 pixel on screen
    // LOD 0 = coarsest (1 voxel total), LOD k = 2^(3*k) total voxels
    let bestLOD = 0;

    for (let lod = 0; lod <= maxLOD; lod++) {
        // At LOD k, brick is divided into 2^k voxels per dimension
        // So each voxel has physical size = brickSize / 2^k
        const voxelSize = brickSize / Math.pow(2, lod);
        const screenPixels = (focalLength * voxelSize) / d;

        // If voxel projects to >= 1 pixel, this LOD is valid
        if (screenPixels >= 1.0) {
            bestLOD = lod;  // Track this as a valid LOD
        } else {
            // voxel is < 1 pixel, so previous LOD was best fit
            break;
        }
    }

    return bestLOD;
}

// Calculate LODs for all bricks using Web Workers for parallelism
export async function calculateAllBrickLODs(bricks, camera, screenWidth, maxLOD, brickSize) {
    // Use Web Workers for parallel LOD calculation
    const numWorkers = navigator.hardwareConcurrency || 4;
    const workers = [];

    // Create worker pool with module support
    for (let i = 0; i < numWorkers; i++) {
        workers.push(new Worker('scripts/lodWorker.js', { type: 'module' }));
    }
    // Split bricks into batches
    const batchSize = Math.ceil(bricks.length / numWorkers);
    const promises = workers.map((worker, workerIdx) => {
        return new Promise((resolve) => {
            const startIdx = workerIdx * batchSize;
            const endIdx = Math.min(startIdx + batchSize, bricks.length);

            // Extract only serializable brick data (position) to avoid BigInt serialization errors
            const batch = bricks.slice(startIdx, endIdx).map(brick => ({
                position: brick.position
            }));

            // Handle worker response
            worker.onmessage = (event) => {
                const results = event.data.lods.map((lod, idx) => ({
                    brickIdx: startIdx + idx,
                    lod: lod,
                    voxelCount: event.data.voxelCounts[idx]
                }));
                worker.terminate();
                resolve(results);
            };

            // Send work to worker
            worker.postMessage({
                bricks: batch,
                camera: camera,
                screenWidth: screenWidth,
                maxLOD: maxLOD,
                brickSize: brickSize
            });
        });
    });

    // Wait for all workers to complete
    const allResults = await Promise.all(promises);

    // Flatten and sort by brick index to maintain order
    const flatResults = allResults.flat().sort((a, b) => a.brickIdx - b.brickIdx);

    // Build LOD array and compute cumulative offsets
    const lodArray = new Uint32Array(bricks.length);
    const offsetArray = new Uint32Array(bricks.length);

    let cumulativeOffset = 0;
    for (const result of flatResults) {
        lodArray[result.brickIdx] = result.lod;
        offsetArray[result.brickIdx] = cumulativeOffset;
        cumulativeOffset += result.voxelCount;
    }

    return { lodArray, offsetArray };
}
