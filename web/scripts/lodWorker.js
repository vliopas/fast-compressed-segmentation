// lodWorker.js - Worker thread for calculating brick LODs in parallel
import { calculateBrickLOD } from './camera.js';

// Worker message handler
self.onmessage = (event) => {
    const { bricks, camera, screenWidth, maxLOD, brickSize } = event.data;

    // Calculate LOD for each brick in this batch
    const lods = bricks.map((brick) => {
        // Brick position is already set from the main thread
        const halfSize = brickSize / 2;
        const brickCenter = {
            x: brick.position.x + halfSize,
            y: brick.position.y + halfSize,
            z: brick.position.z + halfSize
        };

        return calculateBrickLOD(brickCenter, brickSize, camera.position, camera.fov, screenWidth, maxLOD);
    });

    // Calculate voxel counts for each brick based on LOD
    const voxelCounts = lods.map((lod) => {
        return Math.pow(2, 3 * lod);  // 2^(3*LOD) voxels per brick
    });

    // Send results back
    self.postMessage({ lods, voxelCounts });
};
