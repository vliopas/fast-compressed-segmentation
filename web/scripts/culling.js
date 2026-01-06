/**
 * Visibility and Culling Functions
 * Handles frustum culling, transfer function culling, and camera movement detection
 */

export const CULLING_CONFIG = {
    MOVEMENT_THRESHOLD: 0.5,      // Position threshold (world units)
    ROTATION_THRESHOLD: 0.05      // Direction threshold (radians)
};

// Computed constants
const MOVEMENT_THRESHOLD_SQ = CULLING_CONFIG.MOVEMENT_THRESHOLD * CULLING_CONFIG.MOVEMENT_THRESHOLD;
const ROT_DOT_THRESHOLD = Math.cos(CULLING_CONFIG.ROTATION_THRESHOLD);

/**
 * Check if camera has moved significantly since last position
 * @param {Object} camera - Camera object with position and direction
 * @param {Object} lastPosition - Last recorded camera position
 * @param {Object} lastDirection - Last recorded camera direction
 * @returns {{hasMoved: boolean, position: Object, direction: Object}} Movement result
 */
export function checkCameraMovement(camera, lastPosition, lastDirection) {
    if (!camera) return { hasMoved: false, position: lastPosition, direction: lastDirection };

    if (!lastPosition || !lastDirection) {
        // First check - consider camera as moved
        return {
            hasMoved: true,
            position: { ...camera.position },
            direction: { ...camera.direction }
        };
    }

    // Check position change (squared distance to avoid sqrt)
    const dx = camera.position.x - lastPosition.x;
    const dy = camera.position.y - lastPosition.y;
    const dz = camera.position.z - lastPosition.z;
    const posChangeSq = dx * dx + dy * dy + dz * dz;

    // Check direction change using dot product vs cosine threshold
    const dot = (camera.direction.x * lastDirection.x) +
        (camera.direction.y * lastDirection.y) +
        (camera.direction.z * lastDirection.z);
    const hasRotated = dot < ROT_DOT_THRESHOLD;
    const hasMoved = posChangeSq > MOVEMENT_THRESHOLD_SQ || hasRotated;

    if (hasMoved) {
        return {
            hasMoved: true,
            position: { ...camera.position },
            direction: { ...camera.direction }
        };
    }

    return { hasMoved: false, position: lastPosition, direction: lastDirection };
}

/**
 * Check if a brick is visible based on frustum and transfer function
 * @param {Object} brick - Brick object with position and palette
 * @param {number} brickSize - Size of brick
 * @param {Object} camera - Camera object
 * @param {Set} hiddenLabels - Set of hidden label IDs
 * @returns {boolean} True if brick is visible
 */
export function isBrickVisible(brick, brickSize, camera, hiddenLabels) {
    // Frustum culling: check if brick intersects camera frustum
    if (!isBrickInFrustum(brick, brickSize, camera)) {
        return false;
    }

    // Transfer function culling: check if brick has any visible voxels
    if (!brickHasVisibleVoxels(brick, hiddenLabels)) {
        return false;
    }

    return true;
}

/**
 * Frustum culling: Check if brick AABB intersects camera frustum
 * @param {Object} brick - Brick object with position
 * @param {number} brickSize - Size of brick
 * @param {Object} camera - Camera object with position, direction, and fov
 * @returns {boolean} True if brick is in frustum
 */
export function isBrickInFrustum(brick, brickSize, camera) {
    if (!camera) return true; // No camera = assume visible

    // Brick AABB (axis-aligned bounding box)
    const brickMin = brick.position;
    const brickMax = {
        x: brick.position.x + brickSize,
        y: brick.position.y + brickSize,
        z: brick.position.z + brickSize
    };

    // Brick center
    const center = {
        x: (brickMin.x + brickMax.x) / 2,
        y: (brickMin.y + brickMax.y) / 2,
        z: (brickMin.z + brickMax.z) / 2
    };

    // Vector from camera to brick center
    const toCenter = {
        x: center.x - camera.position.x,
        y: center.y - camera.position.y,
        z: center.z - camera.position.z
    };

    // Distance to brick center
    const distance = Math.sqrt(toCenter.x * toCenter.x + toCenter.y * toCenter.y + toCenter.z * toCenter.z);

    // Normalize direction to brick
    const dirToBrick = {
        x: toCenter.x / distance,
        y: toCenter.y / distance,
        z: toCenter.z / distance
    };

    // Dot product with camera direction (how aligned is brick with view direction)
    const dot = dirToBrick.x * camera.direction.x +
        dirToBrick.y * camera.direction.y +
        dirToBrick.z * camera.direction.z;

    // Brick diagonal radius (maximum distance from center to corner)
    const brickRadius = Math.sqrt(3) * (brickSize / 2);

    // FOV-based culling: brick must be within FOV cone + radius margin
    // cos(fov/2) gives the minimum dot product for objects in view
    const cosHalfFov = Math.cos(camera.fov / 2);

    // Angular margin for brick visibility
    // Use atan2 for accurate angular radius even at close distances
    // This ensures bricks slightly outside the cone are still included
    const angularRadius = Math.atan2(brickRadius, distance);

    // Convert angular radius to a dot product threshold margin
    // For small angles, sin(angle) ≈ margin in dot product space
    const margin = Math.sin(angularRadius);

    if (dot < cosHalfFov - margin) {
        return false; // Outside FOV
    }

    // Behind camera check (with margin for partially visible bricks)
    if (distance < -brickRadius) {
        return false;
    }

    return true;
}

/**
 * Transfer function culling: Check if brick palette has visible voxels
 * Empty label is 0 (background from MICrONS dataset)
 * @param {Object} brick - Brick object with palette array
 * @param {Set} hiddenLabels - Set of hidden label IDs
 * @returns {boolean} True if brick has at least one visible voxel
 */
export function brickHasVisibleVoxels(brick, hiddenLabels) {
    // Check if ANY palette entry is non-zero (non-empty) and not hidden
    for (const paletteValue of brick.palette) {
        if (paletteValue !== 0n && !(hiddenLabels && hiddenLabels.has(paletteValue))) {
            return true; // Has at least one non-empty, visible voxel
        }
    }

    return false; // All palette entries are 0 (empty) or hidden
}
