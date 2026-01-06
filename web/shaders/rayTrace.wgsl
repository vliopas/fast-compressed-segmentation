// Ray tracing with transfer function, empty-space skipping, and LOD selection

// Gradient shading and AO always enabled

// f16 disabled for compatibility with GPUs that do not support it

struct DynamicBrickInfo {
    outputOffset: u32,        // Offset into outVoxels buffer
    targetLOD: u32,           // Resolution level (voxels per dimension)
    lodScale: f32,            // Precomputed: lodSize / brickSize (for fast coordinate scaling)
    pad: u32,                 // Padding so array stride is 16 bytes
}

struct StaticBrickInfo {
    nSymbols: u32,
    paletteSize: u32,
    streamOffset: u32,
    paletteOffset: u32,
    flags: u32,               // bit 0 = isEmpty (1 = empty, 0 = non-empty)
    encodedSizeBytes: u32,    // Encoded byte length (kept for struct parity)
    mortonId: u32,            // CPU-provided Morton code for validation
    pad2: u32,
}

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
    invDir: vec3<f32>,  // Precomputed: 1.0 / dir (with epsilon guard)
}

struct VolumeHit {
    hit: bool,
    tEnter: f32,
    tExit: f32,
}

struct DDAState {
    voxel: vec3<i32>,
    tMax: vec3<f32>,
    tDelta: vec3<f32>,
    step: vec3<i32>,
    tExit: f32,
}

struct BrickDDAState {
    brick: vec3<i32>,
    tMax: vec3<f32>,
    tDelta: vec3<f32>,
    step: vec3<i32>,
    tExit: f32,
}

@group(0) @binding(0)
var<storage, read> outVoxels: array<u32>;  // Decoded voxels as packed RGBA8 from decode shader

@group(0) @binding(1)
var<storage, read_write> rayTraceOutput: array<u32>;

@group(0) @binding(2)
var<storage, read> brickData: array<DynamicBrickInfo>;  // Brick metadata with LOD info

@group(0) @binding(3)
var<storage, read> staticBricks: array<StaticBrickInfo>; // Palette info

@group(0) @binding(4)
var<uniform> scene: SceneInfo;

@group(0) @binding(5)
var<uniform> camera: Camera;

@group(0) @binding(6)
var<uniform> screen: Screen;

@group(0) @binding(7)
var<storage, read_write> brickRequests: array<atomic<u32>>;  // GPU marks accessed bricks (1 = accessed, 0 = not accessed)

@group(0) @binding(8)
var<storage, read> paletteBuffer: array<vec2<u32>>;  // Palette for LOD 0 bricks

@group(0) @binding(9)
var<uniform> lighting: Lighting;  // Lighting and shadow/AO parameters

struct SceneInfo {
    gridSize: f32,       // Total volume extent in voxels along one axis (e.g., 1024 for a 1024^3 volume)
    brickSize: f32,      // Full-resolution brick edge length in voxels (e.g., 64 for a 64^3 brick)
    bricksPerAxis: f32,  // Number of bricks per axis in the brick grid (gridSize / brickSize)
    pad0: f32,
    invGridSize: f32,    // 1.0 / gridSize (for slab tests)
    invBrickSize: f32,   // 1.0 / brickSize (for voxel scaling)
    pad1: f32,
    pad2: f32,
}

// Camera inputs are pre-baked on CPU to avoid per-pixel basis math
struct Camera {
    position: vec3<f32>,
    tanHalfFov: f32,      // precomputed tan(fov/2)
    forward: vec3<f32>,   // normalized camera forward
    pad0: f32,
    right: vec3<f32>,     // normalized camera right
    pad1: f32,
    up: vec3<f32>,        // normalized camera up
    pad2: f32,
}

struct Screen {
    width: f32,
    height: f32,
    aspect: f32,
    padding: f32,
}

// Lighting controls
// params = (ambient, unused1, unused2, unused3)
// params2 = (unused, diffuseStrength, pad, pad)
struct Lighting {
    lightDir: vec3<f32>,
    pad0: f32,
    params: vec4<f32>,
    params2: vec4<f32>,
}

const FAR_T: f32 = 1e9;               // Fallback for rays parallel to an axis
const NOT_REQUESTED: u32 = 0u;        // Brick not accessed
const BRICK_ACCESSED: u32 = 1u;       // Brick was accessed by ray

// ==================== Morton Code Utilities ====================

/// Encodes 3D coordinates into a Morton code (Z-order curve).
/// Interleaves bits to preserve spatial locality (supports up to 1024 per axis).
fn morton3D(x: u32, y: u32, z: u32) -> u32 {
    // Interleave bits of x, y, z coordinates into Morton code
    var answer: u32 = 0u;
    for (var i: u32 = 0u; i < 10u; i++) {
        answer |= ((x >> i) & 1u) << (3u * i);
        answer |= ((y >> i) & 1u) << (3u * i + 1u);
        answer |= ((z >> i) & 1u) << (3u * i + 2u);
    }
    return answer;
}

// Calculate appropriate LOD based on distance to brick center
// Closer bricks get higher detail (higher LOD), distant bricks get lower detail
fn calculateLOD(brickPos: vec3<i32>, rayOrigin: vec3<f32>, maxLOD: u32) -> u32 {
    let brickSize = scene.brickSize;
    let brickCenter = (vec3<f32>(brickPos) + vec3<f32>(0.5)) * brickSize;
    let distance = length(brickCenter - rayOrigin);
    
    // Distance-based LOD: closer = higher detail
    // LOD formula: maxLOD - log2(distance / brickSize)
    // Clamp to [0, maxLOD] range
    let lodF = f32(maxLOD) - log2(max(distance / brickSize, 1.0));
    return clamp(u32(lodF), 0u, maxLOD);
}

// Get voxel color from the outVoxels buffer using brick info and local coordinates
fn getVoxelColor(brick: DynamicBrickInfo, staticInfo: StaticBrickInfo, localCoords: vec3<u32>) -> vec4<f32> {
    // LOD system: LOD 0 = coarsest (1 voxel), LOD k = 2^k voxels per dimension
    // Example: LOD 0 = 1 voxel, LOD 6 = 64 voxels (for brickSize=64)
    // lodScale = 2^k / brickSize
    // Example: LOD 0: lodScale = 1/64, LOD 6: lodScale = 64/64 = 1.0
    
    // Special case: LOD 0 bricks (1 voxel) read directly from palette (first entry)
    if (brick.targetLOD == 0u) {
        let paletteIdx = staticInfo.paletteOffset;
        let label = paletteBuffer[paletteIdx];
        // Convert u64 label (stored as vec2<u32>) to u32 RGBA via transfer function
        // For now, just use a simple color based on label value
        let labelLow = label.x;
        let r = f32((labelLow >> 0u) & 0xFFu) / 255.0;
        let g = f32((labelLow >> 8u) & 0xFFu) / 255.0;
        let b = f32((labelLow >> 16u) & 0xFFu) / 255.0;
        let a = f32((labelLow >> 24u) & 0xFFu) / 255.0;
        return vec4<f32>(r, g, b, a);
    }
    
    let lodSize = 1u << brick.targetLOD;  // 2^targetLOD
    
    // Scale from brick-local coordinates (0..brickSize-1) to LOD coordinates (0..lodSize-1)
    let scaled = vec3<f32>(localCoords) * brick.lodScale;
    let coords = vec3<u32>(clamp(scaled, vec3<f32>(0.0), vec3<f32>(f32(lodSize) - 1.0)));
    
    // Use Morton ordering to find index within this brick's output range
    let localIndex = morton3D(coords.x, coords.y, coords.z);
    let globalIndex = brick.outputOffset + localIndex;
    
    // Bounds check - return magenta for out-of-bounds (debug aid)
    if (globalIndex >= arrayLength(&outVoxels)) {
        return vec4<f32>(1.0, 0.0, 1.0, 1.0); // Bright magenta = bounds error
    }
    
    let rgba = outVoxels[globalIndex];
    
    // Unpack RGBA from packed u32
    let r = f32((rgba >> 0u) & 0xFFu) / 255.0;
    let g = f32((rgba >> 8u) & 0xFFu) / 255.0;
    let b = f32((rgba >> 16u) & 0xFFu) / 255.0;
    let a = f32((rgba >> 24u) & 0xFFu) / 255.0;
    
    return vec4<f32>(r, g, b, a);
}

/// Trilinear LOD blending: smoothly interpolates between LOD levels
/// Reduces popping at brick boundaries and cache thrashing
fn getVoxelColorTrilinearLOD(brick: DynamicBrickInfo, localCoords: vec3<f32>) -> vec4<f32> {
    let lodSize = 1u << brick.targetLOD;
    
    // Scale from brick-local coordinates to LOD coordinates
    let lodCoordsF = localCoords * brick.lodScale;
    
    // Convert float coords to integer for LOD sampling
    let coords0 = vec3<u32>(lodCoordsF);
    let frac = lodCoordsF - vec3<f32>(coords0);
    
    // Clamp fractional part for boundary safety
    let fracClamped = clamp(frac, vec3<f32>(0.0), vec3<f32>(1.0));
    
    // Sample 8 corners of the voxel cell at current LOD
    var color = vec4<f32>(0.0);
    var weight_sum = 0.0;
    
    for (var dz: u32 = 0u; dz <= 1u; dz++) {
        for (var dy: u32 = 0u; dy <= 1u; dy++) {
            for (var dx: u32 = 0u; dx <= 1u; dx++) {
                let sampleCoord = vec3<u32>(
                    min(coords0.x + dx, lodSize - 1u),
                    min(coords0.y + dy, lodSize - 1u),
                    min(coords0.z + dz, lodSize - 1u)
                );
                
                // Compute trilinear weight
                let w = (
                    mix(1.0 - fracClamped.x, fracClamped.x, f32(dx)) *
                    mix(1.0 - fracClamped.y, fracClamped.y, f32(dy)) *
                    mix(1.0 - fracClamped.z, fracClamped.z, f32(dz))
                );
                
                // Sample voxel at this corner
                let localIndex = morton3D(sampleCoord.x, sampleCoord.y, sampleCoord.z);
                let globalIndex = brick.outputOffset + localIndex;
                
                if (globalIndex < arrayLength(&outVoxels)) {
                    let rgba = outVoxels[globalIndex];
                    let r = f32((rgba >> 0u) & 0xFFu) / 255.0;
                    let g = f32((rgba >> 8u) & 0xFFu) / 255.0;
                    let b = f32((rgba >> 16u) & 0xFFu) / 255.0;
                    let a = f32((rgba >> 24u) & 0xFFu) / 255.0;
                    
                    color += w * vec4<f32>(r, g, b, a);
                    weight_sum += w;
                }
            }
        }
    }
    
    if (weight_sum > 0.0) {
        return color / weight_sum;
    }
    return vec4<f32>(0.0);
}

// Sample alpha at a world-space voxel coordinate by locating its brick and local coords
fn sampleAlpha(worldVoxel: vec3<i32>) -> f32 {
    // Bounds check against volume extent
    if (any(worldVoxel < vec3<i32>(0)) || any(worldVoxel >= vec3<i32>(i32(scene.gridSize)))) {
        return 0.0;
    }

    let brickSizeI = i32(scene.brickSize);
    let brickCoords = worldVoxel / brickSizeI;
    let numBricks = i32(scene.bricksPerAxis);
    if (any(brickCoords < vec3<i32>(0)) || any(brickCoords >= vec3<i32>(numBricks))) {
        return 0.0;
    }

    // Compute Morton code for this brick position
    let mortonCode = morton3D(u32(brickCoords.x), u32(brickCoords.y), u32(brickCoords.z));

    // Use Morton code directly as brick index (bricks stored in Morton order)
    let brickIdx = mortonCode;
    if (brickIdx >= arrayLength(&brickData)) {
        return 0.0;  // Invalid brick index
    }

    let staticInfo = staticBricks[brickIdx];
    if ((staticInfo.flags & 1u) != 0u) {
        return 0.0;  // Empty brick
    }

    let brick = brickData[brickIdx];
    let brickOrigin = brickCoords * brickSizeI;
    let localVoxel = worldVoxel - brickOrigin;
    let localCoords = vec3<u32>(clamp(localVoxel, vec3<i32>(0), vec3<i32>(i32(scene.brickSize) - 1)));
    return getVoxelColor(brick, staticInfo, localCoords).a;
}

// Fast LOD-aware alpha sampling within a known brick (avoids brick lookup)
fn sampleAlphaInBrick(brick: DynamicBrickInfo, staticInfo: StaticBrickInfo, brickOrigin: vec3<i32>, worldVoxel: vec3<i32>) -> f32 {
    // Bounds check against volume extent
    if (any(worldVoxel < vec3<i32>(0)) || any(worldVoxel >= vec3<i32>(i32(scene.gridSize)))) {
        return 0.0;
    }
    
    let localVoxel = worldVoxel - brickOrigin;
    let brickSizeI = i32(scene.brickSize);
    
    // Check if still within the same brick
    if (any(localVoxel < vec3<i32>(0)) || any(localVoxel >= vec3<i32>(brickSizeI))) {
        // Crossed brick boundary, fall back to full lookup
        return sampleAlpha(worldVoxel);
    }
    
    let localCoords = vec3<u32>(clamp(localVoxel, vec3<i32>(0), vec3<i32>(brickSizeI - 1)));
    return getVoxelColor(brick, staticInfo, localCoords).a;
}

// Cast a shadow ray toward the light with accumulated opacity (1 = clear, 0 = fully occluded)
// Properly integrates accumulated alpha to:
//   - Skip shadow rays for semi-transparent voxels where shadow is barely visible
//   - Reduce ray length based on accumulated density (thick fog doesn't cast long shadows)
//   - Account for light transmission through previously accumulated voxels
// 
// NOTE: Currently samples at full resolution while rendering uses LOD
// This can cause thin occluders to disappear at distance and light leaks through brick interiors
// Ideally shadow rays should use LOD-aware sampling matching the render resolution
fn traceShadowRay(startPos: vec3<f32>, normal: vec3<f32>, lightDir: vec3<f32>, accumulatedAlpha: f32, transmittance: f32) -> f32 {
    const MAX_STEPS: u32 = 64u;           // More steps for better coverage
    const MIN_STEPS: u32 = 24u;           // Minimum steps to ensure shadow quality
    const BASE_STEP_SIZE: f32 = 1.0;      // Step size - increased to reduce LOD mismatch artifacts
    
    // Skip shadow rays entirely for semi-transparent voxels - shadow not visually significant
    // If transmittance is < 0.1 (>90% opaque), shadow is being heavily attenuated anyway
    if (transmittance < 0.1 || accumulatedAlpha < 0.2) {
        return 1.0;  // Skip shadow ray, use full lighting
    }
    
    // Adaptive step count based on accumulated opacity:
    // - Transparent (low alpha): use full quality shadows
    // - Opaque (high alpha): reduce quality since shadow is attenuated heavily
    // But ensure we always do at least MIN_STEPS for quality
    let stepCount = max(MIN_STEPS, u32(f32(MAX_STEPS) * transmittance));
    
    // lightDir points TO the light (same convention as diffuse lighting)
    let dir = normalize(lightDir);
    
    // Start the shadow ray offset along the SURFACE NORMAL to avoid self-shadowing
    // Offsetting along the normal ensures we move away from the surface regardless of light direction
    // This fixes directional bias where negative light directions would move INTO the surface
    let rayStart = startPos + normal * 0.6;  // Offset 0.6 voxels along surface normal
    
    // Start at half-step to sample voxel interiors, avoiding boundary skipping
    // This ensures symmetric sampling regardless of ray direction
    var t = BASE_STEP_SIZE * 0.5;
    var accumulatedOpacity = 0.0;  // Track total occlusion along ray
    
    for (var i: u32 = 0u; i < stepCount; i++) {
        let samplePos = rayStart + dir * t;
        
        // Check if sample position is within volume bounds
        // If we exit the volume, we've escaped to the light (no more occlusion)
        if (any(samplePos < vec3<f32>(0.0)) || any(samplePos >= vec3<f32>(scene.gridSize))) {
            break;  // Exited volume - clear path to light
        }
        
        // Sample the voxel containing this point
        // floor() gives us the voxel index regardless of direction
        // Combined with t starting at 0.5*stepSize, we sample voxel centers
        let voxelCoord = vec3<i32>(floor(samplePos));
        let alpha = sampleAlpha(voxelCoord);
        
        // Accumulate opacity: denser occlusion = darker shadow
        // Each sample contributes based on how opaque it is
        accumulatedOpacity += alpha * BASE_STEP_SIZE;
        
        // Early exit if sufficiently occluded (avoid wasting steps)
        if (accumulatedOpacity > 0.8) {
            return 0.0;  // Ray is fully occluded
        }
        
        t += BASE_STEP_SIZE;
    }
    
    // Return visibility: high accumulated opacity → low visibility → dark shadow
    // Use exponential falloff for smoother shadow gradation
    // Modulate shadow strength by transmittance: already-opaque areas get less shadow contribution
    let shadowVisibility = exp(-accumulatedOpacity * 1.5);  // 1.5 controls shadow darkness falloff
    
    // Mix shadow with fully lit based on transmittance:
    // - If transmittance ≈ 1.0 (fully transparent): use full shadow
    // - If transmittance ≈ 0.0 (fully opaque): shadow barely matters, return near-1.0
    return mix(1.0, shadowVisibility, transmittance);
}

// Compute gradient via central differences (6 neighbor samples)
// Returns unnormalized gradient vector pointing toward increasing alpha
// OLD VERSION: Always samples at full resolution (expensive!)
fn computeGradient(worldPos: vec3<f32>) -> vec3<f32> {
    let voxelPos = vec3<i32>(worldPos);
    
    // Sample 6 neighbors at integer voxel positions
    let alphaXp = sampleAlpha(voxelPos + vec3<i32>(1, 0, 0));
    let alphaXn = sampleAlpha(voxelPos + vec3<i32>(-1, 0, 0));
    let alphaYp = sampleAlpha(voxelPos + vec3<i32>(0, 1, 0));
    let alphaYn = sampleAlpha(voxelPos + vec3<i32>(0, -1, 0));
    let alphaZp = sampleAlpha(voxelPos + vec3<i32>(0, 0, 1));
    let alphaZn = sampleAlpha(voxelPos + vec3<i32>(0, 0, -1));
    
    // Central difference: (forward - backward) / 2
    return vec3<f32>(
        (alphaXp - alphaXn) * 0.5,
        (alphaYp - alphaYn) * 0.5,
        (alphaZp - alphaZn) * 0.5
    );
}

// LOD-AWARE GRADIENT: Samples at lodStride intervals within the current brick
// This is MUCH faster and visually nearly identical since it matches the render resolution
// Benefits:
// - Samples at the same resolution you're already rendering (no wasted detail)
// - Stays within one brick (no morton lookups or brick boundary checks)
// - Cost scales with LOD: close objects (LOD=6, stride=1) are expensive but cover few pixels
//                         far objects (LOD=2, stride=16) are cheap and cover many pixels
// Uses f32 for gradient vector for compatibility
// CRITICAL FIX: Only use non-zero alpha samples for gradient to prevent invisible voxel bleeding
fn computeGradientLOD(brick: DynamicBrickInfo, staticInfo: StaticBrickInfo, brickOrigin: vec3<i32>, worldPos: vec3<f32>, stride: i32, centerAlpha: f32) -> vec3<f32> {
    let voxelPos = vec3<i32>(worldPos);
    let offset = vec3<i32>(stride);
    // Sample 6 neighbors at LOD-stride intervals (e.g., stride=16 for distant bricks)
    let alphaXp = f32(sampleAlphaInBrick(brick, staticInfo, brickOrigin, voxelPos + vec3<i32>(offset.x, 0, 0)));
    let alphaXn = f32(sampleAlphaInBrick(brick, staticInfo, brickOrigin, voxelPos - vec3<i32>(offset.x, 0, 0)));
    let alphaYp = f32(sampleAlphaInBrick(brick, staticInfo, brickOrigin, voxelPos + vec3<i32>(0, offset.y, 0)));
    let alphaYn = f32(sampleAlphaInBrick(brick, staticInfo, brickOrigin, voxelPos - vec3<i32>(0, offset.y, 0)));
    let alphaZp = f32(sampleAlphaInBrick(brick, staticInfo, brickOrigin, voxelPos + vec3<i32>(0, 0, offset.z)));
    let alphaZn = f32(sampleAlphaInBrick(brick, staticInfo, brickOrigin, voxelPos - vec3<i32>(0, 0, offset.z)));
    
    // If any neighbor is invisible (alpha < 0.05), treat it as having the same alpha as center
    // This prevents gradients from pointing toward invisible regions
    let threshold = 0.05;
    let safeXp = select(centerAlpha, alphaXp, alphaXp > threshold);
    let safeXn = select(centerAlpha, alphaXn, alphaXn > threshold);
    let safeYp = select(centerAlpha, alphaYp, alphaYp > threshold);
    let safeYn = select(centerAlpha, alphaYn, alphaYn > threshold);
    let safeZp = select(centerAlpha, alphaZp, alphaZp > threshold);
    let safeZn = select(centerAlpha, alphaZn, alphaZn > threshold);
    
    // Central difference: (forward - backward) / 2
    return vec3<f32>(
        (safeXp - safeXn) * 0.5,
        (safeYp - safeYn) * 0.5,
        (safeZp - safeZn) * 0.5
    );
}

fn intersectVolume(ray: Ray) -> VolumeHit {
    let inv_dir = ray.invDir;  // Use precomputed reciprocal

    let gridSize = scene.gridSize;
    let t0s = (vec3<f32>(0.0, 0.0, 0.0) - ray.origin) * inv_dir;
    let t1s = (vec3<f32>(gridSize, gridSize, gridSize) - ray.origin) * inv_dir;
    let tmin = max(max(min(t0s.x, t1s.x), min(t0s.y, t1s.y)), min(t0s.z, t1s.z));
    let tmax = min(min(max(t0s.x, t1s.x), max(t0s.y, t1s.y)), max(t0s.z, t1s.z));

    if (tmax < 0.0 || tmin > tmax) {
        return VolumeHit(false, 0.0, 0.0);
    }

    return VolumeHit(true, max(tmin, 0.0), tmax);
}

fn initDDA(ray: Ray, tEnter: f32, tExit: f32) -> DDAState {
    // Small adaptive bias to avoid self-intersection without skipping the entry voxel
    // Important: keep all t values in WORLD PARAMETRIC SPACE
    let epsilon = max(tEnter * 1e-6, 1e-5);
    let tStart = tEnter + epsilon;
    let pos = ray.origin + ray.dir * tStart;
    let voxel = vec3<i32>(floor(pos));

    let step = vec3<i32>(
        select(0, 1, ray.dir.x > 0.0) + select(0, -1, ray.dir.x < 0.0),
        select(0, 1, ray.dir.y > 0.0) + select(0, -1, ray.dir.y < 0.0),
        select(0, 1, ray.dir.z > 0.0) + select(0, -1, ray.dir.z < 0.0)
    );

    // Compute the next voxel boundary in WORLD coordinates, then its param t from ray.origin
    let nextBoundaryWorld = vec3<f32>(
        select(f32(voxel.x), f32(voxel.x + 1), ray.dir.x > 0.0),
        select(f32(voxel.y), f32(voxel.y + 1), ray.dir.y > 0.0),
        select(f32(voxel.z), f32(voxel.z + 1), ray.dir.z > 0.0)
    );

    // Use precomputed invDir - eliminates 6 divisions! Now in WORLD param t
    let tMax = (nextBoundaryWorld - ray.origin) * ray.invDir;
    let tDelta = abs(ray.invDir);

    return DDAState(voxel, tMax, tDelta, step, tExit);
}

fn advanceDDA(state: ptr<function, DDAState>) -> f32 {
    var t: f32 = 0.0;
    if ((*state).tMax.x < (*state).tMax.y && (*state).tMax.x < (*state).tMax.z) {
        t = (*state).tMax.x;
        (*state).tMax.x += (*state).tDelta.x;
        (*state).voxel.x += (*state).step.x;
    } else if ((*state).tMax.y < (*state).tMax.z) {
        t = (*state).tMax.y;
        (*state).tMax.y += (*state).tDelta.y;
        (*state).voxel.y += (*state).step.y;
    } else {
        t = (*state).tMax.z;
        (*state).tMax.z += (*state).tDelta.z;
        (*state).voxel.z += (*state).step.z;
    }
    return t;
}

fn initBrickDDA(ray: Ray, tEnter: f32, tExit: f32, brickSize: u32) -> BrickDDAState {
    // Small adaptive bias at brick entry to avoid visible seams
    // t values in WORLD PARAMETRIC SPACE
    let epsilon = max(tEnter * 1e-6, 1e-5);
    let tStart = tEnter + epsilon;
    let pos = ray.origin + ray.dir * tStart;
    let brick = vec3<i32>(floor(pos)) / i32(brickSize);
    
    let step = vec3<i32>(
        select(0, 1, ray.dir.x > 0.0) + select(0, -1, ray.dir.x < 0.0),
        select(0, 1, ray.dir.y > 0.0) + select(0, -1, ray.dir.y < 0.0),
        select(0, 1, ray.dir.z > 0.0) + select(0, -1, ray.dir.z < 0.0)
    );
    
    let brickF = vec3<f32>(brick) * f32(brickSize);
    let nextBrickBoundaryWorld = vec3<f32>(
        select(brickF.x, brickF.x + f32(brickSize), ray.dir.x > 0.0),
        select(brickF.y, brickF.y + f32(brickSize), ray.dir.y > 0.0),
        select(brickF.z, brickF.z + f32(brickSize), ray.dir.z > 0.0)
    );
    
    // Use precomputed invDir - eliminates 6 more divisions! Now in WORLD param t
    let tMax = (nextBrickBoundaryWorld - ray.origin) * ray.invDir;
    let tDelta = abs(vec3<f32>(f32(brickSize)) * ray.invDir);
    
    return BrickDDAState(brick, tMax, tDelta, step, tExit);
}

fn advanceBrickDDA(state: ptr<function, BrickDDAState>) -> f32 {
    var t: f32 = 0.0;
    if ((*state).tMax.x < (*state).tMax.y && (*state).tMax.x < (*state).tMax.z) {
        t = (*state).tMax.x;
        (*state).tMax.x += (*state).tDelta.x;
        (*state).brick.x += (*state).step.x;
    } else if ((*state).tMax.y < (*state).tMax.z) {
        t = (*state).tMax.y;
        (*state).tMax.y += (*state).tDelta.y;
        (*state).brick.y += (*state).step.y;
    } else {
        t = (*state).tMax.z;
        (*state).tMax.z += (*state).tDelta.z;
        (*state).brick.z += (*state).step.z;
    }
    return t;
}

@compute @workgroup_size(8, 8, 1)
fn rayTrace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_x = global_id.x;
    let pixel_y = global_id.y;

    let screen_width = u32(screen.width);
    let screen_height = u32(screen.height);
    
    // if(arrayLength(&brickData) == 512)
    // {
    //     let rgba : u32 =
    //     (255u << 0u)  | // R
    //     (0u   << 8u)  | // G
    //     (255u << 16u) | // B
    //     (255u << 24u);  // A

    //      rayTraceOutput[pixel_y * screen_width + pixel_x] = rgba;
    //     return;
    // }
    // Debug: Log scene uniforms once per frame (from top-left pixel only)
    if (pixel_x == 0u && pixel_y == 0u) {
        // This will show in shader compilation warnings or debug output
        // let debug = vec4<f32>(scene.gridSize, scene.brickSize, scene.bricksPerAxis, 0.0);
    }

    // Critical: Always write to output to avoid stale data artifacts
    // Early return without writing leaves pixels with old frame data
    if (pixel_x >= screen_width || pixel_y >= screen_height) {
        return;  // Out of bounds - buffer should be cleared to black already
    }

    // Normalize pixel coordinates to NDC space (-1..1) with pixel center sampling
    let ndc_x = ((f32(pixel_x) + 0.5) / screen.width) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((f32(pixel_y) + 0.5) / screen.height) * 2.0;  // Flip Y: top=+1, bottom=-1

    // Camera basis and tanHalfFov are precomputed on CPU
    let forward = camera.forward;
    let right = camera.right;
    let up = camera.up;
    let aspectRatio = screen.aspect;
    let tanHalfFov = camera.tanHalfFov;
    let rayDir = normalize(
        forward +
        right * ndc_x * aspectRatio * tanHalfFov +
        up * ndc_y * tanHalfFov
    );
    
    // Precompute ray direction reciprocal once (eliminates 15 divisions per ray!)
    let invRayDir = vec3<f32>(
        sign(rayDir.x) / max(abs(rayDir.x), 1e-6),
        sign(rayDir.y) / max(abs(rayDir.y), 1e-6),
        sign(rayDir.z) / max(abs(rayDir.z), 1e-6)
    );
    
    let ray = Ray(camera.position, rayDir, invRayDir);

    var hit_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);  // Default: transparent background
    var firstHitPos = vec3<f32>(0.0);
    var firstHitAlpha = 0.0;  // Track first hit opacity for AO culling
    var hasFirstHit = false;

    // AABB slab test against volume [0, GRID_SIZE)
    let volumeHit = intersectVolume(ray);
    if (!volumeHit.hit) {
        // Ray missed volume entirely - buffer already cleared to black by compute shader
        return;
    }

    let brickSizeU32 = u32(scene.brickSize);
    var brickDDA = initBrickDDA(ray, volumeHit.tEnter, volumeHit.tExit, brickSizeU32);
    let numBricks = max(i32(scene.bricksPerAxis), 1);
    
    // Calculate tight loop bound: how many bricks does THIS ray cross?
    // Use entry/exit points to compute manhattan distance in brick space
    let brickEnter = vec3<i32>(ray.origin + ray.dir * volumeHit.tEnter) / i32(brickSizeU32);
    let brickExit = vec3<i32>(ray.origin + ray.dir * volumeHit.tExit) / i32(brickSizeU32);
    let brickDelta = abs(brickExit - brickEnter);
    let maxBrickCrossings = u32(brickDelta.x + brickDelta.y + brickDelta.z) + 6u;  // Larger margin to avoid early brick exit
    
    var brickEntryT = volumeHit.tEnter;  // Entry time for first brick

    // Outer loop: traverse brick grid
    for (var brickIter: u32 = 0u; brickIter < maxBrickCrossings; brickIter++) {
        // Check brick bounds
        if (any(brickDDA.brick < vec3<i32>(0)) || any(brickDDA.brick >= vec3<i32>(numBricks))) {
            break;
        }

        // Compute Morton code and look up actual brick index
        let mortonCode = morton3D(u32(brickDDA.brick.x), u32(brickDDA.brick.y), u32(brickDDA.brick.z));

        // Use Morton code directly as brick index (bricks stored in Morton order)
        let brickIdx = mortonCode;
        if (brickIdx >= arrayLength(&brickData)) {
            // Out of range - advance to next brick
            let t = advanceBrickDDA(&brickDDA);
            brickEntryT = t + max(t * 1e-6, 1e-5);
            if (t > brickDDA.tExit) {
                break;
            }
            continue;
        }
        
        if (brickIdx < arrayLength(&brickData)) {
            // Fast empty-brick check using cached static info (no palette access)
            let staticInfo = staticBricks[brickIdx];
            let brick = brickData[brickIdx];

            if ((staticInfo.flags & 1u) != 0u) {  // isEmpty flag set
                // Empty brick: skip to next brick
                let t = advanceBrickDDA(&brickDDA);
                // Small adaptive bias to prevent overlap
                brickEntryT = t + max(t * 1e-6, 1e-5);
                if (t > brickDDA.tExit) {
                    break;
                }
                continue;
            }
            
            // Non-empty brick accessed: mark as accessed for GPU-driven visibility
            // Just write 1 to mark - don't store LOD value in buffer to avoid race conditions
            atomicStore(&brickRequests[brickIdx], BRICK_ACCESSED);
            
            // Non-empty brick: traverse voxels within it
            let brickOriginI = vec3<i32>(brickDDA.brick) * i32(scene.brickSize);
            let brickMinI = brickOriginI;
            let brickMaxI = brickOriginI + vec3<i32>(i32(scene.brickSize));
            
            // Entry is the start of this brick traversal (previous DDA step or initial volume entry)
            // Exit is the minimum of all three tMax values (next brick plane we'll cross)
            let tEnterBrick = max(brickEntryT, volumeHit.tEnter);
            var tExitBrick = min(min(brickDDA.tMax.x, brickDDA.tMax.y), brickDDA.tMax.z);

            // Small adaptive exit bias to prevent boundary overlap
            let exitBias = max(tExitBrick * 1e-6, 1e-5);
            tExitBrick = max(tEnterBrick, tExitBrick - exitBias);
            
            // Initialize voxel DDA within this brick
            var voxelDDA = initDDA(ray, tEnterBrick, tExitBrick);
            
            // Per-brick step cap: never step more times than voxels in this brick at its target LOD
            let lodSize = 1u << brick.targetLOD;

            // Set stride based on LOD to avoid oversampling
            // LOD 6 (64 voxels): stride = 1 (step every voxel)
            // LOD 3 (8 voxels): stride = 8 (step every 8th voxel)
            // LOD 0 (1 voxel): stride = 64 (entire brick is one voxel)
            let lodStride = i32(scene.brickSize) / i32(lodSize);
            
            // OPTIMIZATION: Convert to brick-local coordinate space FIRST
            // --- LOD-aware voxel DDA inside a brick ---
            let brickSizeI = i32(scene.brickSize);

            // Convert voxel to brick-local coordinates
            voxelDDA.voxel = voxelDDA.voxel - brickMinI;

            // Quantize voxel to LOD grid if stride > 1
            if (lodStride > 1) {
                // Align voxel to nearest LOD grid (floor to multiple of stride)
                let lodVoxel = (voxelDDA.voxel / vec3<i32>(lodStride)) * vec3<i32>(lodStride);
                voxelDDA.voxel = clamp(lodVoxel, vec3<i32>(0), vec3<i32>(brickSizeI - 1));

                // Correct step size in world space (always positive so traversal advances correctly for both ray directions)
                voxelDDA.tDelta = abs(ray.invDir) * vec3<f32>(f32(lodStride));

                // Compute tMax correctly using **LOD-aligned voxel world position**
                let lodWorldVoxel = vec3<f32>(brickMinI) + vec3<f32>(voxelDDA.voxel);
                let nextBoundary = vec3<f32>(
                    select(lodWorldVoxel.x, lodWorldVoxel.x + f32(lodStride), ray.dir.x > 0.0),
                    select(lodWorldVoxel.y, lodWorldVoxel.y + f32(lodStride), ray.dir.y > 0.0),
                    select(lodWorldVoxel.z, lodWorldVoxel.z + f32(lodStride), ray.dir.z > 0.0)
                );

                // Set tMax relative to LOD-aligned voxel position, not ray entry
                voxelDDA.tMax = (nextBoundary - ray.origin) * ray.invDir;
            } else {
                // No LOD quantization - use natural voxel-by-voxel DDA
                // Just clamp to brick bounds
                voxelDDA.voxel = clamp(voxelDDA.voxel, vec3<i32>(0), vec3<i32>(brickSizeI - 1));
            }
            
            // Opacity correction factor for variable step size
            // Larger steps need higher opacity to represent the same density
            let opacityScale = f32(lodStride);
            
            // Use a bounded for-loop for stability (prevent potential infinite loops on GPU)
            let voxelEnter = ray.origin + ray.dir * tEnterBrick;
            let voxelExit = ray.origin + ray.dir * tExitBrick;
            
            // CRITICAL: Compute step count in local LOD-stride space to match DDA stepping
            // voxelEnter and voxelExit are in world space; convert to brick-local LOD-stride space
            let localEnterF = (voxelEnter - vec3<f32>(brickMinI)) / f32(lodStride);
            let localExitF = (voxelExit - vec3<f32>(brickMinI)) / f32(lodStride);
            
            // Use floor for enter and ceil for exit to get conservative bounds
            let localEnter = vec3<i32>(floor(localEnterF));
            let localExit = vec3<i32>(ceil(localExitF));
            let delta = abs(localExit - localEnter);
            let maxStepsThisBrick = u32(delta.x + delta.y + delta.z) + 128u;  // Increased margin for safety

            // DEBUG: For first 256 bricks, check iteration count
            var debugStepCount = 0u;
            var debugExitReason = 0u; // 0=none, 1=out_of_bounds, 2=t>tExit, 3=max_iters

            for (var voxelIter: u32 = 0u; voxelIter < maxStepsThisBrick; voxelIter++) {
                let in_bounds = !(any(voxelDDA.voxel < vec3<i32>(0)) || any(voxelDDA.voxel >= vec3<i32>(brickSizeI)));

                if (!in_bounds) {
                    let t = advanceDDA(&voxelDDA);
                    if (mortonCode < 256u) {
                        debugExitReason = 1u; // out of bounds
                    }
                    if (t > voxelDDA.tExit) {
                        break;
                    }
                    continue;
                }

                let staticInfo = staticBricks[brickIdx];
                let localCoords = vec3<u32>(voxelDDA.voxel);
                let color = getVoxelColor(brick, staticInfo, localCoords);

                // DEBUG: Count steps for first 256 bricks
                if (mortonCode < 256u) {
                    debugStepCount += 1u;
                }

                // Skip completely transparent voxels (invisible labels) - don't accumulate
                // Use higher threshold to avoid bleeding from nearly-invisible voxels
                if (color.a > 0.05) {
                    if (!hasFirstHit) {
                        firstHitPos = vec3<f32>(brickMinI + voxelDDA.voxel) + vec3<f32>(0.5);
                        firstHitAlpha = color.a;
                        hasFirstHit = true;
                    }

                    var shadedColor = color;

                    // Only apply shading for sufficiently opaque voxels
                    if (color.a > 0.35) {
                        let worldPos = vec3<f32>(brickMinI + voxelDDA.voxel) + vec3<f32>(0.5);
                        let gradient = computeGradientLOD(brick, staticInfo, brickOriginI, worldPos, lodStride, color.a);
                        let gradLen = length(gradient);
                        let normal = vec3<f32>(-gradient / max(gradLen, 0.001));

                        var shadowFactor = 1.0;
                        if (color.a > 0.4) {
                            let transmittance = 1.0 - hit_color.a;
                            shadowFactor = traceShadowRay(worldPos, normal, lighting.lightDir, hit_color.a, transmittance);
                        }

                        let lightDiffuse = max(0.0, dot(normal, normalize(lighting.lightDir)));
                        let diffuseStrength = lighting.params2.y * shadowFactor;
                        let ambient = lighting.params.x;
                        let isFrontFacing = lightDiffuse > 0.01;
                        let minAmbient = select(0.1, 0.3, isFrontFacing);
                        let shadingFactor = clamp(ambient + lightDiffuse * diffuseStrength, minAmbient, 1.0);
                        let luminance = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
                        let shadowMix = clamp(1.0 - shadowFactor, 0.0, 1.0) * 0.5;
                        let baseColor = mix(color.rgb, vec3<f32>(luminance) * 0.8 + color.rgb * 0.2, shadowMix);
                        shadedColor = vec4<f32>(baseColor * shadingFactor, color.a);
                    }

                    let corrected_alpha = clamp(1.0 - pow(1.0 - shadedColor.a, opacityScale), 0.0, 1.0);
                    let src_alpha = corrected_alpha * (1.0 - hit_color.a);
                    hit_color.r += shadedColor.r * src_alpha;
                    hit_color.g += shadedColor.g * src_alpha;
                    hit_color.b += shadedColor.b * src_alpha;
                    hit_color.a += src_alpha;

                    if (hit_color.a > 0.95) {
                        break;
                    }
                }

                let t = advanceDDA(&voxelDDA);
                if (t > voxelDDA.tExit) {
                    if (mortonCode < 256u) {
                        debugExitReason = 2u; // t > tExit
                    }
                    break;
                }
            }
            
            // DEBUG: For first 256 bricks, show exit reason
            // if (mortonCode < 256u) {
            //     let pixel_index = pixel_y * screen_width + pixel_x;
                
            //     // Show why the loop exited:
            //     // Red = out of bounds after first step
            //     // Blue = t > tExit (ray exited brick)
            //     // Green = hit max iterations (shouldn't happen with current limit)
            //     // Yellow = multiple steps taken (good!)
                
            //     if (debugStepCount == 0u && debugExitReason == 1u) {
            //         rayTraceOutput[pixel_index] = 0xFF0000FFu; // Red = immediately out of bounds
            //         return;
            //     }
                
            //     if (debugStepCount == 1u && debugExitReason == 2u) {
            //         rayTraceOutput[pixel_index] = 0xFF00FFFFu; // Cyan = 1 step then t>tExit
            //         return;
            //     }
                
            //     if (debugStepCount > 1u) {
            //         rayTraceOutput[pixel_index] = 0xFF00FF00u; // Green = multiple steps (correct!)
            //         return;
            //     }
                
            //     // Orange = something else
            //     rayTraceOutput[pixel_index] = 0xFFFF8000u;
            //     return;
            // }
        }
        
        // Early termination if ray is fully opaque
        if (hit_color.a > 0.95) {
            break;
        }
        
        // Advance to next brick
        let t = advanceBrickDDA(&brickDDA);
        // Small adaptive bias to prevent overlap
        brickEntryT = t + max(t * 1e-6, 1e-5);
        if (t > brickDDA.tExit) {
            break;
        }
    }

    // Per-voxel shadows are already applied during compositing above
    // No need for additional AO pass - shadows are baked into the lit color

    // Convert to RGBA8 and store
    let pixel_index = pixel_y * screen_width + pixel_x;
    let rgba = 
        (u32(hit_color.x * 255.0) << 0u) |
        (u32(hit_color.y * 255.0) << 8u) |
        (u32(hit_color.z * 255.0) << 16u) |
        (u32(hit_color.w * 255.0) << 24u);

    rayTraceOutput[pixel_index] = rgba;
}
