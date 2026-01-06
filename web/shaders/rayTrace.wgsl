// Ray tracing with transfer function, empty-space skipping, and LOD selection

// Gradient shading and AO always enabled

// Enable f16 for bandwidth and register pressure reduction
enable f16;

struct DynamicBrickInfo {
    outputOffset: u32,        // Offset into outVoxels buffer
    targetLOD: u32,           // Resolution level (voxels per dimension)
    lodScale: f32,            // Precomputed: lodSize / brickSize (for fast coordinate scaling)
    pad0: u32,                // Padding for 16-byte alignment
}

struct StaticBrickInfo {
    nSymbols: u32,
    paletteSize: u32,
    streamOffset: u32,
    paletteOffset: u32,
    flags: u32,               // bit 0 = isEmpty (1 = empty, 0 = non-empty)
    pad0: u32,                // Padding for 32-byte alignment
    pad1: u32,
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
var<storage, read_write> aoHistory: array<f16>;  // Temporal AO history per pixel (f16 saves 50% bandwidth!)

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

// Lighting + AO controls
// params = (ambient, shadowAlphaThreshold, aoBlend, aoStrength)
// params2 = (gradientShadingEnabled, diffuseStrength, pad, pad)
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
fn getVoxelColor(brick: DynamicBrickInfo, localCoords: vec3<u32>) -> vec4<f32> {
    // Calculate resolution at this LOD: 2^targetLOD voxels per dimension
    let lodSize = 1u << brick.targetLOD;
    
    // Fast vectorized scaling using precomputed lodScale (lodSize / brickSize)
    // This eliminates per-sample bit shift, division, and redundant casts
    let scaled = vec3<u32>(vec3<f32>(localCoords) * brick.lodScale);
    let coords = min(scaled, vec3<u32>(lodSize - 1u));
    
    // Use Morton ordering to find index within this brick's output range
    let localIndex = morton3D(coords.x, coords.y, coords.z);
    let globalIndex = brick.outputOffset + localIndex;
    
    // Bounds check
    if (globalIndex >= arrayLength(&outVoxels)) {
        return vec4<f32>(0.0);
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
    // Convert float coords to integer for LOD0 sampling
    let coords0 = vec3<u32>(localCoords);
    let frac = localCoords - vec3<f32>(coords0);
    let lodSize = 1u << brick.targetLOD;
    
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

    let brickIdx = morton3D(u32(brickCoords.x), u32(brickCoords.y), u32(brickCoords.z));

    if (brickIdx >= arrayLength(&brickData)) {
        return 0.0;
    }

    let staticInfo = staticBricks[brickIdx];
    if ((staticInfo.flags & 1u) != 0u) {
        return 0.0;  // Empty brick
    }

    let brick = brickData[brickIdx];
    let brickOrigin = brickCoords * brickSizeI;
    let localCoords = vec3<u32>(worldVoxel - brickOrigin);
    return getVoxelColor(brick, localCoords).a;
}

// Fast LOD-aware alpha sampling within a known brick (avoids brick lookup)
fn sampleAlphaInBrick(brick: DynamicBrickInfo, brickOrigin: vec3<i32>, worldVoxel: vec3<i32>) -> f32 {
    // Bounds check against volume extent
    if (any(worldVoxel < vec3<i32>(0)) || any(worldVoxel >= vec3<i32>(i32(scene.gridSize)))) {
        return 0.0;
    }
    
    let localCoords = vec3<u32>(worldVoxel - brickOrigin);
    let brickSizeU = u32(scene.brickSize);
    
    // Check if still within the same brick
    if (any(localCoords >= vec3<u32>(brickSizeU))) {
        // Crossed brick boundary, fall back to full lookup
        return sampleAlpha(worldVoxel);
    }
    
    return getVoxelColor(brick, localCoords).a;
}

// Cast a single shadow ray toward the light to estimate visibility (1 = clear, 0 = occluded)
fn traceShadowRay(startPos: vec3<f32>, lightDir: vec3<f32>, alphaThreshold: f32) -> f32 {
    const MAX_STEPS: u32 = 48u;
    const STEP_SIZE: f32 = 1.5;

    let dir = normalize(lightDir);
    var t = STEP_SIZE;  // start slightly off the surface to avoid self-shadow

    for (var i: u32 = 0u; i < MAX_STEPS; i++) {
        let samplePos = startPos + dir * t;
        let alpha = sampleAlpha(vec3<i32>(samplePos));
        if (alpha > alphaThreshold) {
            return 0.0;
        }
        t += STEP_SIZE;
    }
    return 1.0;
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
// Uses f16 for gradient vector to reduce register pressure (normals don't need f32 precision)
fn computeGradientLOD(brick: DynamicBrickInfo, brickOrigin: vec3<i32>, worldPos: vec3<f32>, stride: i32) -> vec3<f16> {
    let voxelPos = vec3<i32>(worldPos);
    let offset = vec3<i32>(stride);
    
    // Sample 6 neighbors at LOD-stride intervals (e.g., stride=16 for distant bricks)
    // Convert to f16 immediately to save register pressure
    let alphaXp = f16(sampleAlphaInBrick(brick, brickOrigin, voxelPos + vec3<i32>(offset.x, 0, 0)));
    let alphaXn = f16(sampleAlphaInBrick(brick, brickOrigin, voxelPos - vec3<i32>(offset.x, 0, 0)));
    let alphaYp = f16(sampleAlphaInBrick(brick, brickOrigin, voxelPos + vec3<i32>(0, offset.y, 0)));
    let alphaYn = f16(sampleAlphaInBrick(brick, brickOrigin, voxelPos - vec3<i32>(0, offset.y, 0)));
    let alphaZp = f16(sampleAlphaInBrick(brick, brickOrigin, voxelPos + vec3<i32>(0, 0, offset.z)));
    let alphaZn = f16(sampleAlphaInBrick(brick, brickOrigin, voxelPos - vec3<i32>(0, 0, offset.z)));
    
    // Central difference: (forward - backward) / 2
    // Note: no need to divide by stride since we only care about direction after normalization
    return vec3<f16>(
        (alphaXp - alphaXn) * 0.5h,
        (alphaYp - alphaYn) * 0.5h,
        (alphaZp - alphaZn) * 0.5h
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
    let pos = ray.origin + ray.dir * tEnter;  // No bias - start exactly at boundary
    let voxel = vec3<i32>(pos);

    let step = vec3<i32>(
        select(0, 1, ray.dir.x > 0.0) + select(0, -1, ray.dir.x < 0.0),
        select(0, 1, ray.dir.y > 0.0) + select(0, -1, ray.dir.y < 0.0),
        select(0, 1, ray.dir.z > 0.0) + select(0, -1, ray.dir.z < 0.0)
    );

    let nextBoundary = vec3<f32>(
        select(f32(voxel.x), f32(voxel.x + 1), ray.dir.x > 0.0),
        select(f32(voxel.y), f32(voxel.y + 1), ray.dir.y > 0.0),
        select(f32(voxel.z), f32(voxel.z + 1), ray.dir.z > 0.0)
    );

    // Use precomputed invDir - eliminates 6 divisions!
    let tMax = (nextBoundary - pos) * ray.invDir;
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
    let pos = ray.origin + ray.dir * (tEnter + f32(brickSize) * 1e-4);
    let brick = vec3<i32>(pos) / i32(brickSize);
    
    let step = vec3<i32>(
        select(0, 1, ray.dir.x > 0.0) + select(0, -1, ray.dir.x < 0.0),
        select(0, 1, ray.dir.y > 0.0) + select(0, -1, ray.dir.y < 0.0),
        select(0, 1, ray.dir.z > 0.0) + select(0, -1, ray.dir.z < 0.0)
    );
    
    let brickF = vec3<f32>(brick) * f32(brickSize);
    let nextBrickBoundary = vec3<f32>(
        select(brickF.x, brickF.x + f32(brickSize), ray.dir.x > 0.0),
        select(brickF.y, brickF.y + f32(brickSize), ray.dir.y > 0.0),
        select(brickF.z, brickF.z + f32(brickSize), ray.dir.z > 0.0)
    );
    
    // Use precomputed invDir - eliminates 6 more divisions!
    let tMax = (nextBrickBoundary - pos) * ray.invDir;
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

    if (pixel_x >= screen_width || pixel_y >= screen_height) {
        return;
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
        select(0.0, 1.0 / rayDir.x, abs(rayDir.x) > 1e-6),
        select(0.0, 1.0 / rayDir.y, abs(rayDir.y) > 1e-6),
        select(0.0, 1.0 / rayDir.z, abs(rayDir.z) > 1e-6)
    );
    
    let ray = Ray(camera.position, rayDir, invRayDir);

    var hit_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);  // Default: transparent background
    var firstHitPos = vec3<f32>(0.0);
    var firstHitAlpha = 0.0;  // Track first hit opacity for AO culling
    var hasFirstHit = false;

    // AABB slab test against volume [0, GRID_SIZE)
    let volumeHit = intersectVolume(ray);
    if (!volumeHit.hit) {
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
    let maxBrickCrossings = u32(brickDelta.x + brickDelta.y + brickDelta.z) + 3u;  // Manhattan distance + margin
    
    var brickEntryT = volumeHit.tEnter;  // Entry time for first brick

    // Outer loop: traverse brick grid
    for (var brickIter: u32 = 0u; brickIter < maxBrickCrossings; brickIter++) {
        // Check brick bounds
        if (any(brickDDA.brick < vec3<i32>(0)) || any(brickDDA.brick >= vec3<i32>(numBricks))) {
            break;
        }

        // Compute Morton index on-the-fly (bit interleaving is cheap ALU vs memory fetch)
        let brickIdx = morton3D(u32(brickDDA.brick.x), u32(brickDDA.brick.y), u32(brickDDA.brick.z));
        
        if (brickIdx < arrayLength(&brickData)) {
            // Fast empty-brick check using cached static info (no palette access)
            let staticInfo = staticBricks[brickIdx];
            if ((staticInfo.flags & 1u) != 0u) {  // isEmpty flag set
                // Empty brick: skip to next brick
                let t = advanceBrickDDA(&brickDDA);
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
            let tEnterBrick = brickEntryT;
            let tExitBrick = min(min(brickDDA.tMax.x, brickDDA.tMax.y), brickDDA.tMax.z);
            
            // Initialize voxel DDA within this brick
            var voxelDDA = initDDA(ray, tEnterBrick, tExitBrick);
            
            // OPTIMIZATION: Convert to brick-local coordinate space ONCE
            // This eliminates repeated (voxelDDA.voxel - brickMinI) in the hot loop
            voxelDDA.voxel = clamp(voxelDDA.voxel - brickMinI, vec3<i32>(0), vec3<i32>(i32(scene.brickSize) - 1));
            
            // Inner loop: traverse voxels within current brick
            let brick = brickData[brickIdx];
            // Per-brick step cap: never step more times than voxels in this brick at its target LOD
            let lodSize = 1u << brick.targetLOD;
            
            // Scale DDA step size to LOD resolution (e.g., if LOD=1, step by 32 instead of 1)
            // This converts from full-resolution (64^3) to LOD-resolution (lodSize^3) traversal
            let lodStride = i32(scene.brickSize) / i32(lodSize);
            voxelDDA.tDelta *= f32(lodStride);  
            voxelDDA.voxel = (voxelDDA.voxel / vec3<i32>(lodStride)) * vec3<i32>(lodStride);
            
            // Opacity correction factor for variable step size
            // Larger steps need higher opacity to represent the same density
            let opacityScale = f32(lodStride);
            
            // OPTIMIZATION: Compute tight voxel step bound based on actual ray traversal
            // Old: 3u * lodSize (very conservative, can be 10-50x too large for grazing rays)
            // New: Manhattan distance in LOD-space + small margin
            let voxelEnter = ray.origin + ray.dir * tEnterBrick;
            let voxelExit = ray.origin + ray.dir * tExitBrick;
            // Convert to brick-local LOD-space coordinates
            let localEnter = (voxelEnter - vec3<f32>(brickMinI)) / f32(lodStride);
            let localExit = (voxelExit - vec3<f32>(brickMinI)) / f32(lodStride);
            let delta = abs(vec3<i32>(localExit) - vec3<i32>(localEnter));
            let maxStepsThisBrick = u32(delta.x + delta.y + delta.z) + 3u;  // Manhattan distance + margin
            
            let brickSizeI = i32(scene.brickSize);
            for (var voxelIter: u32 = 0u; voxelIter < maxStepsThisBrick; voxelIter++) {
                
                // OPTIMIZATION: Simplified bounds check in brick-local space
                // voxelDDA.voxel is now brick-local, so just check [0, brickSize)
                let in_bounds = !(any(voxelDDA.voxel < vec3<i32>(0)) || any(voxelDDA.voxel >= vec3<i32>(brickSizeI)));
                
                if (!in_bounds) {
                    let t = advanceDDA(&voxelDDA);
                    if (t > voxelDDA.tExit) {
                        break;
                    }
                    continue;
                }
                
                // OPTIMIZATION: voxelDDA.voxel is already brick-local - no subtraction needed!
                let localCoords = vec3<u32>(voxelDDA.voxel);
                let color = getVoxelColor(brick, localCoords);
                
                // Front-to-back volumetric compositing with opacity correction
                if (color.a > 0.01) {  // Skip nearly transparent voxels
                    if (!hasFirstHit) {
                        // Reconstruct world position when needed
                        firstHitPos = vec3<f32>(brickMinI + voxelDDA.voxel) + vec3<f32>(0.5);
                        firstHitAlpha = color.a;  // Store first hit opacity
                        hasFirstHit = true;
                    }
                    
                    // Apply gradient-based shading
                    var shadedColor = color;
                    
                    // OPTIMIZATION: Only compute gradient for opaque-enough voxels
                    // Skip gradient for transparent/semi-transparent samples (huge savings!)
                    if (color.a > 0.3) {
                        // Reconstruct world position for gradient computation (only when needed)
                        let worldPos = vec3<f32>(brickMinI + voxelDDA.voxel) + vec3<f32>(0.5);
                        
                        // OPTIMIZATION: Use LOD-aware gradient that samples at lodStride intervals
                        // This is dramatically cheaper and visually nearly identical since it matches
                        // the resolution we're already rendering at
                        // Gradient in f16 reduces register pressure with no visual impact
                        let gradient = computeGradientLOD(brick, brickOriginI, worldPos, lodStride);
                        let gradLen = f16(length(gradient));
                        
                        // OPTIMIZATION: Branchless shading to reduce warp divergence
                        // Use step() to select between gradient normal and ray fallback
                        let useGradient = step(0.05h, gradLen);  // 1.0 if gradLen >= 0.05, else 0.0
                        
                        // Compute both normals (cheap, both are normalize operations)
                        let gradientNormal = vec3<f32>(-gradient / max(gradLen, 0.001h));  // Avoid div-by-zero
                        let rayFallbackNormal = -normalize(rayDir);
                        
                        // Branchless select: use gradient normal if strong, else ray normal
                        let normal = mix(rayFallbackNormal, gradientNormal, f32(useGradient));
                        
                        // Single lighting calculation path (no divergence)
                        let lightDiffuse = f16(max(0.0, dot(normal, normalize(lighting.lightDir))));
                        let diffuseStrength = f16(lighting.params2.y);
                        // Attenuate fallback normal contribution (only when useGradient=0)
                        let fallbackAttenuation = mix(0.5h, 1.0h, useGradient);
                        let shadingFactor = f32(clamp(f16(lighting.params.x) + lightDiffuse * diffuseStrength * fallbackAttenuation, 0.3h, 2.0h));
                        shadedColor = vec4<f32>(color.rgb * shadingFactor, color.a);
                    }
                    
                    // Correct opacity for step size: alpha_corrected = 1 - (1 - alpha)^stepSize
                    // Approximation for small alpha: alpha_corrected ≈ alpha * stepSize
                    let corrected_alpha = clamp(1.0 - pow(1.0 - shadedColor.a, opacityScale), 0.0, 1.0);
                    
                    let src_alpha = corrected_alpha * (1.0 - hit_color.a);  // Attenuate by accumulated opacity
                    hit_color.r += shadedColor.r * src_alpha;
                    hit_color.g += shadedColor.g * src_alpha;
                    hit_color.b += shadedColor.b * src_alpha;
                    hit_color.a += src_alpha;
                    
                    // Early ray termination when accumulated opacity is high
                    if (hit_color.a > 0.95) {
                        break;
                    }
                }
                
                let t = advanceDDA(&voxelDDA);
                if (t > voxelDDA.tExit) {
                    break;
                }
            }
        }
        
        // Early termination if ray is fully opaque
        if (hit_color.a > 0.95) {
            break;
        }
        
        // Advance to next brick
        let t = advanceBrickDDA(&brickDDA);
        brickEntryT = t;  // Entry time for next brick is where current one exits
        if (t > brickDDA.tExit) {
            break;
        }
    }

    // Apply single-shadow-ray AO with temporal accumulation
    // OPTIMIZATION: Skip expensive shadow rays when they won't contribute much
    let pixel_index = pixel_y * screen_width + pixel_x;
    if (hasFirstHit && pixel_index < arrayLength(&aoHistory)) {
        // Calculate distance to first hit for distance-based culling
        let hitDistance = length(firstHitPos - camera.position);
        
        // Skip AO in these cases (saves 48-step shadow ray per pixel!):
        // 1. Ray already opaque (early termination) - AO won't be visible through opacity
        // 2. First hit is very transparent - AO contribution negligible
        // 3. First hit is very distant - AO detail not visible at that distance
        let skipAO = (hit_color.a > 0.9) ||           // Opaque accumulated result
                     (firstHitAlpha < 0.2) ||          // Very transparent first surface
                     (hitDistance > scene.gridSize * 0.6);  // Distant hit (>60% of volume extent)
        
        if (!skipAO) {
            let shadowVis = f16(traceShadowRay(firstHitPos, lighting.lightDir, lighting.params.y));
            let prevAO = aoHistory[pixel_index];
            let blendedAO = mix(prevAO, shadowVis, f16(lighting.params.z));
            aoHistory[pixel_index] = blendedAO;

            // Convert visibility to a modulation factor (convert back to f32 for final color)
            let aoFactor = f32(mix(1.0h - f16(lighting.params.w), 1.0h, blendedAO));
            let litFactor = max(lighting.params.x, aoFactor);
            hit_color = vec4<f32>(hit_color.rgb * litFactor, hit_color.a);
        } else {
            // Skipped AO: preserve temporal history with slow decay toward full visibility
            let prevAO = aoHistory[pixel_index];
            let decayedAO = mix(prevAO, 1.0h, 0.05h);  // Slowly fade to no shadow
            aoHistory[pixel_index] = decayedAO;
        }
    }

    // Convert to RGBA8 and store
    let rgba = 
        (u32(hit_color.x * 255.0) << 0u) |
        (u32(hit_color.y * 255.0) << 8u) |
        (u32(hit_color.z * 255.0) << 16u) |
        (u32(hit_color.w * 255.0) << 24u);

    rayTraceOutput[pixel_index] = rgba;
}
