// Partial clear: clears only specified brick regions

@group(0) @binding(0)
var<storage, read_write> outVoxels : array<u32>;

struct ClearRegion {
    offset: u32,        // Starting voxel index
    voxelCount: u32,    // Number of voxels to clear
}

@group(0) @binding(1)
var<uniform> regionCount: u32;  // Number of regions to clear

@group(0) @binding(2)
var<storage, read> clearRegions: array<ClearRegion>;  // Regions to clear (up to 256)

@group(0) @binding(3)
var<uniform> baseRegion: u32;   // Starting region index for this dispatch chunk

const EMPTY_VALUE : u32 = 0xFFFFFFFFu; // Transparent/empty marker
const MAX_CLEAR_REGIONS : u32 = 256u;  // Maximum regions per frame

// One workgroup per region; threads in the group walk the region span in strides of workgroup_size.
@compute @workgroup_size(256)
fn clearBuffer(
    @builtin(workgroup_id) wg_id : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>
) {
    let regionIdx = baseRegion + wg_id.x;   // global region index
    let localIdx = wg_id.x;                  // index into current chunk buffer
    if (regionIdx >= regionCount || localIdx >= MAX_CLEAR_REGIONS) {
        return;
    }

    let region = clearRegions[localIdx];
    let start = region.offset;
    let count = region.voxelCount;
    let stride = 256u;

    var i = lid.x;
    while (i < count) {
        let idx = start + i;
        if (idx < arrayLength(&outVoxels)) {
            outVoxels[idx] = EMPTY_VALUE;
        }
        i += stride;
    }
}
