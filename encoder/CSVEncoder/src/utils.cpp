#include "utils.hpp"

// View at: https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
uint32_t Utils::morton3D(uint32_t x, uint32_t y, uint32_t z)
{
    // Interleave 10 bits of x, y, z into a 30-bit Morton index
    auto part1by2 = [](uint32_t n)
        {
            // mask to 10 bits -> we only support up to 1024^3 bricks 
            // this is fine bricks typically 8^3 to 64^3 (see section 5.2)
            n &= 0x3ff;

            n = (n | (n << 16)) & 0x30000ff;
            n = (n | (n << 8)) & 0x300f00f;
            n = (n | (n << 4)) & 0x30c30c3;
            n = (n | (n << 2)) & 0x9249249;
            return n;
        };

    return part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2);
}

// Inverse of morton3D: extract x, y, z from a 30-bit Morton code
dim3 Utils::decodeMorton3D(uint32_t code)
{
    auto compact1by2 = [](uint32_t n)
        {
            // inverse of "spread bits" (part1by2)

            n &= 0x9249249;
            n = (n ^ (n >> 2)) & 0x30c30c3;
            n = (n ^ (n >> 4)) & 0x300f00f;
            n = (n ^ (n >> 8)) & 0x30000ff;
            n = (n ^ (n >> 16)) & 0x3ff;
            return n;
        };

    int x = compact1by2(code);
    int y = compact1by2(code >> 1);
    int z = compact1by2(code >> 2);

    return { x, y, z };
}

std::array<size_t, 8> Utils::computeChildMortonIndices(size_t i)
{
    std::array<size_t, 8> outIdx;

    // Decode parent coordinate at output LOD
    dim3 p = decodeMorton3D(i);


    // Child offsets (dx,dy,dz) in Morton order
    static const uint32_t dx[8] = { 0,1,0,1,0,1,0,1 };
    static const uint32_t dy[8] = { 0,0,1,1,0,0,1,1 };
    static const uint32_t dz[8] = { 0,0,0,0,1,1,1,1 };

    for (int c = 0; c < 8; c++)
    {
        uint32_t cx = p.x * 2 + dx[c]; // scale by 2 (go down a level)
        uint32_t cy = p.y * 2 + dy[c]; // scale by 2 (go down a level)
        uint32_t cz = p.z * 2 + dz[c]; // scale by 2 (go down a level)

        outIdx[c] = morton3D(cx, cy, cz);
    }

    return outIdx;
}