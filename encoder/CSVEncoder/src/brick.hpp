#pragma once

#include "types.hpp"
#include "utils.hpp"

#include <algorithm>
#include <unordered_map>
#include <stdexcept>

// Helper function to compute majority vote among 8 labels
inline LabelType majorityVote8(const LabelType* labels)
{
    // count occurrences
    std::unordered_map<LabelType, int> freq;
    for (int i = 0; i < 8; i++)
        freq[labels[i]]++;

    // return the most frequent label
    return std::max_element(
        freq.begin(), freq.end(),
        [](auto& a, auto& b){ return a.second < b.second; }
    )->first;
}

// Pyramid structure for multi-resolution representation - represents a single brick at multiple levels
template<size_t b> // Brick size is b^3
requires is_power_of_two_v<b> // Ensure size is a power of two
struct Brick
{
    static constexpr size_t Levels = std::bit_width(b); // Number of levels in the pyramid is log2(b)

    std::array<std::vector<Node>, Levels> levels; // L_0 finest -> L_N coarsest level

    // Access level from finest
    // returns the l-th finest level (0 = finest)
    // this respects the ordering of the paper where L0 is the finest level
    const std::vector<Node>& level(size_t l) const { return levels[l]; }
    std::vector<Node>& level(size_t l) { return levels[l]; }

    // Access level from coarsest
    // returns the L-th coarsest level (0 = coarsest)
    // this goes reverse to respect the paper ordering
    // its more intuitive to use however for traversing from coarsest to finest (encoding order)
    const std::vector<Node>& coarser(size_t l) const { return levels[Levels - 1 - l]; }
    std::vector<Node>& coarser(size_t l) { return levels[Levels - 1 - l]; }

    // Build all the pyramid levels of the brick
    void build()
    {
        // Level 0 already initiliazed via splitting voxel grid to individual bricks (0th level - finest)

        // Build higher levels (level 1 to N-1 - finer to coarser)
        for (size_t level = 1; level < Brick<b>::Levels; level++)
            buildLevel(level);

    }

private:
    void buildLevel(size_t level)
    {
        assert(level > 0); // level 0 is already initialized

        const size_t nodeCount = Utils::sizeForLevel<b>(Levels - level - 1);
        levels[level].resize(nodeCount);

        for (size_t mortonIdx = 0; mortonIdx < nodeCount; ++mortonIdx)
        {
            auto childrenMortonIndices = Utils::computeChildMortonIndices(mortonIdx);
            
            LabelType childLabels[8];
            bool childConstant[8];

            for (int c = 0; c < 8; ++c)
            {
                const Node& child = levels[level - 1][childrenMortonIndices[c]];
                childLabels[c] = child.label;
                childConstant[c] = child.constantChildren;
            }

            LabelType majLabel = majorityVote8(childLabels);

            bool allSame = true;
            for (size_t i = 0; i < 8; ++i)
            {
                allSame &= (childLabels[i] == majLabel && childConstant[i]);
            }

            levels[level][mortonIdx] = { majLabel, allSame };
        }
    }
};

template <size_t b>
std::vector<Brick<b>> splitGridIntoBricks(const NpyArray& array)
{
    if (array.shape.size() != 3)
        throw std::runtime_error("Expected a 3D array");

    const size_t width  = array.shape[0];
    const size_t height = array.shape[1];
    const size_t depth  = array.shape[2];

    std::vector<Brick<b>> bricks;

    // Iterate over the grid in steps of brick size
    // We are initializing the finest level L0
    for (size_t x0 = 0; x0 < width; x0 += b)
        for (size_t y0 = 0; y0 < height; y0 += b)
            for (size_t z0 = 0; z0 < depth; z0 += b)
            {
                Brick<b> pyramid;

                auto& nodes = pyramid.levels[0]; // L0
                nodes.resize(b * b * b);

                for (size_t dx = 0; dx < b; ++dx)
                    for (size_t dy = 0; dy < b; ++dy)
                        for (size_t dz = 0; dz < b; ++dz)
                        {
                            size_t gx = x0 + dx;
                            size_t gy = y0 + dy;
                            size_t gz = z0 + dz;

                            LabelType label = 0; // default
                            if (gx < width && gy < height && gz < depth)
                                label = array.data[gx + width * (gy + height * gz)];

                            size_t mortonIndex = Utils::morton3D(dx, dy, dz);
                            nodes[mortonIndex] = { label, true };
                        }

                bricks.push_back(std::move(pyramid));
            }

    return bricks;
}