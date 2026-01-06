#pragma once
#include "types.hpp"

namespace Utils
{
    // Morton index from 3D coordinates
    uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z);
    dim3 decodeMorton3D(uint32_t code);

    /**
     * @brief Compute the Morton indices of the 8 child nodes of a parent node.
     *
     * Given a parent node at pyramid level @p l, this function returns the Morton
     * (Z-order) indices of its 8 children at level @p l + 1, ordered according to
     * the Morton traversal used by the encoder and decoder.
     *
     * The returned order matches the processing order assumed by the encoding
     * pseudocode and is essential for consuming the operation stream correctly.
     *
     * @param i  Morton (Z-order) index of the parent node at level @p l.
     *
     * @return std::array<size_t, 8>  Morton indices of the 8 child nodes in Z-order.
     */
    std::array<size_t, 8> computeChildMortonIndices(size_t i);


    // Number of nodes at pyramid level l (0 = coarsest → L-1 = finest)
    template<size_t b>
    constexpr size_t sizeForLevel(size_t level)
    {
        // Finest level is b^3; coarsest is 1^3
        // Level l has (2^l)^3 nodes, because b = 2^(L-1)
        return size_t(1) << (3 * level);  // 2^(3*level)
    }
}