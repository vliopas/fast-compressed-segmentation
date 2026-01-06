#include "brickEncoding.hpp"
#include "utils.hpp"

#include <algorithm>
#include <optional>

Encoding::FrequencyTable Encoding::interiorFreqTable{};
Encoding::FrequencyTable Encoding::leafFreqTable{};

std::vector<uint8_t>
Encoding::packOperationsToNibbles(const std::vector<OpEntry>& operations)
{
    // Bytes = final packed stream     (each byte = 2 nibbles)
    // isLeaf = parallel array tagging each nibble as leaf/interior
    //
    // Always 1 entry per nibble, including delta nibbles.

    std::vector<uint8_t> bytes;

    // Reserve some memory to reduce reallocations
    bytes.reserve((operations.size() * 3) / 2);


    // ------------------------------------------------------------
    // Nibble-pairing state
    //   If haveHigh==false → next nibble becomes the high nibble
    //   If haveHigh==true  → next nibble completes the byte
    bool haveHigh = false;
    uint8_t highNib = 0;
    // ------------------------------------------------------------

    // ------------------------------------------------------------
    // Helper: push one 4-bit nibble
    auto pushNibble = [&](uint8_t nib)
        {
            nib &= 0x0F;                // Keep only 4 bits

            if (!haveHigh)
            {
                // Store as high nibble of next byte
                highNib = nib;
                haveHigh = true;
            }
            else
            {
                // Complete byte by combining high + low nibble
                uint8_t byte = (highNib << 4) | nib;
                bytes.push_back(byte);
                haveHigh = false;
            }
        };
    // ------------------------------------------------------------


    // ------------------------------------------------------------
    // Main packing loop
    for (const auto& e : operations)
    {
        // Build primary nibble:
        //
        //   [ opCode (3 bits) | stopBit (1 bit, LSB) ]
        //
        uint8_t opCode = static_cast<uint8_t>(e.op);
        uint8_t primaryNibble = (opCode << 1) | (e.stopBit & 1);

        // Push primary nibble (always one)
        pushNibble(primaryNibble);

        // If Pδ: push delta nibble (second nibble of the op)
        if (e.op == OpType::PaletteBackD)
        {
            uint8_t deltaNibble = e.delta & 0x0F;
            pushNibble(deltaNibble);
        }
    }
    // ------------------------------------------------------------


    // ------------------------------------------------------------
    // If a high nibble is left without a partner, pad low nibble = 0
    if (haveHigh)
    {
        bytes.push_back(highNib << 4); // low nibble = 0
    }

    return bytes;
    // ------------------------------------------------------------
}


 void Encoding::updateFrequencyTables(const CompressedBrick& compressedBrick)
{
    // auto& bytes = compressedBrick.encodedData;
    // auto& isLeaf = compressedBrick.isLeaf;

    //// Each byte holds TWO nibbles.
    //size_t expectedNibbles = bytes.size() * 2;

    //// Allow last nibble to be padding:
    //assert(isLeaf.size() == expectedNibbles ||
    //       isLeaf.size() == expectedNibbles - 1);

    //// Decode and count
    //size_t leafIndex = 0;

    //for (uint8_t b : bytes)
    //{
    //    uint8_t hi = b >> 4;
    //    uint8_t lo = b & 0x0F;

        //// High nibble
        //if (leafIndex < isLeaf.size())
        //{
        //    if (isLeaf[leafIndex]) leafFreqTable[hi]++;
        //    else interiorFreqTable[hi]++;
        //    leafIndex++;
        //}

        //// Low nibble
        //if (leafIndex < isLeaf.size())
        //{
        //    if (isLeaf[leafIndex]) leafFreqTable[lo]++;
        //    else interiorFreqTable[lo]++;
        //    leafIndex++;
        //}
    //}

     auto& bytes = compressedBrick.encodedData;

     for (uint8_t b : bytes)
     {
            uint8_t hi = b >> 4;
            uint8_t lo = b & 0x0F;

            // High nibble
            interiorFreqTable[hi]++;

            // Low nibble
            interiorFreqTable[lo]++;
     }
}

RansModel Encoding::buildRansModel(const FrequencyTable& rawTable)
{
    const size_t N = rawTable.size();

    RansModel model;
    model.freq.resize(N);
    model.cumulativeFreq.resize(N + 1);

    // Replace zeros with 1
    uint64_t rawSum = 0;
    for (size_t i = 0; i < N; ++i) {
        uint32_t f = rawTable[i];
        if (f == 0) f = 1;
        model.freq[i] = f;
        rawSum += f;
    }

    // Normalize to RANS_TOTAL
    uint64_t normSum = 0;
    for (size_t i = 0; i < N; ++i) {
        uint64_t scaled = (model.freq[i] * (uint64_t)RANS_TOTAL) / rawSum;
        if (scaled == 0) scaled = 1;  // MUST remain ≥1
        model.freq[i] = (uint32_t)scaled;
        normSum += scaled;
    }

    //Fix rounding drift
    while (normSum < RANS_TOTAL) {
        model.freq[0]++;   // add slack somewhere
        normSum++;
    }
    while (normSum > RANS_TOTAL) {
        size_t largest = 0;
        for (size_t i = 1; i < N; ++i)
            if (model.freq[i] > model.freq[largest])
                largest = i;
        if (model.freq[largest] > 1) {
            model.freq[largest]--;
            normSum--;
        } else break;
    }

    // Build cumulative freq
    uint32_t acc = 0;
    for (size_t i = 0; i < N; ++i) {
        model.cumulativeFreq[i] = acc;
        acc += model.freq[i];
    }
    model.cumulativeFreq[N] = acc;

    model.totalFreq = acc;
    assert(model.totalFreq == RANS_TOTAL);

    return model;
}

void Encoding::ransEncodeSymbol(
uint32_t& state,
std::vector<uint8_t>& out,       // <-- OUTPUT BYTE STREAM
uint8_t symbol,
const RansModel& model)
{
    uint32_t freq = model.freq[symbol];
    uint32_t start = model.cumulativeFreq[symbol];

    // Renormalize
    while (state >= ((RANS_LIMIT >> LOG_TOTAL_FREQ) << 8) * freq)
    {
        out.push_back(state & 0xFF);
        state >>= 8;
    }

    state = ((state / freq) * model.totalFreq) + (state % freq) + start;
 }