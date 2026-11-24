#pragma once

#include "lenia/convolve.hpp"

#include <cstdint>

namespace lenia {

// The simulation's two named scalar state fields (docs/state_fields.md):
//
//   Density A   what Lenia evolves; the kernel convolves it into the neighborhood U
//   Nutrient N  a second field a cell rule can read (Nutrient node) and write (the Simulation
//               node's Next nutrient input), e.g. a resource the density consumes
//
// Both are updated simultaneously: a generation computes every cell's next A and next N from
// the current A, N and U, so neither update sees the other's result. A rule that leaves Next
// nutrient unlinked keeps N as it is, and the built-in Lenia update never touches it, so
// single-field experiments run exactly as before the nutrient existed.
struct WorldState {
    Grid density;
    Grid nutrient; // same size as density
    bool operator==(const WorldState&) const = default;
};

// How the nutrient field starts (an experiment's [nutrient] section, docs/projects.md).
enum class NutrientStart {
    Uniform = 0,  // `level` everywhere
    Random = 1,   // independent values in [0, level), from the experiment's random seed
    Gradient = 2, // 0 at the left column rising linearly to `level` at the right one
};

struct NutrientSeed {
    NutrientStart start = NutrientStart::Uniform;
    float level = 1.0f; // in [0, 1]
    bool operator==(const NutrientSeed&) const = default;
};

// The starting nutrient field of a size x size world. Deterministic on every platform: Random
// draws from std::mt19937 (seeded with random_seed XOR 0x4e555452, so not the
// density's random stream) and maps the top 24 bits of each draw to
// [0, 1) itself rather than through std::uniform_real_distribution, whose algorithm the
// standard leaves to the implementation.
Grid seed_nutrient(const NutrientSeed& seed, int size, int random_seed);

// Summary of a field for inspection. Empty grids give zeros.
struct FieldStats {
    float min = 0.0f;
    float max = 0.0f;
    double mean = 0.0;
    double sum = 0.0; // "mass"
};
FieldStats field_stats(const Grid& field);

} // namespace lenia
