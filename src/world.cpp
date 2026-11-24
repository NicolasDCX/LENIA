#include "lenia/world.hpp"

#include <algorithm>
#include <random>

namespace lenia {

namespace {
constexpr std::uint32_t kNutrientStream = 0x4e555452u; // "NUTR"
} // namespace

Grid seed_nutrient(const NutrientSeed& seed, int size, int random_seed) {
    const auto side = static_cast<std::size_t>(std::max(size, 0));
    const float level = std::clamp(seed.level, 0.0f, 1.0f);
    Grid g(side, side, 0.0f);
    switch (seed.start) {
    case NutrientStart::Uniform:
        std::fill(g.cells.begin(), g.cells.end(), level);
        break;
    case NutrientStart::Random: {
        // Not the density's stream (the workbench seeds that with random_seed itself).
        std::mt19937 rng(static_cast<std::uint32_t>(random_seed) ^ kNutrientStream);
        for (float& c : g.cells) c = level * static_cast<float>(rng() >> 8) * (1.0f / 16777216.0f);
        break;
    }
    case NutrientStart::Gradient:
        for (std::size_t y = 0; y < side; ++y) {
            for (std::size_t x = 0; x < side; ++x) {
                g.at(x, y) = side > 1 ? level * static_cast<float>(x) / static_cast<float>(side - 1) : level;
            }
        }
        break;
    }
    return g;
}

FieldStats field_stats(const Grid& field) {
    FieldStats s;
    if (field.cells.empty()) return s;
    s.min = s.max = field.cells.front();
    for (const float v : field.cells) {
        s.min = std::min(s.min, v);
        s.max = std::max(s.max, v);
        s.sum += v;
    }
    s.mean = s.sum / static_cast<double>(field.cells.size());
    return s;
}

} // namespace lenia
