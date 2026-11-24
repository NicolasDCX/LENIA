#include "palettes.hpp"

#include "lenia/display_graph.hpp"

namespace workbench {

namespace {

// The key colors live in lenia (display_graph.cpp), shared with Color ramp presets.
static_assert(PaletteCount == lenia::kDisplayPaletteCount);
static_assert(kPaletteSize == lenia::kPaletteTableSize);

std::array<float, 3> sample(Palette palette, float t) {
    return lenia::palette_sample(static_cast<lenia::DisplayPalette>(palette), t);
}

} // namespace

QStringList palette_names() {
    return {QStringLiteral("Grayscale"), QStringLiteral("Viridis"), QStringLiteral("Inferno"),
            QStringLiteral("Blue-white-red (signed)"), QStringLiteral("Ice-fire (signed)")};
}

std::vector<float> palette_table() {
    std::vector<float> table;
    table.reserve(static_cast<std::size_t>(PaletteCount) * kPaletteSize * 4);
    for (int p = 0; p < PaletteCount; ++p) {
        for (int i = 0; i < kPaletteSize; ++i) {
            const auto rgb = sample(static_cast<Palette>(p), static_cast<float>(i) / (kPaletteSize - 1));
            table.insert(table.end(), {rgb[0], rgb[1], rgb[2], 1.0f});
        }
    }
    return table;
}

QColor palette_color(Palette palette, float t) {
    const auto rgb = sample(palette, t);
    return QColor::fromRgbF(rgb[0], rgb[1], rgb[2]);
}

} // namespace workbench
