#pragma once

#include <QColor>
#include <QStringList>

#include <vector>

namespace workbench {

// Lookup tables used by workbench_view.frag and the kernel editor. Each palette has
// kPaletteSize entries, linear interpolation between sRGB key colors.
constexpr int kPaletteSize = 256;

enum Palette { Grayscale, Viridis, Inferno, RedBlue, IceFire, PaletteCount };

QStringList palette_names();

// All palettes back to back, kPaletteSize RGBA entries each (sRGB-encoded floats in [0, 1]):
// the layout of the Palettes buffer in workbench_view.frag.
std::vector<float> palette_table();

// t in [0, 1], clamped.
QColor palette_color(Palette palette, float t);

} // namespace workbench
