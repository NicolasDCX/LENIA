// Simulation snapshots (snapshot.hpp): a snapshot survives write -> parse bit for bit (density,
// generation, the whole experiment); continuing a parsed snapshot under a time-modulated graph
// reproduces the uninterrupted run exactly, and only because the generation is restored; and
// damaged, inconsistent or malformed files are rejected with std::invalid_argument.

#include "lenia/display_graph.hpp"
#include "lenia/node_graph.hpp"
#include "lenia/pattern.hpp"
#include "lenia/project.hpp"
#include "lenia/snapshot.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAILED: %s\n", what);
        ++failures;
    }
}

bool same(const lenia::Grid& a, const lenia::Grid& b) {
    if (a.width != b.width || a.height != b.height || a.cells.size() != b.cells.size()) return false;
    for (std::size_t i = 0; i < a.cells.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(a.cells[i]) != std::bit_cast<std::uint32_t>(b.cells[i])) return false;
    }
    return true;
}

void expect_rejected(const std::string& bytes, const char* what) {
    try {
        lenia::parse_snapshot(bytes);
        std::printf("FAILED: accepted %s\n", what);
        ++failures;
    } catch (const std::invalid_argument& e) {
        std::printf("  rejected %-44s %s\n", what, e.what());
    } catch (const std::exception& e) {
        std::printf("FAILED: %s threw a non-invalid_argument exception: %s\n", what, e.what());
        ++failures;
    }
}

void expect_write_rejected(const lenia::Snapshot& s, const char* what) {
    try {
        lenia::write_snapshot(s);
        std::printf("FAILED: wrote %s\n", what);
        ++failures;
    } catch (const std::invalid_argument& e) {
        std::printf("  refused to write %-36s %s\n", what, e.what());
    }
}

std::string replace(std::string s, const std::string& from, const std::string& to) {
    const auto at = s.find(from);
    if (at == std::string::npos) {
        std::printf("FAILED: test setup, '%s' not found\n", from.c_str());
        ++failures;
        return s;
    }
    return s.replace(at, from.size(), to);
}

// Recomputes the checksum line after an edit, so the parser gets past it to the check under test.
std::string reseal(const std::string& bytes) {
    const auto at = bytes.rfind("checksum fnv1a64 ");
    const std::string body = bytes.substr(0, at);
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(lenia::snapshot_checksum(body)));
    return body + "checksum fnv1a64 " + hex + "\n";
}

// Offset of the first density byte.
std::size_t density_offset(const std::string& bytes) { return bytes.find("f32le\n") + 6; }

// The line of `text` starting with `prefix` replaced by `with`.
std::string replace_line(const std::string& text, const std::string& prefix, const std::string& with) {
    std::size_t start = 0;
    if (text.rfind(prefix, 0) != 0) {
        const auto at = text.find("\n" + prefix);
        if (at == std::string::npos) {
            std::printf("FAILED: test setup, no line starts with '%s'\n", prefix.c_str());
            ++failures;
            return text;
        }
        start = at + 1;
    }
    return text.substr(0, start) + with + text.substr(text.find('\n', start));
}

// The snapshot with its embedded experiment replaced (byte count updated, checksum resealed).
std::string with_experiment(const std::string& bytes, const std::string& experiment) {
    const auto line = bytes.find("\nexperiment ") + 1;
    const auto data = bytes.find('\n', line) + 1;
    const std::size_t size = std::stoull(bytes.substr(line + 11, data - 1 - (line + 11)));
    return reseal(bytes.substr(0, line) + "experiment " + std::to_string(experiment.size()) + "\n" + experiment +
                  bytes.substr(data + size));
}

} // namespace

int main() {
    using namespace lenia;

    Pattern orbium;
    try {
        orbium = load_pattern_file(std::string(LENIA_PATTERN_DIR) + "/orbium.lenia");
    } catch (const std::exception& ex) {
        std::printf("FAILED: %s\n", ex.what());
        return 1;
    }

    // A time-modulated experiment: the breathing kernel warps the kernel every generation, a
    // pulsing environment adds a spatial term, and the display graph is animated too.
    Project p;
    p.kernel = orbium.kernel;
    p.growth = orbium.growth;
    p.world_size = 48;
    p.seed = SeedKind::Pattern;
    p.pattern = orbium;
    p.graph = make_preset(GraphPreset::BreathingKernel);
    p.graph.set_param(4, environment_node::Strength, 0.01f);
    p.display_graph = make_display_preset(DisplayPreset::GrowthOverlay);

    Grid start(48, 48, 0.0f);
    stamp_pattern(start, orbium.cells, 24, 24);

    // --- Roundtrip: every bit of the density, the generation and the experiment. -------------
    {
        Snapshot s;
        s.project = p;
        s.generation = (std::uint64_t{1} << 40) + 3;
        s.density = start;
        s.density.at(0, 0) = -0.0f;
        s.density.at(1, 0) = std::numeric_limits<float>::denorm_min();
        s.density.at(2, 0) = 0.123456789f;
        s.density.at(3, 0) = 7.5f; // rules may leave [0, 1]; any finite value is a state
        const std::string bytes = write_snapshot(s);
        try {
            const Snapshot r = parse_snapshot(bytes);
            expect(r.generation == s.generation, "generation roundtrips (beyond 32 bits)");
            expect(same(r.density, s.density), "density bit for bit (-0, denormal, > 1 included)");
            expect(write_project(r.project) == write_project(p), "embedded experiment roundtrips");
            expect(write_snapshot(r) == bytes, "write(parse(write(s))) == write(s)");
        } catch (const std::exception& ex) {
            std::printf("FAILED: roundtrip parse: %s\n", ex.what());
            ++failures;
        }
        s.generation = UINT64_MAX;
        expect(parse_snapshot(write_snapshot(s)).generation == UINT64_MAX, "the largest generation roundtrips");
    }

    // --- Continuation: a parsed snapshot continues the run exactly, time modulation included. -
    {
        expect(!(evaluate(p.graph, 17).kernel == evaluate(p.graph, 0).kernel),
               "test setup: the kernel warp depends on the generation");
        constexpr std::uint64_t kSnap = 17, kEnd = 30;
        Grid uninterrupted = start;
        Snapshot saved;
        for (std::uint64_t g = 0; g < kEnd; ++g) {
            if (g == kSnap) saved = {p, g, uninterrupted};
            uninterrupted = step_graph(uninterrupted, p.graph, p.kernel, p.growth, g);
        }
        const Snapshot loaded = parse_snapshot(write_snapshot(saved));
        Grid resumed = loaded.density;
        for (std::uint64_t g = loaded.generation; g < kEnd; ++g) {
            resumed = step_graph(resumed, loaded.project.graph, loaded.project.kernel, loaded.project.growth, g);
        }
        expect(same(resumed, uninterrupted), "snapshot at 17 continued to 30 equals the uninterrupted run bit for bit");

        // The generation is part of the state: restarting the counter at 0 diverges.
        Grid restarted = loaded.density;
        for (std::uint64_t g = 0; g < kEnd - kSnap; ++g) {
            restarted = step_graph(restarted, loaded.project.graph, loaded.project.kernel, loaded.project.growth, g);
        }
        expect(!same(restarted, uninterrupted), "continuing from generation 0 instead diverges (time modulation)");
    }

    // --- Writing refuses snapshots it could not read back. -----------------------------------
    {
        Snapshot s{p, 5, start};
        s.density.at(10, 10) = std::numeric_limits<float>::quiet_NaN();
        expect_write_rejected(s, "a NaN density cell");
        s.density = Grid(32, 32, 0.0f);
        expect_write_rejected(s, "a density not of the world size");
    }

    // --- Invalid files. -----------------------------------------------------------------------
    const std::string base = write_snapshot({p, 123, start});
    try {
        parse_snapshot(base);
    } catch (const std::exception& ex) {
        std::printf("FAILED: base snapshot: %s\n", ex.what());
        return 1;
    }
    std::printf("Invalid files:\n");
    const std::string header = "lenia-snapshot " + std::to_string(kSnapshotVersion);
    const std::string experiment = write_project(p);
    const std::string experiment_line = "experiment " + std::to_string(experiment.size()) + "\n";
    expect(parse_snapshot(with_experiment(base, experiment)).generation == 123, "test setup: with_experiment");
    expect_rejected("", "an empty file");
    expect_rejected(experiment, "an experiment file");
    expect_rejected(replace(base, header, "lenia-snapshot " + std::to_string(kSnapshotVersion + 1)),
                    "a newer format version");
    expect_rejected(replace(base, header, "lenia-snapshot 0"), "format version 0");
    expect_rejected(replace(base, "generation 123", "generation -1"), "a negative generation");
    expect_rejected(replace(base, "generation 123", "generation 12x"), "a malformed generation");
    expect_rejected(replace(base, "generation 123", "generation 18446744073709551616"), "a generation past 2^64 - 1");
    expect_rejected(replace(base, "generation 123\n", ""), "a missing generation");
    expect_rejected(replace(base, "generation 123", "generation 123 4"), "an extra value");
    expect_rejected(replace(base, experiment_line, "experiment 9999999\n"), "an experiment size past the data");
    expect_rejected(replace(base, experiment_line, "experiment 100\n"), "an experiment size too short");
    expect_rejected(replace(base, experiment_line, "experiment 0\n"), "an experiment size of 0");
    expect_rejected(with_experiment(base, replace_line(experiment, "mu ", "mu nan")), "an invalid embedded experiment");
    expect_rejected(with_experiment(base, replace_line(experiment, "lenia-project", "lenia-project 99")),
                    "an embedded experiment too new");
    {
        Project bigger = p;
        bigger.world_size = 64;
        expect_rejected(with_experiment(base, write_project(bigger)), "an experiment of another world size");
    }
    expect_rejected(reseal(replace(base, "density 48 48 f32le", "density 48 48 f16le")), "an unknown encoding");
    expect_rejected(reseal(replace(base, "density 48 48", "density 24 96")), "a density not of the world size");
    expect_rejected(base.substr(0, density_offset(base) + 1000), "a truncated density");
    expect_rejected(base.substr(0, base.size() - 1), "a missing final newline");
    {
        std::string flipped = base;
        flipped[density_offset(base) + 4 * (24 * 48 + 24) + 1] ^= 0x10;
        expect_rejected(flipped, "a changed density byte (checksum)");
        std::string nan = base;
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::quiet_NaN());
        for (int i = 0; i < 4; ++i) nan[density_offset(base) + i] = static_cast<char>((bits >> (8 * i)) & 0xff);
        expect_rejected(reseal(nan), "a NaN density cell");
    }
    expect_rejected(replace(base, "checksum fnv1a64", "checksum crc32"), "an unknown checksum");
    expect_rejected(base + "x", "bytes after the checksum");
    {
        std::string crlf;
        for (const char c : base) crlf += c == '\n' ? std::string("\r\n") : std::string(1, c);
        expect_rejected(crlf, "line endings converted to CRLF");
    }

    if (failures == 0) std::printf("All snapshot tests passed.\n");
    return failures == 0 ? 0 : 1;
}
