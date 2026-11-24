#pragma once

#include "lenia/convolve.hpp"
#include "lenia/project.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace lenia {

// A simulation snapshot (docs/snapshots.md): the running world at one generation together with
// the experiment that continues it. Separate from experiment files (project.hpp), which store
// the setup only and restart at generation 0; the experiment inside a snapshot is the unchanged
// project format, so Reset after loading a snapshot still restarts from the experiment's seed.
//
// Everything a workbench generation depends on is either here or a pure function of the
// generation number (the node graph, the display graph), so stepping a loaded snapshot computes
// exactly the generations the original session would have computed next.

// Format version written by write_snapshot; parse_snapshot reads versions 1..kSnapshotVersion.
constexpr int kSnapshotVersion = 1;

struct Snapshot {
    Project project;              // the experiment, as an experiment file stores it
    std::uint64_t generation = 0; // generation `density` is at: the next step computes generation + 1
    Grid density;                 // project.world_size square, every cell finite
};

// The snapshot as bytes: a text header, the experiment exactly as write_project writes it, the
// density as raw little-endian IEEE 754 float32 (bit for bit) and a checksum. Throws
// std::invalid_argument if the density is not project.world_size square or holds a non-finite
// value, so nothing is written that parse_snapshot would reject.
std::string write_snapshot(const Snapshot& snapshot);

// Parses a snapshot. Strict, like parse_project: the "lenia-snapshot <version>" header, every
// key in order, byte counts that match, a valid embedded experiment (parse_project), a density
// of the experiment's world size with finite cells, a matching checksum and nothing after it.
// Throws std::invalid_argument saying what is wrong.
Snapshot parse_snapshot(std::string_view bytes);

// 64-bit FNV-1a, the snapshot checksum (over every byte before the checksum line).
std::uint64_t snapshot_checksum(std::string_view bytes);

} // namespace lenia
