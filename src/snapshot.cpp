#include "lenia/snapshot.hpp"

#include <bit>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

namespace lenia {

namespace {

// Sanity limits, as in project.cpp: a corrupt header cannot request huge allocations. The
// density's size is also checked against the bytes actually present before allocating.
constexpr std::size_t kMaxExperimentBytes = 64u * 1024u * 1024u;
constexpr int kMaxSide = 8192;
// Header lines are short; a longer "line" means binary data where text was expected.
constexpr std::size_t kMaxLine = 256;

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument("snapshot: " + message); }

std::string hex64(std::uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
    return buf;
}

void append_f32le(std::string& out, float v) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(v);
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<char>((bits >> (8 * i)) & 0xffu));
}

float read_f32le(const char* p) {
    std::uint32_t bits = 0;
    for (int i = 0; i < 4; ++i) bits |= static_cast<std::uint32_t>(static_cast<unsigned char>(p[i])) << (8 * i);
    return std::bit_cast<float>(bits);
}

// Sequential reader over the snapshot bytes.
class Reader {
public:
    explicit Reader(std::string_view bytes) : bytes_(bytes) {}

    std::size_t pos() const { return pos_; }
    std::size_t remaining() const { return bytes_.size() - pos_; }

    // The next text line, split at spaces, skipping comment lines ('#'). `what` names the
    // expected line in errors.
    std::vector<std::string> line(const char* what) {
        for (;;) {
            if (pos_ >= bytes_.size()) fail(std::string("the file ends before the ") + what + " line");
            const std::size_t end = bytes_.find('\n', pos_);
            if (end == std::string_view::npos || end - pos_ > kMaxLine) {
                fail(std::string("expected the ") + what + " line at byte " + std::to_string(pos_));
            }
            const std::string_view text = bytes_.substr(pos_, end - pos_);
            pos_ = end + 1;
            if (!text.empty() && text.front() == '#') continue;
            std::vector<std::string> tokens;
            std::istringstream fields{std::string(text)};
            std::string token;
            while (fields >> token) tokens.push_back(token);
            if (tokens.empty()) fail(std::string("expected the ") + what + " line, got an empty line");
            return tokens;
        }
    }

    std::string_view take(std::size_t n, const char* what) {
        if (n > remaining()) {
            fail(std::string(what) + " is cut short: " + std::to_string(n) + " bytes expected, " +
                 std::to_string(remaining()) + " present");
        }
        const std::string_view s = bytes_.substr(pos_, n);
        pos_ += n;
        return s;
    }

private:
    std::string_view bytes_;
    std::size_t pos_ = 0;
};

void expect_key(const std::vector<std::string>& tokens, const char* key, std::size_t values) {
    if (tokens[0] != key) fail(std::string("expected '") + key + "', got '" + tokens[0] + "'");
    if (tokens.size() != values + 1) {
        fail(std::string(key) + " takes " + std::to_string(values) + (values == 1 ? " value" : " values"));
    }
}

template <class T> T parse_number(const std::string& token, const char* what, T lo, T hi) {
    T v{};
    const auto [end, ec] = std::from_chars(token.data(), token.data() + token.size(), v);
    if (ec != std::errc() || end != token.data() + token.size()) {
        fail(std::string(what) + " '" + token + "' is not a whole number");
    }
    if (v < lo || v > hi) fail(std::string(what) + " '" + token + "' is out of range");
    return v;
}

} // namespace

std::uint64_t snapshot_checksum(std::string_view bytes) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (const char c : bytes) {
        h ^= static_cast<unsigned char>(c);
        h *= 0x100000001b3ull;
    }
    return h;
}

std::string write_snapshot(const Snapshot& s) {
    const auto side = static_cast<std::size_t>(s.project.world_size);
    const auto check_field = [&](const Grid& g, const char* name) {
        if (g.width != side || g.height != side || g.cells.size() != side * side) {
            fail(std::string("the ") + name + " is " + std::to_string(g.width) + " x " + std::to_string(g.height) +
                 " but the experiment's world is " + std::to_string(side) + " x " + std::to_string(side));
        }
        for (std::size_t i = 0; i < g.cells.size(); ++i) {
            if (!std::isfinite(g.cells[i])) {
                fail(std::string(name) + " cell (" + std::to_string(i % g.width) + ", " +
                     std::to_string(i / g.width) + ") is not finite");
            }
        }
    };
    const Grid& d = s.density;
    check_field(d, "density");
    const Grid n = s.nutrient.cells.empty() ? project_nutrient(s.project) : s.nutrient;
    check_field(n, "nutrient");
    const std::string experiment = write_project(s.project);

    std::string out;
    out.reserve(experiment.size() + 2 * d.cells.size() * 4 + 256);
    out += "lenia-snapshot " + std::to_string(kSnapshotVersion) + "\n";
    out += "# Lenia workbench simulation snapshot (docs/snapshots.md); binary data follows, do not edit\n";
    out += "generation " + std::to_string(s.generation) + "\n";
    out += "experiment " + std::to_string(experiment.size()) + "\n";
    out += experiment;
    for (const auto& [name, g] : {std::pair<const char*, const Grid*>{"density", &d}, {"nutrient", &n}}) {
        out += std::string(name) + " " + std::to_string(g->width) + " " + std::to_string(g->height) + " f32le\n";
        for (const float v : g->cells) append_f32le(out, v);
        out += '\n';
    }
    const auto checksum = snapshot_checksum(out);
    out += "checksum fnv1a64 " + hex64(checksum) + "\n";
    return out;
}

Snapshot parse_snapshot(std::string_view bytes) {
    Reader r(bytes);

    const std::vector<std::string> header = r.line("header");
    if (header[0] != "lenia-snapshot" || header.size() != 2) {
        fail("not a Lenia snapshot (the first line must be 'lenia-snapshot <version>')");
    }
    const int version = parse_number<int>(header[1], "format version", 1, 1 << 30);
    if (version > kSnapshotVersion) {
        fail("format version " + header[1] + " is newer than this workbench reads (up to " +
             std::to_string(kSnapshotVersion) + ")");
    }

    Snapshot s;
    const std::vector<std::string> generation = r.line("generation");
    expect_key(generation, "generation", 1);
    s.generation = parse_number<std::uint64_t>(generation[1], "generation", 0, UINT64_MAX);

    const std::vector<std::string> experiment = r.line("experiment");
    expect_key(experiment, "experiment", 1);
    const auto experiment_bytes = parse_number<std::size_t>(experiment[1], "experiment size", 1, kMaxExperimentBytes);
    {
        std::istringstream in{std::string(r.take(experiment_bytes, "the experiment"))};
        try {
            s.project = parse_project(in);
        } catch (const std::invalid_argument& e) {
            fail(std::string("embedded experiment: ") + e.what());
        }
    }

    // A field block: "<name> W H f32le", W x H floats, a newline.
    const auto read_field = [&](const char* name) {
        const std::string n(name);
        const std::vector<std::string> head = r.line(name);
        expect_key(head, name, 3);
        const int w = parse_number<int>(head[1], (n + " width").c_str(), 1, kMaxSide);
        const int h = parse_number<int>(head[2], (n + " height").c_str(), 1, kMaxSide);
        if (head[3] != "f32le") fail("unknown " + n + " encoding '" + head[3] + "' (expected f32le)");
        if (w != s.project.world_size || h != s.project.world_size) {
            fail("the " + n + " is " + head[1] + " x " + head[2] + " but the experiment's world is " +
                 std::to_string(s.project.world_size) + " x " + std::to_string(s.project.world_size));
        }
        const std::size_t count = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
        const std::string_view cells = r.take(count * 4, ("the " + n).c_str());
        Grid g(static_cast<std::size_t>(w), static_cast<std::size_t>(h));
        for (std::size_t i = 0; i < count; ++i) {
            const float v = read_f32le(cells.data() + 4 * i);
            if (!std::isfinite(v)) {
                fail(n + " cell (" + std::to_string(i % g.width) + ", " + std::to_string(i / g.width) +
                     ") is not finite");
            }
            g.cells[i] = v;
        }
        if (r.take(1, ("the line after the " + n).c_str()) != "\n") fail("the " + n + " is longer than its size says");
        return g;
    };
    s.density = read_field("density");
    // Version 1 has no nutrient: its experiment's start, which nothing in a version 1 graph reads.
    s.nutrient = version >= 2 ? read_field("nutrient") : project_nutrient(s.project);

    // Everything up to here is covered by the checksum.
    const std::uint64_t expected = snapshot_checksum(bytes.substr(0, r.pos()));
    const std::vector<std::string> checksum = r.line("checksum");
    expect_key(checksum, "checksum", 2);
    if (checksum[1] != "fnv1a64") fail("unknown checksum '" + checksum[1] + "' (expected fnv1a64)");
    if (checksum[2] != hex64(expected)) {
        fail("checksum mismatch (the file is damaged or was edited): stored " + checksum[2] + ", computed " +
             hex64(expected));
    }
    if (r.remaining() != 0) fail(std::to_string(r.remaining()) + " unexpected bytes after the checksum");
    return s;
}

} // namespace lenia
