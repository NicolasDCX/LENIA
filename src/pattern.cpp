#include "lenia/pattern.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lenia {

namespace {

std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

float parse_float(const std::string& token, const std::string& key) {
    char* end = nullptr;
    const float v = std::strtof(token.c_str(), &end);
    if (token.empty() || end != token.c_str() + token.size()) {
        throw std::invalid_argument("pattern: bad number '" + token + "' for " + key);
    }
    return v;
}

KernelCore parse_core(const std::string& name) {
    if (name == "exponential") return KernelCore::ExponentialBump;
    if (name == "polynomial") return KernelCore::Polynomial;
    if (name == "step") return KernelCore::Step;
    if (name == "gaussian") return KernelCore::GaussianBell;
    throw std::invalid_argument("pattern: unknown core '" + name + "'");
}

GrowthFunction parse_growth(const std::string& name) {
    if (name == "gaussian") return GrowthFunction::Gaussian;
    if (name == "polynomial") return GrowthFunction::Polynomial;
    throw std::invalid_argument("pattern: unknown growth '" + name + "'");
}

Grid parse_cells(std::istream& in) {
    std::vector<std::vector<float>> rows;
    std::vector<float> row;
    std::string token;
    bool brackets = false;

    const auto flush_token = [&] {
        if (!token.empty()) {
            row.push_back(parse_float(token, "cells"));
            token.clear();
        }
    };
    const auto flush_row = [&] {
        flush_token();
        if (!row.empty()) rows.push_back(std::move(row));
        row.clear();
    };

    char c;
    bool comment = false;
    while (in.get(c)) {
        if (comment) {
            if (c == '\n') comment = false;
            else continue;
        }
        if (c == '#') {
            flush_token();
            comment = true;
        } else if (c == '[') {
            flush_token();
            brackets = true;
        } else if (c == ']') {
            flush_row();
        } else if (c == '\n') {
            if (brackets) flush_token();
            else flush_row();
        } else if (c == ',' || std::isspace(static_cast<unsigned char>(c))) {
            flush_token();
        } else {
            token += c;
        }
    }
    flush_row();

    std::size_t width = 0;
    for (const auto& r : rows) width = std::max(width, r.size());
    if (width == 0) {
        throw std::invalid_argument("pattern: empty cell array");
    }
    Grid cells(width, rows.size(), 0.0f);
    for (std::size_t y = 0; y < rows.size(); ++y) {
        std::copy(rows[y].begin(), rows[y].end(), cells.cells.begin() + static_cast<std::ptrdiff_t>(y * width));
    }
    return cells;
}

} // namespace

Pattern parse_pattern(std::istream& in) {
    Pattern p;
    std::string line;
    bool have_cells = false;
    while (std::getline(in, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        line = trim(line);
        if (line.empty()) continue;

        std::istringstream fields(line);
        std::string key;
        fields >> key;
        std::string rest;
        std::getline(fields, rest);
        rest = trim(rest);

        if (key == "cells") {
            // Anything after the keyword on the same line is the start of the array.
            std::string remaining = rest + "\n";
            remaining.append(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
            std::istringstream cells_in(remaining);
            p.cells = parse_cells(cells_in);
            have_cells = true;
            break;
        } else if (key == "name") {
            p.name = rest;
        } else if (key == "source") {
            p.source = rest;
        } else if (key == "license") {
            p.license = rest;
        } else if (key == "R") {
            const float r = parse_float(rest, key);
            if (r < 1.0f || r != static_cast<float>(static_cast<int>(r))) {
                throw std::invalid_argument("pattern: R must be a positive integer");
            }
            p.kernel.radius = static_cast<int>(r);
        } else if (key == "rings") {
            std::istringstream ws(rest);
            std::vector<float> rings;
            std::string w;
            while (ws >> w) rings.push_back(parse_float(w, key));
            if (rings.empty()) throw std::invalid_argument("pattern: rings needs at least one weight");
            p.kernel.rings = rings;
        } else if (key == "core") {
            p.kernel.core = parse_core(rest);
        } else if (key == "growth") {
            p.growth.function = parse_growth(rest);
        } else if (key == "mu") {
            p.growth.mu = parse_float(rest, key);
        } else if (key == "sigma") {
            p.growth.sigma = parse_float(rest, key);
        } else if (key == "dt") {
            p.growth.dt = parse_float(rest, key);
        } else if (key == "T") {
            const float t = parse_float(rest, key);
            if (!(t > 0.0f)) throw std::invalid_argument("pattern: T must be positive");
            p.growth.dt = 1.0f / t;
        } else {
            throw std::invalid_argument("pattern: unknown key '" + key + "'");
        }
    }
    if (!have_cells) {
        throw std::invalid_argument("pattern: missing cells");
    }
    return p;
}

Pattern load_pattern_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open pattern file: " + path);
    }
    return parse_pattern(in);
}

void stamp_pattern(Grid& world, const Grid& cells, std::size_t cx, std::size_t cy) {
    if (world.width == 0 || world.height == 0) return;
    const long long ox = static_cast<long long>(cx) - static_cast<long long>(cells.width / 2);
    const long long oy = static_cast<long long>(cy) - static_cast<long long>(cells.height / 2);
    const long long w = static_cast<long long>(world.width);
    const long long h = static_cast<long long>(world.height);
    for (std::size_t y = 0; y < cells.height; ++y) {
        for (std::size_t x = 0; x < cells.width; ++x) {
            const long long wx = ((ox + static_cast<long long>(x)) % w + w) % w;
            const long long wy = ((oy + static_cast<long long>(y)) % h + h) % h;
            world.at(static_cast<std::size_t>(wx), static_cast<std::size_t>(wy)) = cells.at(x, y);
        }
    }
}

} // namespace lenia
