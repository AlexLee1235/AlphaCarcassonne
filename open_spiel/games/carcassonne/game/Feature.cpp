#include "game.hpp"

#include <array>
#include <bitset>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

constexpr int U = 0;
constexpr int R = 1;
constexpr int D = 2;
constexpr int L = 3;

constexpr std::array<int, 4> dx = {0, 1, 0, -1};
constexpr std::array<int, 4> dy = {-1, 0, 1, 0};
constexpr std::array<int, 4> op = {D, L, U, R};

} // namespace


Feature::Feature(EdgeType feature_type, int id) {
    type = feature_type;
    tile_mask.set(id);
    opens = 1;
    meeple_mask[0] = 0;
    meeple_mask[1] = 0;
}

Feature Feature::operator+(const Feature &other) const {
    Feature res;
    res.type = type;
    res.tile_mask = tile_mask | other.tile_mask;
    res.meeple_mask[0] = meeple_mask[0] | other.meeple_mask[0];
    res.meeple_mask[1] = meeple_mask[1] | other.meeple_mask[1];
    res.opens = opens + other.opens;
    return res;
}

int Feature::countMeeples(uint8_t mask) {
    int count = 0;
    while (mask != 0) {
        count += mask & 1;
        mask >>= 1;
    }
    return count;
}

int Feature::meepleCount(int player) const { return countMeeples(meeple_mask[player]); }

bool Feature::hasMeeples() const { return meeple_mask[0] != 0 || meeple_mask[1] != 0; }

int Feature::getTileCount() const { return static_cast<int>(tile_mask.count()); }

int Feature::getScore() const {
    int tile_count = getTileCount();
    if (type == CITY) {
        int shield_count = static_cast<int>((tile_mask & SHIELD_MASK).count());
        if (opens == 0) {
            return (tile_count + shield_count) * 2;
        }
        return tile_count + shield_count;
    }
    return tile_count;
}