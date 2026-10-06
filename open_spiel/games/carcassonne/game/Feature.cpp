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
    meeple_count[0] = 0;
    meeple_count[1] = 0;
}

Feature Feature::operator+(const Feature &other) const {
    Feature res;
    res.type = type;
    res.tile_mask = tile_mask | other.tile_mask;
    res.meeple_count[0] = meeple_count[0] + other.meeple_count[0];
    res.meeple_count[1] = meeple_count[1] + other.meeple_count[1];
    res.big_meeples[0] = big_meeples[0] + other.big_meeples[0];
    res.big_meeples[1] = big_meeples[1] + other.big_meeples[1];
    res.builders[0] = builders[0] + other.builders[0];
    res.builders[1] = builders[1] + other.builders[1];
    res.opens = opens + other.opens;
    res.shields = shields + other.shields;
    res.inns = inns + other.inns;
    res.cathedrals = cathedrals + other.cathedrals;
    for (int kind = 0; kind < GOODS_KINDS; ++kind) {
        res.goods[kind] = goods[kind] + other.goods[kind];
    }
    return res;
}


bool Feature::hasMeeples() const { return meeple_count[0] != 0 || meeple_count[1] != 0; }

int Feature::getTileCount() const { return static_cast<int>(tile_mask.count()); }

int Feature::getScore() const {
    // Inns & Cathedrals: one inn or cathedral is enough, more change nothing.
    if (type == CITY && cathedrals > 0) {
        return opens == 0 ? 3 * (getTileCount() + shields) : 0;
    }
    if (type == ROAD && inns > 0) {
        return opens == 0 ? 2 * getTileCount() : 0;
    }
    return getBaseScore();
}

int Feature::getBaseScore() const {
    int tile_count = getTileCount();
    if (type == CITY) {
        if (opens == 0) {
            return (tile_count + shields) * 2;
        }
        return tile_count + shields;
    }
    return tile_count;
}