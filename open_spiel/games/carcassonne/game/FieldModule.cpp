#include "game.hpp"

#include <array>
#include <bitset>
#include <cassert>

namespace {

constexpr int U = 0;
constexpr int R = 1;
constexpr int D = 2;
constexpr int L = 3;

constexpr std::array<int, 4> dx = {0, 1, 0, -1};
constexpr std::array<int, 4> dy = {-1, 0, 1, 0};
constexpr std::array<int, 4> op = {D, L, U, R};

} // namespace

Field Field::operator+(const Field &other) const {
    Field res;
    res.city_edges = city_edges | other.city_edges;
    res.tile_mask = tile_mask | other.tile_mask;
    res.farmer_count[0] = farmer_count[0] + other.farmer_count[0];
    res.farmer_count[1] = farmer_count[1] + other.farmer_count[1];
    return res;
}

bool Field::hasFarmers() const { return farmer_count[0] != 0 || farmer_count[1] != 0; }

int Field::getTileCount() const { return static_cast<int>(tile_mask.count()); }

FieldModule::FieldModule() : fieldMap(std::plus<Field>{}) {}

int FieldModule::fieldIndex(int tile_id, int local) const { return (tile_id - 1) * MAX_TILE_FIELDS + local; }

void FieldModule::placeTileOnBoard(int tile_id, int x, int y, const Tile &tile, const BoardModule &board,
                                   const FeatureModule &features) {
    for (int f = 0; f < tile.field_count; ++f) {
        Field field;
        field.tile_mask.set(tile_id);
        for (int side = 0; side < 4; ++side) {
            if (!(tile.field_city_sides[f] & (1 << side))) {
                continue;
            }
            // The sides of one city are already joined; keep one of them.
            bool joined_lower_side = false;
            for (int lower = 0; lower < side; ++lower) {
                joined_lower_side |= (tile.field_city_sides[f] & (1 << lower)) && tile.link[lower] == tile.link[side];
            }
            if (!joined_lower_side) {
                field.city_edges.set(features.edgeIndex(tile_id, side));
            }
        }
        fieldMap.getSetData(fieldIndex(tile_id, f)) = field;
    }

    // Inner fields have no half-edge and never join another field.
    for (int e = 0; e < HALF_EDGE_COUNT; ++e) {
        if (tile.field[e] == -1) {
            continue;
        }
        int side = e / 2;
        int nx = x + dx[side];
        int ny = y + dy[side];
        if (!isInside(nx, ny) || board.board[ny][nx].id == 0) {
            continue;
        }
        const Placement &neighbour = board.board[ny][nx];
        // Placement already matched the terrain, so a field never meets a city.
        int their_field = full_deck[neighbour.id][neighbour.rotation].field[2 * op[side] + 1 - e % 2];
        assert(their_field != -1);
        fieldMap.unionSet(fieldIndex(tile_id, tile.field[e]), fieldIndex(neighbour.id, their_field));
    }
}

void FieldModule::getLegalFarmerMoves(MeepleMoves &ret, int tile_id, const Tile &tile) const {
    int seen_roots[HALF_EDGE_COUNT];
    int root_count = 0;
    for (int e = 0; e < HALF_EDGE_COUNT; ++e) {
        if (tile.field[e] == -1) {
            continue;
        }
        int root = fieldMap.find(fieldIndex(tile_id, tile.field[e]));
        bool seen = false;
        for (int j = 0; j < root_count; ++j) {
            if (seen_roots[j] == root) {
                seen = true;
                break;
            }
        }
        if (seen) {
            continue;
        }
        seen_roots[root_count++] = root;

        if (!fieldMap.getSetData(root).hasFarmers()) {
            ret.push_back(MEEPLE_POS_FIELD + e);
        }
    }
    int inner = tile.innerField();
    if (inner != -1 && !fieldMap.getSetData(fieldIndex(tile_id, inner)).hasFarmers()) {
        ret.push_back(MEEPLE_POS_INNER_FIELD);
    }
}

void FieldModule::placeFarmer(int tile_id, const Tile &tile, int pos, int player) {
    int local = pos == MEEPLE_POS_INNER_FIELD ? tile.innerField() : tile.field[pos - MEEPLE_POS_FIELD];
    assert(local >= 0 && local < tile.field_count);
    int slot = fieldIndex(tile_id, local);
    fieldMap.getSetData(slot).farmer_count[player]++;
    farmed_slots.push_back(static_cast<int16_t>(slot));
}

void FieldModule::getHalfEdgeGroups(int tile_id, const Tile &tile, int8_t groups[HALF_EDGE_COUNT]) const {
    int roots[HALF_EDGE_COUNT];
    for (int e = 0; e < HALF_EDGE_COUNT; ++e) {
        groups[e] = -1;
        if (tile.field[e] == -1) {
            continue;
        }
        roots[e] = fieldMap.find(fieldIndex(tile_id, tile.field[e]));
        groups[e] = static_cast<int8_t>(e);
        for (int j = 0; j < e; ++j) {
            if (groups[j] != -1 && roots[j] == roots[e]) {
                groups[e] = groups[j];
                break;
            }
        }
    }
}

CityCounts FieldModule::adjacentCities(const Field &field, const FeatureModule &features) const {
    CityCounts counts;
    if (field.city_edges.none()) {
        return counts;
    }
    // Cities that grew together are one city, counted once.
    std::bitset<EDGE_SLOT_COUNT> seen_roots;
    auto visit = [&](int slot) {
        int root = features.featureMap.find(slot);
        if (seen_roots[root]) {
            return;
        }
        seen_roots[root] = true;
        if (features.featureMap.getSetData(root).opens == 0) {
            counts.completed++;
        } else {
            counts.open++;
        }
    };
#if defined(__GLIBCXX__)
    // libstdc++ skips whole words of zeros; this runs for every observation.
    for (std::size_t slot = field.city_edges._Find_first(); slot < EDGE_SLOT_COUNT;
         slot = field.city_edges._Find_next(slot)) {
        visit(static_cast<int>(slot));
    }
#else
    for (int slot = 0; slot < EDGE_SLOT_COUNT; ++slot) {
        if (field.city_edges[slot]) {
            visit(slot);
        }
    }
#endif
    return counts;
}

void FieldModule::accumulateScore(int *scores, const FeatureModule &features) const {
    int seen_roots[MAX_FARMERS];
    int root_count = 0;
    for (int slot : farmed_slots) {
        int root = fieldMap.find(slot);
        bool seen = false;
        for (int j = 0; j < root_count; ++j) {
            if (seen_roots[j] == root) {
                seen = true;
                break;
            }
        }
        if (seen) {
            continue;
        }
        seen_roots[root_count++] = root;

        const Field &field = fieldMap.getSetData(root);
        int m0 = field.farmer_count[0];
        int m1 = field.farmer_count[1];
        if (m0 == 0 && m1 == 0) {
            continue;
        }
        int score = FIELD_POINTS_PER_CITY * adjacentCities(field, features).completed;
        if (m0 >= m1) {
            scores[0] += score;
        }
        if (m1 >= m0) {
            scores[1] += score;
        }
    }
}
