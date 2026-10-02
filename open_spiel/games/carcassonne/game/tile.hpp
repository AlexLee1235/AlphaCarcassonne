#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>

using namespace std;

constexpr int PHYSICAL_TILE_COUNT = 72;
constexpr int CANONICAL_TILE_TYPE_COUNT = 24;
constexpr int MAX_PHYSICAL_IDS_PER_TYPE = 9;
constexpr int START_TILE_TYPE = 20;
constexpr int START_TILE_ROTATION = 0;

enum EdgeType { NONE = 0, GRASS = 1, CITY = 2, ROAD = 3 };

// Fields (farms). Each side has two halves; half-edge e = 2 * side + h runs
// clockwise round the tile: 0 N-west, 1 N-east, 2 E-north, 3 E-south,
// 4 S-east, 5 S-west, 6 W-south, 7 W-north. A quarter turn clockwise moves e to
// (e + 2) % 8, and half h of a side meets half 1 - h of the neighbour's
// opposite side.
constexpr int HALF_EDGE_COUNT = 8;
constexpr int MAX_TILE_FIELDS = 4; // RRRR
// A tile may also have one inner field, which touches no half-edge, such as
// grass ringed by four separate cities. None in the base deck.

// Side bits for FieldLayout::city_sides.
constexpr uint8_t SIDE_N = 1 << 0;
constexpr uint8_t SIDE_E = 1 << 1;
constexpr uint8_t SIDE_S = 1 << 2;
constexpr uint8_t SIDE_W = 1 << 3;

struct FieldLayout {
    int8_t half_edge[HALF_EDGE_COUNT];   // local field of each half-edge, -1 on a city side
    uint8_t count;                       // local fields, the inner one (no half-edge) last
    uint8_t city_sides[MAX_TILE_FIELDS]; // the sides of every city each field borders
};

class Tile {
  public:
    EdgeType edge[4];
    int link[4];
    bool shield;
    bool monastery;
    int8_t field[HALF_EDGE_COUNT];
    uint8_t field_count;
    uint8_t field_city_sides[MAX_TILE_FIELDS];

    Tile() = default;

    Tile(EdgeType e1, EdgeType e2, EdgeType e3, EdgeType e4, int l1, int l2, int l3, int l4, const FieldLayout &fields,
         bool sh = false, bool mo = false) {
        edge[0] = e1;
        edge[1] = e2;
        edge[2] = e3;
        edge[3] = e4;
        link[0] = l1;
        link[1] = l2;
        link[2] = l3;
        link[3] = l4;
        shield = sh;
        monastery = mo;
        for (int e = 0; e < HALF_EDGE_COUNT; ++e) {
            field[e] = fields.half_edge[e];
        }
        field_count = fields.count;
        for (int f = 0; f < MAX_TILE_FIELDS; ++f) {
            field_city_sides[f] = fields.city_sides[f];
        }
    }

    // The local field that touches no half-edge (always the last one), or -1.
    int innerField() const {
        int edge_fields = 0;
        for (int e = 0; e < HALF_EDGE_COUNT; ++e) {
            edge_fields = std::max(edge_fields, field[e] + 1);
        }
        return edge_fields < field_count ? edge_fields : -1;
    }

    Tile rotate() const {
        Tile res;
        res.edge[0] = edge[3];
        res.edge[1] = edge[0];
        res.edge[2] = edge[1];
        res.edge[3] = edge[2];
        res.link[0] = link[3];
        res.link[1] = link[0];
        res.link[2] = link[1];
        res.link[3] = link[2];
        res.shield = shield;
        res.monastery = monastery;
        for (int e = 0; e < HALF_EDGE_COUNT; ++e) {
            res.field[(e + 2) % HALF_EDGE_COUNT] = field[e];
        }
        res.field_count = field_count;
        for (int f = 0; f < MAX_TILE_FIELDS; ++f) {
            res.field_city_sides[f] = static_cast<uint8_t>(((field_city_sides[f] << 1) | (field_city_sides[f] >> 3)) & 0xF);
        }
        return res;
    }
};

struct TileBlueprint {
    Tile tile;
    int count;
    int canonical_type;
};

// Fields of each tile: the local field of half-edges 0..7, the number of
// fields, and the city sides each field borders.
const static TileBlueprint base_deck[] = {
    {Tile(GRASS, GRASS, GRASS, GRASS, 0, 1, 2, 3, {{0, 0, 0, 0, 0, 0, 0, 0}, 1, {}}, false, true), 4, 1},
    // The road ends at the monastery, so the field runs round it.
    {Tile(GRASS, GRASS, ROAD, GRASS, 0, 1, 2, 3, {{0, 0, 0, 0, 0, 0, 0, 0}, 1, {}}, false, true), 2, 2},
    {Tile(CITY, CITY, CITY, CITY, 0, 0, 0, 0, {{-1, -1, -1, -1, -1, -1, -1, -1}, 0, {}}, true), 1, 3},
    {Tile(CITY, CITY, GRASS, CITY, 0, 0, 1, 0, {{-1, -1, -1, -1, 0, 0, -1, -1}, 1, {SIDE_N | SIDE_E | SIDE_W}}), 3, 4},
    {Tile(CITY, CITY, GRASS, CITY, 0, 0, 1, 0, {{-1, -1, -1, -1, 0, 0, -1, -1}, 1, {SIDE_N | SIDE_E | SIDE_W}}, true), 1, 5},
    // The road ends at the city gate, with a field on each side of it.
    {Tile(CITY, CITY, ROAD, CITY, 0, 0, 1, 0,
          {{-1, -1, -1, -1, 0, 1, -1, -1}, 2, {SIDE_N | SIDE_E | SIDE_W, SIDE_N | SIDE_E | SIDE_W}}), 1, 6},
    {Tile(CITY, CITY, ROAD, CITY, 0, 0, 1, 0,
          {{-1, -1, -1, -1, 0, 1, -1, -1}, 2, {SIDE_N | SIDE_E | SIDE_W, SIDE_N | SIDE_E | SIDE_W}}, true), 2, 7},
    {Tile(CITY, GRASS, GRASS, CITY, 0, 1, 2, 0, {{-1, -1, 0, 0, 0, 0, -1, -1}, 1, {SIDE_N | SIDE_W}}), 3, 8},
    {Tile(CITY, GRASS, GRASS, CITY, 0, 1, 2, 0, {{-1, -1, 0, 0, 0, 0, -1, -1}, 1, {SIDE_N | SIDE_W}}, true), 2, 9},
    // The inner corner of the bend touches no city.
    {Tile(CITY, ROAD, ROAD, CITY, 0, 1, 1, 0, {{-1, -1, 0, 1, 1, 0, -1, -1}, 2, {SIDE_N | SIDE_W}}), 3, 10},
    {Tile(CITY, ROAD, ROAD, CITY, 0, 1, 1, 0, {{-1, -1, 0, 1, 1, 0, -1, -1}, 2, {SIDE_N | SIDE_W}}, true), 2, 11},
    {Tile(GRASS, CITY, GRASS, CITY, 0, 1, 2, 1, {{0, 0, -1, -1, 1, 1, -1, -1}, 2, {SIDE_E | SIDE_W, SIDE_E | SIDE_W}}), 1, 12},
    {Tile(GRASS, CITY, GRASS, CITY, 0, 1, 2, 1, {{0, 0, -1, -1, 1, 1, -1, -1}, 2, {SIDE_E | SIDE_W, SIDE_E | SIDE_W}}, true),
     2, 13},
    // Two separate cities, both next to the one field.
    {Tile(CITY, GRASS, GRASS, CITY, 0, 1, 2, 3, {{-1, -1, 0, 0, 0, 0, -1, -1}, 1, {SIDE_N | SIDE_W}}), 2, 14},
    {Tile(CITY, GRASS, CITY, GRASS, 0, 1, 2, 3, {{-1, -1, 0, 0, -1, -1, 0, 0}, 1, {SIDE_N | SIDE_S}}), 3, 15},
    {Tile(CITY, GRASS, GRASS, GRASS, 0, 1, 2, 3, {{-1, -1, 0, 0, 0, 0, 0, 0}, 1, {SIDE_N}}), 5, 16},
    {Tile(CITY, GRASS, ROAD, ROAD, 0, 1, 2, 2, {{-1, -1, 0, 0, 0, 1, 1, 0}, 2, {SIDE_N}}), 3, 17},
    {Tile(CITY, ROAD, ROAD, GRASS, 0, 1, 1, 2, {{-1, -1, 0, 1, 1, 0, 0, 0}, 2, {SIDE_N}}), 3, 18},
    {Tile(CITY, ROAD, ROAD, ROAD, 0, 1, 2, 3, {{-1, -1, 0, 1, 1, 2, 2, 0}, 3, {SIDE_N}}), 3, 19},
    {Tile(CITY, ROAD, GRASS, ROAD, 0, 1, 2, 1, {{-1, -1, 0, 1, 1, 1, 1, 0}, 2, {SIDE_N}}), 4, 20},
    {Tile(ROAD, GRASS, ROAD, GRASS, 0, 1, 0, 2, {{0, 1, 1, 1, 1, 0, 0, 0}, 2, {}}), 8, 21},
    {Tile(GRASS, GRASS, ROAD, ROAD, 0, 1, 2, 2, {{0, 0, 0, 0, 0, 1, 1, 0}, 2, {}}), 9, 22},
    {Tile(GRASS, ROAD, ROAD, ROAD, 0, 1, 2, 3, {{0, 0, 0, 1, 1, 2, 2, 0}, 3, {}}), 4, 23},
    {Tile(ROAD, ROAD, ROAD, ROAD, 0, 1, 2, 3, {{0, 1, 1, 2, 2, 3, 3, 0}, 4, {}}), 1, 24},
};

const static auto full_deck = []() {
    std::array<std::array<Tile, 4>, PHYSICAL_TILE_COUNT + 1> result{};
    int current_id = 1;

    for (const auto &bp : base_deck) {
        for (int count = 0; count < bp.count; ++count) {
            result[current_id][0] = bp.tile;
            result[current_id][1] = result[current_id][0].rotate();
            result[current_id][2] = result[current_id][1].rotate();
            result[current_id][3] = result[current_id][2].rotate();
            current_id++;
        }
    }

    return result;
}();

const static auto PHYSICAL_TO_CANONICAL_TYPE = []() {
    std::array<int, PHYSICAL_TILE_COUNT + 1> result{};
    int current_id = 1;

    for (const auto &bp : base_deck) {
        for (int count = 0; count < bp.count; ++count)
            result[current_id++] = bp.canonical_type;
    }

    return result;
}();

struct TileTypeTables {
    std::array<int, PHYSICAL_TILE_COUNT + 1> canonical_type_by_physical_id{};
    std::array<std::array<int, MAX_PHYSICAL_IDS_PER_TYPE>, CANONICAL_TILE_TYPE_COUNT + 1> draw_physical_ids_by_type{};
    std::array<int, CANONICAL_TILE_TYPE_COUNT + 1> draw_count_by_type{};
};

const static auto tile_type_tables = []() {
    TileTypeTables tables{};
    tables.canonical_type_by_physical_id = PHYSICAL_TO_CANONICAL_TYPE;

    for (int physical_id = 1; physical_id <= PHYSICAL_TILE_COUNT; ++physical_id) {
        int canonical_type = PHYSICAL_TO_CANONICAL_TYPE[physical_id];
        int slot = tables.draw_count_by_type[canonical_type]++;
        tables.draw_physical_ids_by_type[canonical_type][slot] = physical_id;
    }

    return tables;
}();

const static std::bitset<73> SHIELD_MASK = []() {
    std::bitset<73> mask;
    for (int i = 1; i <= PHYSICAL_TILE_COUNT; ++i) {
        if (full_deck[i][0].shield)
            mask.set(i);
    }
    return mask;
}();
