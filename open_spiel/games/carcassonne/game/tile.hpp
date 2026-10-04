#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>

using namespace std;

constexpr int START_TILE_TYPE = 20;
constexpr int START_TILE_ROTATION = 0;

// RIVER only meets RIVER. It splits fields like a road, but it is no feature:
// it never scores and takes no meeple.
enum EdgeType { NONE = 0, GRASS = 1, CITY = 2, ROAD = 3, RIVER = 4 };

// A side that belongs to a feature which scores and can hold a meeple.
constexpr bool isFeatureEdge(EdgeType edge) { return edge == CITY || edge == ROAD; }

// Which box each tile comes from. A game deals the base tiles plus the
// expansions it turns on (a bit mask of expansionBit()).
enum Expansion : uint8_t {
    EXP_BASE = 0,
    EXP_INNS_CATHEDRALS = 1,
    EXP_TRADERS_BUILDERS = 2,
    EXP_RIVER = 3,
    EXP_PRINCESS_DRAGON = 4,
};
constexpr int EXPANSION_COUNT = 5;
// Also the names of the game parameters that turn each expansion on.
constexpr const char *EXPANSION_NAMES[EXPANSION_COUNT] = {"base", "inns_cathedrals", "traders_builders", "river",
                                                          "princess_dragon"};
// How many tiles each box has, to check a finished table section against.
constexpr int OFFICIAL_TILE_COUNTS[EXPANSION_COUNT] = {72, 18, 24, 12, 30};
constexpr uint32_t expansionBit(Expansion expansion) { return 1u << expansion; }
constexpr uint32_t BASE_ONLY = 1u << EXP_BASE;
constexpr uint32_t ALL_EXPANSIONS = (1u << EXPANSION_COUNT) - 1;

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
    int8_t half_edge[HALF_EDGE_COUNT] = {};    // local field of each half-edge, -1 on a city side
    uint8_t count = 0;                         // local fields, the inner one (no half-edge) last
    uint8_t city_sides[MAX_TILE_FIELDS] = {};  // the sides of every city each field borders
};

// Marks on one city or road of a tile. Written on any side of that city or
// road (Tile::featureMarks() joins its sides); only the shield has a rule yet.
constexpr uint8_t MARK_SHIELD = 1 << 0;    // city: one more point, two once closed
constexpr uint8_t MARK_PRINCESS = 1 << 1;  // city (The Princess & the Dragon)
constexpr uint8_t MARK_WINE = 1 << 2;      // city goods (Traders & Builders)
constexpr uint8_t MARK_CLOTH = 1 << 3;
constexpr uint8_t MARK_WHEAT = 1 << 4;
constexpr uint8_t MARK_INN = 1 << 5;       // road (Inns & Cathedrals)
constexpr uint8_t GOODS_MARKS = MARK_WINE | MARK_CLOTH | MARK_WHEAT;
constexpr uint8_t CITY_MARKS = MARK_SHIELD | MARK_PRINCESS | GOODS_MARKS;
constexpr uint8_t ROAD_MARKS = MARK_INN;

// Marks on the whole tile. The cathedral needs none: it is one tile type.
constexpr uint8_t TILE_MONASTERY = 1 << 0;
constexpr uint8_t TILE_DRAGON = 1 << 1;   // The Princess & the Dragon; no rule yet
constexpr uint8_t TILE_VOLCANO = 1 << 2;
constexpr uint8_t TILE_PORTAL = 1 << 3;   // magic portal
constexpr uint8_t ALL_TILE_MARKS = TILE_MONASTERY | TILE_DRAGON | TILE_VOLCANO | TILE_PORTAL;

// MARK_* bits per side, in the order N, E, S, W.
struct SideMarks {
    uint8_t side[4] = {};
};

class Tile {
  public:
    EdgeType edge[4] = {NONE, NONE, NONE, NONE};
    int link[4] = {};
    uint8_t marks[4] = {};   // MARK_* bits as written on each side; see featureMarks()
    uint8_t tile_marks = 0;  // TILE_* bits
    bool monastery = false;  // TILE_MONASTERY
    int8_t field[HALF_EDGE_COUNT] = {};
    uint8_t field_count = 0;
    uint8_t field_city_sides[MAX_TILE_FIELDS] = {};

    constexpr Tile() = default;

    constexpr Tile(EdgeType e1, EdgeType e2, EdgeType e3, EdgeType e4, int l1, int l2, int l3, int l4,
                   const FieldLayout &fields, const SideMarks &side_marks = {}, uint8_t flags = 0) {
        edge[0] = e1;
        edge[1] = e2;
        edge[2] = e3;
        edge[3] = e4;
        link[0] = l1;
        link[1] = l2;
        link[2] = l3;
        link[3] = l4;
        for (int side = 0; side < 4; ++side) {
            marks[side] = side_marks.side[side];
        }
        tile_marks = flags;
        monastery = (flags & TILE_MONASTERY) != 0;
        for (int e = 0; e < HALF_EDGE_COUNT; ++e) {
            field[e] = fields.half_edge[e];
        }
        field_count = fields.count;
        for (int f = 0; f < MAX_TILE_FIELDS; ++f) {
            field_city_sides[f] = fields.city_sides[f];
        }
    }

    // The marks of the city or road on side `side`: those written on any of
    // its sides on this tile.
    constexpr uint8_t featureMarks(int side) const {
        uint8_t result = 0;
        for (int s = 0; s < 4; ++s) {
            if (link[s] == link[side]) {
                result |= marks[s];
            }
        }
        return result;
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
        for (int side = 0; side < 4; ++side) {
            res.marks[(side + 1) % 4] = marks[side];
        }
        res.tile_marks = tile_marks;
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
    Expansion expansion = EXP_BASE;
};

// One row per tile type, its canonical type being its row number + 1. The
// base types 1-24 come first and keep their numbers; expansion types follow.
// How to fill a row, with examples: docs/adding_tiles.md. Check the table with
// tools/dump_tiles (game/tile_check.hpp) and look at it drawn over the tile
// images with tools/render_tile_table.py.
//
// Each row: Tile(edges N E S W, links, fields, side marks, tile marks), count,
// type, expansion. Fields: the local field of half-edges 0..7, the number of
// fields, and the city sides each field borders. Side marks: MARK_* on a side
// of the city or road they belong to. Tile marks: TILE_*.
constexpr TileBlueprint all_tiles[] = {
    // ---- Base game: 72 tiles, types 1-24. ----
    {Tile(GRASS, GRASS, GRASS, GRASS, 0, 1, 2, 3, {{0, 0, 0, 0, 0, 0, 0, 0}, 1, {}}, {}, TILE_MONASTERY), 4, 1},
    // The road ends at the monastery, so the field runs round it.
    {Tile(GRASS, GRASS, ROAD, GRASS, 0, 1, 2, 3, {{0, 0, 0, 0, 0, 0, 0, 0}, 1, {}}, {}, TILE_MONASTERY), 2, 2},
    {Tile(CITY, CITY, CITY, CITY, 0, 0, 0, 0, {{-1, -1, -1, -1, -1, -1, -1, -1}, 0, {}}, {MARK_SHIELD, 0, 0, 0}), 1, 3},
    {Tile(CITY, CITY, GRASS, CITY, 0, 0, 1, 0, {{-1, -1, -1, -1, 0, 0, -1, -1}, 1, {SIDE_N | SIDE_E | SIDE_W}}), 3, 4},
    {Tile(CITY, CITY, GRASS, CITY, 0, 0, 1, 0, {{-1, -1, -1, -1, 0, 0, -1, -1}, 1, {SIDE_N | SIDE_E | SIDE_W}}, {MARK_SHIELD, 0, 0, 0}), 1, 5},
    // The road ends at the city gate, with a field on each side of it.
    {Tile(CITY, CITY, ROAD, CITY, 0, 0, 1, 0,
          {{-1, -1, -1, -1, 0, 1, -1, -1}, 2, {SIDE_N | SIDE_E | SIDE_W, SIDE_N | SIDE_E | SIDE_W}}), 1, 6},
    {Tile(CITY, CITY, ROAD, CITY, 0, 0, 1, 0,
          {{-1, -1, -1, -1, 0, 1, -1, -1}, 2, {SIDE_N | SIDE_E | SIDE_W, SIDE_N | SIDE_E | SIDE_W}}, {MARK_SHIELD, 0, 0, 0}), 2, 7},
    {Tile(CITY, GRASS, GRASS, CITY, 0, 1, 2, 0, {{-1, -1, 0, 0, 0, 0, -1, -1}, 1, {SIDE_N | SIDE_W}}), 3, 8},
    {Tile(CITY, GRASS, GRASS, CITY, 0, 1, 2, 0, {{-1, -1, 0, 0, 0, 0, -1, -1}, 1, {SIDE_N | SIDE_W}}, {MARK_SHIELD, 0, 0, 0}), 2, 9},
    // The inner corner of the bend touches no city.
    {Tile(CITY, ROAD, ROAD, CITY, 0, 1, 1, 0, {{-1, -1, 0, 1, 1, 0, -1, -1}, 2, {SIDE_N | SIDE_W}}), 3, 10},
    {Tile(CITY, ROAD, ROAD, CITY, 0, 1, 1, 0, {{-1, -1, 0, 1, 1, 0, -1, -1}, 2, {SIDE_N | SIDE_W}}, {MARK_SHIELD, 0, 0, 0}), 2, 11},
    {Tile(GRASS, CITY, GRASS, CITY, 0, 1, 2, 1, {{0, 0, -1, -1, 1, 1, -1, -1}, 2, {SIDE_E | SIDE_W, SIDE_E | SIDE_W}}), 1, 12},
    {Tile(GRASS, CITY, GRASS, CITY, 0, 1, 2, 1, {{0, 0, -1, -1, 1, 1, -1, -1}, 2, {SIDE_E | SIDE_W, SIDE_E | SIDE_W}},
          {0, MARK_SHIELD, 0, 0}), 2, 13},
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

    // The expansion sections follow the images in tiles/: type N is tiles/N.png.

    // ---- River: 12 tiles, types 25-34 (25 is the spring), rows end with EXP_RIVER. ----
    {Tile(GRASS, GRASS, RIVER, GRASS, 0, 1, 2, 3, {{0, 0, 0, 0, 0, 0, 0, 0}, 1, {}}), 1, 25, EXP_RIVER}, //source
    {Tile(RIVER, GRASS, GRASS, GRASS, 0, 1, 2, 3, {{0, 0, 0, 0, 0, 0, 0, 0}, 1, {}}), 1, 26, EXP_RIVER}, //end
    {Tile(RIVER, CITY, CITY, RIVER, 0, 1, 1, 0, {{0, 1, -1, -1, -1, -1, 1, 0}, 2, {0, SIDE_E | SIDE_S}}), 1, 27, EXP_RIVER},
    {Tile(RIVER, CITY, RIVER, ROAD, 0, 1, 0, 2, {{0, 1, -1, -1, 2, 3, 3, 0}, 4, {0, SIDE_E,SIDE_E,0}}), 1, 28, EXP_RIVER},
    {Tile(CITY, RIVER, CITY, RIVER, 0, 1, 2, 1, {{-1, -1, 0, 1, -1, -1, 1, 0}, 2, {SIDE_N, SIDE_S}}), 1, 29, EXP_RIVER},
    // Two of these: full_imgs/first/r3_c03 is the other one.
    {Tile(GRASS, RIVER, RIVER, GRASS, 0, 1, 1, 2, {{0, 0, 0, 1, 1, 0, 0, 0}, 2, {}}), 2, 30, EXP_RIVER},
    {Tile(GRASS, RIVER, ROAD, RIVER, 0, 1, 2, 1, {{0, 0, 0, 1, 1, 2, 2, 0}, 3, {}}, {}, TILE_MONASTERY), 1, 31, EXP_RIVER},
    // A road bend in the north-east corner, a river bend in the south-west one.
    {Tile(ROAD, ROAD, RIVER, RIVER, 0, 0, 1, 1, {{0, 1, 1, 0, 0, 2, 2, 0}, 3, {}}), 1, 32, EXP_RIVER},
    // The road crosses the river on a bridge: four corner fields.
    {Tile(ROAD, RIVER, ROAD, RIVER, 0, 1, 0, 1, {{0, 1, 1, 2, 2, 3, 3, 0}, 4, {}}), 1, 33, EXP_RIVER},
    // Two of these: full_imgs/first/r3_c07 is the other one. The island changes nothing.
    {Tile(RIVER, GRASS, RIVER, GRASS, 0, 1, 0, 2, {{0, 1, 1, 1, 1, 0, 0, 0}, 2, {}}), 2, 34, EXP_RIVER},

    // ---- Inns & Cathedrals: 18 tiles, types 35-51, rows end with EXP_INNS_CATHEDRALS. ----

    // ---- Traders & Builders: 24 tiles, types 52-75, rows end with EXP_TRADERS_BUILDERS. ----

    // ---- The Princess & the Dragon: 30 tiles, types 76-104, rows end with EXP_PRINCESS_DRAGON. ----
};

constexpr int CANONICAL_TILE_TYPE_COUNT = static_cast<int>(sizeof(all_tiles) / sizeof(all_tiles[0]));
constexpr int BASE_TILE_TYPE_COUNT = 24;

// Physical tiles in a deck of the expansions in `expansions`.
constexpr int tileCountIn(uint32_t expansions) {
    int total = 0;
    for (const TileBlueprint &bp : all_tiles) {
        if (expansions & expansionBit(bp.expansion)) {
            total += bp.count;
        }
    }
    return total;
}

constexpr int maxCopiesOfAType() {
    int most = 0;
    for (const TileBlueprint &bp : all_tiles) {
        most = std::max(most, bp.count);
    }
    return most;
}

constexpr int monasteryTileCount() {
    int total = 0;
    for (const TileBlueprint &bp : all_tiles) {
        total += bp.tile.monastery ? bp.count : 0;
    }
    return total;
}

// Row i is type i + 1, every type has a tile, and the base types come first.
constexpr bool tileTableIsNumbered() {
    for (int i = 0; i < CANONICAL_TILE_TYPE_COUNT; ++i) {
        const TileBlueprint &bp = all_tiles[i];
        if (bp.canonical_type != i + 1 || bp.count < 1 || (bp.expansion == EXP_BASE) != (i < BASE_TILE_TYPE_COUNT)) {
            return false;
        }
    }
    return true;
}

// Every tile of every expansion: the most a game can deal. Physical ids run
// 1..PHYSICAL_TILE_COUNT in table order, so the base tiles keep ids 1-72.
constexpr int PHYSICAL_TILE_COUNT = tileCountIn(ALL_EXPANSIONS);
constexpr int MAX_PHYSICAL_IDS_PER_TYPE = maxCopiesOfAType();
constexpr int MONASTERY_TILE_COUNT = monasteryTileCount();

static_assert(tileTableIsNumbered(), "all_tiles: row i must be canonical type i + 1 with count >= 1, base rows first");
static_assert(tileCountIn(BASE_ONLY) == 72, "the base rows must hold the 72 base tiles");
static_assert(PHYSICAL_TILE_COUNT <= 255, "Placement::id is a uint8_t");
static_assert(all_tiles[START_TILE_TYPE - 1].expansion == EXP_BASE, "the start tile is a base tile");

// A set of physical tiles, indexed by physical id.
using TileMask = std::bitset<PHYSICAL_TILE_COUNT + 1>;

const static auto full_deck = []() {
    std::array<std::array<Tile, 4>, PHYSICAL_TILE_COUNT + 1> result{};
    int current_id = 1;

    for (const auto &bp : all_tiles) {
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

    for (const auto &bp : all_tiles) {
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
