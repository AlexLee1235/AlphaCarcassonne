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
    // Four separate cities round an inner field.
    {Tile(CITY, CITY, CITY, CITY, 0, 1, 2, 3,
          {{-1, -1, -1, -1, -1, -1, -1, -1}, 1, {SIDE_N | SIDE_E | SIDE_S | SIDE_W}}), 1, 35, EXP_INNS_CATHEDRALS},
    // The cathedral: two in the box.
    {Tile(CITY, CITY, CITY, CITY, 0, 0, 0, 0, {{-1, -1, -1, -1, -1, -1, -1, -1}, 0, {}}), 2, 36, EXP_INNS_CATHEDRALS},
    // Three separate cities, grass east.
    {Tile(CITY, GRASS, CITY, CITY, 0, 1, 2, 3, {{-1, -1, 0, 0, -1, -1, -1, -1}, 1, {SIDE_N | SIDE_S | SIDE_W}}),
     1, 37, EXP_INNS_CATHEDRALS},
    // A north-west corner city with the shield, a separate city south.
    {Tile(CITY, GRASS, CITY, CITY, 0, 1, 2, 0, {{-1, -1, 0, 0, -1, -1, -1, -1}, 1, {SIDE_N | SIDE_S | SIDE_W}},
          {MARK_SHIELD, 0, 0, 0}), 1, 38, EXP_INNS_CATHEDRALS},
    // A north-west corner city; a road leaves its gate south, the inn's lake beside it.
    {Tile(CITY, GRASS, ROAD, CITY, 0, 1, 2, 0, {{-1, -1, 0, 0, 0, 1, -1, -1}, 2, {SIDE_N | SIDE_W, SIDE_N | SIDE_W}},
          {0, 0, MARK_INN, 0}), 1, 39, EXP_INNS_CATHEDRALS},
    // A north-west corner city; a road leaves its gate east.
    {Tile(CITY, ROAD, GRASS, CITY, 0, 1, 2, 0, {{-1, -1, 0, 1, 1, 1, -1, -1}, 2, {SIDE_N | SIDE_W, SIDE_N | SIDE_W}}),
     1, 40, EXP_INNS_CATHEDRALS},
    // Type 11 (corner city with the shield, road bend east to south) with an inn.
    {Tile(CITY, ROAD, ROAD, CITY, 0, 1, 1, 0, {{-1, -1, 0, 1, 1, 0, -1, -1}, 2, {SIDE_N | SIDE_W}},
          {MARK_SHIELD, MARK_INN, 0, 0}), 1, 41, EXP_INNS_CATHEDRALS},
    // A city on the east side only, but its walls run corner to corner, so it
    // splits the grass in two. Looks like type 16 to the observation, whose
    // grass is one field (docs/carcassonne_field_observation.md §2.2).
    {Tile(GRASS, CITY, GRASS, GRASS, 0, 1, 2, 3, {{0, 0, -1, -1, 1, 1, 0, 0}, 2, {SIDE_E, SIDE_E}}), 1, 42,
     EXP_INNS_CATHEDRALS},
    // A city north; a road leaves its gate south.
    {Tile(CITY, GRASS, ROAD, GRASS, 0, 1, 2, 3, {{-1, -1, 0, 0, 0, 1, 1, 1}, 2, {SIDE_N, SIDE_N}}), 1, 43,
     EXP_INNS_CATHEDRALS},
    // Type 17 (city north, road bend south to west) with an inn.
    {Tile(CITY, GRASS, ROAD, ROAD, 0, 1, 2, 2, {{-1, -1, 0, 0, 0, 1, 1, 0}, 2, {SIDE_N}}, {0, 0, MARK_INN, 0}), 1, 44,
     EXP_INNS_CATHEDRALS},
    // One city east to west with the shield; a road ends at it from the north
    // and another from the south.
    {Tile(ROAD, CITY, ROAD, CITY, 0, 1, 2, 1,
          {{0, 1, -1, -1, 2, 3, -1, -1}, 4, {SIDE_E | SIDE_W, SIDE_E | SIDE_W, SIDE_E | SIDE_W, SIDE_E | SIDE_W}},
          {0, MARK_SHIELD, 0, 0}), 1, 45, EXP_INNS_CATHEDRALS},
    // Type 22 (road bend south to west) with an inn.
    {Tile(GRASS, GRASS, ROAD, ROAD, 0, 1, 2, 2, {{0, 0, 0, 0, 0, 1, 1, 0}, 2, {}}, {0, 0, MARK_INN, 0}), 1, 46,
     EXP_INNS_CATHEDRALS},
    // A monastery the road runs past, east to west.
    {Tile(GRASS, ROAD, GRASS, ROAD, 0, 1, 2, 3, {{0, 0, 0, 1, 1, 1, 1, 0}, 2, {}}, {}, TILE_MONASTERY), 1, 47,
     EXP_INNS_CATHEDRALS},
    // A straight road east to west with an inn.
    {Tile(GRASS, ROAD, GRASS, ROAD, 0, 1, 2, 1, {{0, 0, 0, 1, 1, 1, 1, 0}, 2, {}}, {0, MARK_INN, 0, 0}), 1, 48,
     EXP_INNS_CATHEDRALS},
    // Type 23 (junction east, south, west) with an inn on the east road.
    {Tile(GRASS, ROAD, ROAD, ROAD, 0, 1, 2, 3, {{0, 0, 0, 1, 1, 2, 2, 0}, 3, {}}, {0, MARK_INN, 0, 0}), 1, 49,
     EXP_INNS_CATHEDRALS},
    // Two road bends, north to west and east to south.
    {Tile(ROAD, ROAD, ROAD, ROAD, 0, 1, 1, 0, {{0, 1, 1, 2, 2, 1, 1, 0}, 3, {}}), 1, 50, EXP_INNS_CATHEDRALS},
    // Separate cities east and west; roads from north and south meet two short
    // roads to the city gates. Those two touch no side, so the engine has no
    // feature for them.
    {Tile(ROAD, CITY, ROAD, CITY, 0, 1, 2, 3, {{0, 1, -1, -1, 2, 3, -1, -1}, 4, {SIDE_W, SIDE_E, SIDE_E, SIDE_W}}),
     1, 51, EXP_INNS_CATHEDRALS},

    // ---- Traders & Builders: 24 tiles, types 52-75, rows end with EXP_TRADERS_BUILDERS. ----
    // A north-west corner city (cloth) and caps east and south round an inner field.
    {Tile(CITY, CITY, CITY, CITY, 0, 1, 2, 0,
          {{-1, -1, -1, -1, -1, -1, -1, -1}, 1, {SIDE_N | SIDE_E | SIDE_S | SIDE_W}}, {MARK_CLOTH, 0, 0, 0}), 1, 52,
     EXP_TRADERS_BUILDERS},
    // Corner cities north-west (wine) and south-east; the grass between runs
    // corner to corner and touches no side.
    {Tile(CITY, CITY, CITY, CITY, 0, 1, 1, 0,
          {{-1, -1, -1, -1, -1, -1, -1, -1}, 1, {SIDE_N | SIDE_E | SIDE_S | SIDE_W}}, {MARK_WINE, 0, 0, 0}), 1, 53,
     EXP_TRADERS_BUILDERS},
    // One city north to south (wine), grass west, a separate cap east. The
    // grass between that city and the cap touches no side: an inner field.
    {Tile(CITY, CITY, CITY, GRASS, 0, 1, 0, 2,
          {{-1, -1, -1, -1, -1, -1, 0, 0}, 2, {SIDE_N | SIDE_S, SIDE_N | SIDE_E | SIDE_S}}, {MARK_WINE, 0, 0, 0}), 1, 54,
     EXP_TRADERS_BUILDERS},
    // A cap north; one city east to west (cloth); grass south. The grass
    // between the cap and that city touches no side: an inner field.
    {Tile(CITY, CITY, GRASS, CITY, 0, 1, 2, 1,
          {{-1, -1, -1, -1, 0, 0, -1, -1}, 2, {SIDE_E | SIDE_W, SIDE_N | SIDE_E | SIDE_W}}, {0, MARK_CLOTH, 0, 0}), 1,
     55, EXP_TRADERS_BUILDERS},
    // One city north, south and west (wheat), grass east.
    {Tile(CITY, GRASS, CITY, CITY, 0, 1, 0, 0, {{-1, -1, 0, 0, -1, -1, -1, -1}, 1, {SIDE_N | SIDE_S | SIDE_W}},
          {MARK_WHEAT, 0, 0, 0}), 1, 56, EXP_TRADERS_BUILDERS},
    // A cap north, a south-east corner city (wheat); a road from the west ends
    // at its gate.
    {Tile(CITY, CITY, CITY, ROAD, 0, 1, 1, 2,
          {{-1, -1, -1, -1, -1, -1, 0, 1}, 2, {SIDE_E | SIDE_S, SIDE_N | SIDE_E | SIDE_S}}, {0, MARK_WHEAT, 0, 0}), 1,
     57, EXP_TRADERS_BUILDERS},
    // A south-west corner city (cloth) with a road from the north at its gate; a cap east.
    {Tile(ROAD, CITY, CITY, CITY, 0, 1, 2, 2,
          {{0, 1, -1, -1, -1, -1, -1, -1}, 2, {SIDE_S | SIDE_W, SIDE_E | SIDE_S | SIDE_W}}, {0, 0, MARK_CLOTH, 0}), 1,
     58, EXP_TRADERS_BUILDERS},
    // Type 6 (city north, east and west, a road south from its gate) with wine.
    {Tile(CITY, CITY, ROAD, CITY, 0, 0, 1, 0,
          {{-1, -1, -1, -1, 0, 1, -1, -1}, 2, {SIDE_N | SIDE_E | SIDE_W, SIDE_N | SIDE_E | SIDE_W}},
          {MARK_WINE, 0, 0, 0}), 1, 59, EXP_TRADERS_BUILDERS},
    // Type 8 (north-west corner city) with wine.
    {Tile(CITY, GRASS, GRASS, CITY, 0, 1, 2, 0, {{-1, -1, 0, 0, 0, 0, -1, -1}, 1, {SIDE_N | SIDE_W}},
          {MARK_WINE, 0, 0, 0}), 1, 60, EXP_TRADERS_BUILDERS},
    // A south-east corner city with wheat.
    {Tile(GRASS, CITY, CITY, GRASS, 0, 1, 1, 2, {{0, 0, -1, -1, -1, -1, 0, 0}, 1, {SIDE_E | SIDE_S}},
          {0, MARK_WHEAT, 0, 0}), 1, 61, EXP_TRADERS_BUILDERS},
    // A city east and south (wheat), a road from the north at its gate. Its
    // wall reaches the north-west corner, so the grass west is a field apart.
    {Tile(ROAD, CITY, CITY, GRASS, 0, 1, 1, 2,
          {{0, 1, -1, -1, -1, -1, 2, 2}, 3, {SIDE_E | SIDE_S, SIDE_E | SIDE_S, SIDE_E | SIDE_S}}, {0, MARK_WHEAT, 0, 0}),
     1, 62, EXP_TRADERS_BUILDERS},
    // A south-west corner city (wheat), a road from the east at its gate.
    {Tile(GRASS, ROAD, CITY, CITY, 0, 1, 2, 2, {{0, 0, 0, 1, -1, -1, -1, -1}, 2, {SIDE_S | SIDE_W, SIDE_S | SIDE_W}},
          {0, 0, MARK_WHEAT, 0}), 1, 63, EXP_TRADERS_BUILDERS},
    // A north-east corner city (cloth), a road south from its gate.
    {Tile(CITY, CITY, ROAD, GRASS, 0, 0, 1, 2, {{-1, -1, -1, -1, 0, 1, 1, 1}, 2, {SIDE_N | SIDE_E, SIDE_N | SIDE_E}},
          {MARK_CLOTH, 0, 0, 0}), 1, 64, EXP_TRADERS_BUILDERS},
    // A city south and west (wine), a road from the north at its gate. Its
    // wall reaches the north-east corner, so the grass east is a field apart.
    {Tile(ROAD, GRASS, CITY, CITY, 0, 1, 2, 2,
          {{0, 1, 2, 2, -1, -1, -1, -1}, 3, {SIDE_S | SIDE_W, SIDE_S | SIDE_W, SIDE_S | SIDE_W}}, {0, 0, MARK_WINE, 0}),
     1, 65, EXP_TRADERS_BUILDERS},
    // A city south and west (wine); roads from the north and from the east end at its gates.
    {Tile(ROAD, ROAD, CITY, CITY, 0, 1, 2, 2,
          {{0, 1, 1, 2, -1, -1, -1, -1}, 3, {SIDE_S | SIDE_W, SIDE_S | SIDE_W, SIDE_S | SIDE_W}}, {0, 0, MARK_WINE, 0}),
     1, 66, EXP_TRADERS_BUILDERS},
    // A city north and east (cloth); roads from the south and from the west end
    // at its gates. Its wall reaches the south-west corner: four fields.
    {Tile(CITY, CITY, ROAD, ROAD, 0, 0, 1, 2,
          {{-1, -1, -1, -1, 0, 1, 2, 3}, 4, {SIDE_N | SIDE_E, SIDE_N | SIDE_E, SIDE_N | SIDE_E, SIDE_N | SIDE_E}},
          {MARK_CLOTH, 0, 0, 0}), 1, 67, EXP_TRADERS_BUILDERS},
    // One city north to south (wine), grass east and west.
    {Tile(CITY, GRASS, CITY, GRASS, 0, 1, 0, 2, {{-1, -1, 0, 0, -1, -1, 1, 1}, 2, {SIDE_N | SIDE_S, SIDE_N | SIDE_S}},
          {MARK_WINE, 0, 0, 0}), 1, 68, EXP_TRADERS_BUILDERS},
    // One city north to south (wheat), a road from the east at its gate, grass west.
    {Tile(CITY, ROAD, CITY, GRASS, 0, 1, 0, 2,
          {{-1, -1, 0, 1, -1, -1, 2, 2}, 3, {SIDE_N | SIDE_S, SIDE_N | SIDE_S, SIDE_N | SIDE_S}}, {MARK_WHEAT, 0, 0, 0}),
     1, 69, EXP_TRADERS_BUILDERS},
    // The same with wine.
    {Tile(CITY, ROAD, CITY, GRASS, 0, 1, 0, 2,
          {{-1, -1, 0, 1, -1, -1, 2, 2}, 3, {SIDE_N | SIDE_S, SIDE_N | SIDE_S, SIDE_N | SIDE_S}}, {MARK_WINE, 0, 0, 0}),
     1, 70, EXP_TRADERS_BUILDERS},
    // A road from the north crosses the west road on a bridge and ends at a
    // house; the west road ends at the gate of a cap east.
    {Tile(ROAD, CITY, GRASS, ROAD, 0, 1, 2, 3, {{0, 1, -1, -1, 2, 2, 2, 0}, 3, {0, SIDE_E, SIDE_E}}), 1, 71,
     EXP_TRADERS_BUILDERS},
    // One city north to south (wine); roads from east and west end at its gates.
    {Tile(CITY, ROAD, CITY, ROAD, 0, 1, 0, 2,
          {{-1, -1, 0, 1, -1, -1, 2, 3}, 4, {SIDE_N | SIDE_S, SIDE_N | SIDE_S, SIDE_N | SIDE_S, SIDE_N | SIDE_S}},
          {MARK_WINE, 0, 0, 0}), 1, 72, EXP_TRADERS_BUILDERS},
    // A cap north; a road leaves its gate and bends east.
    {Tile(CITY, ROAD, GRASS, GRASS, 0, 1, 2, 3, {{-1, -1, 0, 1, 1, 1, 1, 1}, 2, {SIDE_N, SIDE_N}}), 1, 73,
     EXP_TRADERS_BUILDERS},
    // Roads from north, south and west end at a monastery; like type 23, they
    // split the grass in three.
    {Tile(ROAD, GRASS, ROAD, ROAD, 0, 1, 2, 3, {{0, 1, 1, 1, 1, 2, 2, 0}, 3, {}}, {}, TILE_MONASTERY), 1, 74,
     EXP_TRADERS_BUILDERS},
    // Two roads crossing on a bridge, north-south and east-west: four corner fields.
    {Tile(ROAD, ROAD, ROAD, ROAD, 0, 1, 0, 1, {{0, 1, 1, 2, 2, 3, 3, 0}, 4, {}}), 1, 75, EXP_TRADERS_BUILDERS},

    // ---- The Princess & the Dragon: 30 tiles, types 76-104, rows end with EXP_PRINCESS_DRAGON. ----
    // One city east to west; roads from the north and from the south end in
    // the grass before they reach it, so the grass runs round each: two fields.
    {Tile(ROAD, CITY, ROAD, CITY, 0, 1, 2, 1, {{0, 0, -1, -1, 1, 1, -1, -1}, 2, {SIDE_E | SIDE_W, SIDE_E | SIDE_W}}, {},
          TILE_DRAGON), 1, 76, EXP_PRINCESS_DRAGON},
    // Type 13 (city east to west with the shield).
    {Tile(GRASS, CITY, GRASS, CITY, 0, 1, 2, 1, {{0, 0, -1, -1, 1, 1, -1, -1}, 2, {SIDE_E | SIDE_W, SIDE_E | SIDE_W}},
          {0, MARK_SHIELD, 0, 0}, TILE_DRAGON), 1, 77, EXP_PRINCESS_DRAGON},
    // Type 4 (city north, east and west) with a monastery inside the city.
    {Tile(CITY, CITY, GRASS, CITY, 0, 0, 1, 0, {{-1, -1, -1, -1, 0, 0, -1, -1}, 1, {SIDE_N | SIDE_E | SIDE_W}}, {},
          TILE_MONASTERY | TILE_DRAGON), 1, 78, EXP_PRINCESS_DRAGON},
    // Type 8 (north-west corner city).
    {Tile(CITY, GRASS, GRASS, CITY, 0, 1, 2, 0, {{-1, -1, 0, 0, 0, 0, -1, -1}, 1, {SIDE_N | SIDE_W}}, {}, TILE_DRAGON),
     1, 79, EXP_PRINCESS_DRAGON},
    // Type 17 (cap north, road bend south to west).
    {Tile(CITY, GRASS, ROAD, ROAD, 0, 1, 2, 2, {{-1, -1, 0, 0, 0, 1, 1, 0}, 2, {SIDE_N}}, {}, TILE_DRAGON), 1, 80,
     EXP_PRINCESS_DRAGON},
    // Type 18 (cap north, road bend east to south).
    {Tile(CITY, ROAD, ROAD, GRASS, 0, 1, 1, 2, {{-1, -1, 0, 1, 1, 0, 0, 0}, 2, {SIDE_N}}, {}, TILE_DRAGON), 1, 81,
     EXP_PRINCESS_DRAGON},
    // Type 16 (cap north); the garden changes nothing.
    {Tile(CITY, GRASS, GRASS, GRASS, 0, 1, 2, 3, {{-1, -1, 0, 0, 0, 0, 0, 0}, 1, {SIDE_N}}, {}, TILE_DRAGON), 1, 82,
     EXP_PRINCESS_DRAGON},
    // Type 23 (roads east, south and west end at a village).
    {Tile(GRASS, ROAD, ROAD, ROAD, 0, 1, 2, 3, {{0, 0, 0, 1, 1, 2, 2, 0}, 3, {}}, {}, TILE_DRAGON), 1, 83,
     EXP_PRINCESS_DRAGON},
    // Roads east, south and west end at a monastery; like type 74.
    {Tile(GRASS, ROAD, ROAD, ROAD, 0, 1, 2, 3, {{0, 0, 0, 1, 1, 2, 2, 0}, 3, {}}, {}, TILE_MONASTERY | TILE_DRAGON), 1,
     84, EXP_PRINCESS_DRAGON},
    // Type 22 (road bend south to west). Two: second edition r8_c08 is the other one.
    {Tile(GRASS, GRASS, ROAD, ROAD, 0, 1, 2, 2, {{0, 0, 0, 0, 0, 1, 1, 0}, 2, {}}, {}, TILE_DRAGON), 2, 85,
     EXP_PRINCESS_DRAGON},
    // Type 21 (straight road north to south).
    {Tile(ROAD, GRASS, ROAD, GRASS, 0, 1, 0, 2, {{0, 1, 1, 1, 1, 0, 0, 0}, 2, {}}, {}, TILE_DRAGON), 1, 86,
     EXP_PRINCESS_DRAGON},
    // Volcano; type 22 (road bend south to west).
    {Tile(GRASS, GRASS, ROAD, ROAD, 0, 1, 2, 2, {{0, 0, 0, 0, 0, 1, 1, 0}, 2, {}}, {}, TILE_VOLCANO), 1, 87,
     EXP_PRINCESS_DRAGON},
    // Volcano; type 21 (straight road north to south).
    {Tile(ROAD, GRASS, ROAD, GRASS, 0, 1, 0, 2, {{0, 1, 1, 1, 1, 0, 0, 0}, 2, {}}, {}, TILE_VOLCANO), 1, 88,
     EXP_PRINCESS_DRAGON},
    // Volcano; a road from the south ends at it, so the grass runs round.
    {Tile(GRASS, GRASS, ROAD, GRASS, 0, 1, 2, 3, {{0, 0, 0, 0, 0, 0, 0, 0}, 1, {}}, {}, TILE_VOLCANO), 1, 89,
     EXP_PRINCESS_DRAGON},
    // Volcano alone.
    {Tile(GRASS, GRASS, GRASS, GRASS, 0, 1, 2, 3, {{0, 0, 0, 0, 0, 0, 0, 0}, 1, {}}, {}, TILE_VOLCANO), 1, 90,
     EXP_PRINCESS_DRAGON},
    // Volcano; type 16 (cap north).
    {Tile(CITY, GRASS, GRASS, GRASS, 0, 1, 2, 3, {{-1, -1, 0, 0, 0, 0, 0, 0}, 1, {SIDE_N}}, {}, TILE_VOLCANO), 1, 91,
     EXP_PRINCESS_DRAGON},
    // Volcano; type 14 (separate caps north and west).
    {Tile(CITY, GRASS, GRASS, CITY, 0, 1, 2, 3, {{-1, -1, 0, 0, 0, 0, -1, -1}, 1, {SIDE_N | SIDE_W}}, {},
          TILE_VOLCANO), 1, 92, EXP_PRINCESS_DRAGON},
    // Type 4 (city north, east and west) with the princess.
    {Tile(CITY, CITY, GRASS, CITY, 0, 0, 1, 0, {{-1, -1, -1, -1, 0, 0, -1, -1}, 1, {SIDE_N | SIDE_E | SIDE_W}},
          {MARK_PRINCESS, 0, 0, 0}), 1, 93, EXP_PRINCESS_DRAGON},
    // A north-west corner city (princess) whose wall reaches the south-east
    // corner, so the grass east and the grass south are two fields.
    {Tile(CITY, GRASS, GRASS, CITY, 0, 1, 2, 0, {{-1, -1, 0, 0, 1, 1, -1, -1}, 2, {SIDE_N | SIDE_W, SIDE_N | SIDE_W}},
          {MARK_PRINCESS, 0, 0, 0}), 1, 94, EXP_PRINCESS_DRAGON},
    // A city east (princess) and a separate south-west corner city (shield);
    // grass north, and between the two cities a diagonal strip of grass that
    // touches no side: an inner field.
    {Tile(GRASS, CITY, CITY, CITY, 0, 1, 2, 2,
          {{0, 0, -1, -1, -1, -1, -1, -1}, 2, {SIDE_E, SIDE_E | SIDE_S | SIDE_W}}, {0, MARK_PRINCESS, MARK_SHIELD, 0}),
     1, 95, EXP_PRINCESS_DRAGON},
    // Type 8 (north-west corner city) with the princess; the garden changes nothing.
    {Tile(CITY, GRASS, GRASS, CITY, 0, 1, 2, 0, {{-1, -1, 0, 0, 0, 0, -1, -1}, 1, {SIDE_N | SIDE_W}},
          {MARK_PRINCESS, 0, 0, 0}), 1, 96, EXP_PRINCESS_DRAGON},
    // Type 10 (corner city, road bend east to south) with the princess.
    {Tile(CITY, ROAD, ROAD, CITY, 0, 1, 1, 0, {{-1, -1, 0, 1, 1, 0, -1, -1}, 2, {SIDE_N | SIDE_W}},
          {MARK_PRINCESS, 0, 0, 0}), 1, 97, EXP_PRINCESS_DRAGON},
    // Type 19 (cap north, roads east, south and west meet) with the princess.
    {Tile(CITY, ROAD, ROAD, ROAD, 0, 1, 2, 3, {{-1, -1, 0, 1, 1, 2, 2, 0}, 3, {SIDE_N}}, {MARK_PRINCESS, 0, 0, 0}), 1,
     98, EXP_PRINCESS_DRAGON},
    // Magic portal; type 4 (city north, east and west).
    {Tile(CITY, CITY, GRASS, CITY, 0, 0, 1, 0, {{-1, -1, -1, -1, 0, 0, -1, -1}, 1, {SIDE_N | SIDE_E | SIDE_W}}, {},
          TILE_PORTAL), 1, 99, EXP_PRINCESS_DRAGON},
    // Magic portal; type 10 (corner city, road bend east to south).
    {Tile(CITY, ROAD, ROAD, CITY, 0, 1, 1, 0, {{-1, -1, 0, 1, 1, 0, -1, -1}, 2, {SIDE_N | SIDE_W}}, {}, TILE_PORTAL),
     1, 100, EXP_PRINCESS_DRAGON},
    // Magic portal; type 17 (cap north, road bend south to west).
    {Tile(CITY, GRASS, ROAD, ROAD, 0, 1, 2, 2, {{-1, -1, 0, 0, 0, 1, 1, 0}, 2, {SIDE_N}}, {}, TILE_PORTAL), 1, 101,
     EXP_PRINCESS_DRAGON},
    // Magic portal; type 18 (cap north, road bend east to south).
    {Tile(CITY, ROAD, ROAD, GRASS, 0, 1, 1, 2, {{-1, -1, 0, 1, 1, 0, 0, 0}, 2, {SIDE_N}}, {}, TILE_PORTAL), 1, 102,
     EXP_PRINCESS_DRAGON},
    // Magic portal; two road bends, north to west and east to south (like type 50).
    {Tile(ROAD, ROAD, ROAD, ROAD, 0, 1, 1, 0, {{0, 1, 1, 2, 2, 1, 1, 0}, 3, {}}, {}, TILE_PORTAL), 1, 103,
     EXP_PRINCESS_DRAGON},
    // Magic portal; type 23 (roads east, south and west meet).
    {Tile(GRASS, ROAD, ROAD, ROAD, 0, 1, 2, 3, {{0, 0, 0, 1, 1, 2, 2, 0}, 3, {}}, {}, TILE_PORTAL), 1, 104,
     EXP_PRINCESS_DRAGON},
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
