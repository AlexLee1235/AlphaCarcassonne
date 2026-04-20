#pragma once

#include <array>
#include <bitset>

using namespace std;

constexpr int PHYSICAL_TILE_COUNT = 72;
constexpr int CANONICAL_TILE_TYPE_COUNT = 24;
constexpr int MAX_PHYSICAL_IDS_PER_TYPE = 9;
constexpr int START_TILE_TYPE = 20;
constexpr int START_TILE_ROTATION = 0;

enum EdgeType { NONE = 0, GRASS = 1, CITY = 2, ROAD = 3 };

class Tile {
  public:
    EdgeType edge[4];
    int link[4];
    bool shield;
    bool monastery;

    Tile() = default;

    Tile(EdgeType e1, EdgeType e2, EdgeType e3, EdgeType e4, int l1, int l2, int l3, int l4, bool sh = false, bool mo = false) {
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
        return res;
    }
};

struct TileBlueprint {
    Tile tile;
    int count;
    int canonical_type;
};

const static TileBlueprint base_deck[] = {
    {Tile(GRASS, GRASS, GRASS, GRASS, 0, 1, 2, 3, false, true), 4, 1},
    {Tile(GRASS, GRASS, ROAD, GRASS, 0, 1, 2, 3, false, true), 2, 2},
    {Tile(CITY, CITY, CITY, CITY, 0, 0, 0, 0, true), 1, 3},
    {Tile(CITY, CITY, GRASS, CITY, 0, 0, 1, 0), 3, 4},
    {Tile(CITY, CITY, GRASS, CITY, 0, 0, 1, 0, true), 1, 5},
    {Tile(CITY, CITY, ROAD, CITY, 0, 0, 1, 0), 1, 6},
    {Tile(CITY, CITY, ROAD, CITY, 0, 0, 1, 0, true), 2, 7},
    {Tile(CITY, GRASS, GRASS, CITY, 0, 1, 2, 0), 3, 8},
    {Tile(CITY, GRASS, GRASS, CITY, 0, 1, 2, 0, true), 2, 9},
    {Tile(CITY, ROAD, ROAD, CITY, 0, 1, 1, 0), 3, 10},
    {Tile(CITY, ROAD, ROAD, CITY, 0, 1, 1, 0, true), 2, 11},
    {Tile(GRASS, CITY, GRASS, CITY, 0, 1, 2, 1), 1, 12},
    {Tile(GRASS, CITY, GRASS, CITY, 0, 1, 2, 1, true), 2, 13},
    {Tile(CITY, GRASS, GRASS, CITY, 0, 1, 2, 3), 2, 14},
    {Tile(CITY, GRASS, CITY, GRASS, 0, 1, 2, 3), 3, 15},
    {Tile(CITY, GRASS, GRASS, GRASS, 0, 1, 2, 3), 5, 16},
    {Tile(CITY, GRASS, ROAD, ROAD, 0, 1, 2, 2), 3, 17},
    {Tile(CITY, ROAD, ROAD, GRASS, 0, 1, 1, 2), 3, 18},
    {Tile(CITY, ROAD, ROAD, ROAD, 0, 1, 2, 3), 3, 19},
    {Tile(CITY, ROAD, GRASS, ROAD, 0, 1, 2, 1), 4, 20},
    {Tile(ROAD, GRASS, ROAD, GRASS, 0, 1, 0, 2), 8, 21},
    {Tile(GRASS, GRASS, ROAD, ROAD, 0, 1, 2, 2), 9, 22},
    {Tile(GRASS, ROAD, ROAD, ROAD, 0, 1, 2, 3), 4, 23},
    {Tile(ROAD, ROAD, ROAD, ROAD, 0, 1, 2, 3), 1, 24},
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
