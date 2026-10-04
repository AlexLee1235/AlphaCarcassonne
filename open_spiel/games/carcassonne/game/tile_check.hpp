#pragma once

// Checks on the tile table (all_tiles in tile.hpp) that its types cannot
// express. carcassonne_test runs them, and so does tools/dump_tiles, which is
// quick to build while filling in the table. How to fill it:
// docs/adding_tiles.md.

#include "tile.hpp"

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace tile_check {

inline const char *EdgeName(EdgeType edge) {
    switch (edge) {
    case GRASS:
        return "GRASS";
    case CITY:
        return "CITY";
    case ROAD:
        return "ROAD";
    case RIVER:
        return "RIVER";
    case NONE:
        break;
    }
    return "NONE";
}

// How many sides of the tile share side `side`'s link (itself included).
inline int SidesLinkedTo(const Tile &tile, int side) {
    int sides = 0;
    for (int other = 0; other < 4; ++other) {
        sides += tile.link[other] == tile.link[side] ? 1 : 0;
    }
    return sides;
}

// Everything wrong with one tile as written (rotation 0); empty if nothing.
inline std::vector<std::string> CheckTile(const Tile &tile) {
    std::vector<std::string> errors;
    auto error = [&](const std::string &message) { errors.push_back(message); };
    const char *side_names[4] = {"N", "E", "S", "W"};

    for (int side = 0; side < 4; ++side) {
        if (tile.edge[side] == NONE) {
            error(std::string("side ") + side_names[side] + " has no edge type");
        }
    }
    if (!errors.empty()) {
        return errors;
    }

    // Links: one number per feature on the tile.
    for (int a = 0; a < 4; ++a) {
        for (int b = a + 1; b < 4; ++b) {
            if (tile.link[a] != tile.link[b]) {
                continue;
            }
            if (tile.edge[a] != tile.edge[b]) {
                error(std::string("sides ") + side_names[a] + " and " + side_names[b] + " share link " +
                      std::to_string(tile.link[a]) + " but are " + EdgeName(tile.edge[a]) + " and " +
                      EdgeName(tile.edge[b]) + "; give each feature its own link number");
            } else if (tile.edge[a] == GRASS) {
                error(std::string("grass sides ") + side_names[a] + " and " + side_names[b] +
                      " share a link; every grass side needs its own link number");
            }
        }
    }

    // Fields.
    if (tile.field_count > MAX_TILE_FIELDS) {
        error("field_count " + std::to_string(tile.field_count) + " is more than " + std::to_string(MAX_TILE_FIELDS));
        return errors;
    }
    int edge_fields = 0;
    std::array<bool, MAX_TILE_FIELDS> on_half_edge{};
    for (int side = 0; side < 4; ++side) {
        const int a = tile.field[2 * side];
        const int b = tile.field[2 * side + 1];
        const std::string where = std::string("side ") + side_names[side] + " (half-edges " +
                                  std::to_string(2 * side) + ", " + std::to_string(2 * side + 1) + ")";
        if (tile.edge[side] == CITY) {
            if (a != -1 || b != -1) {
                error(where + " is a city: both half-edges must be -1");
            }
            continue;
        }
        if (a < 0 || b < 0 || a >= tile.field_count || b >= tile.field_count) {
            error(where + " is " + EdgeName(tile.edge[side]) + ": its half-edges need fields 0.." +
                  std::to_string(tile.field_count - 1) + ", not " + std::to_string(a) + ", " + std::to_string(b));
            continue;
        }
        on_half_edge[a] = on_half_edge[b] = true;
        edge_fields = std::max(edge_fields, std::max(a, b) + 1);
        if (tile.edge[side] == GRASS && a != b) {
            error(where + " is grass: both half-edges must be the same field");
        }
        // A road or river that carries on to another side cuts the tile, so the
        // grass on its two sides cannot be one field here.
        if ((tile.edge[side] == ROAD || tile.edge[side] == RIVER) && a == b && SidesLinkedTo(tile, side) > 1) {
            error(where + ": the " + EdgeName(tile.edge[side]) +
                  " carries on to another side, so its two half-edges must be different fields");
        }
    }
    for (int field = 0; field < edge_fields; ++field) {
        if (!on_half_edge[field]) {
            error("field " + std::to_string(field) + " is on no half-edge; number the fields on half-edges 0, 1, 2... "
                  "and put the inner field last");
        }
    }
    if (tile.field_count > edge_fields + 1) {
        error("field_count " + std::to_string(tile.field_count) + " leaves " +
              std::to_string(tile.field_count - edge_fields) + " fields on no half-edge; at most one inner field");
    }

    // Each field lists every side of every city it borders.
    for (int field = 0; field < MAX_TILE_FIELDS; ++field) {
        const int sides = tile.field_city_sides[field];
        if (field >= tile.field_count) {
            if (sides != 0) {
                error("field " + std::to_string(field) + " does not exist but has city sides");
            }
            continue;
        }
        for (int side = 0; side < 4; ++side) {
            if (!(sides & (1 << side))) {
                continue;
            }
            if (tile.edge[side] != CITY) {
                error("field " + std::to_string(field) + " lists side " + side_names[side] + ", which is not a city");
                continue;
            }
            for (int other = 0; other < 4; ++other) {
                if (tile.edge[other] == CITY && tile.link[other] == tile.link[side] && !(sides & (1 << other))) {
                    error("field " + std::to_string(field) + " borders the city on side " + side_names[side] +
                          ", so it must list that city's side " + side_names[other] + " too");
                }
            }
        }
    }

    // Marks sit on the city or road they belong to.
    for (int side = 0; side < 4; ++side) {
        const uint8_t marks = tile.marks[side];
        const std::string where = std::string("side ") + side_names[side] + " (" + EdgeName(tile.edge[side]) + ")";
        if (marks & ~(CITY_MARKS | ROAD_MARKS)) {
            error(where + " has an unknown mark bit");
        }
        if (tile.edge[side] == CITY && (marks & ~CITY_MARKS)) {
            error(where + ": MARK_INN goes on a road side");
        } else if (tile.edge[side] == ROAD && (marks & ~ROAD_MARKS)) {
            error(where + ": shield, princess and goods go on a city side");
        } else if ((tile.edge[side] == GRASS || tile.edge[side] == RIVER) && marks) {
            error(where + " has a mark; marks go on a side of the city or road they belong to");
        }
        const uint8_t goods = tile.featureMarks(side) & GOODS_MARKS;
        if (goods & (goods - 1)) {
            error(where + ": one city with more than one kind of goods");
        }
    }
    if (tile.tile_marks & ~ALL_TILE_MARKS) {
        error("unknown tile mark bit");
    }
    return errors;
}

// CheckTile() for every row, and that each expansion's tiles are of that
// expansion's box; each message starts with its type.
inline std::vector<std::string> CheckTileTable() {
    std::vector<std::string> errors;
    for (int row = 0; row < CANONICAL_TILE_TYPE_COUNT; ++row) {
        const TileBlueprint &bp = all_tiles[row];
        for (const std::string &message : CheckTile(bp.tile)) {
            errors.push_back("type " + std::to_string(bp.canonical_type) + " (" + EXPANSION_NAMES[bp.expansion] +
                             "): " + message);
        }
    }
    return errors;
}

// Expansions whose table section is started but does not add up to the box.
// An empty section is fine: that expansion is not in the table yet.
inline std::vector<std::string> IncompleteExpansions() {
    std::vector<std::string> problems;
    for (int expansion = 0; expansion < EXPANSION_COUNT; ++expansion) {
        const int tiles = tileCountIn(expansionBit(static_cast<Expansion>(expansion)));
        if (tiles != 0 && tiles != OFFICIAL_TILE_COUNTS[expansion]) {
            problems.push_back(std::string(EXPANSION_NAMES[expansion]) + ": " + std::to_string(tiles) +
                               " tiles in the table, the box has " + std::to_string(OFFICIAL_TILE_COUNTS[expansion]));
        }
    }
    return problems;
}

// Fields renamed in order of first appearance (half-edges, then the inner one),
// followed by the field count and each field's city sides: equal for two
// tiles exactly when their field layouts are.
inline std::vector<int> CanonicalFieldLayout(const Tile &tile) {
    std::array<int, MAX_TILE_FIELDS> renamed;
    renamed.fill(-1);
    int next = 0;
    std::vector<int> layout;
    for (int half_edge = 0; half_edge < HALF_EDGE_COUNT; ++half_edge) {
        const int field = tile.field[half_edge];
        if (field != -1 && renamed[field] == -1) renamed[field] = next++;
        layout.push_back(field == -1 ? -1 : renamed[field]);
    }
    if (tile.innerField() != -1) renamed[tile.innerField()] = next++;
    layout.push_back(tile.field_count);
    std::array<int, MAX_TILE_FIELDS> city_sides{};
    for (int field = 0; field < tile.field_count; ++field) {
        city_sides[renamed[field]] = tile.field_city_sides[field];
    }
    layout.insert(layout.end(), city_sides.begin(), city_sides.end());
    return layout;
}

// What the observation shows of a tile's shape: its terrain and which pairs
// of non-grass sides it joins.
inline std::vector<int> TileLook(const Tile &tile) {
    std::vector<int> look(tile.edge, tile.edge + 4);
    for (int a = 0; a < 4; ++a) {
        for (int b = a + 1; b < 4; ++b) {
            look.push_back(tile.edge[a] != GRASS && tile.edge[b] != GRASS && tile.link[a] == tile.link[b]);
        }
    }
    return look;
}

// Pairs of types (lower first) that look the same to the observation in some
// rotations but have different field layouts there. The observation has no
// planes for a tile's own field layout (docs/carcassonne_field_observation.md
// §2.2), so a network cannot tell these apart.
inline std::vector<std::pair<int, int>> TileLookConflicts() {
    std::vector<std::pair<int, int>> conflicts;
    for (int a = 0; a < CANONICAL_TILE_TYPE_COUNT; ++a) {
        for (int b = a; b < CANONICAL_TILE_TYPE_COUNT; ++b) {
            bool conflict = false;
            Tile ta = all_tiles[a].tile;
            for (int ra = 0; ra < 4 && !conflict; ++ra, ta = ta.rotate()) {
                Tile tb = all_tiles[b].tile;
                for (int rb = 0; rb < 4 && !conflict; ++rb, tb = tb.rotate()) {
                    conflict = TileLook(ta) == TileLook(tb) && CanonicalFieldLayout(ta) != CanonicalFieldLayout(tb);
                }
            }
            if (conflict) {
                conflicts.emplace_back(a + 1, b + 1);
            }
        }
    }
    return conflicts;
}

} // namespace tile_check
