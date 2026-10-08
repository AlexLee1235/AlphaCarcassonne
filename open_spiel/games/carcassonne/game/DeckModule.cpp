#include "game.hpp"
#include "tile.hpp"

#include <array>
#include <bitset>
#include <cstdlib>
#include <utility>
#include <vector>



int DeckModule::consumeType(int type_id) {
    int slot = type_counts[type_id] - 1;
    int physical_id = tile_type_tables.draw_physical_ids_by_type[type_id][slot];
    type_counts[type_id]--;
    total_remaining--;
    return physical_id;
}

void DeckModule::initializeTypeCounts(uint32_t expansions) {
    total_remaining = 0;
    for (int type_id = 1; type_id <= CANONICAL_TILE_TYPE_COUNT; ++type_id) {
        const bool dealt = expansions & expansionBit(all_tiles[type_id - 1].expansion);
        type_counts[type_id] = dealt ? tile_type_tables.draw_count_by_type[type_id] : 0;
        total_remaining += type_counts[type_id];
    }
    river_first = (expansions & expansionBit(EXP_RIVER)) != 0;
    if (river_first) {
        // The spring starts the game; the base start tile is left out.
        type_counts[START_TILE_TYPE]--;
        total_remaining--;
    }
    initial_total = total_remaining;
}

void DeckModule::getAvailableDraws(ChanceBranch *out, int &count) const {
    if (total_remaining == 0)
        return;
    if (river_first) {
        // The river tiles other than the lake, then the lake, then the rest.
        int river_left = 0;
        for (int type_id = 1; type_id <= CANONICAL_TILE_TYPE_COUNT; ++type_id) {
            if (all_tiles[type_id - 1].expansion == EXP_RIVER && type_id != RIVER_LAKE_TYPE) {
                river_left += type_counts[type_id];
            }
        }
        if (river_left > 0) {
            for (int type_id = 1; type_id <= CANONICAL_TILE_TYPE_COUNT; ++type_id) {
                if (all_tiles[type_id - 1].expansion == EXP_RIVER && type_id != RIVER_LAKE_TYPE &&
                    type_counts[type_id] > 0) {
                    out[count++] = {type_id, static_cast<double>(type_counts[type_id]) / river_left};
                }
            }
            return;
        }
        if (type_counts[RIVER_LAKE_TYPE] > 0) {
            out[count++] = {RIVER_LAKE_TYPE, 1.0};
            return;
        }
    }
    const int drawable = drawableRemaining();
    for (int type_id = 1; type_id <= CANONICAL_TILE_TYPE_COUNT; ++type_id) {
        if (type_counts[type_id] > 0 && !(hold_dragon_tiles && (all_tiles[type_id - 1].tile.tile_marks & TILE_DRAGON))) {
            out[count++] = {type_id, static_cast<double>(type_counts[type_id]) / drawable};
        }
    }
}

int DeckModule::drawableRemaining() const {
    if (!hold_dragon_tiles) {
        return total_remaining;
    }
    int held = 0;
    for (int type_id = 1; type_id <= CANONICAL_TILE_TYPE_COUNT; ++type_id) {
        if (all_tiles[type_id - 1].tile.tile_marks & TILE_DRAGON) {
            held += type_counts[type_id];
        }
    }
    return total_remaining - held;
}
