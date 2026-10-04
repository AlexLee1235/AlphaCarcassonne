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
    initial_total = total_remaining;
}

void DeckModule::getAvailableDraws(ChanceBranch *out, int &count) const {
    if (total_remaining == 0)
        return;
    for (int type_id = 1; type_id <= CANONICAL_TILE_TYPE_COUNT; ++type_id) {
        if (type_counts[type_id] > 0) {
            out[count++] = {type_id, static_cast<double>(type_counts[type_id]) / total_remaining};
        }
    }
}
