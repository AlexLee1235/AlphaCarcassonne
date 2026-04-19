#include "game.hpp"
#include "games/bridge/double_dummy_solver/include/dll.h"
#include "games/carcassonne/tile.hpp"

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


int DeckModule::consumeType(int type_id) {
    int slot = type_counts[type_id] - 1;
    int physical_id = tile_type_tables.draw_physical_ids_by_type[type_id][slot];
    type_counts[type_id]--;
    total_remaining--;
    return physical_id;
}

void DeckModule::initializeTypeCounts() {
    total_remaining = TOTAL_TILE_COUNT;
    for (int type_id = 1; type_id <= CANONICAL_TILE_TYPE_COUNT; ++type_id) {
        type_counts[type_id] = tile_type_tables.draw_count_by_type[type_id];
    }
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
