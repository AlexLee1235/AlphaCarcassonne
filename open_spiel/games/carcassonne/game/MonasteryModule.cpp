#include "game.hpp"

#include <array>
#include <bitset>
#include <cstdlib>
#include <utility>
#include <vector>


void MonasteryModule::settleCompletedMonasteries(int *player_scores, int *holding_meeples) {
    for (int i = active_monasteries.size() - 1; i >= 0; --i) {
        if (active_monasteries[i].tile_count == 9) {
            player_scores[active_monasteries[i].owner] += 9;
            holding_meeples[active_monasteries[i].owner]++;
            active_monasteries.swap_pop_erase_at(i);
        }
    }
}

void MonasteryModule::resolveEndGameScore(int *player_scores) {
    for (int i = 0; i < active_monasteries.size(); ++i) {
        player_scores[active_monasteries[i].owner] += active_monasteries[i].tile_count;
    }
}

void MonasteryModule::placeTileOnBoard(int tile_id, int x, int y, int rot) {
    for (int i = 0; i < active_monasteries.size(); ++i) {
        if (std::abs(active_monasteries[i].x - x) <= 1 && std::abs(active_monasteries[i].y - y) <= 1) {
            active_monasteries[i].tile_count++;
        }
    }
}

void MonasteryModule::placeMeeple(int x, int y, int pos, int player, const BoardModule &board, int *player_scores,
                                  int *holding_meeples) {
    active_monasteries.push_back({x, y, board.count3x3(x, y), player});
}
