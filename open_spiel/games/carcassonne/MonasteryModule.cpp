#include "game.hpp"

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


void MonasteryModule::settleCompletedMonasteries(int* player_scores) {
    for (int i = active_monasteries.size() - 1; i >= 0; --i) {
        if (active_monasteries[i].tile_count == 9) {
            player_scores[active_monasteries[i].owner] += 9;
            releaseMeepleToken(active_monasteries[i].owner, active_monasteries[i].token_id);
            active_monasteries.swap_pop_erase_at(i);
        }
    }
}


void MonasteryModule::resolveEndGameScore(int* player_scores) {
    for (int i = 0; i < active_monasteries.size(); ++i) {
        player_scores[active_monasteries[i].owner] += active_monasteries[i].tile_count;
    }
}

void MonasteryModule::placeTileOnBoard(int tile_id, int x, int y, int rot){
    for (int i = 0; i < active_monasteries.size(); ++i) {
        if (std::abs(active_monasteries[i].x - x) <= 1 && std::abs(active_monasteries[i].y - y) <= 1) {
            active_monasteries[i].tile_count++;
        }
    }
}
