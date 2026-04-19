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


MeepleModule::MeepleModule(){
    for (int player = 0; player < 2; ++player) {
        for (int token_id = 0; token_id < 7; ++token_id) {
            free_meeple_ids[player][token_id] = static_cast<uint8_t>(token_id);
        }
    }
}

int MeepleModule::acquireMeepleToken(int player) { return free_meeple_ids[player][--free_meeple_count[player]]; }

void MeepleModule::releaseMeepleToken(int player, int token_id) {
    MeepleTokenState &token = meeple_tokens[player][token_id];
    if (!token.active) {
        return;
    }
    token.active = false;
    token.x = -1;
    token.y = -1;
    token.pos = -1;
    free_meeple_ids[player][free_meeple_count[player]++] = static_cast<uint8_t>(token_id);
}

void MeepleModule::releaseFeatureMeeples(Feature &feature) {
    for (int player = 0; player < 2; ++player) {
        uint8_t mask = feature.meeple_mask[player];
        while (mask != 0) {
            int token_id = 0;
            while (((mask >> token_id) & 1u) == 0) {
                token_id++;
            }
            releaseMeepleToken(player, token_id);
            mask &= static_cast<uint8_t>(mask - 1);
        }
        feature.meeple_mask[player] = 0;
    }
}