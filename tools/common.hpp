// AlphaCarcassonne 診斷工具共用程式碼。
//
// 這些工具用 `#define private public` 直接存取 Carcassonne 的內部狀態
// (features / monasteries / logs)。這是刻意的：診斷工具需要看到引擎內部，
// 但不應該為了診斷而放寬正式程式碼的封裝。
#pragma once

#define private public
#include "game.hpp"
#undef private

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace diag {

// 依真實機率抽一張牌。回傳 false 表示牌堆已空。
inline bool SampleDraw(Carcassonne &g, std::mt19937 &rng) {
    ChanceBranch d[CANONICAL_TILE_TYPE_COUNT];
    int c = 0;
    g.getAvailableDraws(d, c);
    if (c == 0) return false;
    double r = std::uniform_real_distribution<double>(0, 1)(rng), acc = 0;
    int pick = d[c - 1].type_id;
    for (int i = 0; i < c; ++i) {
        acc += d[i].probability;
        if (r <= acc) { pick = d[i].type_id; break; }
    }
    g.drawTile(pick);
    return true;
}

// 隨機選一個合法落點。回傳 false 表示無合法手。
inline bool RandomPlaceTile(Carcassonne &g, std::mt19937 &rng, TileMove *out = nullptr) {
    static std::vector<TileMove> buf;
    buf.resize(BOARD_SIZE * BOARD_SIZE * 4);
    int c = 0;
    g.getLegalTileMoves(buf.data(), c);
    if (c == 0) return false;
    const TileMove &m = buf[std::uniform_int_distribution<int>(0, c - 1)(rng)];
    if (out) *out = m;
    g.placeTile(m.x, m.y, m.rot);
    return true;
}

inline bool RandomPlaceMeeple(Carcassonne &g, std::mt19937 &rng) {
    FixedVector<int, 6> mm = g.getLegalMeepleMoves();
    if (mm.size() == 0) return false;
    g.placeMeeple(mm[std::uniform_int_distribution<int>(0, mm.size() - 1)(rng)]);
    return true;
}

// 「若現在立刻結束」的終局補分（不含已入袋分數），從 player 視角的分差。
inline int PendingDiff(const Carcassonne &g, int player) {
    Carcassonne copy = g;
    int b0 = copy.player_scores[0], b1 = copy.player_scores[1];
    copy.resolveEndGameScore();
    int p0 = copy.player_scores[0] - b0, p1 = copy.player_scores[1] - b1;
    return player == 0 ? p0 - p1 : p1 - p0;
}

inline int BankedDiff(const Carcassonne &g, int player) {
    return g.player_scores[player] - g.player_scores[1 - player];
}

} // namespace diag
