// diag_pending_scale —— 每位玩家的 pending（終局待結分）量級,用來定 §9.2 的正規化尺度。
// §9.2 原本寫 pending/20 且沒有 clip;這支程式檢查 20 是不是對的除數。
// 用法: ./diag_pending_scale [局數=3000]
#include "common.hpp"
#include <algorithm>

int main(int argc, char **argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 3000;
    std::mt19937 rng(20240918);
    std::vector<int> per;            // 單一玩家的 pending
    std::vector<int> diff;           // pending 分差
    std::vector<int> lastPer, lastDiff;
    for (int g = 0; g < N; ++g) {
        Carcassonne game;
        int prevPer[2] = {0,0}, prevDiff = 0;
        while (game.current_phase != PHASE_TERMINAL) {
            if (game.current_phase == PHASE_CHANCE)      { if (!diag::SampleDraw(game, rng)) break; }
            else if (game.current_phase == PHASE_TILE)   { if (!diag::RandomPlaceTile(game, rng)) break; }
            else {
                Carcassonne copy = game;
                int b0 = copy.player_scores[0], b1 = copy.player_scores[1];
                copy.resolveEndGameScore();
                int p0 = copy.player_scores[0] - b0, p1 = copy.player_scores[1] - b1;
                per.push_back(p0); per.push_back(p1);
                diff.push_back(p0 - p1);
                prevPer[0] = p0; prevPer[1] = p1; prevDiff = p0 - p1;
                if (!diag::RandomPlaceMeeple(game, rng)) break;
            }
        }
        lastPer.push_back(prevPer[0]); lastPer.push_back(prevPer[1]);
        lastDiff.push_back(prevDiff);
    }
    auto q = [](std::vector<int> &v, double p) { return v[(size_t)(p * (v.size() - 1))]; };
    auto rep = [&](std::vector<int> v, const char *name) {
        std::sort(v.begin(), v.end());
        double s = 0, a = 0; for (int x : v) { s += x; a += std::abs(x); }
        long over20 = 0, over30 = 0;
        for (int x : v) { if (std::abs(x) > 20) over20++; if (std::abs(x) > 30) over30++; }
        printf("%-22s n=%-8zu mean %6.2f  mean|.| %6.2f  p50 %3d  p90 %3d  p99 %3d  max %3d   |x|>20: %5.2f%%  |x|>30: %5.2f%%\n",
               name, v.size(), s / v.size(), a / v.size(), q(v,.50), q(v,.90), q(v,.99), v.back(),
               100.0*over20/v.size(), 100.0*over30/v.size());
    };
    printf("=== pending 量級 (%d 局隨機對局) ===\n", N);
    rep(per,      "單一玩家 pending");
    rep(diff,     "pending 分差");
    rep(lastPer,  "最後一手 單一玩家");
    rep(lastDiff, "最後一手 分差");
    return 0;
}
