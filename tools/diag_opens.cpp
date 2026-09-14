// diag_opens —— 兩件事：
//   (A) 有盾的磚型是不是恆為「單一城市元件」？（決定元件盾牌數要不要廣播）
//   (B) opens 對「最終是否封口」的預測力（決定 opens 要不要廣播）
//
// 參考結果：
//   (A) type 3/5/7/9/11/13 全部都只有 1 個城市元件
//       -> 盾的歸屬沒有歧義，本地 shield plane 就夠，元件盾牌數不需要廣播
//   (B) 牌堆剩 30 張時取樣、帶 meeple 的未完成元件 (4000 局隨機對局)：
//       opens=1 城市封口率 7.6%，opens=2 是 1.0%，opens>=3 基本為零
//       -> 單調關係非常乾淨，而觀測裡完全沒有 opens。必須加。
//       (隨機下法不會刻意封城，絕對封口率被嚴重低估；有意識的對局差距只會更大)
//
// 用法: ./diag_opens [局數=4000]
#include "common.hpp"

int main(int argc, char **argv) {
    printf("=== (A) 有盾磚型：單張磚上有幾個相異的城市元件 ===\n");
    for (const auto &bp : base_deck) {
        if (!bp.tile.shield) continue;
        int links[4], n = 0;
        for (int i = 0; i < 4; ++i) {
            if (bp.tile.edge[i] != CITY) continue;
            bool dup = false;
            for (int k = 0; k < n; ++k) if (links[k] == bp.tile.link[i]) dup = true;
            if (!dup) links[n++] = bp.tile.link[i];
        }
        printf("  type %2d  count=%d  相異城市元件 = %d\n", bp.canonical_type, bp.count, n);
    }

    const int N = argc > 1 ? atoi(argv[1]) : 4000;
    std::mt19937 rng(31337);
    constexpr int OB = 8;
    long long cityN[OB] = {0}, cityClosed[OB] = {0}, roadN[OB] = {0}, roadClosed[OB] = {0};
    double cityGain[OB] = {0};

    for (int g = 0; g < N; ++g) {
        Carcassonne game;
        struct Obs { int rep, opens, type, scoreNow; };
        std::vector<Obs> obs;
        while (game.current_phase != PHASE_TERMINAL) {
            if (game.current_phase == PHASE_CHANCE) {
                if (!diag::SampleDraw(game, rng)) break;
            } else if (game.current_phase == PHASE_TILE) {
                if (game.getTotalRemaining() == 30) {   // 固定取樣點，避免同一元件被重複計數
                    for (int id = 1; id <= PHYSICAL_TILE_COUNT; ++id) {
                        if (game.logs.tile_x[id] < 0) continue;
                        for (int s = 0; s < 4; ++s) {
                            const int e = game.features.edgeIndex(id, s);
                            if (game.features.featureMap.find(e) != e) continue;  // 只取 root，天然去重
                            const Feature &f = game.features.featureMap.getSetData(e);
                            if (f.type == GRASS || f.type == NONE || f.opens == 0) continue;
                            if (f.meeple_count[0] == 0 && f.meeple_count[1] == 0) continue;
                            obs.push_back({e, (int)f.opens, (int)f.type, f.getScore()});
                        }
                    }
                }
                if (!diag::RandomPlaceTile(game, rng)) break;
            } else {
                if (!diag::RandomPlaceMeeple(game, rng)) break;
            }
        }
        for (const Obs &o : obs) {
            const Feature &f = game.features.featureMap.getSetData(o.rep);
            const bool closed = (f.opens == 0);
            const int b = std::min(o.opens, OB - 1);
            const int finalPts = closed ? 2 * f.getScore() : f.getScore();
            if (o.type == CITY) { cityN[b]++; if (closed) cityClosed[b]++; cityGain[b] += finalPts - o.scoreNow; }
            else                { roadN[b]++; if (closed) roadClosed[b]++; }
        }
    }

    printf("\n=== (B) 牌堆剩 30 張時、帶 meeple 的未完成元件，最終是否封口 (%d 局) ===\n", N);
    printf("%-6s %8s %10s %14s | %8s %10s\n", "opens", "城市 n", "封口%", "E[分數增益]", "道路 n", "封口%");
    for (int b = 1; b < OB; ++b) {
        if (!cityN[b] && !roadN[b]) continue;
        printf("%-6d %8lld %9.1f%% %14.2f | %8lld %9.1f%%\n", b,
               cityN[b], cityN[b] ? 100.0 * cityClosed[b] / cityN[b] : 0.0,
               cityN[b] ? cityGain[b] / cityN[b] : 0.0,
               roadN[b], roadN[b] ? 100.0 * roadClosed[b] / roadN[b] : 0.0);
    }
    return 0;
}
