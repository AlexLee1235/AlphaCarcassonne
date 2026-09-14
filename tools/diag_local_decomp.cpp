// diag_local_decomp —— 現有 80 個 plane 的資訊夠不夠算出終局待結分？
//
// 模擬「一個完美的 per-cell CNN + 全域求和」能算出什麼：每格只用
//   - 邊地形 / shield / monastery      (plane 0-13, 本地)
//   - 可從磚型還原的磚內 link          (plane 14 的作用)
//   - 已廣播的元件 meeple 數           (plane 15-18 / 20-23, LogModule::getMeepleMap)
// 每格輸出 sign(多數方) x (1 + shield)，最後求和，對照真實 pending。
//
// 參考結果 (3000 局 / 212,938 個狀態)：
//   exact = 99.17%   sign = 99.96%   MAE = 0.0083 分   max|err| = 1 分
// -> 資訊是夠的、分解是局部的。value 學不起來的原因在架構與監督，不在表示法。
//    (那 0.83% 的 ±1 誤差來自「同一元件離開一張磚又繞回來」的外部迴圈。)
//
// 用法: ./diag_local_decomp [局數=3000]
#include "common.hpp"

// 只用網路看得到的資訊做的估計
static int LocalEstimateDiff(const Carcassonne &g, int player) {
    const int opp = 1 - player;
    int total = 0;
    for (int y = 0; y < BOARD_SIZE; ++y) {
        for (int x = 0; x < BOARD_SIZE; ++x) {
            const Placement p = g.getPlacement(x, y);
            if (!p.id) continue;
            const Tile &t = full_deck[p.id][p.rotation];
            int seenLink[4], nSeen = 0;
            for (int i = 0; i < 4; ++i) {
                if (t.edge[i] == GRASS) continue;
                bool dup = false;
                for (int k = 0; k < nSeen; ++k) if (seenLink[k] == t.link[i]) dup = true;
                if (dup) continue;              // 只靠磚內 link 去重 —— 網路能做到的極限
                seenLink[nSeen++] = t.link[i];
                const Feature &f = g.features.featureMap.getSetData(g.features.edgeIndex(p.id, i));
                const int mm = f.meeple_count[player], mo = f.meeple_count[opp];
                const int sgn = (mm > mo) ? 1 : ((mo > mm) ? -1 : 0);
                if (!sgn) continue;
                total += sgn * (1 + ((t.edge[i] == CITY && t.shield) ? 1 : 0));
            }
        }
    }
    for (int i = 0; i < g.monasteries.active_monasteries.size(); ++i) {
        const MonasteryTracker &m = g.monasteries.active_monasteries[i];
        total += (m.owner == player ? 1 : -1) * m.tile_count;
    }
    return total;
}

int main(int argc, char **argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 3000;
    std::mt19937 rng(777);
    long long n = 0, exactAll = 0, signAll = 0, nLate = 0, exactLate = 0;
    double sae = 0, saeLate = 0, maxae = 0;

    for (int g = 0; g < N; ++g) {
        Carcassonne game;
        while (game.current_phase != PHASE_TERMINAL) {
            if (game.current_phase == PHASE_CHANCE) {
                if (!diag::SampleDraw(game, rng)) break;
            } else if (game.current_phase == PHASE_TILE) {
                const int me = game.currentPlayer;
                const int truePend = diag::PendingDiff(game, me);
                const int est = LocalEstimateDiff(game, me);
                const int ae = std::abs(est - truePend);
                n++; sae += ae;
                if (ae == 0) exactAll++;
                if ((est >= 0) == (truePend >= 0)) signAll++;
                if (ae > maxae) maxae = ae;
                if (game.getTotalRemaining() < 12) { nLate++; saeLate += ae; if (ae == 0) exactLate++; }
                if (!diag::RandomPlaceTile(game, rng)) break;
            } else {
                if (!diag::RandomPlaceMeeple(game, rng)) break;
            }
        }
    }
    printf("states=%lld\n", n);
    printf("local-decomposable estimate vs true pending:  exact=%.2f%%  sign=%.2f%%  MAE=%.4f  max|err|=%.0f\n",
           100.0 * exactAll / n, 100.0 * signAll / n, sae / n, maxae);
    printf("  (late game, <12 tiles left): exact=%.2f%%  MAE=%.4f  n=%lld\n",
           100.0 * exactLate / nLate, saeLate / nLate, nLate);
    return 0;
}
