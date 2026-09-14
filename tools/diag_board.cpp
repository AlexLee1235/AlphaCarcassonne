// diag_board —— 15x15 的棋盤夠不夠大？分支因子多少？終局待結分的組成？
//
// 建議編譯兩份：一份用 repo 的 BOARD_SIZE，一份用 25（Makefile 的 diag_board25），
// 對照就能看出 15x15 裁掉了多少。
//
// 參考結果 (3000 局隨機對局)：
//   BOARD_SIZE=15: avg bbox 13.1x13.2   碰到邊界 97.7%   avg 合法落點 30.8
//   BOARD_SIZE=25: avg bbox 13.7x13.8   碰到邊界  1.3%   avg 合法落點 32.6
//   真實最大跨度 > 15 的對局佔 31.1%  (>17: 4.9%, >19: 0.4%, >21: 0.0%)
//   -> 建議 BOARD_SIZE = 21（覆蓋 99.9%）
//   終局待結分組成（雙方合計）: 未完成城市 19.8 / 未完成道路 15.7 / 修道院 7.7
//
// 用法: ./diag_board [局數=3000]
#include "common.hpp"

int main(int argc, char **argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 3000;
    std::mt19937 rng(999);
    long long games = 0, borderGames = 0, branchN = 0, maxB = 0;
    double sumW = 0, sumH = 0, sumBranch = 0;
    double sumOpenCity = 0, sumOpenRoad = 0, sumMon = 0;
    std::vector<long long> spanHist(64, 0);
    long long over15 = 0, over17 = 0, over19 = 0, over21 = 0;

    for (int g = 0; g < N; ++g) {
        Carcassonne game;
        bool border = false;
        while (game.current_phase != PHASE_TERMINAL) {
            if (game.current_phase == PHASE_CHANCE) {
                if (!diag::SampleDraw(game, rng)) break;
            } else if (game.current_phase == PHASE_TILE) {
                std::vector<TileMove> buf(BOARD_SIZE * BOARD_SIZE * 4);
                int c = 0;
                game.getLegalTileMoves(buf.data(), c);
                if (c == 0) break;
                sumBranch += c; branchN++;
                if (c > maxB) maxB = c;
                const TileMove &m = buf[std::uniform_int_distribution<int>(0, c - 1)(rng)];
                if (m.x == 0 || m.y == 0 || m.x == BOARD_SIZE - 1 || m.y == BOARD_SIZE - 1) border = true;
                game.placeTile(m.x, m.y, m.rot);
            } else {
                if (!diag::RandomPlaceMeeple(game, rng)) break;
            }
        }
        games++;
        if (border) borderGames++;

        int x0 = 9999, x1 = -1, y0 = 9999, y1 = -1;
        for (int y = 0; y < BOARD_SIZE; ++y)
            for (int x = 0; x < BOARD_SIZE; ++x)
                if (game.getPlacement(x, y).id) {
                    x0 = std::min(x0, x); x1 = std::max(x1, x);
                    y0 = std::min(y0, y); y1 = std::max(y1, y);
                }
        const int w = x1 - x0 + 1, h = y1 - y0 + 1;
        sumW += w; sumH += h;
        const int span = std::max(w, h);
        if (span < (int)spanHist.size()) spanHist[span]++;
        if (span > 15) over15++;
        if (span > 17) over17++;
        if (span > 19) over19++;
        if (span > 21) over21++;

        // 終局待結分組成
        double oc = 0, orr = 0;
        for (auto it = game.features.featureMap.begin(); it != game.features.featureMap.end(); ++it) {
            Feature &f = *it;
            if (f.opens == 0 || f.type == GRASS) continue;
            if (f.meeple_count[0] == 0 && f.meeple_count[1] == 0) continue;
            (f.type == CITY ? oc : orr) += f.getScore();
        }
        double mo = 0;
        for (int i = 0; i < game.monasteries.active_monasteries.size(); ++i)
            mo += game.monasteries.active_monasteries[i].tile_count;
        sumOpenCity += oc; sumOpenRoad += orr; sumMon += mo;
    }

    printf("BOARD_SIZE = %d,  games = %lld\n", BOARD_SIZE, games);
    printf("  avg bounding box = %.1f x %.1f    碰到盤面邊界的對局 = %.1f%%\n",
           sumW / games, sumH / games, 100.0 * borderGames / games);
    printf("  avg 合法落點/手 = %.1f    max = %lld\n", sumBranch / branchN, maxB);
    printf("  終局待結分（雙方合計）: 未完成城市 %.2f  未完成道路 %.2f  修道院 %.2f\n",
           sumOpenCity / games, sumOpenRoad / games, sumMon / games);
    printf("  P(最大跨度 > 15) = %.1f%%   >17 = %.1f%%   >19 = %.1f%%   >21 = %.1f%%\n",
           100.0 * over15 / games, 100.0 * over17 / games, 100.0 * over19 / games, 100.0 * over21 / games);
    printf("  跨度分佈:\n");
    for (size_t i = 0; i < spanHist.size(); ++i)
        if (spanHist[i]) printf("    %2zu: %6lld (%.1f%%)\n", i, spanHist[i], 100.0 * spanHist[i] / games);
    if (BOARD_SIZE <= 21)
        printf("  ** 注意: BOARD_SIZE=%d 會截斷對局，上面的跨度分佈是被裁過的。\n"
               "     用 diag_board25 看真實分佈。\n", BOARD_SIZE);
    return 0;
}
