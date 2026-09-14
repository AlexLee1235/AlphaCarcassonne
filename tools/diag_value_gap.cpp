// diag_value_gap —— 觀測裡的分數訊號離「足以判斷勝負」還差多少
//
// 在每個 tile 決策點比較兩個量預測最終勝者的準確率：
//   banked = player_scores 差              <- 這正是 kScoreDiffPlane 餵給網路的東西
//   static = banked + 現在就結束的終局補分差 <- 網路看不到的那一半
//
// 參考結果 (3000 局隨機對局)：
//   最後一個決策點  acc(banked) = 0.717    acc(static) = 0.992
//   E|pending| 從第 20% 之後穩定在 6 分上下，而平均最終分差只有 8.62
//
// 用法: ./diag_value_gap [局數=3000]
#include "common.hpp"

int main(int argc, char **argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 3000;
    std::mt19937 rng(12345);
    constexpr int B = 10;

    struct Rec { int cur; double banked, stat; };
    std::vector<long long> nb(B, 0), okB(B, 0), okS(B, 0);
    std::vector<double> pend(B, 0.0), absB(B, 0.0);
    std::vector<double> sx(B, 0), sy(B, 0), sxx(B, 0), syy(B, 0), sxy(B, 0);
    long long lastN = 0, lastOkB = 0, lastOkS = 0, games = 0, draws = 0;
    double sumFinalAbs = 0;

    for (int g = 0; g < N; ++g) {
        Carcassonne game;
        std::vector<Rec> recs;
        while (game.current_phase != PHASE_TERMINAL) {
            if (game.current_phase == PHASE_CHANCE) {
                if (!diag::SampleDraw(game, rng)) break;
            } else if (game.current_phase == PHASE_TILE) {
                const int cur = game.currentPlayer;
                const double banked = diag::BankedDiff(game, cur);
                recs.push_back({cur, banked, banked + diag::PendingDiff(game, cur)});
                if (!diag::RandomPlaceTile(game, rng)) break;
            } else {
                if (!diag::RandomPlaceMeeple(game, rng)) break;
            }
        }
        if (recs.empty()) continue;
        games++;
        const int fd = game.player_scores[0] - game.player_scores[1];
        if (fd == 0) draws++;
        sumFinalAbs += std::abs(fd);
        const int T = recs.size();
        for (int i = 0; i < T; ++i) {
            const Rec &r = recs[i];
            const double finalDiff = (r.cur == 0 ? fd : -fd);
            const int b = std::min(B - 1, (int)((double)i / T * B));
            nb[b]++;
            if ((r.banked >= 0) == (finalDiff >= 0)) okB[b]++;
            if ((r.stat   >= 0) == (finalDiff >= 0)) okS[b]++;
            pend[b] += std::abs(r.stat - r.banked);
            absB[b] += std::abs(r.banked);
            sx[b] += r.stat; sy[b] += finalDiff;
            sxx[b] += r.stat * r.stat; syy[b] += finalDiff * finalDiff; sxy[b] += r.stat * finalDiff;
            if (i == T - 1) {
                lastN++;
                if ((r.banked >= 0) == (finalDiff >= 0)) lastOkB++;
                if ((r.stat   >= 0) == (finalDiff >= 0)) lastOkS++;
            }
        }
    }

    printf("games=%lld draws=%lld(%.1f%%) mean|final score diff|=%.2f\n",
           games, draws, 100.0 * draws / games, sumFinalAbs / games);
    printf("%-8s %8s %10s %10s %11s %9s %8s\n",
           "bucket", "n", "acc(bank)", "acc(stat)", "E|pending|", "E|bank|", "corr");
    for (int b = 0; b < B; ++b) {
        const double n = nb[b];
        if (n == 0) continue;
        const double num = sxy[b] - sx[b] * sy[b] / n;
        const double den = std::sqrt((sxx[b] - sx[b] * sx[b] / n) * (syy[b] - sy[b] * sy[b] / n));
        printf("%3d-%3d%% %8.0f %10.3f %10.3f %11.2f %9.2f %8.3f\n",
               b * 10, (b + 1) * 10, n, okB[b] / n, okS[b] / n, pend[b] / n, absB[b] / n,
               den > 0 ? num / den : 0.0);
    }
    printf("LAST decision ply: n=%lld acc(banked)=%.3f acc(static)=%.3f\n",
           lastN, (double)lastOkB / lastN, (double)lastOkS / lastN);
    return 0;
}
