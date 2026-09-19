// diag_branching —— 合法手數 n 的分佈（分 PHASE_TILE / PHASE_MEEPLE），
// 用來決定 Dirichlet 的 policy_alpha 要固定還是 10/n。
//
// 輸出:
//   1. 兩個 phase 的 n 分佈（mean / 分位數 / 直方圖）
//   2. n 隨對局進度的變化（10 個 bucket）
//   3. results/branching.csv: phase,n  的原始樣本，給 Python 算 Dirichlet 統計
//
// 用法: ./diag_branching [局數=3000]
#include "common.hpp"
#include <map>

int main(int argc, char **argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 3000;
    std::mt19937 rng(4242);
    std::map<int, long long> histT, histM;
    std::vector<double> sumT(10, 0), sumM(10, 0);
    std::vector<long long> cntT(10, 0), cntM(10, 0);
    std::vector<int> allT, allM;

    for (int g = 0; g < N; ++g) {
        Carcassonne game;
        // 先跑一次拿總決策數，用來算進度 bucket
        std::vector<std::pair<int,int>> rec;   // (phase, n)
        while (game.current_phase != PHASE_TERMINAL) {
            if (game.current_phase == PHASE_CHANCE) {
                if (!diag::SampleDraw(game, rng)) break;
            } else if (game.current_phase == PHASE_TILE) {
                std::vector<TileMove> buf(BOARD_SIZE * BOARD_SIZE * 4);
                int c = 0;
                game.getLegalTileMoves(buf.data(), c);
                if (c == 0) break;
                rec.push_back({1, c});
                const TileMove &m = buf[std::uniform_int_distribution<int>(0, c - 1)(rng)];
                game.placeTile(m.x, m.y, m.rot);
            } else {
                FixedVector<int, 6> mm = game.getLegalMeepleMoves();
                if (mm.size() == 0) break;
                rec.push_back({2, (int)mm.size()});
                game.placeMeeple(mm[std::uniform_int_distribution<int>(0, mm.size() - 1)(rng)]);
            }
        }
        int T = (int)rec.size();
        for (int i = 0; i < T; ++i) {
            int b = T > 1 ? (int)((double)i / (T - 1) * 9.999) : 0;
            if (b > 9) b = 9;
            if (rec[i].first == 1) { histT[rec[i].second]++; sumT[b] += rec[i].second; cntT[b]++; allT.push_back(rec[i].second); }
            else                   { histM[rec[i].second]++; sumM[b] += rec[i].second; cntM[b]++; allM.push_back(rec[i].second); }
        }
    }

    auto stats = [](std::vector<int> &v, const char *name) {
        std::sort(v.begin(), v.end());
        double s = 0; for (int x : v) s += x;
        auto q = [&](double p) { return v[(size_t)(p * (v.size() - 1))]; };
        printf("%-14s n=%zu  mean %.2f   p5 %d  p25 %d  median %d  p75 %d  p95 %d   min %d  max %d\n",
               name, v.size(), s / v.size(), q(.05), q(.25), q(.50), q(.75), q(.95), v.front(), v.back());
    };
    printf("=== 合法手數 n 的分佈 (%d 局隨機對局) ===\n", N);
    stats(allT, "PHASE_TILE");
    stats(allM, "PHASE_MEEPLE");

    printf("\n=== PHASE_MEEPLE 的 n 直方圖 ===\n");
    long long tm = 0; for (auto &kv : histM) tm += kv.second;
    for (auto &kv : histM) printf("  n=%d : %6.2f%%\n", kv.first, 100.0 * kv.second / tm);

    printf("\n=== n 隨對局進度 (10 bucket) ===\n");
    printf("  bucket:   ");
    for (int b = 0; b < 10; ++b) printf("%6d", b);
    printf("\n  tile  n:  ");
    for (int b = 0; b < 10; ++b) printf("%6.1f", cntT[b] ? sumT[b] / cntT[b] : 0);
    printf("\n  meeple n: ");
    for (int b = 0; b < 10; ++b) printf("%6.2f", cntM[b] ? sumM[b] / cntM[b] : 0);
    printf("\n");

    FILE *f = fopen("branching.csv", "w");
    if (f) {
        fprintf(f, "phase,n\n");
        for (int x : allT) fprintf(f, "tile,%d\n", x);
        for (int x : allM) fprintf(f, "meeple,%d\n", x);
        fclose(f);
        printf("\n-> branching.csv (in cwd)\n");
    }
    return 0;
}
