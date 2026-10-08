// diag_branching —— 合法手數 n 的分佈（依決策階段分開），
// 用來決定 Dirichlet 的 policy_alpha 要固定還是 10/n。
// n 是 LegalActions 的動作數（diag::LegalMoves）：meeple 期含魔法門、公主、仙女的格子。
//
// 輸出:
//   1. 每個階段的 n 分佈（mean / 分位數），放 meeple 期另有直方圖
//   2. n 隨對局進度的變化（10 個 bucket；放磚、放 meeple 兩期）
//   3. branching.csv: phase,n 的原始樣本（phase 是 tile/meeple/spot/dragon），給 Python 算 Dirichlet 統計
//
// 用法: ./diag_branching [局數=3000] [牌組=base]
//   牌組:base、river、all(所有擴充都 on),或像 inns_cathedrals=on,princess_dragon=tiles 的寫法。
#include "common.hpp"
#include <map>

namespace {

// 決策階段 → 統計用的編號與名稱
constexpr int kPhases = 4;
const char *const kPhaseNames[kPhases] = {"tile", "meeple", "spot", "dragon"};
int PhaseIndex(GamePhase phase) {
    switch (phase) {
    case PHASE_TILE: return 0;
    case PHASE_MEEPLE: return 1;
    case PHASE_SPOT: return 2;
    case PHASE_DRAGON: return 3;
    default: return -1;
    }
}

} // namespace

int main(int argc, char **argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 3000;
    const diag::GameRules rules = diag::DeckArg(argc > 2 ? argv[2] : "base");
    std::mt19937 rng(4242);
    std::map<int, long long> histM;
    std::vector<double> sumT(10, 0), sumM(10, 0);
    std::vector<long long> cntT(10, 0), cntM(10, 0);
    std::vector<int> all[kPhases];

    for (int g = 0; g < N; ++g) {
        Carcassonne game = diag::NewGame(rules);
        // 先跑完一局拿總決策數，用來算進度 bucket
        std::vector<std::pair<int,int>> rec;   // (phase index, n)
        while (game.current_phase != PHASE_TERMINAL) {
            if (game.current_phase == PHASE_CHANCE) {
                if (!diag::SampleDraw(game, rng)) break;
                continue;
            }
            const int n = static_cast<int>(diag::LegalMoves(game).size());
            if (n == 0) break;
            rec.push_back({PhaseIndex(game.current_phase), n});
            diag::RandomStep(game, rng);
        }
        int T = (int)rec.size();
        for (int i = 0; i < T; ++i) {
            int b = T > 1 ? (int)((double)i / (T - 1) * 9.999) : 0;
            if (b > 9) b = 9;
            const auto [phase, n] = rec[i];
            all[phase].push_back(n);
            if (phase == 0) { sumT[b] += n; cntT[b]++; }
            if (phase == 1) { histM[n]++; sumM[b] += n; cntM[b]++; }
        }
    }

    auto stats = [](std::vector<int> v, const char *name) {
        if (v.empty()) return;
        std::sort(v.begin(), v.end());
        double s = 0; for (int x : v) s += x;
        auto q = [&](double p) { return v[(size_t)(p * (v.size() - 1))]; };
        printf("%-14s n=%zu  mean %.2f   p5 %d  p25 %d  median %d  p75 %d  p95 %d   min %d  max %d\n",
               name, v.size(), s / v.size(), q(.05), q(.25), q(.50), q(.75), q(.95), v.front(), v.back());
    };
    printf("=== 合法手數 n 的分佈 (%d 局隨機對局, %s) ===\n", N, diag::GameName(rules).c_str());
    const char *const labels[kPhases] = {"PHASE_TILE", "PHASE_MEEPLE", "PHASE_SPOT", "PHASE_DRAGON"};
    for (int p = 0; p < kPhases; ++p) stats(all[p], labels[p]);

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
        for (int p = 0; p < kPhases; ++p)
            for (int x : all[p]) fprintf(f, "%s,%d\n", kPhaseNames[p], x);
        fclose(f);
        printf("\n-> branching.csv (in cwd)\n");
    }
    return 0;
}
