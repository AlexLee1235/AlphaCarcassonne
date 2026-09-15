// diag_residual —— 餵了 static_diff 之後，還剩多少東西要模型學？
//
// 把最終分差拆成兩項：
//     final  =  static(t)  +  Δ
//               ~~~~~~~~~     ~
//               純規則算術     策略：封口(×2)、元件成長、合併翻盤
//               O(n) 可算      只能學
//
// 這支量每個階段的 std(static) / std(Δ) / corr(static, Δ)，回答：
//   1. 用 static 當 baseline 能消掉多少變異數？（= 網路省下多少工作）
//   2. 剩下的 Δ 有多大？（= 網路還要學的策略內容有多少）
//   3. corr(static, Δ) 是不是負的？（= static 是不是有系統性偏差，需要均值回歸修正）
//
// 用法: ./diag_residual [局數=3000]
#include "common.hpp"

int main(int argc, char **argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 3000;
    std::mt19937 rng(20260914);
    constexpr int B = 10;
    struct Acc { long long n=0; double sx=0,sy=0,sxx=0,syy=0,sxy=0,sf=0,sff=0; };
    std::vector<Acc> acc(B);
    Acc last;

    auto add = [](Acc &a, double st, double d, double fin) {
        a.n++; a.sx+=st; a.sy+=d; a.sxx+=st*st; a.syy+=d*d; a.sxy+=st*d; a.sf+=fin; a.sff+=fin*fin;
    };

    for (int g = 0; g < N; ++g) {
        Carcassonne game;
        struct Rec { int cur; double stat; };
        std::vector<Rec> recs;
        while (game.current_phase != PHASE_TERMINAL) {
            if (game.current_phase == PHASE_CHANCE) {
                if (!diag::SampleDraw(game, rng)) break;
            } else if (game.current_phase == PHASE_TILE) {
                const int cur = game.currentPlayer;
                recs.push_back({cur, (double)diag::BankedDiff(game, cur) + diag::PendingDiff(game, cur)});
                if (!diag::RandomPlaceTile(game, rng)) break;
            } else {
                if (!diag::RandomPlaceMeeple(game, rng)) break;
            }
        }
        if (recs.empty()) continue;
        const int fd = game.player_scores[0] - game.player_scores[1];
        const int T = recs.size();
        for (int i = 0; i < T; ++i) {
            const double fin = (recs[i].cur == 0 ? fd : -fd);
            const double st = recs[i].stat, d = fin - st;
            add(acc[std::min(B-1, (int)((double)i / T * B))], st, d, fin);
            if (i == T - 1) add(last, st, d, fin);
        }
    }

    auto report = [](const char *label, const Acc &a) {
        const double n = a.n;
        const double vs = a.sxx/n - (a.sx/n)*(a.sx/n);
        const double vd = a.syy/n - (a.sy/n)*(a.sy/n);
        const double vf = a.sff/n - (a.sf/n)*(a.sf/n);
        const double cov = a.sxy/n - (a.sx/n)*(a.sy/n);
        const double r = (vs > 0 && vd > 0) ? cov/std::sqrt(vs*vd) : 0.0;
        printf("%-9s %8lld %8.2f %8.2f %8.2f %9.2f %9.3f %11.1f%%\n",
               label, a.n, std::sqrt(vf), std::sqrt(vs), std::sqrt(vd), a.sy/n, r,
               100.0*(1.0 - vd/vf));
    };

    printf("%-9s %8s %8s %8s %8s %9s %9s %11s\n",
           "bucket", "n", "std(fin)", "std(st)", "std(D)", "mean(D)", "corr(st,D)", "變異數消除");
    for (int b = 0; b < B; ++b) {
        char buf[16]; snprintf(buf, sizeof buf, "%2d-%3d%%", b*10, (b+1)*10);
        if (acc[b].n) report(buf, acc[b]);
    }
    report("LAST", last);
    return 0;
}
