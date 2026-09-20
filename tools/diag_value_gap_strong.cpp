// diag_value_gap_strong —— §2.1 的量測換成「強玩家」的對局分佈。
//
// 為什麼要重做:§2.1 的 acc(static) 是在「雙方亂下」的對局上量的。
// 亂下的人不會翻盤,所以中盤的分差特別能預測勝負。訓練好的網路是兩個
// 勢均力敵的強玩家在下,局面會咬得更緊 —— 同樣的分差不一定同樣決定性。
// 用 greedy(最大化 banked+pending)當強玩家的代理,重量一次。
//
// 用法: ./diag_value_gap_strong [局數=300]
#include "common.hpp"

static void PlayTileGreedy(Carcassonne &g, int me) {
    static std::vector<TileMove> buf;
    buf.resize(BOARD_SIZE * BOARD_SIZE * 4);
    int c = 0; g.getLegalTileMoves(buf.data(), c);
    if (c == 0) return;
    int best = 0, bs = -1000000;
    for (int i = 0; i < c; ++i) {
        Carcassonne t = g; t.placeTile(buf[i].x, buf[i].y, buf[i].rot);
        int s = diag::BankedDiff(t, me) + diag::PendingDiff(t, me);
        if (s > bs) { bs = s; best = i; }
    }
    g.placeTile(buf[best].x, buf[best].y, buf[best].rot);
}
static void PlayMeepleGreedy(Carcassonne &g, int me) {
    FixedVector<int, 6> mm = g.getLegalMeepleMoves();
    if (mm.size() == 0) return;
    int best = mm[0], bs = -1000000;
    for (int i = 0; i < mm.size(); ++i) {
        Carcassonne t = g; t.placeMeeple(mm[i]);
        int s = diag::BankedDiff(t, me) + diag::PendingDiff(t, me);
        if (s > bs) { bs = s; best = mm[i]; }
    }
    g.placeMeeple(best);
}

int main(int argc, char **argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 300;
    constexpr int B = 10;
    for (int mode = 0; mode < 2; ++mode) {
        std::mt19937 rng(12345);
        struct Rec { int cur; double banked, stat; };
        std::vector<long long> nb(B,0), okB(B,0), okS(B,0);
        std::vector<double> pend(B,0.0);
        long long lastN=0,lastOkB=0,lastOkS=0,draws=0; double sumFinalAbs=0;
        for (int g = 0; g < N; ++g) {
            Carcassonne game; std::vector<Rec> recs;
            while (game.current_phase != PHASE_TERMINAL) {
                if (game.current_phase == PHASE_CHANCE) { if (!diag::SampleDraw(game, rng)) break; }
                else if (game.current_phase == PHASE_TILE) {
                    const int cur = game.currentPlayer;
                    const double banked = diag::BankedDiff(game, cur);
                    recs.push_back({cur, banked, banked + diag::PendingDiff(game, cur)});
                    if (mode == 0) { if (!diag::RandomPlaceTile(game, rng)) break; }
                    else PlayTileGreedy(game, cur);
                } else {
                    if (mode == 0) { if (!diag::RandomPlaceMeeple(game, rng)) break; }
                    else PlayMeepleGreedy(game, game.currentPlayer);
                }
            }
            int s0 = game.player_scores[0], s1 = game.player_scores[1];
            sumFinalAbs += std::abs(s0 - s1);
            if (s0 == s1) { draws++; continue; }
            int winner = s0 > s1 ? 0 : 1;
            int T = (int)recs.size();
            for (int i = 0; i < T; ++i) {
                int b = T > 1 ? (int)((double)i / (T - 1) * 9.999) : 0; if (b > 9) b = 9;
                int sgnW = (recs[i].cur == winner) ? 1 : -1;
                nb[b]++;
                if ((recs[i].banked > 0) == (sgnW > 0) && recs[i].banked != 0) okB[b]++;
                else if (recs[i].banked == 0 && sgnW > 0) {}
                if ((recs[i].stat > 0) == (sgnW > 0) && recs[i].stat != 0) okS[b]++;
                pend[b] += std::abs(recs[i].stat - recs[i].banked);
                if (i == T - 1) { lastN++; if ((recs[i].banked>0)==(sgnW>0)&&recs[i].banked!=0) lastOkB++;
                                  if ((recs[i].stat>0)==(sgnW>0)&&recs[i].stat!=0) lastOkS++; }
            }
        }
        printf("=== %s (%d 局)  draws %.1f%%  mean|final diff| %.2f ===\n",
               mode==0?"random vs random":"greedy vs greedy", N, 100.0*draws/N, sumFinalAbs/N);
        printf("%9s %8s %11s %11s %12s\n","bucket","n","acc(bank)","acc(static)","E|pending|");
        for (int b = 0; b < B; ++b) {
            double n = nb[b]; if (n < 1) continue;
            printf("%3d-%3d%% %8.0f %11.3f %11.3f %12.2f\n",
                   b*10,(b+1)*10,n,okB[b]/n,okS[b]/n,pend[b]/n);
        }
        printf("LAST ply  %8lld %11.3f %11.3f\n\n",lastN,(double)lastOkB/lastN,(double)lastOkS/lastN);
    }
    return 0;
}
