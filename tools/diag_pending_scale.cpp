// diag_pending_scale —— 每位玩家的 pending（終局待結分）與已入袋分數的量級,用來定
// global vector 的正規化尺度(carcassonne.cc 的 kPendingNormalization /
// kFieldPendingNormalization / kScoreNormalization / kScoreDiffNormalization)。
//
// 量的就是觀測寫進 global 的值:getPendingScore()(總 pending)、
// getPendingFieldScore()(其中農田的部分)與 player_scores(已入袋),
// 在每個決策點(tile 期與 meeple 期)取樣。
// 兩種對局分佈:雙方亂下,以及 greedy(最大化 banked + pending,強玩家代理,
// 見 CLAUDE.md §2.1 的警告框)。
//
// 用法: ./diag_pending_scale [隨機局數=3000] [greedy 局數=300]
#include "common.hpp"
#include <algorithm>

namespace {

struct Samples {
    std::vector<int> per, diff, last_per, last_diff;            // 總 pending
    std::vector<int> field_per, field_last_per;                  // 農田 pending
    std::vector<int> banked, banked_diff, last_banked, last_banked_diff;  // 已入袋
    std::vector<int> static_diff, last_static_diff;              // 已入袋分差 + pending 分差
};

void Play(int games, bool greedy, std::mt19937 &rng, Samples *s) {
    for (int g = 0; g < games; ++g) {
        Carcassonne game;
        int last[2] = {0, 0}, last_field[2] = {0, 0}, last_banked[2] = {0, 0}, last_static = 0;
        while (game.current_phase != PHASE_TERMINAL) {
            if (game.current_phase == PHASE_CHANCE) {
                if (!diag::SampleDraw(game, rng)) break;
                continue;
            }
            int pending[2], field[2];
            game.getPendingScore(pending);
            game.getPendingFieldScore(field);
            for (int p = 0; p < 2; ++p) {
                s->per.push_back(pending[p]);
                s->field_per.push_back(field[p]);
                s->banked.push_back(game.player_scores[p]);
                last[p] = pending[p];
                last_field[p] = field[p];
                last_banked[p] = game.player_scores[p];
            }
            s->diff.push_back(pending[0] - pending[1]);
            s->banked_diff.push_back(game.player_scores[0] - game.player_scores[1]);
            last_static = game.player_scores[0] - game.player_scores[1] + pending[0] - pending[1];
            s->static_diff.push_back(last_static);
            const int me = game.currentPlayer;
            if (game.current_phase == PHASE_TILE) {
                if (greedy) diag::GreedyPlaceTile(game, me);
                else if (!diag::RandomPlaceTile(game, rng)) break;
            } else {
                if (greedy) diag::GreedyPlaceMeeple(game, me);
                else if (!diag::RandomPlaceMeeple(game, rng)) break;
            }
        }
        for (int p = 0; p < 2; ++p) {
            s->last_per.push_back(last[p]);
            s->field_last_per.push_back(last_field[p]);
            s->last_banked.push_back(last_banked[p]);
        }
        s->last_diff.push_back(last[0] - last[1]);
        s->last_banked_diff.push_back(last_banked[0] - last_banked[1]);
        s->last_static_diff.push_back(last_static);
    }
}

void Report(std::vector<int> v, const char *name) {
    std::sort(v.begin(), v.end());
    auto q = [&](double p) { return v[(size_t)(p * (v.size() - 1))]; };
    double sum = 0, abs_sum = 0;
    for (int x : v) { sum += x; abs_sum += std::abs(x); }
    printf("  %-20s n=%-8zu mean %6.2f  mean|.| %6.2f  p50 %3d  p90 %3d  p99 %3d  max %3d\n", name, v.size(),
           sum / v.size(), abs_sum / v.size(), q(.50), q(.90), q(.99), v.back());
}

void ReportAll(const Samples &s, const char *title, int games) {
    printf("=== %s (%d 局) ===\n", title, games);
    Report(s.per, "總 pending 單一玩家");
    Report(s.last_per, "  最後一手");
    Report(s.diff, "總 pending 分差");
    Report(s.last_diff, "  最後一手");
    Report(s.field_per, "農田 pending 單一玩家");
    Report(s.field_last_per, "  最後一手");
    Report(s.banked, "已入袋 單一玩家");
    Report(s.last_banked, "  最後一手");
    Report(s.banked_diff, "已入袋 分差");
    Report(s.last_banked_diff, "  最後一手");
    Report(s.static_diff, "static_diff");
    Report(s.last_static_diff, "  最後一手");
}

} // namespace

int main(int argc, char **argv) {
    const int random_games = argc > 1 ? atoi(argv[1]) : 3000;
    const int greedy_games = argc > 2 ? atoi(argv[2]) : 300;
    std::mt19937 rng(20240918);
    Samples random_samples, greedy_samples;
    Play(random_games, /*greedy=*/false, rng, &random_samples);
    ReportAll(random_samples, "隨機對局", random_games);
    Play(greedy_games, /*greedy=*/true, rng, &greedy_samples);
    ReportAll(greedy_samples, "greedy 對局", greedy_games);
    return 0;
}
