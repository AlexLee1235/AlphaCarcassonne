// diag_pending_scale —— 每位玩家的 pending(終局待結分)與已入袋分數的量級,用來定
// global vector 的正規化尺度(carcassonne.cc 的 kPendingNormalization /
// kFieldPendingNormalization / kScoreNormalization / kScoreDiffNormalization)。
//
// 量的就是觀測寫進 global 的值,在每個決策點取樣(scale_stats.hpp)。
// 兩種對局分佈:雙方亂下,以及 greedy(最大化 banked + pending,強玩家代理,
// 見 docs/carcassonne_design_history.md §2.1 的警告框)。自我對弈的分佈用 diag_replay_scale 量。
//
// 用法: ./diag_pending_scale [隨機局數=3000] [greedy 局數=300] [牌組=base]
//   牌組:base、river、all(所有擴充都 on),或像 inns_cathedrals=on,princess_dragon=tiles 的寫法。
#include "scale_stats.hpp"

int main(int argc, char **argv) {
    const int random_games = argc > 1 ? atoi(argv[1]) : 3000;
    const int greedy_games = argc > 2 ? atoi(argv[2]) : 300;
    const diag::GameRules rules = diag::DeckArg(argc > 3 ? argv[3] : "base");
    std::mt19937 rng(20240918);
    scale::Collector random_samples(/*plane_every=*/0), greedy_samples(/*plane_every=*/0);
    scale::PlayGames(random_games, /*greedy=*/false, rng, &random_samples, rules);
    printf("=== 隨機對局 (%d 局, %s) ===\n", random_games, diag::GameName(rules).c_str());
    scale::ReportScores(random_samples.scores);
    scale::PlayGames(greedy_games, /*greedy=*/true, rng, &greedy_samples, rules);
    printf("=== greedy 對局 (%d 局, %s) ===\n", greedy_games, diag::GameName(rules).c_str());
    scale::ReportScores(greedy_samples.scores);
    return 0;
}
