// diag_plane_scale —— 空間平面的正規化尺度(carcassonne.cc 的 kMaxOpens /
// kFeatureScoreNormalization / kFieldScoreNormalization / kFieldSizeNormalization /
// kFieldOpenCitiesNormalization)。
//
// 照觀測的寫法逐格取值(scale_stats.hpp)。十個決策點取一個,另外每局一定取
// 最後一個決策點。兩種對局分佈:亂下與 greedy。自我對弈的分佈用 diag_replay_scale 量。
//
// 用法: ./diag_plane_scale [隨機局數=1000] [greedy 局數=300] [牌組=base]
//   牌組:base、river、all(所有擴充都 on),或像 inns_cathedrals=on,princess_dragon=tiles 的寫法。
#include "scale_stats.hpp"

int main(int argc, char **argv) {
    const int random_games = argc > 1 ? atoi(argv[1]) : 1000;
    const int greedy_games = argc > 2 ? atoi(argv[2]) : 300;
    const diag::GameRules rules = diag::DeckArg(argc > 3 ? argv[3] : "base");
    std::mt19937 rng(20261005);
    scale::Collector random_samples(/*plane_every=*/10), greedy_samples(/*plane_every=*/10);
    scale::PlayGames(random_games, /*greedy=*/false, rng, &random_samples, rules);
    printf("=== 隨機對局 (%d 局, %s) ===\n", random_games, diag::GameName(rules).c_str());
    scale::ReportPlanes(random_samples.planes);
    scale::PlayGames(greedy_games, /*greedy=*/true, rng, &greedy_samples, rules);
    printf("=== greedy 對局 (%d 局, %s) ===\n", greedy_games, diag::GameName(rules).c_str());
    scale::ReportPlanes(greedy_samples.planes);
    return 0;
}
