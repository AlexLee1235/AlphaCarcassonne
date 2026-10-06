// diag_plane_scale —— 空間平面的正規化尺度(carcassonne.cc 的 kMaxOpens /
// kFeatureScoreNormalization / kFieldScoreNormalization / kFieldSizeNormalization /
// kFieldOpenCitiesNormalization)。
//
// 照觀測的寫法逐格取值(scale_stats.hpp)。十個決策點取一個,另外每局一定取
// 最後一個決策點。兩種對局分佈:亂下與 greedy。自我對弈的分佈用 diag_replay_scale 量。
//
// 用法: ./diag_plane_scale [隨機局數=1000] [greedy 局數=300]
#include "scale_stats.hpp"

int main(int argc, char **argv) {
    const int random_games = argc > 1 ? atoi(argv[1]) : 1000;
    const int greedy_games = argc > 2 ? atoi(argv[2]) : 300;
    std::mt19937 rng(20261005);
    scale::Collector random_samples(/*plane_every=*/10), greedy_samples(/*plane_every=*/10);
    scale::PlayGames(random_games, /*greedy=*/false, rng, &random_samples);
    printf("=== 隨機對局 (%d 局) ===\n", random_games);
    scale::ReportPlanes(random_samples.planes);
    scale::PlayGames(greedy_games, /*greedy=*/true, rng, &greedy_samples);
    printf("=== greedy 對局 (%d 局) ===\n", greedy_games);
    scale::ReportPlanes(greedy_samples.planes);
    return 0;
}
