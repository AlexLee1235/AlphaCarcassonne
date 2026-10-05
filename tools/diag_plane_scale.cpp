// diag_plane_scale —— 空間平面的正規化尺度(carcassonne.cc 的 kMaxOpens /
// kFeatureScoreNormalization / kFieldScoreNormalization / kFieldSizeNormalization /
// kFieldOpenCitiesNormalization)。
//
// 照觀測的寫法逐格取值:每張已放磚的每個非草地邊(城/路元件的 opens 與
// getScore()),每個半邊(所屬農田的磚數、相鄰未完成城數、3 × 相鄰已完成城數)。
// 十個決策點取一個,另外每局一定取最後一個決策點。兩種對局分佈:亂下與 greedy。
//
// 用法: ./diag_plane_scale [隨機局數=1000] [greedy 局數=300]
#include "common.hpp"
#include <algorithm>

namespace {

struct Samples {
    std::vector<int> opens, feature_score, field_size, field_open, field_score;
    std::vector<int> last_opens, last_feature_score, last_field_size, last_field_open, last_field_score;
};

void Sample(const Carcassonne &game, bool last, Samples *s) {
    for (int y = 0; y < BOARD_SIZE; ++y) {
        for (int x = 0; x < BOARD_SIZE; ++x) {
            const Placement p = game.getPlacement(x, y);
            if (p.id == 0) continue;
            const Tile &tile = full_deck[p.id][p.rotation];
            for (int side = 0; side < 4; ++side) {
                if (tile.edge[side] == GRASS) continue;
                const Feature &feature = game.featureAt(p.id, side);
                s->opens.push_back(feature.opens);
                s->feature_score.push_back(feature.getScore());
                if (last) {
                    s->last_opens.push_back(feature.opens);
                    s->last_feature_score.push_back(feature.getScore());
                }
            }
            for (int e = 0; e < HALF_EDGE_COUNT; ++e) {
                if (tile.field[e] == -1) continue;
                const Field &field = game.fieldAtRoot(game.fieldRoot(p.id, tile.field[e]));
                const CityCounts cities = game.citiesNextTo(field);
                const int score = FIELD_POINTS_PER_CITY * cities.completed;
                s->field_size.push_back(field.getTileCount());
                s->field_open.push_back(cities.open);
                s->field_score.push_back(score);
                if (last) {
                    s->last_field_size.push_back(field.getTileCount());
                    s->last_field_open.push_back(cities.open);
                    s->last_field_score.push_back(score);
                }
            }
        }
    }
}

void Play(int games, bool greedy, std::mt19937 &rng, Samples *s) {
    for (int g = 0; g < games; ++g) {
        Carcassonne game;
        while (game.current_phase != PHASE_TERMINAL) {
            if (game.current_phase == PHASE_CHANCE) {
                if (!diag::SampleDraw(game, rng)) break;
                continue;
            }
            const bool last = game.getTotalRemaining() == 0 && game.current_phase == PHASE_MEEPLE;
            if (last || std::uniform_int_distribution<int>(0, 9)(rng) == 0) Sample(game, last, s);
            const int me = game.currentPlayer;
            if (game.current_phase == PHASE_TILE) {
                if (greedy) diag::GreedyPlaceTile(game, me);
                else if (!diag::RandomPlaceTile(game, rng)) break;
            } else {
                if (greedy) diag::GreedyPlaceMeeple(game, me);
                else if (!diag::RandomPlaceMeeple(game, rng)) break;
            }
        }
    }
}

void Report(std::vector<int> v, const char *name) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    auto q = [&](double p) { return v[(size_t)(p * (v.size() - 1))]; };
    double sum = 0;
    for (int x : v) sum += x;
    printf("  %-26s n=%-9zu mean %6.2f  p50 %3d  p90 %3d  p99 %3d  max %3d\n", name, v.size(), sum / v.size(),
           q(.50), q(.90), q(.99), v.back());
}

void ReportAll(const Samples &s, const char *title, int games) {
    printf("=== %s (%d 局) ===\n", title, games);
    Report(s.opens, "城/路 opens(每個非草地邊)");
    Report(s.last_opens, "  最後一手");
    Report(s.feature_score, "城/路 getScore()");
    Report(s.last_feature_score, "  最後一手");
    Report(s.field_size, "田的磚數(每個半邊)");
    Report(s.last_field_size, "  最後一手");
    Report(s.field_open, "相鄰未完成城數");
    Report(s.last_field_open, "  最後一手");
    Report(s.field_score, "3 × 相鄰已完成城數");
    Report(s.last_field_score, "  最後一手");
}

} // namespace

int main(int argc, char **argv) {
    const int random_games = argc > 1 ? atoi(argv[1]) : 1000;
    const int greedy_games = argc > 2 ? atoi(argv[2]) : 300;
    std::mt19937 rng(20261005);
    Samples random_samples, greedy_samples;
    Play(random_games, /*greedy=*/false, rng, &random_samples);
    ReportAll(random_samples, "隨機對局", random_games);
    Play(greedy_games, /*greedy=*/true, rng, &greedy_samples);
    ReportAll(greedy_samples, "greedy 對局", greedy_games);
    return 0;
}
