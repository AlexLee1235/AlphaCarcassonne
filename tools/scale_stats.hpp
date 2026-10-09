// 正規化尺度的量測(CLAUDE.md §4)。diag_pending_scale、diag_plane_scale、
// diag_replay_scale 共用這裡的取樣,保證三支工具量的是同一個東西,只差對局分佈。
//
// 用法:每個決策點(放磚、放 meeple、選格、選位置、龍)呼叫 Collector::Decision(),每局結束
// 呼叫 Collector::EndGame()。「最後一手」是每局最後一個決策點。
#pragma once

#include "common.hpp"

namespace scale {

// global vector 的分數類(kScoreNormalization、kScoreDiffNormalization、
// kPendingNormalization、kFieldPendingNormalization、static_diff 的尺度)。
// 量的就是觀測寫進 global 的值:getPendingScore()(總 pending)、
// getPendingFieldScore()(其中農田的部分)與 player_scores(已入袋)。每個決策點都取。
// 另外 kLegalPlacementNormalization:放磚決策的合法落點數(沒有「最後一手」,看全程)。
struct ScoreSamples {
    std::vector<int> per, diff, last_per, last_diff;                      // 總 pending
    std::vector<int> field_per, last_field_per;                           // 農田 pending
    std::vector<int> banked, banked_diff, last_banked, last_banked_diff;  // 已入袋
    std::vector<int> static_diff, last_static_diff;                       // 已入袋分差 + pending 分差
    std::vector<int> legal_placements;                                    // 放磚決策的合法落點數

    // 取一個決策點,並把它記成這局目前的最後一手。
    void Decision(const Carcassonne &game) {
        int pending[2], field[2];
        game.getPendingScore(pending);
        game.getPendingFieldScore(field);
        for (int p = 0; p < 2; ++p) {
            per.push_back(pending[p]);
            field_per.push_back(field[p]);
            banked.push_back(game.player_scores[p]);
            last_pending_[p] = pending[p];
            last_field_[p] = field[p];
            last_banked_[p] = game.player_scores[p];
        }
        diff.push_back(pending[0] - pending[1]);
        banked_diff.push_back(game.player_scores[0] - game.player_scores[1]);
        last_static_ = game.player_scores[0] - game.player_scores[1] + pending[0] - pending[1];
        static_diff.push_back(last_static_);
        if (game.current_phase == PHASE_TILE) {
            std::vector<TileMove> moves(BOARD_SIZE * BOARD_SIZE * 4);
            int count = 0;
            game.getLegalTileMoves(moves.data(), count);
            legal_placements.push_back(count);
        }
    }

    void EndGame() {
        for (int p = 0; p < 2; ++p) {
            last_per.push_back(last_pending_[p]);
            last_field_per.push_back(last_field_[p]);
            last_banked.push_back(last_banked_[p]);
        }
        last_diff.push_back(last_pending_[0] - last_pending_[1]);
        last_banked_diff.push_back(last_banked_[0] - last_banked_[1]);
        last_static_diff.push_back(last_static_);
    }

  private:
    int last_pending_[2] = {}, last_field_[2] = {}, last_banked_[2] = {}, last_static_ = 0;
};

// 空間平面(kMaxOpens、kFeatureScoreNormalization、kFieldScoreNormalization、
// kFieldSizeNormalization、kFieldOpenCitiesNormalization)。照觀測的寫法逐格取值:
// 每張已放磚的每個非草地邊(城/路元件的 opens、getBaseScore() 與 getScore()),每個半邊
// (所屬農田的磚數、相鄰未完成城數、3 × 相鄰已完成城數)。一個決策點就有上百個值。
// 元件的分數平面寫 getBaseScore()(不含旅館、大教堂),有號分數平面寫 getScore(),兩者同一個分母。
struct PlaneSamples {
    std::vector<int> opens, feature_score, feature_base_score, field_size, field_open, field_score;
    std::vector<int> last_opens, last_feature_score, last_feature_base_score, last_field_size, last_field_open,
        last_field_score;

    // last 為真時取進「最後一手」那組,否則取進全程那組。
    void Sample(const Carcassonne &game, bool last) {
        std::vector<int> &o = last ? last_opens : opens;
        std::vector<int> &fs = last ? last_feature_score : feature_score;
        std::vector<int> &base = last ? last_feature_base_score : feature_base_score;
        std::vector<int> &size = last ? last_field_size : field_size;
        std::vector<int> &open = last ? last_field_open : field_open;
        std::vector<int> &score = last ? last_field_score : field_score;
        for (int y = 0; y < BOARD_SIZE; ++y) {
            for (int x = 0; x < BOARD_SIZE; ++x) {
                const Placement p = game.getPlacement(x, y);
                if (p.id == 0) continue;
                const Tile &tile = full_deck[p.id][p.rotation];
                for (int side = 0; side < 4; ++side) {
                    if (!isFeatureEdge(tile.edge[side])) continue;
                    const Feature &feature = game.featureAt(p.id, side);
                    o.push_back(feature.opens);
                    fs.push_back(feature.getScore());
                    base.push_back(feature.getBaseScore());
                }
                for (int e = 0; e < HALF_EDGE_COUNT; ++e) {
                    if (tile.field[e] == -1) continue;
                    const Field &field = game.fieldAtRoot(game.fieldRoot(p.id, tile.field[e]));
                    const CityCounts cities = game.citiesNextTo(field);
                    size.push_back(field.getTileCount());
                    open.push_back(cities.open);
                    score.push_back(FIELD_POINTS_PER_CITY * cities.completed);
                }
            }
        }
    }
};

// plane_every = 0 不量空間平面;否則每 plane_every 個決策點取一次全程,
// 並在 EndGame() 取最後一手。
class Collector {
  public:
    explicit Collector(int plane_every) : plane_every_(plane_every) {}

    void Decision(const Carcassonne &game) {
        scores.Decision(game);
        if (plane_every_ == 0) return;
        if (decisions_++ % plane_every_ == 0) planes.Sample(game, /*last=*/false);
        last_decision_ = game;
        has_last_decision_ = true;
    }

    void EndGame() {
        ++games;
        scores.EndGame();
        if (has_last_decision_) planes.Sample(last_decision_, /*last=*/true);
        has_last_decision_ = false;
    }

    ScoreSamples scores;
    PlaneSamples planes;
    int games = 0;

  private:
    int plane_every_;
    long long decisions_ = 0;
    Carcassonne last_decision_;
    bool has_last_decision_ = false;
};

// 雙方都亂下,或雙方都 greedy(diag::GreedyStep,最大化 banked + pending,強玩家代理,§2.1 警告框)。
// rules 是牌組與規則(diag::DeckArg)。
inline void PlayGames(int games, bool greedy, std::mt19937 &rng, Collector *c,
                      const diag::GameRules &rules = diag::GameRules()) {
    for (int g = 0; g < games; ++g) {
        Carcassonne game = diag::NewGame(rules);
        while (game.current_phase != PHASE_TERMINAL) {
            if (game.current_phase == PHASE_CHANCE) {
                if (!diag::SampleDraw(game, rng)) break;
                continue;
            }
            c->Decision(game);
            if (!(greedy ? diag::GreedyStep(game) : diag::RandomStep(game, rng))) break;
        }
        c->EndGame();
    }
}

inline int Quantile(std::vector<int> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[(size_t)(q * (v.size() - 1))];
}

inline void Report(std::vector<int> v, const char *name) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    auto q = [&](double p) { return v[(size_t)(p * (v.size() - 1))]; };
    double sum = 0, abs_sum = 0;
    for (int x : v) { sum += x; abs_sum += std::abs(x); }
    printf("  %-26s n=%-9zu mean %6.2f  mean|.| %6.2f  p50 %3d  p90 %3d  p99 %3d  max %3d\n", name, v.size(),
           sum / v.size(), abs_sum / v.size(), q(.50), q(.90), q(.99), v.back());
}

inline void ReportScores(const ScoreSamples &s) {
    Report(s.banked, "已入袋 單一玩家");
    Report(s.last_banked, "  最後一手");
    Report(s.banked_diff, "已入袋 分差");
    Report(s.last_banked_diff, "  最後一手");
    Report(s.per, "總 pending 單一玩家");
    Report(s.last_per, "  最後一手");
    Report(s.diff, "總 pending 分差");
    Report(s.last_diff, "  最後一手");
    Report(s.field_per, "農田 pending 單一玩家");
    Report(s.last_field_per, "  最後一手");
    Report(s.static_diff, "static_diff");
    Report(s.last_static_diff, "  最後一手");
    Report(s.legal_placements, "合法落點數(放磚決策)");
}

inline void ReportPlanes(const PlaneSamples &s) {
    Report(s.opens, "城/路 opens(每個非草地邊)");
    Report(s.last_opens, "  最後一手");
    Report(s.feature_base_score, "城/路 getBaseScore()");
    Report(s.last_feature_base_score, "  最後一手");
    Report(s.feature_score, "城/路 getScore()");
    Report(s.last_feature_score, "  最後一手");
    Report(s.field_size, "田的磚數(每個半邊)");
    Report(s.last_field_size, "  最後一手");
    Report(s.field_open, "相鄰未完成城數");
    Report(s.last_field_open, "  最後一手");
    Report(s.field_score, "3 × 相鄰已完成城數");
    Report(s.last_field_score, "  最後一手");
}

} // namespace scale
