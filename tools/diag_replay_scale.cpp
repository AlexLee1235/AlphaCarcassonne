// diag_replay_scale —— 在自我對弈的分佈上量正規化尺度:重播訓練的 actor log
// (actor_log.hpp),量的東西與 diag_pending_scale / diag_plane_scale 完全相同
// (scale_stats.hpp)。
//
// 網路訓練吃的就是這些局面,所以比 greedy 代理準。但尺度要涵蓋整個訓練過程,
// 前期的網路接近亂下,所以定值時仍要跟隨機對局比(CLAUDE.md §4.1)。
// 依每局結束時 learner 已完成的步數分段,看分佈怎麼隨訓練漂移。
// 只有 actor 0..19 寫 log(alpha_zero.cc),所以這是自我對弈的一個抽樣。
//
// 用法: ./diag_replay_scale <訓練目錄> [分段數=2]
//   訓練目錄要有 log-actor-*.txt;有 log-learner.txt 才能依步數分段。
//   牌組與規則照 config.json 的 "game"(actor_log.hpp)。
#include "actor_log.hpp"
#include "scale_stats.hpp"

#include <functional>
#include <map>

namespace {

struct Row {
    const char *constant;
    const char *name;
    std::function<const std::vector<int> &(const scale::Collector &)> samples;
};

// 每列一個尺度,最後一手的 p99 / max。中文名稱放最後,欄位才對得齊。
const Row kRows[] = {
    {"kScoreNormalization", "已入袋 單一玩家", [](const scale::Collector &c) -> const std::vector<int> & { return c.scores.last_banked; }},
    {"kScoreDiffNormalization", "已入袋 分差", [](const scale::Collector &c) -> const std::vector<int> & { return c.scores.last_banked_diff; }},
    {"kPendingNormalization", "總 pending 單一玩家", [](const scale::Collector &c) -> const std::vector<int> & { return c.scores.last_per; }},
    {"kFieldPendingNormalization", "農田 pending 單一玩家", [](const scale::Collector &c) -> const std::vector<int> & { return c.scores.last_field_per; }},
    {"kStaticDiffNormalizations", "static_diff(/30 那個)", [](const scale::Collector &c) -> const std::vector<int> & { return c.scores.last_static_diff; }},
    {"kMaxOpens", "城/路 opens", [](const scale::Collector &c) -> const std::vector<int> & { return c.planes.last_opens; }},
    {"kFeatureScoreNormalization", "城/路 getScore()", [](const scale::Collector &c) -> const std::vector<int> & { return c.planes.last_feature_score; }},
    {"kFieldSizeNormalization", "田的磚數", [](const scale::Collector &c) -> const std::vector<int> & { return c.planes.last_field_size; }},
    {"kFieldOpenCitiesNormalization", "相鄰未完成城數", [](const scale::Collector &c) -> const std::vector<int> & { return c.planes.last_field_open; }},
    {"kFieldScoreNormalization", "3 × 相鄰已完成城數", [](const scale::Collector &c) -> const std::vector<int> & { return c.planes.last_field_score; }},
};

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "用法: %s <訓練目錄> [分段數=2]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    int buckets = argc > 2 ? std::max(1, atoi(argv[2])) : 2;
    const auto steps = diag::ReadLearnerSteps(dir + "/log-learner.txt");
    if (steps.empty() && buckets > 1) {
        fprintf(stderr, "沒有 log-learner.txt 的步數,不分段\n");
        buckets = 1;
    }
    const std::vector<diag::LoggedGame> games = diag::ReadActorLogs(dir);
    if (games.empty()) {
        fprintf(stderr, "%s 裡沒有 log-actor-*.txt 的對局\n", dir.c_str());
        return 2;
    }

    // 第一遍:確認每局都能照現在的規則重播完,勝負也跟 log 一致。
    std::vector<size_t> ok;
    std::vector<int> game_step(games.size());
    int errors = 0, unfinished = 0;
    for (size_t i = 0; i < games.size(); ++i) {
        game_step[i] = diag::StepAt(steps, games[i].time);
        std::string error;
        switch (diag::Replay(games[i], [](const Carcassonne &) {}, &error)) {
        case diag::ReplayResult::kOk:
            ok.push_back(i);
            break;
        case diag::ReplayResult::kError:
            if (errors++ < 5) fprintf(stderr, "重播失敗 %s: %s\n", games[i].source.c_str(), error.c_str());
            break;
        case diag::ReplayResult::kUnfinished:
            if (unfinished++ < 5) fprintf(stderr, "略過 %s: %s\n", games[i].source.c_str(), error.c_str());
            break;
        }
    }
    printf("讀到 %zu 局:重播成功 %zu,失敗 %d,沒下完 %d\n", games.size(), ok.size(), errors, unfinished);
    std::map<std::string, int> decks;
    for (const diag::LoggedGame &g : games) decks[diag::GameName(g.rules)]++;
    for (const auto &[name, count] : decks) printf("  牌組 %s: %d 局\n", name.c_str(), count);
    if (errors > 0) printf("!! 有對局重播失敗:log 與現在的規則或動作格式不一致,下面的數字只含成功的局\n");
    if (ok.empty()) return 1;

    int max_step = 0;
    for (size_t i : ok) max_step = std::max(max_step, game_step[i]);
    auto bucket_of = [&](int step) { return std::min(buckets - 1, step * buckets / (max_step + 1)); };

    // 第二遍:量。
    std::vector<scale::Collector> per_bucket(buckets, scale::Collector(/*plane_every=*/10));
    std::vector<int> lo(buckets, max_step), hi(buckets, 0);
    scale::Collector all(/*plane_every=*/10);
    for (size_t i : ok) {
        const int b = bucket_of(game_step[i]);
        lo[b] = std::min(lo[b], game_step[i]);
        hi[b] = std::max(hi[b], game_step[i]);
        std::string error;
        diag::Replay(games[i], [&](const Carcassonne &game) {
            all.Decision(game);
            per_bucket[b].Decision(game);
        }, &error);
        all.EndGame();
        per_bucket[b].EndGame();
    }

    printf("\n=== 最後一手 p99 / max,依局結束時 learner 的步數分段 ===\n");
    std::vector<const scale::Collector *> columns;
    char label[64];
    for (int b = 0; b < buckets; ++b) {
        if (per_bucket[b].games == 0) continue;
        snprintf(label, sizeof(label), "step %d-%d (%d)", lo[b], hi[b], per_bucket[b].games);
        printf("%18s", label);
        columns.push_back(&per_bucket[b]);
    }
    if (columns.size() > 1) {
        snprintf(label, sizeof(label), "all (%d)", all.games);
        printf("%18s", label);
        columns.push_back(&all);
    }
    printf("\n");
    for (const Row &row : kRows) {
        for (const scale::Collector *c : columns) {
            const std::vector<int> &v = row.samples(*c);
            snprintf(label, sizeof(label), "%d / %d", scale::Quantile(v, .99),
                     v.empty() ? 0 : *std::max_element(v.begin(), v.end()));
            printf("%18s", label);
        }
        printf("   %-30s %s\n", row.constant, row.name);
    }

    printf("\n=== 全部 (%d 局) ===\n", all.games);
    scale::ReportScores(all.scores);
    scale::ReportPlanes(all.planes);
    return errors > 0 ? 1 : 0;
}
