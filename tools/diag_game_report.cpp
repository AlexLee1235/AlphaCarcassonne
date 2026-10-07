// 重播 actor log 格式的對局(GUI 存的 log-actor-gui.txt、訓練的 log-actor-*.txt),
// 把每個玩家的分數拆成來源,並列出 meeple 怎麼用掉。
//
//   ./build/diag_game_report <目錄或檔案> [最多逐局印幾局=20]
//
// 分數來源:局中完成的城 / 路 / 修道院(+),終局沒完成的城 / 路 / 修道院(~),農田,貨物獎金。
// 旅館、大教堂算在城/路裡(getScore() 已含),小豬算在農田裡。
// 歸因不改引擎:
//   - 放 meeple 時引擎只結算剛放的那張磚碰到的城/路(settleAfterPlaceMeeple),
//     所以在放之前看那張磚的城/路邊,已封口(opens == 0)的就依多數(含這手新放的,大米寶算 2)記 getScore()。
//   - 修道院 = 這手的分差扣掉城/路(與終局結算)之後的剩餘,必須是 9 的非負倍數。
//   - 終局照 FeatureModule::resolveEndGameScore 的迴圈分城/路;修道院、農田、貨物直接呼叫(都不改狀態)。
// 各項加總對不上最終分數、或勝負跟 log 的 Returns 不一致,就印 WARN —— 那一局的數字不能用。
#include "actor_log.hpp"

#include <filesystem>
#include <string>

namespace {

enum Source { CITY_DONE, ROAD_DONE, MONASTERY_DONE, CITY_END, ROAD_END, MONASTERY_END, FARM, GOODS, SOURCES };
const char *kSourceNames[SOURCES] = {"city+", "road+", "mon+", "city~", "road~", "mon~", "farm", "goods"};

struct PlayerReport {
    int points[SOURCES] = {};
    int final_score = 0;
    int turns = 0, hand_sum = 0, empty_hand_turns = 0;
    // 大米寶依位置算進前三類;建築師、小豬另計
    int on_feature = 0, on_monastery = 0, farmers = 0, skipped = 0, builders_pigs = 0;
    std::vector<int> farmer_turns;  // 自己的第幾手(從 1 起)放下農夫
    int quarter_hand[4] = {}, quarter_turns[4] = {};  // 自己第 1–9、10–18、19–27、28– 手前手上的 meeple

    int Total() const {
        int total = 0;
        for (int p : points) total += p;
        return total;
    }
};

// 動作字串 → meeple 位置(含大米寶、建築師、小豬的偏移),對應 ActionToString;看不懂回傳 -2。
int ParseMeeplePos(const std::string &a) {
    if (a == "place_meeple(skip)") return MEEPLE_POS_SKIP;
    struct Verb {
        const char *prefix;
        int offset;
    };
    const Verb verbs[] = {{"place_meeple(", 0},
                          {"place_big_meeple(", MEEPLE_POS_BIG},
                          {"place_builder(", MEEPLE_POS_BUILDER},
                          {"place_pig(", MEEPLE_POS_PIG - MEEPLE_POS_FIELD}};
    for (const Verb &verb : verbs) {
        const size_t len = strlen(verb.prefix);
        if (a.compare(0, len, verb.prefix) != 0) continue;
        const std::string arg = a.substr(len);
        int k = 0;
        if (arg == "monastery)") return verb.offset + MEEPLE_POS_MONASTERY;
        if (arg == "inner_field)") return verb.offset + MEEPLE_POS_INNER_FIELD;
        if (sscanf(arg.c_str(), "field=%d)", &k) == 1) return verb.offset + MEEPLE_POS_FIELD + k;
        if (sscanf(arg.c_str(), "edge=%d)", &k) == 1) return verb.offset + k;
        return -2;
    }
    return -2;
}

void Credit(int score, const int meeples[2], Source source, PlayerReport rep[2], int credited[2]) {
    for (int p = 0; p < 2; ++p) {
        if (meeples[p] >= meeples[1 - p]) {
            rep[p].points[source] += score;
            credited[p] += score;
        }
    }
}

// 這手 meeple 放下後 settleAfterPlaceMeeple 會結算的城/路。
void CreditCompletedFeatures(const Carcassonne &game, int pos, PlayerReport rep[2], int credited[2]) {
    Carcassonne g = game;  // find() 會壓縮路徑
    const int x = g.last_x, y = g.last_y, id = g.board.board[y][x].id, player = g.currentPlayer;
    // 新放的 follower 在哪個元件、算多少強度;建築師、小豬不是 follower,不算多數
    const bool follower = pos != MEEPLE_POS_SKIP && !isBuilderPos(pos) && !isPigPos(pos);
    const int spot = meepleSpot(pos);
    const int placed_root =
        follower && spot >= 0 && spot < 4 ? g.features.featureMap.find(g.features.edgeIndex(id, spot)) : -1;
    const int placed_strength = isBigMeeplePos(pos) ? 2 : 1;
    int seen[4], seen_count = 0;
    for (int side = 0; side < 4; ++side) {
        if (!isFeatureEdge(g.board.edge[y][x][side])) continue;
        const int root = g.features.featureMap.find(g.features.edgeIndex(id, side));
        if (std::find(seen, seen + seen_count, root) != seen + seen_count) continue;
        seen[seen_count++] = root;
        const Feature &feature = g.features.featureMap.getSetData(root);
        if (feature.opens != 0) continue;
        int meeples[2] = {feature.meeple_count[0], feature.meeple_count[1]};
        if (root == placed_root) meeples[player] += placed_strength;
        if (meeples[0] == 0 && meeples[1] == 0) continue;
        Credit(feature.getScore(), meeples, feature.type == CITY ? CITY_DONE : ROAD_DONE, rep, credited);
    }
}

// 終局結算(Carcassonne::resolveEndGameScore)的四部分。
void CreditEndGame(const Carcassonne &game, PlayerReport rep[2], int credited[2]) {
    Carcassonne g = game;
    for (auto it = g.features.featureMap.begin(); it != g.features.featureMap.end(); ++it) {
        const Feature &feature = *it;
        if (feature.opens == 0 || !isFeatureEdge(feature.type)) continue;
        const int meeples[2] = {feature.meeple_count[0], feature.meeple_count[1]};
        if (meeples[0] == 0 && meeples[1] == 0) continue;
        Credit(feature.getScore(), meeples, feature.type == CITY ? CITY_END : ROAD_END, rep, credited);
    }
    int monastery[2] = {0, 0}, farm[2] = {0, 0}, goods[2] = {0, 0};
    g.monasteries.resolveEndGameScore(monastery);
    g.fields.accumulateScore(farm, g.features);
    g.accumulateGoodsScore(goods);
    for (int p = 0; p < 2; ++p) {
        rep[p].points[MONASTERY_END] += monastery[p];
        rep[p].points[FARM] += farm[p];
        rep[p].points[GOODS] += goods[p];
        credited[p] += monastery[p] + farm[p] + goods[p];
    }
}

// 回傳空字串表示成功,否則是不能用這局的原因。
std::string ReportGame(const diag::LoggedGame &logged, PlayerReport rep[2]) {
    if (logged.truncated) return "行被截斷";
    Carcassonne g;
    std::string error;
    int own_turn[2] = {0, 0};
    bool end_credited = false;
    for (const std::string &action : logged.actions) {
        if (g.current_phase != PHASE_MEEPLE) {
            if (!diag::ApplyLoggedAction(g, action, &error, logged.board_shift)) return error;
            continue;
        }
        const int pl = g.currentPlayer, pos = ParseMeeplePos(action);
        if (pos == -2) return "看不懂的 meeple 動作: " + action;
        PlayerReport &me = rep[pl];
        const int hand = g.holding_meeples[pl];
        const int quarter = std::min(own_turn[pl] / 9, 3);
        own_turn[pl]++;
        me.turns++;
        me.hand_sum += hand;
        me.empty_hand_turns += hand == 0;
        me.quarter_hand[quarter] += hand;
        me.quarter_turns[quarter]++;
        if (pos == MEEPLE_POS_SKIP) {
            me.skipped++;
        } else if (isBuilderPos(pos) || isPigPos(pos)) {
            me.builders_pigs++;
        } else if (meepleSpot(pos) == MEEPLE_POS_MONASTERY) {
            me.on_monastery++;
        } else if (meepleSpot(pos) >= MEEPLE_POS_FIELD) {
            me.farmers++;
            me.farmer_turns.push_back(own_turn[pl]);
        } else {
            me.on_feature++;
        }

        int credited[2] = {0, 0};
        CreditCompletedFeatures(g, pos, rep, credited);
        const int before[2] = {g.player_scores[0], g.player_scores[1]};
        if (!diag::ApplyLoggedAction(g, action, &error, logged.board_shift)) return error;
        if (g.current_phase == PHASE_TERMINAL) {
            CreditEndGame(g, rep, credited);
            end_credited = true;
        }
        for (int p = 0; p < 2; ++p) {
            const int monastery = g.player_scores[p] - before[p] - credited[p];
            if (monastery < 0 || monastery % 9 != 0)
                return "P" + std::to_string(p + 1) + " 這手的分差拆不開(剩 " + std::to_string(monastery) + "): " + action;
            rep[p].points[MONASTERY_DONE] += monastery;
        }
    }
    if (g.current_phase != PHASE_TERMINAL) return "對局沒有下完";
    if (!end_credited) {  // 最後幾張牌都放不下,終局在抽牌時結算
        int ignored[2] = {0, 0};
        CreditEndGame(g, rep, ignored);
    }
    for (int p = 0; p < 2; ++p) {
        rep[p].final_score = g.player_scores[p];
        if (rep[p].Total() != rep[p].final_score)
            return "P" + std::to_string(p + 1) + " 拆出來的總和 " + std::to_string(rep[p].Total()) + " ≠ 最終分數 " +
                   std::to_string(rep[p].final_score);
    }
    const int diff = g.player_scores[0] - g.player_scores[1];
    const int sign = (diff > 0) - (diff < 0);
    const int logged_sign = (logged.returns[0] > 0) - (logged.returns[0] < 0);
    if (sign != logged_sign) return "勝負跟 log 的 Returns 不一致";
    return "";
}

void PrintHeader() {
    printf("        ");
    for (const char *name : kSourceNames) printf(" %6s", name);
    printf(" | %5s | farmers(own turn)  feature mon skip b+p | hand avg, by quarter          0-hand\n", "total");
}

void PrintRow(const char *label, const PlayerReport &r, double games) {
    printf("  %-5s ", label);
    for (int s = 0; s < SOURCES; ++s) printf(" %6.1f", r.points[s] / games);
    std::string turns;
    if (games == 1) {
        for (size_t i = 0; i < r.farmer_turns.size(); ++i) turns += (i ? "," : "(") + std::to_string(r.farmer_turns[i]);
        if (!turns.empty()) turns += ")";
    }
    printf(" | %5.1f | %4.1f %-13s", r.Total() / games, r.farmers / games, turns.c_str());
    printf(" %7.1f %3.1f %4.1f %3.1f | %4.2f,", r.on_feature / games, r.on_monastery / games, r.skipped / games,
           r.builders_pigs / games,
           (double)r.hand_sum / std::max(1, r.turns));
    for (int q = 0; q < 4; ++q) printf(" %4.2f", (double)r.quarter_hand[q] / std::max(1, r.quarter_turns[q]));
    printf("   %3.0f%%\n", 100.0 * r.empty_hand_turns / std::max(1, r.turns));
}

void Add(PlayerReport &sum, const PlayerReport &r) {
    for (int s = 0; s < SOURCES; ++s) sum.points[s] += r.points[s];
    sum.final_score += r.final_score;
    sum.turns += r.turns;
    sum.hand_sum += r.hand_sum;
    sum.empty_hand_turns += r.empty_hand_turns;
    sum.on_feature += r.on_feature;
    sum.on_monastery += r.on_monastery;
    sum.farmers += r.farmers;
    sum.skipped += r.skipped;
    sum.builders_pigs += r.builders_pigs;
    for (int q = 0; q < 4; ++q) {
        sum.quarter_hand[q] += r.quarter_hand[q];
        sum.quarter_turns[q] += r.quarter_turns[q];
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "用法: %s <目錄或 log 檔> [最多逐局印幾局=20]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int max_print = argc > 2 ? atoi(argv[2]) : 20;
    const std::vector<diag::LoggedGame> games =
        std::filesystem::is_directory(path) ? diag::ReadActorLogs(path) : diag::ReadActorLog(path);
    printf("%s: %zu 局。+ = 局中完成, ~ = 終局沒完成; hand = 自己回合開始時手上的 meeple,\n"
           "quarter = 自己第 1-9 / 10-18 / 19-27 / 28- 手; 0-hand = 手上沒有 meeple 的回合比例\n\n",
           path.c_str(), games.size());

    PlayerReport sum[2];
    int ok = 0, failed = 0;
    for (size_t i = 0; i < games.size(); ++i) {
        PlayerReport rep[2];
        const std::string error = ReportGame(games[i], rep);
        if (!error.empty()) {
            printf("WARN %s (%s): %s\n", games[i].source.c_str(), games[i].time.c_str(), error.c_str());
            failed++;
            continue;
        }
        ok++;
        Add(sum[0], rep[0]);
        Add(sum[1], rep[1]);
        if (ok <= max_print) {
            printf("%s (%s): P1 %d, P2 %d\n", games[i].source.c_str(), games[i].time.c_str(), rep[0].final_score,
                   rep[1].final_score);
            PrintHeader();
            PrintRow("P1", rep[0], 1);
            PrintRow("P2", rep[1], 1);
            printf("\n");
        }
    }
    if (ok > 1) {
        printf("平均(%d 局):\n", ok);
        PrintHeader();
        PrintRow("P1", sum[0], ok);
        PrintRow("P2", sum[1], ok);
    }
    if (failed) printf("\n%d 局拆解失敗(見上面的 WARN)\n", failed);
    return failed ? 1 : 0;
}
