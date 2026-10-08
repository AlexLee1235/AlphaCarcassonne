// 讀訓練的 actor log,把自我對弈的對局重播到引擎上。
//
// alpha_zero.cc 的 PlayGame 每局結束寫一行
//   [2026-10-04 09:20:45.850] Game 1: Returns: -1 1; Actions: draw_type(16) place_tile(x=10, y=11, rot=1) ...
// 動作含抽牌,字串是 CarcassonneState::ActionToString 的格式,ApplyLoggedAction 照它解析。
// 格式若改了,解析或合法性檢查會失敗,不會默默重播錯。
#pragma once

#include "common.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace diag {

struct LoggedGame {
    std::string source;  // "log-actor-3.txt Game 7"
    std::string time;    // 這局結束的時間,與 log 同格式,可直接比字串
    double returns[2] = {0, 0};
    std::vector<std::string> actions;
    bool truncated = false;  // 行被截斷(例如下載時 log 還在寫)
    int board_shift = 0;     // 加到 place_tile 的座標上,見 BoardShift
};

// 盤面從 21x21 加大成 BOARD_SIZE 以前的 log,起始磚在 (10, 10),格子都比現在少
// BOARD_SIZE / 2 - 10。第一張磚一定貼著起始磚(或河源),看它就分得出來。
inline int BoardShift(const std::vector<std::string> &actions) {
    constexpr int kOldCentre = 10;
    for (const std::string &a : actions) {
        int x = 0, y = 0, rot = 0;
        if (sscanf(a.c_str(), "place_tile(x=%d, y=%d, rot=%d)", &x, &y, &rot) == 3)
            return std::abs(x - kOldCentre) + std::abs(y - kOldCentre) == 1 ? BOARD_SIZE / 2 - kOldCentre : 0;
    }
    return 0;
}

// 一個檔案裡所有 "Game N: Returns: ...; Actions: ..." 行。
inline std::vector<LoggedGame> ReadActorLog(const std::string &path) {
    std::vector<LoggedGame> games;
    std::ifstream in(path);
    const std::string file = std::filesystem::path(path).filename().string();
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t close = line.find("] Game ");
        const size_t actions = line.find("; Actions: ");
        if (line.empty() || line[0] != '[' || close == std::string::npos || actions == std::string::npos) continue;
        LoggedGame g;
        g.time = line.substr(1, close - 1);
        int number = 0;
        if (sscanf(line.c_str() + close, "] Game %d: Returns: %lf %lf", &number, &g.returns[0], &g.returns[1]) != 3)
            continue;
        g.source = file + " Game " + std::to_string(number);
        size_t i = actions + strlen("; Actions: ");
        while (i < line.size()) {
            while (i < line.size() && line[i] == ' ') ++i;
            if (i == line.size()) break;
            const size_t end = line.find(')', i);
            if (end == std::string::npos) {
                g.truncated = true;
                break;
            }
            g.actions.push_back(line.substr(i, end - i + 1));
            i = end + 1;
        }
        g.board_shift = BoardShift(g.actions);
        games.push_back(std::move(g));
    }
    return games;
}

// 目錄裡所有 log-actor-*.txt。
inline std::vector<LoggedGame> ReadActorLogs(const std::string &dir) {
    std::vector<std::string> paths;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("log-actor-", 0) == 0 && name.size() > 4 && name.compare(name.size() - 4, 4, ".txt") == 0)
            paths.push_back(entry.path().string());
    }
    std::sort(paths.begin(), paths.end());
    std::vector<LoggedGame> games;
    for (const std::string &path : paths) {
        std::vector<LoggedGame> more = ReadActorLog(path);
        games.insert(games.end(), more.begin(), more.end());
    }
    return games;
}

// log-learner.txt 的 "[時間] Step: N",依時間排序。
inline std::vector<std::pair<std::string, int>> ReadLearnerSteps(const std::string &path) {
    std::vector<std::pair<std::string, int>> steps;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const size_t close = line.find("] Step: ");
        if (line.empty() || line[0] != '[' || close == std::string::npos) continue;
        steps.emplace_back(line.substr(1, close - 1), atoi(line.c_str() + close + strlen("] Step: ")));
    }
    return steps;
}

// time 當下 learner 已完成的步數(之前沒有任何一步就是 0)。
inline int StepAt(const std::vector<std::pair<std::string, int>> &steps, const std::string &time) {
    int step = 0;
    for (const auto &[t, s] : steps) {
        if (t > time) break;
        step = s;
    }
    return step;
}

// 把一個 ActionToString 字串套到引擎上。階段不對、看不懂或不合法就回傳 false。
// board_shift 是 LoggedGame::board_shift。
inline bool ApplyLoggedAction(Carcassonne &g, const std::string &a, std::string *error, int board_shift = 0) {
    auto fail = [&](const char *why) {
        *error = std::string(why) + ": " + a;
        return false;
    };
    int k = 0, x = 0, y = 0, rot = 0;
    if (sscanf(a.c_str(), "draw_type(%d)", &k) == 1) {
        if (g.current_phase != PHASE_CHANCE) return fail("不在抽牌階段");
        ChanceBranch draws[CANONICAL_TILE_TYPE_COUNT];
        int count = 0;
        g.getAvailableDraws(draws, count);
        if (std::none_of(draws, draws + count, [&](const ChanceBranch &d) { return d.type_id == k; }))
            return fail("牌堆裡沒有這種牌");
        g.drawTile(k);
        return true;
    }
    if (sscanf(a.c_str(), "place_tile(x=%d, y=%d, rot=%d)", &x, &y, &rot) == 3) {
        if (g.current_phase != PHASE_TILE) return fail("不在放磚階段");
        x += board_shift;
        y += board_shift;
        std::vector<TileMove> moves(BOARD_SIZE * BOARD_SIZE * 4);
        int count = 0;
        g.getLegalTileMoves(moves.data(), count);
        if (std::none_of(moves.begin(), moves.begin() + count,
                         [&](const TileMove &m) { return m.x == x && m.y == y && m.rot == rot; }))
            return fail("不合法的落點");
        g.placeTile(x, y, rot);
        return true;
    }
    const std::string prefix = "place_meeple(";
    const std::string big_prefix = "place_big_meeple(";
    const std::string builder_prefix = "place_builder(";
    const std::string pig_prefix = "place_pig(";
    const bool big = a.rfind(big_prefix, 0) == 0;
    const bool builder = a.rfind(builder_prefix, 0) == 0;
    const bool pig = a.rfind(pig_prefix, 0) == 0;
    if ((big || builder || pig || a.rfind(prefix, 0) == 0) && a.back() == ')') {
        const size_t start = (big ? big_prefix : builder ? builder_prefix : pig ? pig_prefix : prefix).size();
        const std::string arg = a.substr(start, a.size() - start - 1);
        int pos;
        if (pig) {
            // 小豬只放半邊所屬的田
            if (sscanf(arg.c_str(), "field=%d", &k) != 1) return fail("看不懂的小豬位置");
            pos = MEEPLE_POS_PIG + k;
        }
        else if (arg == "skip" && !big && !builder) pos = MEEPLE_POS_SKIP;
        else if (sscanf(arg.c_str(), "edge=%d", &k) == 1) pos = k;
        else if (builder) return fail("看不懂的建築師位置");  // 建築師只放城、路
        else if (arg == "monastery") pos = MEEPLE_POS_MONASTERY;
        else if (arg == "inner_field") pos = MEEPLE_POS_INNER_FIELD;
        else if (sscanf(arg.c_str(), "field=%d", &k) == 1) pos = MEEPLE_POS_FIELD + k;
        else return fail("看不懂的 meeple 位置");
        if (big) pos += MEEPLE_POS_BIG;  // 大米寶、建築師放在同樣的位置
        if (builder) pos += MEEPLE_POS_BUILDER;
        if (g.current_phase != PHASE_MEEPLE) return fail("不在放 meeple 階段");
        const MeepleMoves moves = g.getLegalMeepleMoves();
        bool legal = false;
        for (int i = 0; i < moves.size(); ++i) legal |= moves[i] == pos;
        if (!legal) return fail("不合法的 meeple 位置");
        g.placeMeeple(pos);
        return true;
    }
    char dir = 0;
    if (sscanf(a.c_str(), "move_dragon(dir=%c)", &dir) == 1) {
        const char *sides = "NESW";
        const char *side = std::strchr(sides, dir);
        if (dir == 0 || side == nullptr) return fail("看不懂的龍的方向");
        if (g.current_phase != PHASE_DRAGON) return fail("不在龍移動的階段");
        const FixedVector<int, 4> moves = g.getLegalDragonMoves();
        if (std::none_of(moves.begin(), moves.end(), [&](int m) { return m == side - sides; }))
            return fail("龍不能往這邊走");
        g.moveDragon(static_cast<int>(side - sides));
        return true;
    }
    return fail("看不懂的動作");
}

enum class ReplayResult { kOk, kError, kUnfinished };

// 重播一局,每個決策點(放磚與放 meeple)套用前呼叫 on_decision(game)。
// kOk 表示下完,而且勝負與 log 記的 Returns 一致。
template <typename OnDecision>
ReplayResult Replay(const LoggedGame &logged, OnDecision on_decision, std::string *error) {
    Carcassonne game;
    for (const std::string &action : logged.actions) {
        if (game.current_phase == PHASE_TERMINAL) {
            *error = "終局後還有動作: " + action;
            return ReplayResult::kError;
        }
        if (game.current_phase != PHASE_CHANCE) on_decision(static_cast<const Carcassonne &>(game));
        if (!ApplyLoggedAction(game, action, error, logged.board_shift)) return ReplayResult::kError;
    }
    if (game.current_phase != PHASE_TERMINAL) {
        *error = logged.truncated ? "行被截斷" : "沒下完(提前認輸?)";
        return ReplayResult::kUnfinished;
    }
    auto sign = [](double v) { return (v > 0) - (v < 0); };
    if (sign(game.player_scores[0] - game.player_scores[1]) != sign(logged.returns[0])) {
        *error = "終局分數 " + std::to_string(game.player_scores[0]) + ":" + std::to_string(game.player_scores[1]) +
                 " 與 log 的 Returns 不符";
        return ReplayResult::kError;
    }
    return ReplayResult::kOk;
}

} // namespace diag
