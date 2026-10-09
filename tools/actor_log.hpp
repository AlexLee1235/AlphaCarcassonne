// 讀訓練的 actor log,把自我對弈的對局重播到引擎上。
//
// alpha_zero.cc 的 PlayGame 每局結束寫一行
//   [2026-10-04 09:20:45.850] Game 1: Returns: -1 1; Actions: draw_type(16) place_tile(x=10, y=11, rot=1) ...
// 動作含抽牌,字串是 CarcassonneState::ActionToString 的格式,ParseLoggedMove 照它解析。
// 格式若改了,解析或合法性檢查會失敗,不會默默重播錯。
//
// 牌組與規則:每一行的 "; Game: <遊戲字串>"(alpha_zero.cc 2026-10-09 起每局都寫,混合規則的 run
// 每局不同);沒有的話,訓練的 log 看同目錄 config.json 的 "game"(alpha_zero.cc 寫的),
// GUI 的 log 看檔名(play/engine/adapter.py 的 game_log_file)。
#pragma once

#include "common.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace diag {

struct LoggedGame {
    std::string source;  // "log-actor-3.txt Game 7"
    std::string time;    // 這局結束的時間,與 log 同格式,可直接比字串
    double returns[2] = {0, 0};
    std::vector<std::string> actions;
    bool truncated = false;  // 行被截斷(例如下載時 log 還在寫)
    int board_shift = 0;     // 加到 place_tile 與選格的座標上,見 BoardShift
    GameRules rules;         // 這局的牌組與規則:行裡的 "; Game: ",沒有就看 LogGameString
    std::string rules_error; // 讀不出規則時的原因;不是空的就不重播
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

// GUI 的 log 檔名帶著擴充:log-actor-gui.txt 是基本版,
// log-actor-gui-inns_cathedrals-tiles-river.txt 是 inns_cathedrals=tiles、river=on
// (擴充名照字母排,後面跟著 tiles 或 off,沒跟就是 on)。不是 GUI 的檔名就回傳 false。
inline bool GuiLogGameString(const std::string &file, std::string *game) {
    const std::string prefix = "log-actor-gui", suffix = ".txt";
    if (file.rfind(prefix, 0) != 0 || file.size() < prefix.size() + suffix.size() ||
        file.compare(file.size() - suffix.size(), suffix.size(), suffix) != 0)
        return false;
    std::vector<std::string> tokens;
    std::stringstream rest(file.substr(prefix.size(), file.size() - prefix.size() - suffix.size()));
    for (std::string token; std::getline(rest, token, '-');)
        if (!token.empty()) tokens.push_back(token);
    game->clear();
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::string mode = "on";
        if (i + 1 < tokens.size() && (tokens[i + 1] == "tiles" || tokens[i + 1] == "off")) mode = tokens[++i];
        *game += (game->empty() ? "" : ",") + tokens[i] + "=" + mode;
    }
    return true;
}

// 訓練目錄 config.json 裡的 "game"。沒有這個檔回傳 false。
inline bool RunGameString(const std::string &dir, std::string *game) {
    std::ifstream in(dir + "/config.json");
    if (!in) return false;
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    const size_t key = text.find("\"game\"");
    const size_t colon = key == std::string::npos ? key : text.find(':', key);
    const size_t open = colon == std::string::npos ? colon : text.find('"', colon);
    const size_t close = open == std::string::npos ? open : text.find('"', open + 1);
    if (close == std::string::npos) return false;
    *game = text.substr(open + 1, close - open - 1);
    return true;
}

// 一個 log 檔的遊戲字串:GUI 看檔名,其他看同目錄的 config.json;都沒有就當基本版,並在 *note 說明。
inline std::string LogGameString(const std::string &path, std::string *note) {
    const std::filesystem::path p(path);
    std::string game;
    if (GuiLogGameString(p.filename().string(), &game)) return game;
    const std::string dir = p.parent_path().empty() ? "." : p.parent_path().string();
    if (RunGameString(dir, &game)) return game;
    *note = p.filename().string() + " 旁邊沒有 config.json,當成基本版";
    return "";
}

// 一個檔案裡所有 "Game N: Returns: ...; Actions: ..." 行。
inline std::vector<LoggedGame> ReadActorLog(const std::string &path) {
    std::vector<LoggedGame> games;
    std::ifstream in(path);
    const std::string file = std::filesystem::path(path).filename().string();
    std::string note, rules_error;
    GameRules rules;
    if (!ParseGameString(LogGameString(path, &note), &rules, &rules_error)) rules_error = file + ": " + rules_error;
    bool note_printed = false;
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
        const size_t game_field = line.find("; Game: ");
        if (game_field != std::string::npos && game_field < actions) {
            const size_t start = game_field + strlen("; Game: ");
            if (!ParseGameString(line.substr(start, actions - start), &g.rules, &g.rules_error))
                g.rules_error = g.source + ": " + g.rules_error;
        } else {
            g.rules = rules;
            g.rules_error = rules_error;
            if (!note.empty() && !note_printed) fprintf(stderr, "%s\n", note.c_str());
            note_printed = true;
        }
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

inline const char *PhaseName(GamePhase phase) {
    switch (phase) {
    case PHASE_CHANCE: return "抽牌";
    case PHASE_TILE: return "放磚";
    case PHASE_MEEPLE: return "放 meeple";
    case PHASE_TERMINAL: return "終局";
    case PHASE_DRAGON: return "龍移動";
    case PHASE_SPOT: return "選位置";
    }
    return "?";
}

// 把一個決策的 ActionToString 字串解析成 Move,並確認它在 g 合法。
// 看不懂、階段不對或不合法就回傳 false。board_shift 是 LoggedGame::board_shift。
inline bool ParseLoggedMove(const Carcassonne &g, const std::string &a, Move *move, std::string *error,
                            int board_shift = 0) {
    auto fail = [&](const std::string &why) {
        *error = why + ": " + a;
        return false;
    };
    // 選格:魔法門要放 meeple 的格、公主要移走騎士的格、仙女要去的格
    const std::pair<const char *, SpotChoice> cells[] = {{"portal(x=%d, y=%d)", SPOT_PORTAL},
                                                         {"princess(x=%d, y=%d)", SPOT_PRINCESS},
                                                         {"move_fairy(x=%d, y=%d)", SPOT_FAIRY}};
    Move m;
    int k = 0, x = 0, y = 0, rot = 0;
    char dir = 0;
    const auto cell = std::find_if(std::begin(cells), std::end(cells),
                                   [&](const auto &c) { return sscanf(a.c_str(), c.first, &x, &y) == 2; });
    if (sscanf(a.c_str(), "place_tile(x=%d, y=%d, rot=%d)", &x, &y, &rot) == 3) {
        m.kind = Move::TILE;
        m.x = x + board_shift, m.y = y + board_shift, m.rot = rot;
    } else if (cell != std::end(cells)) {
        m.kind = Move::CELL;
        m.choice = cell->second;
        m.x = x + board_shift, m.y = y + board_shift;
    } else if (sscanf(a.c_str(), "move_dragon(dir=%c)", &dir) == 1) {
        const char *sides = "NESW";
        const char *side = dir == 0 ? nullptr : std::strchr(sides, dir);
        if (side == nullptr) return fail("看不懂的龍的方向");
        m.kind = Move::DRAGON;
        m.side = static_cast<int>(side - sides);
    } else {
        // 放 meeple 與其他棋子;選格之後的第二段(魔法門放的 meeple、公主移走的騎士、仙女的 meeple)
        struct Verb {
            const char *prefix;
            int offset;  // 加到位置上:大米寶、建築師、小豬放在同樣的位置再往後
        };
        const Verb verbs[] = {{"place_meeple(", 0},
                              {"place_big_meeple(", MEEPLE_POS_BIG},
                              {"place_builder(", MEEPLE_POS_BUILDER},
                              {"place_pig(", MEEPLE_POS_PIG - MEEPLE_POS_FIELD},
                              {"remove_knight(", 0},
                              {"fairy_meeple(", 0}};
        const Verb *verb = nullptr;
        for (const Verb &v : verbs)
            if (a.rfind(v.prefix, 0) == 0) verb = &v;
        if (verb == nullptr || a.back() != ')') return fail("看不懂的動作");
        const std::string name = verb->prefix;
        const std::string arg = a.substr(name.size(), a.size() - name.size() - 1);
        const bool pig = name == "place_pig(", builder = name == "place_builder(";
        int pos;
        if (arg == "skip" && name == "place_meeple(") pos = MEEPLE_POS_SKIP;
        else if (sscanf(arg.c_str(), "edge=%d", &k) == 1 && !pig) pos = k;
        else if (arg == "monastery" && !pig && !builder) pos = MEEPLE_POS_MONASTERY;
        else if (arg == "inner_field" && !pig && !builder) pos = MEEPLE_POS_INNER_FIELD;
        else if (sscanf(arg.c_str(), "field=%d", &k) == 1 && !builder) pos = MEEPLE_POS_FIELD + k;
        else return fail("看不懂的位置");
        if (pos != MEEPLE_POS_SKIP) pos += verb->offset;
        m.pos = pos;
        if (name == "remove_knight(" || name == "fairy_meeple(") {
            m.kind = Move::SPOT;
            m.choice = name == "remove_knight(" ? SPOT_PRINCESS : SPOT_FAIRY;
        } else if (g.current_phase == PHASE_SPOT) {
            m.kind = Move::SPOT;  // 魔法門的第二段
            m.choice = SPOT_PORTAL;
        } else {
            m.kind = Move::MEEPLE;
        }
    }
    const std::vector<Move> legal = LegalMoves(g);
    if (std::find(legal.begin(), legal.end(), m) == legal.end())
        return fail(std::string("不合法(現在是") + PhaseName(g.current_phase) + "階段)");
    *move = m;
    return true;
}

// 把一個 ActionToString 字串套到引擎上。階段不對、看不懂或不合法就回傳 false。
// board_shift 是 LoggedGame::board_shift。
inline bool ApplyLoggedAction(Carcassonne &g, const std::string &a, std::string *error, int board_shift = 0) {
    int k = 0;
    if (sscanf(a.c_str(), "draw_type(%d)", &k) == 1) {
        if (g.current_phase != PHASE_CHANCE) {
            *error = "不在抽牌階段: " + a;
            return false;
        }
        ChanceBranch draws[CANONICAL_TILE_TYPE_COUNT];
        int count = 0;
        g.getAvailableDraws(draws, count);
        if (std::none_of(draws, draws + count, [&](const ChanceBranch &d) { return d.type_id == k; })) {
            *error = "牌堆裡沒有這種牌: " + a;
            return false;
        }
        g.drawTile(k);
        return true;
    }
    Move m;
    if (!ParseLoggedMove(g, a, &m, error, board_shift)) return false;
    ApplyMove(g, m);
    return true;
}

enum class ReplayResult { kOk, kError, kUnfinished };

// 重播一局,每個決策點(放磚、放 meeple、選格、龍等)套用前呼叫 on_decision(game)。
// kOk 表示下完,而且勝負與 log 記的 Returns 一致。
template <typename OnDecision>
ReplayResult Replay(const LoggedGame &logged, OnDecision on_decision, std::string *error) {
    if (!logged.rules_error.empty()) {
        *error = logged.rules_error;
        return ReplayResult::kError;
    }
    Carcassonne game = NewGame(logged.rules);
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
