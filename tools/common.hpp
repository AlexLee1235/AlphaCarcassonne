// AlphaCarcassonne 診斷工具共用程式碼。
//
// 這些工具用 `#define private public` 直接存取 Carcassonne 的內部狀態
// (features / monasteries / logs)。這是刻意的：診斷工具需要看到引擎內部，
// 但不應該為了診斷而放寬正式程式碼的封裝。
#pragma once

#define private public
#include "game.hpp"
#undef private

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace diag {

// 一局的牌組與規則，同 CarcassonneGame 的參數（carcassonne.cc）。
struct GameRules {
    uint32_t expansions = BASE_ONLY;  // 發哪些擴充的牌（expansionBit() 遮罩）
    uint32_t rules = 0;               // 其中開了規則的
    int max_turns = 0;
};

inline Carcassonne NewGame(const GameRules &rules) {
    return Carcassonne(rules.max_turns, START_TILE_ROTATION, rules.expansions, rules.rules);
}

// 解析遊戲字串，例如 "carcassonne(river=on,inns_cathedrals=tiles,max_turns=40)"；
// 外層的 "carcassonne(...)" 可以省略，空字串是基本版。擴充收 off、tiles（只發牌）、
// on（加規則），跟 CarcassonneGame 的建構子做一樣的檢查。
inline bool ParseGameString(std::string text, GameRules *out, std::string *error) {
    text.erase(std::remove(text.begin(), text.end(), ' '), text.end());
    const std::string name = "carcassonne";
    if (text.rfind(name, 0) == 0) {
        text = text.substr(name.size());
        if (!text.empty()) {
            if (text.size() < 2 || text.front() != '(' || text.back() != ')') {
                *error = "看不懂的遊戲字串: carcassonne" + text;
                return false;
            }
            text = text.substr(1, text.size() - 2);
        }
    }
    GameRules rules;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find(',', start);
        if (end == std::string::npos) end = text.size();
        const std::string item = text.substr(start, end - start);
        start = end + 1;
        const size_t eq = item.find('=');
        if (eq == std::string::npos) {
            *error = "參數要寫成 名稱=值: " + item;
            return false;
        }
        const std::string key = item.substr(0, eq), value = item.substr(eq + 1);
        if (key == "max_turns") {
            rules.max_turns = std::atoi(value.c_str());
            continue;
        }
        int expansion = -1;
        for (int e = 1; e < EXPANSION_COUNT; ++e) {
            if (key == EXPANSION_NAMES[e]) expansion = e;
        }
        if (expansion < 0) {
            *error = "不認得的參數: " + key;
            return false;
        }
        const uint32_t bit = expansionBit(static_cast<Expansion>(expansion));
        const bool tiles_mode = (RULES_REQUIRED_EXPANSIONS & bit) == 0;
        const bool on_mode = (RULED_EXPANSIONS & bit) != 0;
        if (value == "tiles" && tiles_mode) {
            rules.expansions |= bit;
        } else if (value == "on" && on_mode) {
            rules.expansions |= bit;
            rules.rules |= bit;
        } else if (value != "off") {
            *error = item + ": 這個擴充不收 " + value;
            return false;
        }
    }
    *out = rules;
    return true;
}

// 給人看的名字："base"，或 "inns_cathedrals=on,river=on" 這種 ParseGameString 讀得回去的寫法。
inline std::string GameName(const GameRules &rules) {
    std::string name;
    for (int e = 1; e < EXPANSION_COUNT; ++e) {
        const uint32_t bit = expansionBit(static_cast<Expansion>(e));
        if (!(rules.expansions & bit)) continue;
        name += (name.empty() ? "" : ",") + std::string(EXPANSION_NAMES[e]) + ((rules.rules & bit) ? "=on" : "=tiles");
    }
    if (rules.max_turns > 0) name += (name.empty() ? "" : ",") + std::string("max_turns=") + std::to_string(rules.max_turns);
    return name.empty() ? "base" : name;
}

// 命令列的牌組參數：base、river（開河流）、all（所有擴充都 on），或 ParseGameString 的寫法。
// 看不懂就印出原因並結束。
inline GameRules DeckArg(const char *arg) {
    GameRules rules;
    const std::string deck = arg;
    if (deck == "base") return rules;
    if (deck == "river") {
        rules.expansions |= expansionBit(EXP_RIVER);
        rules.rules |= expansionBit(EXP_RIVER);
        return rules;
    }
    if (deck == "all") {
        rules.expansions = ALL_EXPANSIONS;
        rules.rules = RULED_EXPANSIONS;
        return rules;
    }
    std::string error;
    if (!ParseGameString(deck, &rules, &error)) {
        std::fprintf(stderr, "牌組參數: %s\n（base、river、all，或像 inns_cathedrals=on,princess_dragon=tiles 的寫法）\n",
                     error.c_str());
        std::exit(2);
    }
    return rules;
}

// 依真實機率抽一張牌。回傳 false 表示牌堆已空。
inline bool SampleDraw(Carcassonne &g, std::mt19937 &rng) {
    ChanceBranch d[CANONICAL_TILE_TYPE_COUNT];
    int c = 0;
    g.getAvailableDraws(d, c);
    if (c == 0) return false;
    double r = std::uniform_real_distribution<double>(0, 1)(rng), acc = 0;
    int pick = d[c - 1].type_id;
    for (int i = 0; i < c; ++i) {
        acc += d[i].probability;
        if (r <= acc) { pick = d[i].type_id; break; }
    }
    g.drawTile(pick);
    return true;
}

// 一個決策，對應 CarcassonneState::LegalActions 的一個動作：放磚、放 meeple（含跳過）、
// 選格（魔法門、公主、仙女）、選格之後的位置、龍走一步。用不到的欄位留預設值，== 才比得準。
struct Move {
    enum Kind : uint8_t { TILE, MEEPLE, CELL, SPOT, DRAGON };
    Kind kind = TILE;
    int x = -1, y = -1, rot = -1;     // TILE：落點；CELL：格子
    int pos = -1;                     // MEEPLE、SPOT：位置（公主、仙女的 SPOT 是棋子的 spot）
    SpotChoice choice = SPOT_PORTAL;  // CELL、SPOT：哪一種選擇
    int side = -1;                    // DRAGON：0 N、1 E、2 S、3 W

    bool operator==(const Move &o) const {
        return kind == o.kind && x == o.x && y == o.y && rot == o.rot && pos == o.pos && choice == o.choice &&
               side == o.side;
    }
};

// 目前玩家所有的合法決策，組成同 LegalActions：PHASE_MEEPLE 先是 getLegalMeepleMoves()，
// 再接魔法門、公主、仙女的格子。抽牌與終局時是空的。
inline std::vector<Move> LegalMoves(const Carcassonne &g) {
    std::vector<Move> moves;
    Move m;
    switch (g.current_phase) {
    case PHASE_TILE: {
        std::vector<TileMove> buf(BOARD_SIZE * BOARD_SIZE * 4);
        int count = 0;
        g.getLegalTileMoves(buf.data(), count);
        m.kind = Move::TILE;
        for (int i = 0; i < count; ++i) {
            m.x = buf[i].x, m.y = buf[i].y, m.rot = buf[i].rot;
            moves.push_back(m);
        }
        break;
    }
    case PHASE_MEEPLE: {
        const MeepleMoves meeples = g.getLegalMeepleMoves();
        m.kind = Move::MEEPLE;
        for (int i = 0; i < meeples.size(); ++i) {
            m.pos = meeples[i];
            moves.push_back(m);
        }
        m = Move();
        m.kind = Move::CELL;
        for (const SpotChoice choice : {SPOT_PORTAL, SPOT_PRINCESS, SPOT_FAIRY}) {
            const Cells cells = choice == SPOT_PORTAL     ? g.getLegalPortalCells()
                                : choice == SPOT_PRINCESS ? g.getLegalPrincessCells()
                                                          : g.getLegalFairyCells();
            m.choice = choice;
            for (const auto &[x, y] : cells) {
                m.x = x, m.y = y;
                moves.push_back(m);
            }
        }
        break;
    }
    case PHASE_SPOT: {
        const MeepleMoves spots = g.getLegalSpotMoves();
        m.kind = Move::SPOT;
        m.choice = g.spot_choice;
        for (int i = 0; i < spots.size(); ++i) {
            m.pos = spots[i];
            moves.push_back(m);
        }
        break;
    }
    case PHASE_DRAGON:
        m.kind = Move::DRAGON;
        for (const int side : g.getLegalDragonMoves()) {
            m.side = side;
            moves.push_back(m);
        }
        break;
    default:
        break;
    }
    return moves;
}

inline void ApplyMove(Carcassonne &g, const Move &m) {
    switch (m.kind) {
    case Move::TILE: g.placeTile(m.x, m.y, m.rot); break;
    case Move::MEEPLE: g.placeMeeple(m.pos); break;
    case Move::CELL: g.chooseCell(m.choice, m.x, m.y); break;
    case Move::SPOT: g.chooseSpot(m.pos); break;
    case Move::DRAGON: g.moveDragon(m.side); break;
    }
}

// 隨機選一個合法落點。回傳 false 表示無合法手。
inline bool RandomPlaceTile(Carcassonne &g, std::mt19937 &rng, TileMove *out = nullptr) {
    static std::vector<TileMove> buf;
    buf.resize(BOARD_SIZE * BOARD_SIZE * 4);
    int c = 0;
    g.getLegalTileMoves(buf.data(), c);
    if (c == 0) return false;
    const TileMove &m = buf[std::uniform_int_distribution<int>(0, c - 1)(rng)];
    if (out) *out = m;
    g.placeTile(m.x, m.y, m.rot);
    return true;
}

// 只處理放 meeple：基本版的工具用。有龍、選格的牌組用 RandomStep。
inline bool RandomPlaceMeeple(Carcassonne &g, std::mt19937 &rng) {
    MeepleMoves mm = g.getLegalMeepleMoves();
    if (mm.size() == 0) return false;
    g.placeMeeple(mm[std::uniform_int_distribution<int>(0, mm.size() - 1)(rng)]);
    return true;
}

// 任何決策階段都均勻隨機選一個合法決策（同 OpenSpiel 的隨機 bot）。回傳 false 表示沒有決策可下。
// 放磚與基本版的放 meeple 跟 RandomPlaceTile、RandomPlaceMeeple 用一樣的亂數，結果相同。
inline bool RandomStep(Carcassonne &g, std::mt19937 &rng) {
    const std::vector<Move> moves = LegalMoves(g);
    if (moves.empty()) return false;
    ApplyMove(g, moves[std::uniform_int_distribution<int>(0, moves.size() - 1)(rng)]);
    return true;
}

// 「若現在立刻結束」還會加的分（不含已入袋分數），從 player 視角的分差：觀測用的
// getPendingScore()，含最後一張磚剛完成、回合結束才結算的城、路、修道院；終局時是 0。
// 2026-10-09 以前是對複本呼叫 resolveEndGameScore()：它跳過已完成的城、路，所以放磚之後
// 剛完成的元件不算分，greedy 因此不愛完成自己的城、路；終局狀態還會再加一次終局分。
inline int PendingDiff(const Carcassonne &g, int player) {
    int pending[2];
    g.getPendingScore(pending);
    return pending[player] - pending[1 - player];
}

inline int BankedDiff(const Carcassonne &g, int player) {
    return g.player_scores[player] - g.player_scores[1 - player];
}

// greedy：每個決策選讓目前玩家的 banked + pending 分差最大的，當「強玩家」的代理
// （§2.1 警告框）。pending 含農田，所以 greedy 也會主動放農夫。龍每一步由輪到的玩家選。
// 同分時 ties 為 nullptr 取第一個，否則用它隨機挑。回傳 false 表示沒有決策可下。
inline bool GreedyStep(Carcassonne &g, std::mt19937 *ties = nullptr) {
    const std::vector<Move> moves = LegalMoves(g);
    if (moves.empty()) return false;
    const int me = g.currentPlayer;
    int best_score = INT_MIN;
    std::vector<int> best;
    for (int i = 0; i < static_cast<int>(moves.size()); ++i) {
        Carcassonne t = g;
        ApplyMove(t, moves[i]);
        const int s = BankedDiff(t, me) + PendingDiff(t, me);
        if (s > best_score) best_score = s, best.clear();
        if (s == best_score) best.push_back(i);
    }
    const int pick = ties ? best[std::uniform_int_distribution<int>(0, best.size() - 1)(*ties)] : best[0];
    ApplyMove(g, moves[pick]);
    return true;
}

} // namespace diag
