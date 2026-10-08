// 重播 actor log 格式的對局(GUI 存的 log-actor-gui*.txt、訓練的 log-actor-*.txt),
// 把每個玩家的分數拆成來源,並列出 meeple 怎麼用掉。牌組與規則由 actor_log.hpp 從檔名或 config.json 讀。
//
//   ./build/diag_game_report <目錄或檔案> [最多逐局印幾局=20]
//
// 分數來源:局中完成的城 / 路 / 修道院(+),終局沒完成的城 / 路 / 修道院(~),農田,貨物獎金,
// 仙女(回合開始 +1、它旁邊的 meeple 計分 +3)。
// 旅館、大教堂算在城/路裡(getScore() 已含),小豬算在農田裡。
// 歸因不改引擎:
//   - 回合在結束它的那一手結算(放 meeple、選格、選位置,龍牌是龍的最後一步;Carcassonne::finishTurn)。
//     BeforeSettle 照 game.cpp 算出那一手的效果、還沒結算的狀態,再從它依 settleTurn 的規則分:
//     最後一張磚已封口(opens == 0)的城/路依多數(大米寶算 2)記 getScore(),滿 9 格的修道院記 9,
//     仙女旁邊的 meeple 計分記 3。對那個狀態呼叫 finishTurn 必須得到引擎實際的結果。
//   - 終局照 FeatureModule::resolveEndGameScore 的迴圈分城/路;修道院、農田、貨物直接呼叫(都不改狀態)。
// 每一手拆出來的分要等於那一手實際的分差;對不上、或勝負跟 log 的 Returns 不一致,就印 WARN
// —— 那一局的數字不能用。
#include "actor_log.hpp"

#include <filesystem>
#include <map>
#include <string>

namespace {

enum Source { CITY_DONE, ROAD_DONE, MONASTERY_DONE, CITY_END, ROAD_END, MONASTERY_END, FARM, GOODS, FAIRY, SOURCES };
const char *kSourceNames[SOURCES] = {"city+", "road+", "mon+", "city~", "road~", "mon~", "farm", "goods", "fairy"};

struct PlayerReport {
    int points[SOURCES] = {};
    int final_score = 0;
    int turns = 0, hand_sum = 0, empty_hand_turns = 0;
    // 大米寶依位置算進前三類(魔法門放的也算);建築師、小豬另計;公主移走騎士、移仙女另計
    int on_feature = 0, on_monastery = 0, farmers = 0, skipped = 0, builders_pigs = 0, princess_fairy = 0;
    std::vector<int> farmer_turns;  // 自己的第幾手(從 1 起)放下農夫
    int quarter_hand[4] = {}, quarter_turns[4] = {};  // 自己第 1–9、10–18、19–27、28– 手前手上的 meeple

    int Total() const {
        int total = 0;
        for (int p : points) total += p;
        return total;
    }
};

// 這回合放下的棋子,依位置(含大米寶、建築師、小豬的偏移)分類。
void CountPiece(PlayerReport &me, int pos, int own_turn) {
    if (pos == MEEPLE_POS_SKIP) {
        me.skipped++;
    } else if (isBuilderPos(pos) || isPigPos(pos)) {
        me.builders_pigs++;
    } else if (meepleSpot(pos) == MEEPLE_POS_MONASTERY) {
        me.on_monastery++;
    } else if (meepleSpot(pos) >= MEEPLE_POS_FIELD) {
        me.farmers++;
        me.farmer_turns.push_back(own_turn);
    } else {
        me.on_feature++;
    }
}

void Credit(int score, const int meeples[2], Source source, PlayerReport rep[2], int credited[2]) {
    for (int p = 0; p < 2; ++p) {
        if (meeples[p] >= meeples[1 - p]) {
            rep[p].points[source] += score;
            credited[p] += score;
        }
    }
}

void CreditTo(int player, int score, Source source, PlayerReport rep[2], int credited[2]) {
    rep[player].points[source] += score;
    credited[player] += score;
}

// 結束回合的那一手 m 套在 g 上、還沒結算的狀態:引擎在 endPiecePhase 與 moveDragon 的最後
// 呼叫 finishTurn,它先結算(settleTurn)再換人。照 game.cpp 寫,ReportGame 用 finishTurn 核對。
Carcassonne BeforeSettle(const Carcassonne &g, const diag::Move &m) {
    Carcassonne p = g;
    switch (m.kind) {
    case diag::Move::TILE:
        break;  // 放磚不會結束回合
    case diag::Move::MEEPLE:
        if (m.pos != MEEPLE_POS_SKIP) p.putPiece(p.last_x, p.last_y, m.pos);
        break;
    case diag::Move::CELL:
        // 結束回合的選格只有一個選項(不然會進 PHASE_SPOT)
        if (m.choice == SPOT_PORTAL) p.putPiece(m.x, m.y, p.portalMovesAt(m.x, m.y)[0]);
        else if (m.choice == SPOT_PRINCESS) p.sendHome(p.princessKnightsAt(m.x, m.y)[0]);
        else p.assignFairy(p.fairyMeeplesAt(m.x, m.y)[0]);
        break;
    case diag::Move::SPOT:
        if (m.choice == SPOT_PORTAL) {
            p.putPiece(p.spot_x, p.spot_y, m.pos);
            break;
        }
        for (const int index : m.choice == SPOT_PRINCESS ? p.princessKnightsAt(p.spot_x, p.spot_y)
                                                         : p.fairyMeeplesAt(p.spot_x, p.spot_y)) {
            if (p.pieces[index].spot != m.pos) continue;
            if (m.choice == SPOT_PRINCESS) p.sendHome(index);
            else p.assignFairy(index);
            break;
        }
        break;
    case diag::Move::DRAGON:
        // 龍的最後一步:吃掉那格的棋子,回合還給放龍牌的玩家(方向同 game.cpp 的 dx、dy)
        p.dragon_x += (m.side == 1) - (m.side == 3);
        p.dragon_y += (m.side == 2) - (m.side == 0);
        p.eatPiecesAt(p.dragon_x, p.dragon_y);
        p.currentPlayer = p.dragon_turn_player;
        p.dragon_turn_player = -1;
        break;
    }
    return p;
}

// 回合結算(Carcassonne::settleTurn)的三部分,從結算前的狀態算。
void CreditSettle(const Carcassonne &before_settle, PlayerReport rep[2], int credited[2]) {
    Carcassonne g = before_settle;  // find() 會壓縮路徑
    const int fairy = g.fairyPiece();
    if (fairy >= 0) {
        const Piece &piece = g.pieces[fairy];
        const bool scored = piece.spot == MEEPLE_POS_MONASTERY ? g.board.count3x3(piece.x, piece.y) == 9
                                                               : piece.spot < MEEPLE_POS_MONASTERY &&
                                                                     g.featureAt(piece.tile_id, piece.spot).opens == 0;
        if (scored) CreditTo(piece.owner, FAIRY_SCORE_POINTS, FAIRY, rep, credited);
    }
    const int x = g.last_x, y = g.last_y, id = g.board.board[y][x].id;
    int seen[4], seen_count = 0;
    for (int side = 0; side < 4; ++side) {
        if (!isFeatureEdge(g.board.edge[y][x][side])) continue;
        const int root = g.features.featureMap.find(g.features.edgeIndex(id, side));
        if (std::find(seen, seen + seen_count, root) != seen + seen_count) continue;
        seen[seen_count++] = root;
        const Feature &feature = g.features.featureMap.getSetData(root);
        if (feature.opens != 0) continue;
        const int meeples[2] = {feature.meeple_count[0], feature.meeple_count[1]};
        if (meeples[0] == 0 && meeples[1] == 0) continue;
        Credit(feature.getScore(), meeples, feature.type == CITY ? CITY_DONE : ROAD_DONE, rep, credited);
    }
    for (int i = 0; i < g.monasteries.active_monasteries.size(); ++i) {
        const auto &monastery = g.monasteries.active_monasteries[i];
        if (monastery.tile_count == 9) CreditTo(monastery.owner, 9, MONASTERY_DONE, rep, credited);
    }
}

// 終局結算(Carcassonne::resolveEndGameScore)的五部分。
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
    if (g.fairyOwner() >= 0) CreditTo(g.fairyOwner(), FAIRY_SCORE_POINTS, FAIRY, rep, credited);
}

// 回傳空字串表示成功,否則是不能用這局的原因。
std::string ReportGame(const diag::LoggedGame &logged, PlayerReport rep[2]) {
    if (logged.truncated) return "行被截斷";
    if (!logged.rules_error.empty()) return logged.rules_error;
    Carcassonne g = diag::NewGame(logged.rules);
    std::string error;
    int own_turn[2] = {0, 0};
    for (const std::string &action : logged.actions) {
        if (g.current_phase == PHASE_TERMINAL) return "終局後還有動作: " + action;
        const Carcassonne before = g;
        int credited[2] = {0, 0};
        if (g.current_phase == PHASE_CHANCE) {
            if (!diag::ApplyLoggedAction(g, action, &error, logged.board_shift)) return error;
        } else {
            diag::Move m;
            if (!diag::ParseLoggedMove(g, action, &m, &error, logged.board_shift)) return error;
            const int pl = g.currentPlayer;
            if (g.current_phase == PHASE_MEEPLE) {
                // 這回合放 meeple 的決策:手上的 meeple 與放了什麼
                PlayerReport &me = rep[pl];
                const int hand = g.holding_meeples[pl];
                const int quarter = std::min(own_turn[pl] / 9, 3);
                own_turn[pl]++;
                me.turns++;
                me.hand_sum += hand;
                me.empty_hand_turns += hand == 0;
                me.quarter_hand[quarter] += hand;
                me.quarter_turns[quarter]++;
                if (m.kind == diag::Move::MEEPLE) {
                    CountPiece(me, m.pos, own_turn[pl]);
                } else if (m.choice != SPOT_PORTAL) {
                    me.princess_fairy++;
                } else if (g.portalMovesAt(m.x, m.y).size() == 1) {
                    CountPiece(me, g.portalMovesAt(m.x, m.y)[0], own_turn[pl]);
                }  // 否則魔法門的位置在下一手(PHASE_SPOT)
            } else if (m.kind == diag::Move::SPOT && m.choice == SPOT_PORTAL) {
                CountPiece(rep[pl], m.pos, own_turn[pl]);
            }
            diag::ApplyMove(g, m);
            if (g.completed_turns != before.completed_turns) {
                // 這手結束回合
                Carcassonne settled = BeforeSettle(before, m);
                CreditSettle(settled, rep, credited);
                settled.finishTurn();
                if (settled.player_scores[0] != g.player_scores[0] || settled.player_scores[1] != g.player_scores[1] ||
                    settled.current_phase != g.current_phase || settled.currentPlayer != g.currentPlayer)
                    return "BeforeSettle 跟引擎的結果不一致(game.cpp 的流程改了?): " + action;
                // 仙女:新回合開始時給它旁邊 meeple 的主人(finishTurn 的最後)
                if (g.current_phase != PHASE_TERMINAL && !g.builder_second_tile && g.fairyOwner() == g.currentPlayer)
                    CreditTo(g.currentPlayer, FAIRY_TURN_POINTS, FAIRY, rep, credited);
            }
        }
        // 終局:回合結束時沒牌了,或最後幾張牌都放不下、在抽牌時結算
        if (g.current_phase == PHASE_TERMINAL) CreditEndGame(g, rep, credited);
        for (int p = 0; p < 2; ++p) {
            const int delta = g.player_scores[p] - before.player_scores[p];
            if (delta != credited[p])
                return "P" + std::to_string(p + 1) + " 這手的分數拆不開(實際 " + std::to_string(delta) + ",拆出 " +
                       std::to_string(credited[p]) + "): " + action;
        }
    }
    if (g.current_phase != PHASE_TERMINAL) return "對局沒有下完";
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
    printf(" | %5s | farmers(own turn)  feature mon skip b+p p/f | hand avg, by quarter          0-hand\n", "total");
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
    printf(" %7.1f %3.1f %4.1f %3.1f %3.1f | %4.2f,", r.on_feature / games, r.on_monastery / games, r.skipped / games,
           r.builders_pigs / games, r.princess_fairy / games, (double)r.hand_sum / std::max(1, r.turns));
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
    sum.princess_fairy += r.princess_fairy;
    for (int q = 0; q < 4; ++q) {
        sum.quarter_hand[q] += r.quarter_hand[q];
        sum.quarter_turns[q] += r.quarter_turns[q];
    }
}

struct DeckSum {
    PlayerReport sum[2];
    int games = 0;
};

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
           "quarter = 自己第 1-9 / 10-18 / 19-27 / 28- 手; 0-hand = 手上沒有 meeple 的回合比例;\n"
           "p/f = 用公主移走騎士或移仙女(沒放棋子)的回合\n\n",
           path.c_str(), games.size());

    std::map<std::string, DeckSum> decks;  // 依牌組分開平均
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
        DeckSum &deck = decks[diag::GameName(games[i].rules)];
        Add(deck.sum[0], rep[0]);
        Add(deck.sum[1], rep[1]);
        deck.games++;
        if (ok <= max_print) {
            printf("%s (%s, %s): P1 %d, P2 %d\n", games[i].source.c_str(), games[i].time.c_str(),
                   diag::GameName(games[i].rules).c_str(), rep[0].final_score, rep[1].final_score);
            PrintHeader();
            PrintRow("P1", rep[0], 1);
            PrintRow("P2", rep[1], 1);
            printf("\n");
        }
    }
    for (const auto &[name, deck] : decks) {
        if (deck.games <= 1 && decks.size() == 1) continue;
        printf("平均(%s, %d 局):\n", name.c_str(), deck.games);
        PrintHeader();
        PrintRow("P1", deck.sum[0], deck.games);
        PrintRow("P2", deck.sum[1], deck.games);
    }
    if (failed) printf("\n%d 局拆解失敗(見上面的 WARN)\n", failed);
    return failed ? 1 : 0;
}
