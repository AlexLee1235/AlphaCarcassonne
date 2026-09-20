// diag_luck —— 卡卡頌的「運氣天花板」有多高?
//
// 方法:同一副牌序打兩腿、交換座位(跟 alpha_zero_torch_game_example 的成對對局同構)。
//   2-0 橫掃 = 那一對由「棋力」決定;1-1 分帳 = 由「牌序 / 座位」決定。
// 用兩種棋力差當標尺:
//   (a) random vs random     —— 棋力差 0,橫掃率 = 純雜訊底線
//   (b) greedy vs random     —— 很大的棋力差,橫掃率 = 棋力能壓過牌序的程度
// greedy = 選讓 (banked + pending) 分差最大的落點,然後盡量放 meeple。
//
// 用法: ./diag_luck [對數=400]
#include "common.hpp"

enum Agent { RANDOM, GREEDY };

// 回傳選定的落點。greedy 用「走完之後自己的 banked+pending 分差」當分數。
static bool PlayTile(Carcassonne &g, Agent a, int me, std::mt19937 &rng) {
    static std::vector<TileMove> buf;
    buf.resize(BOARD_SIZE * BOARD_SIZE * 4);
    int c = 0;
    g.getLegalTileMoves(buf.data(), c);
    if (c == 0) return false;
    if (a == RANDOM) {
        const TileMove &m = buf[std::uniform_int_distribution<int>(0, c - 1)(rng)];
        g.placeTile(m.x, m.y, m.rot);
        return true;
    }
    int best = -1, bestScore = -1000000;
    for (int i = 0; i < c; ++i) {
        Carcassonne t = g;
        t.placeTile(buf[i].x, buf[i].y, buf[i].rot);
        int s = diag::BankedDiff(t, me) + diag::PendingDiff(t, me);
        if (s > bestScore) { bestScore = s; best = i; }
    }
    g.placeTile(buf[best].x, buf[best].y, buf[best].rot);
    return true;
}

static bool PlayMeeple(Carcassonne &g, Agent a, int me, std::mt19937 &rng) {
    FixedVector<int, 6> mm = g.getLegalMeepleMoves();
    if (mm.size() == 0) return false;
    if (a == RANDOM) {
        g.placeMeeple(mm[std::uniform_int_distribution<int>(0, mm.size() - 1)(rng)]);
        return true;
    }
    int best = mm[0], bestScore = -1000000;
    for (int i = 0; i < mm.size(); ++i) {
        Carcassonne t = g;
        t.placeMeeple(mm[i]);
        int s = diag::BankedDiff(t, me) + diag::PendingDiff(t, me);
        if (s > bestScore) { bestScore = s; best = mm[i]; }
    }
    g.placeMeeple(best);
    return true;
}

// 用預先抽好的牌序打一局。seat_of_agent0 = agent0 坐哪一個座位。
// 回傳 agent0 的結果:+1 / 0 / -1
static int PlayOne(const std::vector<int> &deck, Agent a0, Agent a1, int seatOfA0, std::mt19937 &rng) {
    Carcassonne g;
    size_t di = 0;
    while (g.current_phase != PHASE_TERMINAL) {
        if (g.current_phase == PHASE_CHANCE) {
            // 依預定牌序抽,跳過現在不可抽的型別(牌堆已空該型別)
            ChanceBranch d[CANONICAL_TILE_TYPE_COUNT];
            int c = 0;
            g.getAvailableDraws(d, c);
            if (c == 0) break;
            int pick = -1;
            while (di < deck.size() && pick < 0) {
                for (int i = 0; i < c; ++i)
                    if (d[i].type_id == deck[di]) { pick = deck[di]; break; }
                ++di;
            }
            if (pick < 0) pick = d[0].type_id;
            g.drawTile(pick);
            continue;
        }
        int seat = g.currentPlayer;
        Agent who = (seat == seatOfA0) ? a0 : a1;
        bool ok = (g.current_phase == PHASE_TILE) ? PlayTile(g, who, seat, rng)
                                                  : PlayMeeple(g, who, seat, rng);
        if (!ok) break;
    }
    int s0 = g.player_scores[0], s1 = g.player_scores[1];
    int a0seat = seatOfA0;
    int mine = a0seat == 0 ? s0 : s1, theirs = a0seat == 0 ? s1 : s0;
    return mine > theirs ? 1 : (mine < theirs ? -1 : 0);
}

static void Run(const char *label, Agent a0, Agent a1, int pairs) {
    std::mt19937 deckRng(1234), playRng(9876);
    int sweep = 0, split = 0, half = 0;
    double score = 0;   // agent0 的平均得分(勝1/和0.5/負0)
    for (int p = 0; p < pairs; ++p) {
        // 產生一副牌序:把每個 type 依初始張數展開後洗牌
        std::vector<int> deck;
        {
            Carcassonne probe;
            ChanceBranch d[CANONICAL_TILE_TYPE_COUNT];
            int c = 0;
            probe.getAvailableDraws(d, c);
            for (int i = 0; i < c; ++i)
                for (int k = 0; k < 12; ++k) deck.push_back(d[i].type_id);
            std::shuffle(deck.begin(), deck.end(), deckRng);
        }
        int r0 = PlayOne(deck, a0, a1, 0, playRng);   // agent0 坐 seat 0
        int r1 = PlayOne(deck, a0, a1, 1, playRng);   // agent0 坐 seat 1
        double s = (r0 > 0 ? 1 : r0 < 0 ? 0 : 0.5) + (r1 > 0 ? 1 : r1 < 0 ? 0 : 0.5);
        score += s;
        if (s == 2.0 || s == 0.0) sweep++;
        else if (s == 1.0) split++;
        else half++;
    }
    printf("%-22s agent0 平均得分 %.3f   橫掃(2-0/0-2) %5.1f%%   分帳(1-1) %5.1f%%   含和局 %5.1f%%\n",
           label, score / (2.0 * pairs), 100.0 * sweep / pairs, 100.0 * split / pairs, 100.0 * half / pairs);
}

int main(int argc, char **argv) {
    const int pairs = argc > 1 ? atoi(argv[1]) : 400;
    printf("=== 同牌序成對對局 %d 對(%d 局)===\n", pairs, pairs * 2);
    printf("橫掃 = 棋力壓過牌序;分帳 = 牌序/座位決定\n\n");
    Run("random vs random", RANDOM, RANDOM, pairs);
    Run("greedy vs random", GREEDY, RANDOM, pairs);
    Run("greedy vs greedy",  GREEDY, GREEDY, pairs);
    return 0;
}
