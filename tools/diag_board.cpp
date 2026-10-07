// diag_board —— 視窗(VIEW_SIZE)夠不夠大？分支因子多少？終局待結分的組成？
//
// 引擎只在視窗裡放磚:VIEW_SIZE x VIEW_SIZE,每手都置中在已放磚的外框上(game.hpp)。
// 兩種編譯:
//   diag_board       repo 的引擎:看視窗實際擋掉多少(外框貼到視窗、視窗外還有 frontier)。
//   diag_board_free  VIEW_SIZE = BOARD_SIZE = 41 的副本(Makefile):什麼都不擋,
//                    量對局本來會長多大,並算出各種視窗放法要多大才放得下。
//
// 參考結果(diag_board_free,隨機 10000 局 / greedy 2000 局,21 格放得下的比例):
//   基本版   起始磚置中 95.5% / 97.6%   每手依外框置中 99.96% / 100%
//   開河流   起始磚置中 28.4% / 47.6%   每手依外框置中 98.5% / 99.7%
//   河流 p99 往下游延伸 16 格、往上游 8 格:起始磚置中要 29 格才有基本版 21 格的覆蓋率。
//   實際的 21 格視窗(diag_board,隨機):河流對局 3.9% 放磚碰到視窗邊、5.2% 視窗外還有 frontier。
//
// 用法: ./diag_board [局數=3000] [base|river|all] [random|greedy]
//   all 發牌表裡所有擴充的牌(規則照引擎預設)。greedy 每手選 banked + pending 分差最大的動作
//   (平手隨機挑),當強玩家的代理;一局要算很多份複本,局數給少一點。
#include "common.hpp"

#include <cstring>

namespace {

struct Extent {
    // 相對起始磚(河源)的延伸格數:上、下、左、右
    int up = 0, down = 0, left = 0, right = 0;
    int w = 0, h = 0;
    // 起始磚置中、每手前依外框置中時,磚離視窗中心最遠幾格
    int from_start = 0, following = 0;
};

void GreedyTile(Carcassonne &g, std::mt19937 &rng) {
    std::vector<TileMove> buf(BOARD_SIZE * BOARD_SIZE * 4);
    int c = 0;
    g.getLegalTileMoves(buf.data(), c);
    const int me = g.currentPlayer;
    int best = -1000000;
    std::vector<int> ties;
    for (int i = 0; i < c; ++i) {
        Carcassonne t = g;
        t.placeTile(buf[i].x, buf[i].y, buf[i].rot);
        const int s = diag::BankedDiff(t, me) + diag::PendingDiff(t, me);
        if (s > best) best = s, ties.clear();
        if (s == best) ties.push_back(i);
    }
    const TileMove &m = buf[ties[std::uniform_int_distribution<int>(0, ties.size() - 1)(rng)]];
    g.placeTile(m.x, m.y, m.rot);
}

void GreedyMeeple(Carcassonne &g, std::mt19937 &rng) {
    const MeepleMoves mm = g.getLegalMeepleMoves();
    const int me = g.currentPlayer;
    int best = -1000000;
    std::vector<int> ties;
    for (int i = 0; i < mm.size(); ++i) {
        Carcassonne t = g;
        t.placeMeeple(mm[i]);
        const int s = diag::BankedDiff(t, me) + diag::PendingDiff(t, me);
        if (s > best) best = s, ties.clear();
        if (s == best) ties.push_back(mm[i]);
    }
    g.placeMeeple(ties[std::uniform_int_distribution<int>(0, ties.size() - 1)(rng)]);
}

int Percentile(std::vector<int> v, double p) {
    std::sort(v.begin(), v.end());
    return v[std::min<size_t>(v.size() - 1, static_cast<size_t>(p * v.size()))];
}

} // namespace

int main(int argc, char **argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 3000;
    const char *deck = argc > 2 ? argv[2] : "base";
    const bool greedy = argc > 3 && std::strcmp(argv[3], "greedy") == 0;
    const uint32_t expansions = std::strcmp(deck, "all") == 0     ? ALL_EXPANSIONS
                                : std::strcmp(deck, "river") == 0 ? BASE_ONLY | expansionBit(EXP_RIVER)
                                                                  : BASE_ONLY;
    std::mt19937 rng(999);
    const int c0 = BOARD_SIZE / 2;
    long long branchN = 0, maxB = 0, at_view_edge = 0, outside_frontier = 0;
    double sumBranch = 0, sumOpenCity = 0, sumOpenRoad = 0, sumMon = 0;
    std::vector<Extent> extents;

    for (int n = 0; n < N; ++n) {
        Carcassonne game(0, START_TILE_ROTATION, expansions);
        Extent e;
        int x0 = c0, x1 = c0, y0 = c0, y1 = c0;
        bool edge = false, outside = false;
        while (game.current_phase != PHASE_TERMINAL) {
            if (game.current_phase == PHASE_CHANCE) {
                if (!diag::SampleDraw(game, rng)) break;
            } else if (game.current_phase == PHASE_TILE) {
                std::vector<TileMove> buf(BOARD_SIZE * BOARD_SIZE * 4);
                int c = 0;
                game.getLegalTileMoves(buf.data(), c);
                if (c == 0) break;
                sumBranch += c;
                branchN++;
                maxB = std::max<long long>(maxB, c);
                for (int y = 0; y < BOARD_SIZE && !outside; ++y)
                    for (int x = 0; x < BOARD_SIZE && !outside; ++x)
                        outside = game.isFrontier(x, y) && !game.inView(x, y);
                if (greedy) GreedyTile(game, rng);
                else diag::RandomPlaceTile(game, rng);
                // 每手前依外框置中:放下的磚與外框離中心最遠幾格(中心取法同 game.cpp 的 viewOrigin)
                const int x = game.last_x, y = game.last_y;
                const int cx = (x0 + x1) / 2 + ((x0 + x1) % 2 != 0 && x0 + x1 < 2 * c0);
                const int cy = (y0 + y1) / 2 + ((y0 + y1) % 2 != 0 && y0 + y1 < 2 * c0);
                e.following = std::max({e.following, std::abs(x - cx), std::abs(y - cy), x1 - cx, cx - x0, y1 - cy,
                                        cy - y0});
                x0 = std::min(x0, x), x1 = std::max(x1, x), y0 = std::min(y0, y), y1 = std::max(y1, y);
                edge |= x == game.view_x0 || x == game.view_x0 + VIEW_SIZE - 1 || y == game.view_y0 ||
                        y == game.view_y0 + VIEW_SIZE - 1;
            } else if (greedy) {
                GreedyMeeple(game, rng);
            } else if (!diag::RandomPlaceMeeple(game, rng)) {
                break;
            }
        }
        at_view_edge += edge;
        outside_frontier += outside;
        e.up = c0 - y0, e.down = y1 - c0, e.left = c0 - x0, e.right = x1 - c0;
        e.w = x1 - x0 + 1, e.h = y1 - y0 + 1;
        e.from_start = std::max({e.up, e.down, e.left, e.right});
        extents.push_back(e);

        // 終局待結分組成
        double oc = 0, orr = 0;
        for (auto it = game.features.featureMap.begin(); it != game.features.featureMap.end(); ++it) {
            Feature &f = *it;
            if (f.opens == 0 || !isFeatureEdge(f.type)) continue;
            if (f.meeple_count[0] == 0 && f.meeple_count[1] == 0) continue;
            (f.type == CITY ? oc : orr) += f.getScore();
        }
        double mo = 0;
        for (int i = 0; i < game.monasteries.active_monasteries.size(); ++i)
            mo += game.monasteries.active_monasteries[i].tile_count;
        sumOpenCity += oc, sumOpenRoad += orr, sumMon += mo;
    }

    printf("BOARD_SIZE %d, VIEW_SIZE %d, %s, %d 局, 牌組 %d 張(%s)\n", BOARD_SIZE, VIEW_SIZE,
           greedy ? "greedy" : "隨機", N, deckSizeOf(expansions), deck);
    printf("  放磚碰到視窗邊的對局 %.1f%%,視窗外還有 frontier 的對局 %.1f%%\n", 100.0 * at_view_edge / N,
           100.0 * outside_frontier / N);
    printf("  avg 合法落點/手 = %.1f    max = %lld\n", sumBranch / branchN, maxB);
    printf("  終局待結分(雙方合計): 未完成城市 %.2f  未完成道路 %.2f  修道院 %.2f\n", sumOpenCity / N,
           sumOpenRoad / N, sumMon / N);

    auto column = [&](int Extent::*field) {
        std::vector<int> v;
        for (const Extent &e : extents) v.push_back(e.*field);
        return v;
    };
    const std::pair<const char *, int Extent::*> rows[] = {
        {"上", &Extent::up},       {"下", &Extent::down},        {"左", &Extent::left},
        {"右", &Extent::right},    {"寬", &Extent::w},           {"高", &Extent::h},
        {"起始磚置中", &Extent::from_start}, {"每手依外框置中", &Extent::following}};
    printf("  相對起始磚(河源)的延伸、外框、離視窗中心最遠:\n");
    printf("    %-16s %5s %5s %5s %6s %5s\n", "", "p50", "p90", "p99", "p99.9", "max");
    for (const auto &[name, field] : rows) {
        const std::vector<int> v = column(field);
        printf("    %-16s %5d %5d %5d %6d %5d\n", name, Percentile(v, .5), Percentile(v, .9), Percentile(v, .99),
               Percentile(v, .999), Percentile(v, 1));
    }

    // 只有什麼都不擋(diag_board_free)時,這張表才是「要多大」:否則對局已經被視窗擋過了。
    printf("  S 格的視窗放得下的對局: 起始磚置中 | 每手依外框置中 | 終局外框置中\n");
    for (int S = 15; S <= BOARD_SIZE; S += 2) {
        int start = 0, following = 0, final_box = 0;
        for (const Extent &e : extents) {
            start += e.from_start <= S / 2;
            following += e.following <= S / 2;
            final_box += e.w <= S && e.h <= S;
        }
        printf("    %2d: %6.2f%% | %6.2f%% | %6.2f%%\n", S, 100.0 * start / N, 100.0 * following / N,
               100.0 * final_box / N);
        if (start == N) break;
    }
    if (VIEW_SIZE < BOARD_SIZE)
        printf("  ** 這份引擎只在 %d 格的視窗裡放磚,上面的延伸與表是被擋過的;沒被擋的分佈用 diag_board_free。\n",
               VIEW_SIZE);
    return 0;
}
