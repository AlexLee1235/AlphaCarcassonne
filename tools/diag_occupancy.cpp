// diag_occupancy —— 每一格「有磚」的機率熱圖
//
// 為什麼要量：現在的 value head 是 flatten(225) -> Linear(225, 32)，
// 每個位置有自己獨立的權重。位置 (x,y) 的權重只有在那格有磚時才拿得到梯度。
//
// 參考結果 (1500 局 / 106,468 個狀態)：
//   中心 (7,7) = 100%   邊 (7,1) = 9.1%   角 (1,1) = 0.5%   ->  185x 失衡
//   只有 15 格的占用率 >= 50%，但終局盤上有 72 格磚
//   對照 checkpoint-193 的權重：占用率 <=5% 的 93 格，W1 per-position std = 0.0345
//   （libtorch Linear 初始化 std ~= 0.0385）-> 那些權重完全沒被訓練過
//
// 同時輸出 occupancy.csv 供 check_value_head.py / erf.py 使用。
//
// 用法: ./diag_occupancy [局數=1500]
#include "common.hpp"

int main(int argc, char **argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 1500;
    std::mt19937 rng(2024);
    static double occ[BOARD_SIZE][BOARD_SIZE] = {};
    long long states = 0;

    for (int g = 0; g < N; ++g) {
        Carcassonne game;
        while (game.current_phase != PHASE_TERMINAL) {
            if (game.current_phase == PHASE_CHANCE) {
                if (!diag::SampleDraw(game, rng)) break;
            } else if (game.current_phase == PHASE_TILE) {
                states++;
                for (int y = 0; y < BOARD_SIZE; ++y)
                    for (int x = 0; x < BOARD_SIZE; ++x)
                        if (game.getPlacement(x, y).id) occ[y][x] += 1;
                if (!diag::RandomPlaceTile(game, rng)) break;
            } else {
                if (!diag::RandomPlaceMeeple(game, rng)) break;
            }
        }
    }

    printf("states=%lld   每格「有磚」的機率 (%%):\n", states);
    for (int y = 0; y < BOARD_SIZE; ++y) {
        printf("  ");
        for (int x = 0; x < BOARD_SIZE; ++x) printf("%4.0f", 100.0 * occ[y][x] / states);
        printf("\n");
    }
    const int c = BOARD_SIZE / 2;
    const double ctr = 100.0 * occ[c][c] / states;
    const double corner = 100.0 * occ[1][1] / states;
    printf("\n中心(%d,%d)=%.1f%%   邊(%d,1)=%.1f%%   角(1,1)=%.1f%%   中心/角 = %.0fx\n",
           c, c, ctr, c, 100.0 * occ[c][1] / states, corner, ctr / std::max(corner, 1e-9));

    FILE *f = fopen("occupancy.csv", "w");
    for (int y = 0; y < BOARD_SIZE; ++y) {
        for (int x = 0; x < BOARD_SIZE; ++x)
            fprintf(f, "%s%.6f", x ? "," : "", 100.0 * occ[y][x] / states);
        fprintf(f, "\n");
    }
    fclose(f);
    printf("已寫出 occupancy.csv\n");
    return 0;
}
