#include "game.hpp"

#include <array>
#include <bitset>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

constexpr int U = 0;
constexpr int R = 1;
constexpr int D = 2;
constexpr int L = 3;

constexpr std::array<int, 4> dx = {0, 1, 0, -1};
constexpr std::array<int, 4> dy = {-1, 0, 1, 0};
constexpr std::array<int, 4> op = {D, L, U, R};

} // namespace


int BoardModule::count3x3(int x, int y) const {
    int count = 0;
    for (int yy = y - 1; yy <= y + 1; ++yy) {
        for (int xx = x - 1; xx <= x + 1; ++xx) {
            if (isInside(xx, yy) && board[yy][xx].id != 0) {
                count++;
            }
        }
    }
    return count;
}

bool BoardModule::canPlaceTileAt(int x, int y, const Tile &tile) const {
    if (board[y][x].id != 0) {
        return false;
    }
    if (y > 0 && board[y - 1][x].id != 0 && edge[y - 1][x][op[U]] != tile.edge[U]) {
        return false;
    }
    if (x < BOARD_SIZE - 1 && board[y][x + 1].id != 0 && edge[y][x + 1][op[R]] != tile.edge[R]) {
        return false;
    }
    if (y < BOARD_SIZE - 1 && board[y + 1][x].id != 0 && edge[y + 1][x][op[D]] != tile.edge[D]) {
        return false;
    }
    if (x > 0 && board[y][x - 1].id != 0 && edge[y][x - 1][op[L]] != tile.edge[L]) {
        return false;
    }
    return true;
}

void BoardModule::placeTileOnBoard(int tile_id, int x, int y, int rot, const Tile &tile){
    board[y][x].id = static_cast<uint8_t>(tile_id);
    board[y][x].rotation = static_cast<uint8_t>(rot);
    for (int i = 0; i < 4; ++i) {
        edge[y][x][i] = tile.edge[i];
    }
}