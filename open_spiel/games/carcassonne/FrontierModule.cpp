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



void FrontierModule::addFrontierCell(int x, int y, const BoardModule& board) {
    if (!isInside(x, y) || board.board[y][x].id != 0 || frontier[y][x]) {
        return;
    }
    frontier[y][x] = true;
    frontier_cells.push_back({static_cast<uint8_t>(x), static_cast<uint8_t>(y)});
}

void FrontierModule::removeFrontierCell(int x, int y) {
    if (!isInside(x, y) || !frontier[y][x]) {
        return;
    }
    frontier[y][x] = false;
    for (int i = 0; i < frontier_cells.size(); ++i) {
        if (frontier_cells[i].first == x && frontier_cells[i].second == y) {
            frontier_cells.swap_pop_erase_at(i);
            return;
        }
    }
}

void FrontierModule::placeTileOnBoard(int tile_id, int x, int y, int rot, const BoardModule& board){
    removeFrontierCell(x, y);
    for (int i = 0; i < 4; ++i) {
        addFrontierCell(x + dx[i], y + dy[i], board);
    }
}