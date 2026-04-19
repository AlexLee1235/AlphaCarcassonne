#include "game.hpp"
#include "games/bridge/double_dummy_solver/include/dll.h"
#include "games/carcassonne/tile.hpp"

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

bool Carcassonne::hasValidMove(int tile_id) const {
    for (int i = 0; i < frontier.frontier_cells.size(); ++i) {
        int x = frontier.frontier_cells[i].first;
        int y = frontier.frontier_cells[i].second;
        for (int rot = 0; rot < 4; ++rot) {
            if (board.canPlaceTileAt(x, y, full_deck[tile_id][rot])) {
                return true;
            }
        }
    }
    return false;
}

void Carcassonne::settleScore(int x, int y) {
    for (int i = 0; i < 4; ++i) 
        if (board.edge[y][x][i] != GRASS)
            features.settleCompletedFeatures(board.board[y][x].id, i, player_scores);
    monasteries.settleCompletedMonasteries(player_scores);
}

void Carcassonne::resolveEndGameScore() {
    features.resolveEndGameScore(player_scores);
    monasteries.resolveEndGameScore(player_scores);
}

void Carcassonne::resolveNoMoreDraws() {
    deck.current_tile_in_hand = 0;
    current_phase = PHASE_TERMINAL;
    is_game_over = true;
    resolveEndGameScore();
}



void Carcassonne::placeTileOnBoard(int tile_id, int x, int y, int rot) {
    const Tile &tile = full_deck[tile_id][rot];
    last_x = x;
    last_y = y;
    frontier.placeTileOnBoard(tile_id, x, y, rot, board);
    board.placeTileOnBoard(tile_id, x, y, rot, tile);
    features.placeTileOnBoard(tile_id, x, y, rot, tile, board);
    monasteries.placeTileOnBoard(tile_id, x, y, rot);
}

Carcassonne::Carcassonne() {
    deck.initializeTypeCounts();
    int start_tile_id = deck.consumeType(START_TILE_TYPE);
    placeTileOnBoard(start_tile_id, BOARD_SIZE / 2, BOARD_SIZE / 2, START_TILE_ROTATION);
    current_phase = PHASE_CHANCE;
}

int Carcassonne::currentTileType() const {
    return deck.current_tile_in_hand == 0 ? 0 : PHYSICAL_TO_CANONICAL_TYPE[deck.current_tile_in_hand];
}

float Carcassonne::terminalValue() const {
    if (current_phase != PHASE_TERMINAL) {
        return 0.0f;
    }
    if (player_scores[currentPlayer] > player_scores[1 - currentPlayer]) {
        return 1.0f;
    }
    if (player_scores[currentPlayer] < player_scores[1 - currentPlayer]) {
        return -1.0f;
    }
    return 0.0f;
}

void Carcassonne::getAvailableDraws(ChanceBranch *out, int &count) const {
    count = 0;
    if (current_phase != PHASE_CHANCE)
        return;
    deck.getAvailableDraws(out, count);
}

void Carcassonne::drawTile(int type_id) {
    int physical_id = deck.consumeType(type_id);
    deck.current_tile_in_hand = 0;
    if (hasValidMove(physical_id)) {
        deck.current_tile_in_hand = physical_id;
        current_phase = PHASE_TILE;
        return;
    }
    if (deck.total_remaining == 0) {
        resolveNoMoreDraws();
        return;
    }
    current_phase = PHASE_CHANCE;
}

void Carcassonne::getLegalTileMoves(TileMove *out, int &count) const {
    count = 0;
    if (current_phase != PHASE_TILE || deck.current_tile_in_hand == 0) {
        return;
    }

    for (int i = 0; i < frontier.frontier_cells.size(); ++i) {
        int x = frontier.frontier_cells[i].first;
        int y = frontier.frontier_cells[i].second;
        for (int rot = 0; rot < 4; ++rot) {
            if (board.canPlaceTileAt(x, y, full_deck[deck.current_tile_in_hand][rot])) {
                out[count++] = {static_cast<uint8_t>(x), static_cast<uint8_t>(y), static_cast<uint8_t>(rot)};
            }
        }
    }
}

void Carcassonne::placeTile(int x, int y, int rot) {
    int tile_id = deck.current_tile_in_hand;
    placeTileOnBoard(tile_id, x, y, rot);
    deck.current_tile_in_hand = 0;
    current_phase = PHASE_MEEPLE;
}

FixedVector<int, 6> Carcassonne::getLegalMeepleMoves() const {
    FixedVector<int, 6> ret;
    if (current_phase != PHASE_MEEPLE) {
        return ret;
    }

    ret.push_back(-1);
    if (holding_meeples[currentPlayer] == 0) {
        return ret;
    }

    int x = last_x;
    int y = last_y;
    const Tile &tile = full_deck[board.board[y][x].id][board.board[y][x].rotation];
    features.getLegalMeepleMoves(ret, x, y, board, tile);

    if (tile.monastery) {
        ret.push_back(4);
    }
    return ret;
}

void Carcassonne::placeMeeple(int pos) {
    int x = last_x;
    int y = last_y;
    if (pos != -1) {
        holding_meeples[currentPlayer]--;
        if (pos == 4) {
            monasteries.active_monasteries.push_back({x, y, board.count3x3(x, y), currentPlayer});
        } else {
            features.featureMap.getSetData(edgeIndex(board[y][x].id, pos)).meeple_mask[currentPlayer]++;
        }
    }

    settleScore(x, y);
    currentPlayer = 1 - currentPlayer;
    if (deck.total_remaining == 0) {
        resolveNoMoreDraws();
        return;
    }
    current_phase = PHASE_CHANCE;
}

Carcassonne Carcassonne::clone() const { return *this; }
