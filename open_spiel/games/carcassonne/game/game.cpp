#include "game.hpp"
#include "tile.hpp"

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

bool Carcassonne::isLegalPlacement(int tile_id, int x, int y, int rot) const {
    const Tile &tile = full_deck[tile_id][rot];
    if (!board.canPlaceTileAt(x, y, tile)) {
        return false;
    }
    if (!river_rules || riverEdgeCount(tile) == 0) {
        return true;
    }
    // A river tile continues the river: it goes where the river flows, whose
    // last tile has a river side facing it, so canPlaceTileAt has already
    // matched this tile's river side on that side.
    if (x != river_x || y != river_y) {
        return false;
    }
    for (int side = 0; side < 4; ++side) {
        if (tile.edge[side] != RIVER || side == op[river_heading]) {
            continue;
        }
        // Where it flows out: on the board, and not turning the way the last
        // bend turned.
        if (!isInside(x + dx[side], y + dy[side])) {
            return false;
        }
        const int turn = (side - river_heading + 4) % 4;
        if (turn != 0 && turn == river_last_turn) {
            return false;
        }
    }
    return true;
}

void Carcassonne::advanceRiver(int x, int y, const Tile &tile) {
    // The spring has no river flowing in; the lake none flowing out.
    const int in_side = river_heading < 0 ? -1 : op[river_heading];
    int out_side = -1;
    for (int side = 0; side < 4; ++side) {
        if (tile.edge[side] == RIVER && side != in_side) {
            out_side = side;
        }
    }
    if (out_side < 0) {
        river_x = river_y = -1;
        return;
    }
    if (river_heading >= 0) {
        const int turn = (out_side - river_heading + 4) % 4;
        if (turn != 0) {
            river_last_turn = turn;
        }
    }
    river_heading = out_side;
    river_x = x + dx[out_side];
    river_y = y + dy[out_side];
}

bool Carcassonne::hasValidMove(int tile_id) const {
    for (int i = 0; i < frontier.frontier_cells.size(); ++i) {
        int x = frontier.frontier_cells[i].first;
        int y = frontier.frontier_cells[i].second;
        for (int rot = 0; rot < 4; ++rot) {
            if (isLegalPlacement(tile_id, x, y, rot)) {
                return true;
            }
        }
    }
    return false;
}

void Carcassonne::resolveEndGameScore() {
    features.resolveEndGameScore(player_scores);
    monasteries.resolveEndGameScore(player_scores);
    fields.accumulateScore(player_scores, features);
}

void Carcassonne::resolveNoMoreDraws() {
    current_tile_in_hand = 0;
    current_phase = PHASE_TERMINAL;
    resolveEndGameScore();
}

void Carcassonne::placeTileOnBoard(int tile_id, int x, int y, int rot) {
    const Tile &tile = full_deck[tile_id][rot];
    last_x = x;
    last_y = y;
    frontier.placeTileOnBoard(tile_id, x, y, rot, board);
    board.placeTileOnBoard(tile_id, x, y, rot, tile);
    features.placeTileOnBoard(tile_id, x, y, rot, tile, board);
    fields.placeTileOnBoard(tile_id, x, y, tile, board, features);
    monasteries.placeTileOnBoard(tile_id, x, y, rot);
    logs.placeTileOnBoard(tile_id, x, y, rot);
    if (river_rules && riverEdgeCount(tile) > 0) {
        advanceRiver(x, y, tile);
    }
}

Carcassonne::Carcassonne(int max_turns) : Carcassonne(max_turns, START_TILE_ROTATION) {}

Carcassonne::Carcassonne(int max_turns, int start_rotation, uint32_t expansions) : max_turns(max_turns) {
    deck.initializeTypeCounts(expansions | BASE_ONLY);
    river_rules = deck.river_first;
    // With the river the spring starts the game and the base start tile is
    // dealt like any other.
    int start_tile_id = deck.consumeType(river_rules ? RIVER_SPRING_TYPE : START_TILE_TYPE);
    placeTileOnBoard(start_tile_id, BOARD_SIZE / 2, BOARD_SIZE / 2, start_rotation);
    current_phase = PHASE_CHANCE;
}

int Carcassonne::currentTileType() const {
    return current_tile_in_hand == 0 ? 0 : PHYSICAL_TO_CANONICAL_TYPE[current_tile_in_hand];
}

void Carcassonne::WriteMeepleMap(int player, float *span) const { logs.getMeepleMap(features, monasteries, player, span); }

void Carcassonne::getAvailableDraws(ChanceBranch *out, int &count) const {
    count = 0;
    if (current_phase != PHASE_CHANCE)
        return;
    deck.getAvailableDraws(out, count);
}

void Carcassonne::drawTile(int type_id) {
    int physical_id = deck.consumeType(type_id);
    current_tile_in_hand = 0;
    if (hasValidMove(physical_id)) {
        current_tile_in_hand = physical_id;
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
    if (current_phase != PHASE_TILE || current_tile_in_hand == 0) {
        return;
    }

    for (int i = 0; i < frontier.frontier_cells.size(); ++i) {
        int x = frontier.frontier_cells[i].first;
        int y = frontier.frontier_cells[i].second;
        for (int rot = 0; rot < 4; ++rot) {
            if (isLegalPlacement(current_tile_in_hand, x, y, rot)) {
                out[count++] = {static_cast<uint8_t>(x), static_cast<uint8_t>(y), static_cast<uint8_t>(rot)};
            }
        }
    }
}

void Carcassonne::placeTile(int x, int y, int rot) {
    int tile_id = current_tile_in_hand;
    placeTileOnBoard(tile_id, x, y, rot);
    current_tile_in_hand = 0;
    current_phase = PHASE_MEEPLE;
}

MeepleMoves Carcassonne::getLegalMeepleMoves() const {
    MeepleMoves ret;
    if (current_phase != PHASE_MEEPLE) {
        return ret;
    }
    ret.push_back(MEEPLE_POS_SKIP);
    if (holding_meeples[currentPlayer] == 0) {
        return ret;
    }
    int x = last_x;
    int y = last_y;
    const Placement &placement = board.board[y][x];
    const Tile &tile = full_deck[placement.id][placement.rotation];
    features.getLegalMeepleMoves(ret, x, y, board, tile);
    if (tile.monastery) {
        ret.push_back(MEEPLE_POS_MONASTERY);
    }
    fields.getLegalFarmerMoves(ret, placement.id, tile);
    return ret;
}

void Carcassonne::getLastTileSideGroups(int8_t groups[4]) const {
    for (int i = 0; i < 4; ++i) {
        groups[i] = -1;
    }
    if (last_x < 0 || last_y < 0) {
        return;
    }
    const Placement &placement = board.board[last_y][last_x];
    const Tile &tile = full_deck[placement.id][placement.rotation];
    int roots[4];
    for (int i = 0; i < 4; ++i) {
        if (!isFeatureEdge(tile.edge[i])) {
            continue;
        }
        roots[i] = features.featureMap.find(features.edgeIndex(placement.id, i));
        groups[i] = static_cast<int8_t>(i);
        for (int j = 0; j < i; ++j) {
            if (groups[j] != -1 && roots[j] == roots[i]) {
                groups[i] = groups[j];
                break;
            }
        }
    }
}

void Carcassonne::getLastTileFieldGroups(int8_t groups[HALF_EDGE_COUNT]) const {
    for (int e = 0; e < HALF_EDGE_COUNT; ++e) {
        groups[e] = -1;
    }
    if (last_x < 0 || last_y < 0) {
        return;
    }
    const Placement &placement = board.board[last_y][last_x];
    fields.getHalfEdgeGroups(placement.id, full_deck[placement.id][placement.rotation], groups);
}

void Carcassonne::getPendingScore(int pending[2]) const {
    pending[0] = pending[1] = 0;
    if (current_phase == PHASE_TERMINAL) {
        return;  // The end-game scoring is already in player_scores.
    }
    features.accumulatePendingScore(pending);
    monasteries.accumulatePendingScore(pending);
    fields.accumulateScore(pending, features);
}

void Carcassonne::getPendingFieldScore(int pending[2]) const {
    pending[0] = pending[1] = 0;
    if (current_phase == PHASE_TERMINAL) {
        return;
    }
    fields.accumulateScore(pending, features);
}

void Carcassonne::getPendingScoreByResolving(int pending[2]) const {
    pending[0] = pending[1] = 0;
    if (current_phase == PHASE_TERMINAL) {
        return;
    }
    Carcassonne copy = *this;
    if (copy.current_phase == PHASE_MEEPLE) {
        // What placeMeeple settles whatever the move is.
        copy.features.settleAfterPlaceMeeple(last_x, last_y, copy.board, copy.player_scores, copy.holding_meeples);
        copy.monasteries.settleCompletedMonasteries(copy.player_scores, copy.holding_meeples);
    }
    copy.resolveEndGameScore();
    pending[0] = copy.player_scores[0] - player_scores[0];
    pending[1] = copy.player_scores[1] - player_scores[1];
}

void Carcassonne::placeMeeple(int pos) {
    int x = last_x;
    int y = last_y;
    if (pos != MEEPLE_POS_SKIP) {
        holding_meeples[currentPlayer]--;
        if (pos == MEEPLE_POS_MONASTERY) {
            monasteries.placeMeeple(x, y, pos, currentPlayer, board, player_scores, holding_meeples);
        } else if (pos >= MEEPLE_POS_FIELD) {
            // A farmer is never settled or returned.
            const Placement &placement = board.board[y][x];
            fields.placeFarmer(placement.id, full_deck[placement.id][placement.rotation], pos, currentPlayer);
        } else {
            features.placeMeeple(x, y, pos, currentPlayer, board, player_scores, holding_meeples);
        }
    }
    features.settleAfterPlaceMeeple(x, y, board, player_scores, holding_meeples);
    monasteries.settleCompletedMonasteries(player_scores, holding_meeples);

    completed_turns++;
    currentPlayer = 1 - currentPlayer;
    if (max_turns > 0 && completed_turns >= max_turns) {
        resolveNoMoreDraws();
        return;
    }
    if (deck.total_remaining == 0) {
        resolveNoMoreDraws();
        return;
    }
    current_phase = PHASE_CHANCE;
}

Carcassonne Carcassonne::clone() const { return *this; }
