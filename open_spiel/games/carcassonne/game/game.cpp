#include "game.hpp"
#include "tile.hpp"

#include <algorithm>
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
    // Here, not only in the moves offered: a tile that fits only outside the
    // view is unplaceable (drawTile discards it).
    if (!inView(x, y) || !board.canPlaceTileAt(x, y, tile)) {
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

void Carcassonne::accumulateGoodsScore(int *scores) const {
    for (int kind = 0; kind < GOODS_KINDS; ++kind) {
        const int most = std::max(goods_tokens[0][kind], goods_tokens[1][kind]);
        for (int player = 0; player < 2; ++player) {
            if (most > 0 && goods_tokens[player][kind] == most) {
                scores[player] += GOODS_POINTS;
            }
        }
    }
}

void Carcassonne::resolveEndGameScore() {
    features.resolveEndGameScore(player_scores);
    monasteries.resolveEndGameScore(player_scores);
    fields.accumulateScore(player_scores, features);
    accumulateGoodsScore(player_scores);
}

void Carcassonne::resolveNoMoreDraws() {
    current_tile_in_hand = 0;
    current_phase = PHASE_TERMINAL;
    builder_extra_tile = builder_second_tile = false;  // no tile left to play
    resolveEndGameScore();
}

namespace {

// The view's first cell along one axis, for tiles spanning lo..hi on it:
// centred on them; when they span an even count, half a cell towards the
// centre cell, so that a game rotated about it gets the view rotated too.
// Kept on the board.
int viewOrigin(int lo, int hi) {
    const int twice_centre = lo + hi;
    int centre = twice_centre / 2;
    if (twice_centre % 2 != 0 && twice_centre < 2 * (BOARD_SIZE / 2)) {
        ++centre;
    }
    return std::clamp(centre - VIEW_SIZE / 2, 0, BOARD_SIZE - VIEW_SIZE);
}

} // namespace

void Carcassonne::updateView(int x, int y) {
    tiles_x0 = std::min(tiles_x0, x);
    tiles_x1 = std::max(tiles_x1, x);
    tiles_y0 = std::min(tiles_y0, y);
    tiles_y1 = std::max(tiles_y1, y);
    view_x0 = viewOrigin(tiles_x0, tiles_x1);
    view_y0 = viewOrigin(tiles_y0, tiles_y1);
}

void Carcassonne::placeTileOnBoard(int tile_id, int x, int y, int rot) {
    const Tile &tile = full_deck[tile_id][rot];
    last_x = x;
    last_y = y;
    updateView(x, y);
    frontier.placeTileOnBoard(tile_id, x, y, rot, board);
    board.placeTileOnBoard(tile_id, x, y, rot, tile);
    features.placeTileOnBoard(tile_id, x, y, rot, tile, board);
    fields.placeTileOnBoard(tile_id, x, y, tile, board, features);
    monasteries.placeTileOnBoard(tile_id, x, y, rot);
    logs.placeTileOnBoard(tile_id, x, y, rot);
    if (river_rules && riverEdgeCount(tile) > 0) {
        advanceRiver(x, y, tile);
        river_tiles_placed++;
    }
}

Carcassonne::Carcassonne(int max_turns) : Carcassonne(max_turns, START_TILE_ROTATION) {}

Carcassonne::Carcassonne(int max_turns, int start_rotation, uint32_t expansions, uint32_t rules)
    : max_turns(max_turns), expansions(expansions | BASE_ONLY) {
    this->rules = (rules & RULED_EXPANSIONS & this->expansions) | (this->expansions & RULES_REQUIRED_EXPANSIONS);
    deck.initializeTypeCounts(this->expansions);
    river_rules = deck.river_first;
    big_meeple_rules = (this->rules & expansionBit(EXP_INNS_CATHEDRALS)) != 0;
    if (big_meeple_rules) {
        holding_big_meeples[0] = holding_big_meeples[1] = 1;
    }
    features.inns_cathedrals = big_meeple_rules;
    builder_rules = (this->rules & expansionBit(EXP_TRADERS_BUILDERS)) != 0;
    pig_rules = builder_rules;
    goods_rules = builder_rules;
    features.goods_rules = goods_rules;
    if (builder_rules) {
        holding_builders[0] = holding_builders[1] = 1;
        holding_pigs[0] = holding_pigs[1] = 1;
    }
    dragon_rules = (this->rules & expansionBit(EXP_PRINCESS_DRAGON)) != 0;
    portal_rules = dragon_rules;
    princess_rules = dragon_rules;
    deck.hold_dragon_tiles = dragon_rules;
    // With the river the spring starts the game instead of the base start tile,
    // which the deck leaves out.
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
    if (deck.drawableRemaining() == 0) {
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
    // A builder goes down only in the meeple phase, so one in a feature of this
    // tile was there before: the tile extends it. Even if it completes the
    // feature, which sends the builder home, the extra tile stands.
    builder_extra_tile = builder_rules && !builder_second_tile &&
                         features.hasBuilderOf(tile_id, full_deck[tile_id][rot], currentPlayer);
    // The cities this tile completes hand their goods to whoever placed it.
    if (goods_rules) {
        features.collectGoods(tile_id, full_deck[tile_id][rot], goods_tokens[currentPlayer]);
    }
    // A volcano brings the dragon into play, or takes it there; on the way it
    // eats nothing, and nothing is on a tile just placed.
    if (dragon_rules && (full_deck[tile_id][rot].tile_marks & TILE_VOLCANO)) {
        dragon_x = x;
        dragon_y = y;
        deck.hold_dragon_tiles = false;
    }
}

MeepleMoves Carcassonne::getLegalMeepleMoves() const {
    MeepleMoves ret;
    if (current_phase != PHASE_MEEPLE) {
        return ret;
    }
    ret.push_back(MEEPLE_POS_SKIP);
    const bool meeple = holding_meeples[currentPlayer] > 0;
    const bool big = holding_big_meeples[currentPlayer] > 0;
    const bool builder = holding_builders[currentPlayer] > 0;
    const bool pig = holding_pigs[currentPlayer] > 0;
    if (!meeple && !big && !builder && !pig) {
        return ret;
    }
    int x = last_x;
    int y = last_y;
    const Placement &placement = board.board[y][x];
    const Tile &tile = full_deck[placement.id][placement.rotation];
    // No piece shares a tile with the dragon, which a volcano has just brought.
    if (dragon_rules && (tile.tile_marks & TILE_VOLCANO)) {
        return ret;
    }
    MeepleMoves spots;
    features.getLegalMeepleMoves(spots, x, y, board, tile);
    if (tile.monastery) {
        spots.push_back(MEEPLE_POS_MONASTERY);
    }
    fields.getLegalFarmerMoves(spots, placement.id, tile);
    for (int i = 0; meeple && i < spots.size(); ++i) {
        ret.push_back(spots[i]);
    }
    for (int i = 0; big && i < spots.size(); ++i) {
        ret.push_back(spots[i] + MEEPLE_POS_BIG);
    }
    if (builder) {
        features.getLegalBuilderMoves(ret, x, y, board, tile, currentPlayer);
    }
    if (pig) {
        fields.getLegalPigMoves(ret, placement.id, tile, currentPlayer);
    }
    return ret;
}

void Carcassonne::getFocusSpotGroups(int8_t sides[4], int8_t half_edges[HALF_EDGE_COUNT]) const {
    std::fill(sides, sides + 4, -1);
    std::fill(half_edges, half_edges + HALF_EDGE_COUNT, -1);
    if (current_phase != PHASE_MEEPLE && current_phase != PHASE_SPOT) {
        return;
    }
    const int x = focusX();
    const int y = focusY();
    if (current_phase == PHASE_SPOT && spot_choice == SPOT_PRINCESS) {
        // A piece is named by its spot, the lowest of its spots.
        for (const Piece &piece : pieces) {
            if (piece.x != x || piece.y != y) {
                continue;
            }
            for (int side = 0; side < 4; ++side) {
                if (piece.spots >> side & 1) {
                    sides[side] = piece.spot;
                }
            }
            for (int e = 0; e < HALF_EDGE_COUNT; ++e) {
                if (piece.spots >> (MEEPLE_POS_FIELD + e) & 1) {
                    half_edges[e] = static_cast<int8_t>(piece.spot - MEEPLE_POS_FIELD);
                }
            }
        }
        return;
    }
    const Placement &placement = board.board[y][x];
    const Tile &tile = full_deck[placement.id][placement.rotation];
    int roots[4];
    for (int i = 0; i < 4; ++i) {
        if (!isFeatureEdge(tile.edge[i])) {
            continue;
        }
        roots[i] = features.featureMap.find(features.edgeIndex(placement.id, i));
        sides[i] = static_cast<int8_t>(i);
        for (int j = 0; j < i; ++j) {
            if (sides[j] != -1 && roots[j] == roots[i]) {
                sides[i] = sides[j];
                break;
            }
        }
    }
    fields.getHalfEdgeGroups(placement.id, tile, half_edges);
}

void Carcassonne::getPendingScore(int pending[2]) const {
    pending[0] = pending[1] = 0;
    if (current_phase == PHASE_TERMINAL) {
        return;  // The end-game scoring is already in player_scores.
    }
    features.accumulatePendingScore(pending);
    monasteries.accumulatePendingScore(pending);
    fields.accumulateScore(pending, features);
    accumulateGoodsScore(pending);
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
    if (copy.current_phase == PHASE_MEEPLE || copy.current_phase == PHASE_DRAGON ||
        copy.current_phase == PHASE_SPOT) {
        // What the end of the turn settles whatever the moves are.
        copy.features.settleAfterPlaceMeeple(last_x, last_y, copy.board, copy.player_scores, copy.holding_meeples,
                                             copy.holding_big_meeples, copy.holding_builders);
        copy.monasteries.settleCompletedMonasteries(copy.player_scores, copy.holding_meeples,
                                                    copy.holding_big_meeples);
    }
    copy.resolveEndGameScore();
    pending[0] = copy.player_scores[0] - player_scores[0];
    pending[1] = copy.player_scores[1] - player_scores[1];
}

uint16_t Carcassonne::spotsOf(int x, int y, int spot) const {
    if (spot == MEEPLE_POS_MONASTERY || spot == MEEPLE_POS_INNER_FIELD) {
        return static_cast<uint16_t>(1u << spot);
    }
    const Placement &placement = board.board[y][x];
    const Tile &tile = full_deck[placement.id][placement.rotation];
    uint16_t spots = 0;
    if (spot < MEEPLE_POS_MONASTERY) {
        const int root = features.featureMap.find(features.edgeIndex(placement.id, spot));
        for (int side = 0; side < 4; ++side) {
            if (isFeatureEdge(tile.edge[side]) && features.featureMap.find(features.edgeIndex(placement.id, side)) == root) {
                spots |= static_cast<uint16_t>(1u << side);
            }
        }
        return spots;
    }
    const int root = fields.fieldMap.find(fields.fieldIndex(placement.id, tile.field[spot - MEEPLE_POS_FIELD]));
    for (int e = 0; e < HALF_EDGE_COUNT; ++e) {
        if (tile.field[e] != -1 && fields.fieldMap.find(fields.fieldIndex(placement.id, tile.field[e])) == root) {
            spots |= static_cast<uint16_t>(1u << (MEEPLE_POS_FIELD + e));
        }
    }
    return spots;
}

void Carcassonne::putPiece(int x, int y, int pos) {
    const int spot = meepleSpot(pos);
    const PieceKind kind = isBuilderPos(pos)       ? PIECE_BUILDER
                           : isPigPos(pos)         ? PIECE_PIG
                           : isBigMeeplePos(pos)   ? PIECE_BIG_MEEPLE
                                                   : PIECE_MEEPLE;
    const Placement &placement = board.board[y][x];
    const Tile &tile = full_deck[placement.id][placement.rotation];
    pieces.push_back({static_cast<int8_t>(x), static_cast<int8_t>(y), placement.id, static_cast<int8_t>(spot),
                      static_cast<uint8_t>(currentPlayer), kind, spotsOf(x, y, spot)});
    if (kind == PIECE_BUILDER) {
        holding_builders[currentPlayer]--;
        features.placeBuilder(x, y, spot, currentPlayer, board);
    } else if (kind == PIECE_PIG) {
        // Like a farmer, a pig is never settled or returned.
        holding_pigs[currentPlayer]--;
        fields.placePig(placement.id, tile, pos - MEEPLE_POS_PIG, currentPlayer);
    } else {
        const bool big = kind == PIECE_BIG_MEEPLE;
        (big ? holding_big_meeples : holding_meeples)[currentPlayer]--;
        if (spot == MEEPLE_POS_MONASTERY) {
            monasteries.placeMeeple(x, y, currentPlayer, big, board);
        } else if (spot >= MEEPLE_POS_FIELD) {
            // A farmer is never settled or returned.
            fields.placeFarmer(placement.id, tile, spot, currentPlayer, big);
        } else {
            features.placeMeeple(x, y, spot, currentPlayer, big, board);
        }
    }
}

void Carcassonne::placeMeeple(int pos) {
    if (pos != MEEPLE_POS_SKIP) {
        putPiece(last_x, last_y, pos);
    }
    endPiecePhase();
}

int Carcassonne::princessCityRoot() const {
    if (!princess_rules || last_x < 0) {
        return -1;
    }
    const Placement &placement = board.board[last_y][last_x];
    const Tile &tile = full_deck[placement.id][placement.rotation];
    for (int side = 0; side < 4; ++side) {
        if (tile.edge[side] == CITY && (tile.marks[side] & MARK_PRINCESS)) {
            return features.featureMap.find(features.edgeIndex(placement.id, side));
        }
    }
    return -1;
}

FixedVector<int, MAX_PIECES> Carcassonne::princessKnightsAt(int x, int y) const {
    FixedVector<int, MAX_PIECES> ret;
    const int root = princessCityRoot();
    if (root < 0) {
        return ret;
    }
    for (int i = 0; i < pieces.size(); ++i) {
        const Piece &p = pieces[i];
        if (p.x == x && p.y == y && (p.kind == PIECE_MEEPLE || p.kind == PIECE_BIG_MEEPLE) &&
            p.spot < MEEPLE_POS_MONASTERY && features.featureMap.find(features.edgeIndex(p.tile_id, p.spot)) == root) {
            ret.push_back(i);
        }
    }
    return ret;
}

MeepleMoves Carcassonne::portalMovesAt(int x, int y) const {
    MeepleMoves ret;
    const bool meeple = holding_meeples[currentPlayer] > 0;
    const bool big = holding_big_meeples[currentPlayer] > 0;
    const Placement &placement = board.board[y][x];
    // No piece shares a tile with the dragon.
    if ((!meeple && !big) || placement.id == 0 || (x == dragon_x && y == dragon_y)) {
        return ret;
    }
    const Tile &tile = full_deck[placement.id][placement.rotation];
    // As on the tile just placed, but nothing complete: no closed road or city
    // (even one the tile just placed closed), no monastery with all 8 tiles
    // round it. Fields are never complete.
    MeepleMoves features_free;
    features.getLegalMeepleMoves(features_free, x, y, board, tile);
    MeepleMoves spots;
    for (int side : features_free) {
        if (featureAt(placement.id, side).opens > 0) {
            spots.push_back(side);
        }
    }
    if (tile.monastery && monasteries.ownerAt(x, y) == -1 && board.count3x3(x, y) < 9) {
        spots.push_back(MEEPLE_POS_MONASTERY);
    }
    fields.getLegalFarmerMoves(spots, placement.id, tile);
    for (int i = 0; meeple && i < spots.size(); ++i) {
        ret.push_back(spots[i]);
    }
    for (int i = 0; big && i < spots.size(); ++i) {
        ret.push_back(spots[i] + MEEPLE_POS_BIG);
    }
    return ret;
}

Cells Carcassonne::getLegalPortalCells() const {
    Cells ret;
    if (current_phase != PHASE_MEEPLE || !portal_rules) {
        return ret;
    }
    const Placement &last = board.board[last_y][last_x];
    if (!(full_deck[last.id][last.rotation].tile_marks & TILE_PORTAL)) {
        return ret;
    }
    for (int y = tiles_y0; y <= tiles_y1; ++y) {
        for (int x = tiles_x0; x <= tiles_x1; ++x) {
            if ((x != last_x || y != last_y) && portalMovesAt(x, y).size() > 0) {
                ret.push_back({static_cast<int8_t>(x), static_cast<int8_t>(y)});
            }
        }
    }
    return ret;
}

Cells Carcassonne::getLegalPrincessCells() const {
    Cells ret;
    const int root = current_phase == PHASE_MEEPLE ? princessCityRoot() : -1;
    if (root < 0) {
        return ret;
    }
    for (const Piece &p : pieces) {
        if ((p.kind != PIECE_MEEPLE && p.kind != PIECE_BIG_MEEPLE) || p.spot >= MEEPLE_POS_MONASTERY ||
            features.featureMap.find(features.edgeIndex(p.tile_id, p.spot)) != root) {
            continue;
        }
        const std::pair<int8_t, int8_t> cell = {p.x, p.y};
        if (std::find(ret.begin(), ret.end(), cell) == ret.end()) {
            ret.push_back(cell);
        }
    }
    std::sort(ret.begin(), ret.end(), [](const auto &a, const auto &b) {
        return std::make_pair(a.second, a.first) < std::make_pair(b.second, b.first);
    });
    return ret;
}

void Carcassonne::chooseCell(SpotChoice choice, int x, int y) {
    if (choice == SPOT_PORTAL) {
        const MeepleMoves moves = portalMovesAt(x, y);
        if (moves.size() == 1) {
            putPiece(x, y, moves[0]);
            endPiecePhase();
            return;
        }
    } else {
        const FixedVector<int, MAX_PIECES> knights = princessKnightsAt(x, y);
        if (knights.size() == 1) {
            sendHome(knights[0]);
            endPiecePhase();
            return;
        }
    }
    spot_choice = choice;
    spot_x = x;
    spot_y = y;
    current_phase = PHASE_SPOT;
}

MeepleMoves Carcassonne::getLegalSpotMoves() const {
    if (current_phase != PHASE_SPOT) {
        return {};
    }
    if (spot_choice == SPOT_PORTAL) {
        return portalMovesAt(spot_x, spot_y);
    }
    MeepleMoves ret;
    for (int index : princessKnightsAt(spot_x, spot_y)) {
        ret.push_back(pieces[index].spot);
    }
    std::sort(ret.begin(), ret.end());
    return ret;
}

void Carcassonne::chooseSpot(int pos) {
    if (spot_choice == SPOT_PORTAL) {
        putPiece(spot_x, spot_y, pos);
    } else {
        for (int index : princessKnightsAt(spot_x, spot_y)) {
            if (pieces[index].spot == pos) {
                sendHome(index);
                break;
            }
        }
    }
    endPiecePhase();
}

void Carcassonne::endPiecePhase() {
    spot_x = spot_y = -1;
    // A dragon tile sets the dragon moving before the turn is scored.
    const Placement &placed = board.board[last_y][last_x];
    if (dragon_rules && dragonInPlay() && (full_deck[placed.id][placed.rotation].tile_marks & TILE_DRAGON)) {
        dragon_visited = {};
        dragon_visited.push_back({static_cast<int8_t>(dragon_x), static_cast<int8_t>(dragon_y)});
        dragon_steps = 0;
        if (getLegalDragonMoves().size() > 0) {
            dragon_turn_player = currentPlayer;
            current_phase = PHASE_DRAGON;
            return;
        }
    }
    finishTurn();
}

FixedVector<int, 4> Carcassonne::getLegalDragonMoves() const {
    FixedVector<int, 4> ret;
    for (int side = 0; side < 4; ++side) {
        if (dragonCanEnter(dragon_x + dx[side], dragon_y + dy[side])) {
            ret.push_back(side);
        }
    }
    return ret;
}

bool Carcassonne::dragonCanEnter(int x, int y) const {
    if (!isInside(x, y) || board.board[y][x].id == 0) {
        return false;
    }
    for (const auto &cell : dragon_visited) {
        if (cell.first == x && cell.second == y) {
            return false;
        }
    }
    return true;
}

void Carcassonne::moveDragon(int side) {
    dragon_x += dx[side];
    dragon_y += dy[side];
    dragon_visited.push_back({static_cast<int8_t>(dragon_x), static_cast<int8_t>(dragon_y)});
    dragon_steps++;
    eatPiecesAt(dragon_x, dragon_y);
    if (dragon_steps < DRAGON_STEPS && getLegalDragonMoves().size() > 0) {
        currentPlayer = 1 - currentPlayer;
        return;
    }
    currentPlayer = dragon_turn_player;
    dragon_turn_player = -1;
    finishTurn();
}

Piece Carcassonne::removePiece(int index) {
    const Piece piece = pieces[index];
    pieces.swap_pop_erase_at(index);
    const int owner = piece.owner;
    const Tile &tile = full_deck[piece.tile_id][board.board[piece.y][piece.x].rotation];
    switch (piece.kind) {
    case PIECE_BUILDER:
        features.featureMap.getSetData(features.edgeIndex(piece.tile_id, piece.spot)).builders[owner] = 0;
        holding_builders[owner]++;
        return piece;
    case PIECE_PIG:
        fields.fieldMap.getSetData(fields.fieldIndex(piece.tile_id, tile.field[piece.spot - MEEPLE_POS_FIELD]))
            .pigs[owner] = 0;
        holding_pigs[owner]++;
        return piece;
    case PIECE_MEEPLE:
    case PIECE_BIG_MEEPLE:
        break;
    }
    const bool big = piece.kind == PIECE_BIG_MEEPLE;
    (big ? holding_big_meeples : holding_meeples)[owner]++;
    if (piece.spot == MEEPLE_POS_MONASTERY) {
        for (int i = 0; i < monasteries.active_monasteries.size(); ++i) {
            if (monasteries.active_monasteries[i].x == piece.x && monasteries.active_monasteries[i].y == piece.y) {
                monasteries.active_monasteries.swap_pop_erase_at(i);
                break;
            }
        }
    } else if (piece.spot >= MEEPLE_POS_FIELD) {
        fields.removeFarmer(piece.tile_id, tile, piece.spot, owner, big);
    } else {
        Feature &feature = features.featureMap.getSetData(features.edgeIndex(piece.tile_id, piece.spot));
        feature.meeple_count[owner] -= big ? 2 : 1;
        feature.big_meeples[owner] -= big ? 1 : 0;
    }
    return piece;
}

void Carcassonne::sendHome(int index) {
    const Piece gone = removePiece(index);
    if (gone.kind != PIECE_MEEPLE && gone.kind != PIECE_BIG_MEEPLE) {
        return;
    }
    // The builder (on a road or city) and the pig (on a field) stay only
    // with a follower of their owner.
    const int owner = gone.owner;
    const Tile &tile = full_deck[gone.tile_id][board.board[gone.y][gone.x].rotation];
    if (gone.spot < MEEPLE_POS_MONASTERY) {
        const int root = features.featureMap.find(features.edgeIndex(gone.tile_id, gone.spot));
        if (features.featureMap.getSetData(root).meeple_count[owner] > 0) {
            return;
        }
        for (int i = 0; i < pieces.size(); ++i) {
            const Piece &p = pieces[i];
            if (p.kind == PIECE_BUILDER && p.owner == owner &&
                features.featureMap.find(features.edgeIndex(p.tile_id, p.spot)) == root) {
                removePiece(i);
                return;
            }
        }
    } else if (gone.spot >= MEEPLE_POS_FIELD) {
        const int local =
            gone.spot == MEEPLE_POS_INNER_FIELD ? tile.innerField() : tile.field[gone.spot - MEEPLE_POS_FIELD];
        const int root = fields.fieldMap.find(fields.fieldIndex(gone.tile_id, local));
        if (fields.fieldMap.getSetData(root).farmer_count[owner] > 0) {
            return;
        }
        for (int i = 0; i < pieces.size(); ++i) {
            const Piece &p = pieces[i];
            if (p.kind != PIECE_PIG || p.owner != owner) {
                continue;
            }
            const Tile &pig_tile = full_deck[p.tile_id][board.board[p.y][p.x].rotation];
            if (fields.fieldMap.find(fields.fieldIndex(p.tile_id, pig_tile.field[p.spot - MEEPLE_POS_FIELD])) ==
                root) {
                removePiece(i);
                return;
            }
        }
    }
}

void Carcassonne::eatPiecesAt(int x, int y) {
    while (true) {
        int index = -1;
        for (int i = 0; i < pieces.size() && index < 0; ++i) {
            if (pieces[i].x == x && pieces[i].y == y) {
                index = i;
            }
        }
        if (index < 0) {
            return;
        }
        sendHome(index);
    }
}

void Carcassonne::forgetSettledPieces() {
    for (int i = pieces.size() - 1; i >= 0; --i) {
        const Piece &piece = pieces[i];
        // Farmers and pigs stay till the end. A road or city is scored, and its
        // pieces sent home, in the turn it is closed: the turn just ending.
        const bool settled = piece.spot == MEEPLE_POS_MONASTERY
                                 ? monasteries.ownerAt(piece.x, piece.y) == -1
                                 : piece.spot < MEEPLE_POS_MONASTERY &&
                                       features.featureMap.getSetData(features.edgeIndex(piece.tile_id, piece.spot))
                                               .opens == 0;
        if (settled) {
            pieces.swap_pop_erase_at(i);
        }
    }
}

void Carcassonne::finishTurn() {
    features.settleAfterPlaceMeeple(last_x, last_y, board, player_scores, holding_meeples, holding_big_meeples,
                                    holding_builders);
    monasteries.settleCompletedMonasteries(player_scores, holding_meeples, holding_big_meeples);
    forgetSettledPieces();

    completed_turns++;
    // The builder's double turn: the same player draws once more, never a
    // third time.
    builder_second_tile = builder_extra_tile;
    builder_extra_tile = false;
    if (!builder_second_tile) {
        currentPlayer = 1 - currentPlayer;
    }
    if (max_turns > 0 && completed_turns >= max_turns) {
        resolveNoMoreDraws();
        return;
    }
    if (deck.drawableRemaining() == 0) {
        resolveNoMoreDraws();
        return;
    }
    current_phase = PHASE_CHANCE;
}

Carcassonne Carcassonne::clone() const { return *this; }
