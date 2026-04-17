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

Feature::Feature(EdgeType feature_type, int id) {
    type = feature_type;
    tile_mask.set(id);
    opens = 1;
    meeple_mask[0] = 0;
    meeple_mask[1] = 0;
}

Feature Feature::operator+(const Feature &other) const {
    Feature res;
    res.type = type;
    res.tile_mask = tile_mask | other.tile_mask;
    res.meeple_mask[0] = meeple_mask[0] | other.meeple_mask[0];
    res.meeple_mask[1] = meeple_mask[1] | other.meeple_mask[1];
    res.opens = opens + other.opens;
    return res;
}

int Feature::countMeeples(uint8_t mask) {
    int count = 0;
    while (mask != 0) {
        count += mask & 1;
        mask >>= 1;
    }
    return count;
}

int Feature::meepleCount(int player) const { return countMeeples(meeple_mask[player]); }

bool Feature::hasMeeples() const { return meeple_mask[0] != 0 || meeple_mask[1] != 0; }

int Feature::getTileCount() const { return static_cast<int>(tile_mask.count()); }

int Feature::getScore() const {
    int tile_count = getTileCount();
    if (type == CITY) {
        int shield_count = static_cast<int>((tile_mask & SHIELD_MASK).count());
        if (opens == 0) {
            return (tile_count + shield_count) * 2;
        }
        return tile_count + shield_count;
    }
    return tile_count;
}

int Carcassonne::edgeIndex(int tile_id, int side) const { return (tile_id - 1) * 4 + side; }

bool Carcassonne::isInside(int x, int y) const { return x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE; }

void Carcassonne::addFrontierCell(int x, int y) {
    if (!isInside(x, y) || board[y][x].id != 0 || frontier[y][x]) {
        return;
    }
    frontier[y][x] = true;
    frontier_cells.push_back({static_cast<uint8_t>(x), static_cast<uint8_t>(y)});
}

void Carcassonne::removeFrontierCell(int x, int y) {
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

int Carcassonne::count3x3(int x, int y) const {
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

bool Carcassonne::canPlaceTileAt(int x, int y, const Tile &tile) const {
    if (board[y][x].id != 0 || !frontier[y][x]) {
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

bool Carcassonne::hasValidMove(int tile_id) const {
    for (int i = 0; i < frontier_cells.size(); ++i) {
        int x = frontier_cells[i].first;
        int y = frontier_cells[i].second;
        for (int rot = 0; rot < 4; ++rot) {
            if (canPlaceTileAt(x, y, full_deck[tile_id][rot])) {
                return true;
            }
        }
    }
    return false;
}

int Carcassonne::acquireMeepleToken(int player) { return free_meeple_ids[player][--free_meeple_count[player]]; }

void Carcassonne::releaseMeepleToken(int player, int token_id) {
    MeepleTokenState &token = meeple_tokens[player][token_id];
    if (!token.active) {
        return;
    }
    token.active = false;
    token.x = -1;
    token.y = -1;
    token.pos = -1;
    free_meeple_ids[player][free_meeple_count[player]++] = static_cast<uint8_t>(token_id);
}

void Carcassonne::releaseFeatureMeeples(Feature &feature) {
    for (int player = 0; player < 2; ++player) {
        uint8_t mask = feature.meeple_mask[player];
        while (mask != 0) {
            int token_id = 0;
            while (((mask >> token_id) & 1u) == 0) {
                token_id++;
            }
            releaseMeepleToken(player, token_id);
            mask &= static_cast<uint8_t>(mask - 1);
        }
        feature.meeple_mask[player] = 0;
    }
}

void Carcassonne::settleCompletedFeatures(int x, int y) {
    int tile_id = board[y][x].id;
    int seen_roots[4];
    int seen_count = 0;

    for (int i = 0; i < 4; ++i) {
        if (edge[y][x][i] == GRASS) {
            continue;
        }
        int root = featureMap.find(edgeIndex(tile_id, i));
        bool seen = false;
        for (int j = 0; j < seen_count; ++j) {
            if (seen_roots[j] == root) {
                seen = true;
                break;
            }
        }
        if (seen) {
            continue;
        }
        seen_roots[seen_count++] = root;

        Feature &feature = featureMap.getSetData(root);
        if (feature.opens != 0) {
            continue;
        }
        int m0 = feature.meepleCount(0);
        int m1 = feature.meepleCount(1);
        if (m0 == 0 && m1 == 0) {
            continue;
        }
        int score = feature.getScore();
        if (m0 >= m1) {
            player_scores[0] += score;
        }
        if (m1 >= m0) {
            player_scores[1] += score;
        }
        holding_meeples[0] += m0;
        holding_meeples[1] += m1;
        releaseFeatureMeeples(feature);
    }
}

void Carcassonne::settleCompletedMonasteries() {
    for (int i = active_monasteries.size() - 1; i >= 0; --i) {
        if (active_monasteries[i].tile_count == 9) {
            player_scores[active_monasteries[i].owner] += 9;
            holding_meeples[active_monasteries[i].owner]++;
            releaseMeepleToken(active_monasteries[i].owner, active_monasteries[i].token_id);
            active_monasteries.swap_pop_erase_at(i);
        }
    }
}

void Carcassonne::settleScore(int x, int y) {
    settleCompletedFeatures(x, y);
    settleCompletedMonasteries();
}

void Carcassonne::resolveEndGameScore() {
    for (auto it = featureMap.begin(); it != featureMap.end(); ++it) {
        Feature &feature = *it;
        if (feature.opens == 0 || feature.type == GRASS) {
            continue;
        }
        int m0 = feature.meepleCount(0);
        int m1 = feature.meepleCount(1);
        if (m0 == 0 && m1 == 0) {
            continue;
        }
        int score = feature.getScore();
        if (m0 >= m1) {
            player_scores[0] += score;
        }
        if (m1 >= m0) {
            player_scores[1] += score;
        }
    }

    for (int i = 0; i < active_monasteries.size(); ++i) {
        player_scores[active_monasteries[i].owner] += active_monasteries[i].tile_count;
    }
}

void Carcassonne::resolveNoMoreDraws() {
    current_tile_in_hand = 0;
    current_phase = PHASE_TERMINAL;
    is_game_over = true;
    resolveEndGameScore();
}

int Carcassonne::consumeType(int type_id) {
    int slot = type_counts[type_id] - 1;
    int physical_id = tile_type_tables.draw_physical_ids_by_type[type_id][slot];
    type_counts[type_id]--;
    total_remaining--;
    return physical_id;
}

void Carcassonne::initializeTypeCounts() {
    total_remaining = TOTAL_TILE_COUNT;
    for (int type_id = 1; type_id <= CANONICAL_TILE_TYPE_COUNT; ++type_id) {
        type_counts[type_id] = tile_type_tables.draw_count_by_type[type_id];
    }
}

void Carcassonne::placeTileOnBoard(int tile_id, int x, int y, int rot) {
    const Tile &tile = full_deck[tile_id][rot];
    last_x = x;
    last_y = y;

    removeFrontierCell(x, y);
    for (int i = 0; i < 4; ++i) {
        addFrontierCell(x + dx[i], y + dy[i]);
    }

    board[y][x].id = static_cast<uint8_t>(tile_id);
    board[y][x].rotation = static_cast<uint8_t>(rot);
    for (int i = 0; i < 4; ++i) {
        edge[y][x][i] = tile.edge[i];
    }

    for (int i = 0; i < 4; ++i) {
        featureMap.getSetData(edgeIndex(tile_id, i)) = Feature(tile.edge[i], tile_id);
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            if (tile.link[i] == tile.link[j] && tile.edge[i] != GRASS) {
                featureMap.unionSet(edgeIndex(tile_id, i), edgeIndex(tile_id, j));
            }
        }
    }

    for (int i = 0; i < 4; ++i) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (isInside(nx, ny) && board[ny][nx].id != 0 && tile.edge[i] != GRASS) {
            int my_edge = edgeIndex(tile_id, i);
            int their_edge = edgeIndex(board[ny][nx].id, op[i]);
            featureMap.unionSet(my_edge, their_edge);
            featureMap.getSetData(my_edge).opens -= 2;
        }
    }

    for (int i = 0; i < active_monasteries.size(); ++i) {
        if (std::abs(active_monasteries[i].x - x) <= 1 && std::abs(active_monasteries[i].y - y) <= 1) {
            active_monasteries[i].tile_count++;
        }
    }
}

Carcassonne::Carcassonne() : featureMap(std::plus<Feature>{}) {
    for (int player = 0; player < 2; ++player) {
        for (int token_id = 0; token_id < 7; ++token_id) {
            free_meeple_ids[player][token_id] = static_cast<uint8_t>(token_id);
        }
    }
    initializeTypeCounts();
    int start_tile_id = consumeType(START_TILE_TYPE);
    placeTileOnBoard(start_tile_id, BOARD_SIZE / 2, BOARD_SIZE / 2, START_TILE_ROTATION);
    current_phase = PHASE_CHANCE;
}

int Carcassonne::currentTileType() const {
    return current_tile_in_hand == 0 ? 0 : PHYSICAL_TO_CANONICAL_TYPE[current_tile_in_hand];
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
    if (current_phase != PHASE_CHANCE || total_remaining == 0) {
        return;
    }

    for (int type_id = 1; type_id <= CANONICAL_TILE_TYPE_COUNT; ++type_id) {
        if (type_counts[type_id] > 0) {
            out[count++] = {type_id, static_cast<double>(type_counts[type_id]) / total_remaining};
        }
    }
}

void Carcassonne::drawTile(int type_id) {
    int physical_id = consumeType(type_id);
    current_tile_in_hand = 0;
    if (hasValidMove(physical_id)) {
        current_tile_in_hand = physical_id;
        current_phase = PHASE_TILE;
        return;
    }
    if (total_remaining == 0) {
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

    for (int i = 0; i < frontier_cells.size(); ++i) {
        int x = frontier_cells[i].first;
        int y = frontier_cells[i].second;
        for (int rot = 0; rot < 4; ++rot) {
            if (canPlaceTileAt(x, y, full_deck[current_tile_in_hand][rot])) {
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
    const Tile &tile = full_deck[board[y][x].id][board[y][x].rotation];
    int seen_roots[4];
    int root_count = 0;
    for (int i = 0; i < 4; ++i) {
        if (tile.edge[i] == GRASS) {
            continue;
        }
        int root = featureMap.find(edgeIndex(board[y][x].id, i));
        bool seen = false;
        for (int j = 0; j < root_count; ++j) {
            if (seen_roots[j] == root) {
                seen = true;
                break;
            }
        }
        if (seen) {
            continue;
        }
        seen_roots[root_count++] = root;

        const Feature &feature = featureMap.getSetData(root);
        if (!feature.hasMeeples()) {
            ret.push_back(i);
        }
    }

    if (tile.monastery) {
        ret.push_back(4);
    }
    return ret;
}

void Carcassonne::placeMeeple(int pos) {
    int x = last_x;
    int y = last_y;
    if (pos != -1) {
        int token_id = acquireMeepleToken(currentPlayer);
        MeepleTokenState &token = meeple_tokens[currentPlayer][token_id];
        token.active = true;
        token.x = static_cast<int8_t>(x);
        token.y = static_cast<int8_t>(y);
        token.pos = static_cast<int8_t>(pos);
        holding_meeples[currentPlayer]--;
        if (pos == 4) {
            active_monasteries.push_back({x, y, count3x3(x, y), currentPlayer, token_id});
        } else {
            featureMap.getSetData(edgeIndex(board[y][x].id, pos)).meeple_mask[currentPlayer] |=
                static_cast<uint8_t>(1u << token_id);
        }
    }

    settleScore(x, y);
    currentPlayer = 1 - currentPlayer;
    if (total_remaining == 0) {
        resolveNoMoreDraws();
        return;
    }
    current_phase = PHASE_CHANCE;
}

Carcassonne Carcassonne::clone() const { return *this; }
