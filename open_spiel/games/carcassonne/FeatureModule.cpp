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

int FeatureModule::edgeIndex(int tile_id, int side) const { return (tile_id - 1) * 4 + side; }

FeatureModule::FeatureModule() : featureMap(std::plus<Feature>{}) {}

void FeatureModule::settleCompletedFeatures(int tile_id, int side, int *player_scores) {
    int root = featureMap.find(edgeIndex(tile_id, side));
    Feature &feature = featureMap.getSetData(root);
    if (feature.opens != 0) {
        return;
    }
    int m0 = feature.meepleCount(0);
    int m1 = feature.meepleCount(1);
    if (m0 == 0 && m1 == 0) {
        return;
    }
    int score = feature.getScore();
    if (m0 >= m1) {
        player_scores[0] += score;
    }
    if (m1 >= m0) {
        player_scores[1] += score;
    }
    releaseFeatureMeeples(feature);
}

void FeatureModule::resolveEndGameScore(int *player_scores) {
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
}

void FeatureModule::placeTileOnBoard(int tile_id, int x, int y, int rot, const Tile &tile, const BoardModule &board) {
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
        if (isInside(nx, ny) && board.board[ny][nx].id != 0 && tile.edge[i] != GRASS) {
            int my_edge = edgeIndex(tile_id, i);
            int their_edge = edgeIndex(board.board[ny][nx].id, op[i]);
            featureMap.unionSet(my_edge, their_edge);
            featureMap.getSetData(my_edge).opens -= 2;
        }
    }
}

void FeatureModule::getLegalMeepleMoves(FixedVector<int, 6> &ret, int x, int y, const BoardModule &board, const Tile &tile) const {
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
}