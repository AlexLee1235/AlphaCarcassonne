#pragma once

#include "DisjointSet.hpp"
#include "FixedVector.hpp"
#include "tile.hpp"

#include <array>
#include <bitset>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

constexpr int BOARD_SIZE = 15;
constexpr int TOTAL_TILE_COUNT = PHYSICAL_TILE_COUNT;
constexpr int MAX_FRONTIER_CELLS = TOTAL_TILE_COUNT * 2 + 2;
constexpr int EDGE_SLOT_COUNT = TOTAL_TILE_COUNT * 4;

enum GamePhase { PHASE_CHANCE = 0, PHASE_TILE = 1, PHASE_MEEPLE = 2, PHASE_TERMINAL = 3 };

inline bool isInside(int x, int y) { return x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE; }

struct TileMove {
    uint8_t x = 0;
    uint8_t y = 0;
    uint8_t rot = 0;
};

struct ChanceBranch {
    int type_id = 0;
    double probability = 0.0;
};

struct Placement {
    uint8_t id = 0;
    uint8_t rotation = 0;
};

struct MeepleTokenState {
    bool active = false;
    int8_t x = -1;
    int8_t y = -1;
    int8_t pos = -1;
};

struct MonasteryTracker {
    int x = 0;
    int y = 0;
    int tile_count = 0;
    int owner = 0;
};

class Feature {
  public:
    EdgeType type = NONE;
    std::bitset<73> tile_mask;
    uint8_t opens = 0;
    uint8_t meeple_count[2] = {};

    Feature() = default;
    Feature(EdgeType type, int id);

    Feature operator+(const Feature &other) const;

    bool hasMeeples() const;
    int getTileCount() const;
    int getScore() const;
};

class BoardModule {
  public:
    Placement board[BOARD_SIZE][BOARD_SIZE] = {};
    EdgeType edge[BOARD_SIZE][BOARD_SIZE][4] = {};
    int count3x3(int x, int y) const;
    bool canPlaceTileAt(int x, int y, const Tile &tile) const;
    void placeTileOnBoard(int tile_id, int x, int y, int rot, const Tile &tile);
};

class FeatureModule {
    void settleCompletedFeatures(int tile_id, int side, int *player_scores, int *holding_meeples);

  public:
    DisjointSet<Feature, std::plus<Feature>, EDGE_SLOT_COUNT> featureMap;
    FeatureModule();
    int edgeIndex(int tile_id, int side) const;
    void resolveEndGameScore(int *player_scores);
    void placeTileOnBoard(int tile_id, int x, int y, int rot, const Tile &tile, const BoardModule &board);
    void getLegalMeepleMoves(FixedVector<int, 6> &ret, int x, int y, const BoardModule &board, const Tile &tile) const;
    void placeMeeple(int x, int y, int pos, int player, const BoardModule &board, int *player_scores, int *holding_meeples);
    void settleAfterPlaceMeeple(int x, int y, const BoardModule &board, int *player_scores, int *holding_meeples);
};

class MonasteryModule {
  public:
    FixedVector<MonasteryTracker, 6> active_monasteries;
    void placeTileOnBoard(int tile_id, int x, int y, int rot);
    void resolveEndGameScore(int *player_scores);
    void placeMeeple(int x, int y, int pos, int player, const BoardModule &board, int *player_scores, int *holding_meeples);
    void settleCompletedMonasteries(int *player_scores, int *holding_meeples);
};

class FrontierModule {
    void addFrontierCell(int x, int y, const BoardModule &board);
    void removeFrontierCell(int x, int y);

  public:
    bool frontier[BOARD_SIZE][BOARD_SIZE] = {};
    FixedVector<std::pair<uint8_t, uint8_t>, MAX_FRONTIER_CELLS> frontier_cells;
    void placeTileOnBoard(int tile_id, int x, int y, int rot, const BoardModule &board);
};

class DeckModule {
  public:
    int total_remaining = 0;
    int type_counts[CANONICAL_TILE_TYPE_COUNT + 1] = {};
    int consumeType(int type_id);
    void initializeTypeCounts();
    void getAvailableDraws(ChanceBranch *out, int &count) const;
};

class LogModule {
  public:
    int tile_x[73], tile_y[73];
    LogModule() {
        fill(tile_x, tile_x + 73, -1);
        fill(tile_y, tile_y + 73, -1);
    }
    void placeTileOnBoard(int tile_id, int x, int y, int rot) {
        tile_x[tile_id] = x;
        tile_y[tile_id] = y;
    }
    void getMeepleMap(const FeatureModule &features, const MonasteryModule &monasteries, int player, float *span) const {
        int opponent = 1 - player;
        for (int i = 1; i < 73; i++) {
            int x = tile_x[i], y = tile_y[i];
            if (x == -1 || y == -1)
                continue;
            for (int j = 0; j < 4; j++) {
                int index0 = (j * BOARD_SIZE + y) * BOARD_SIZE + x;
                int index1 = ((5 + j) * BOARD_SIZE + y) * BOARD_SIZE + x;
                const Feature &f = features.featureMap.getSetData(features.edgeIndex(i, j));
                int my_meeples = f.meeple_count[player], opponent_meeples = f.meeple_count[opponent];
                span[index0] = 1.0f / 7.0f * my_meeples;
                span[index1] = 1.0f / 7.0f * opponent_meeples;
            }
        }
        for (const MonasteryTracker &tr : monasteries.active_monasteries) {
            int x = tr.x, y = tr.y;
            int plane = tr.owner == player ? 4 : 9;
            int index = (plane * BOARD_SIZE + y) * BOARD_SIZE + x;
            span[index] = 1.0f;
        }
    }
};

class Carcassonne {
  private:
    FeatureModule features;
    MonasteryModule monasteries;
    FrontierModule frontier;
    BoardModule board;
    DeckModule deck;
    LogModule logs;

    void placeTileOnBoard(int tile_id, int x, int y, int rot);
    bool hasValidMove(int tile_id) const;
    void resolveEndGameScore();
    void resolveNoMoreDraws();

  public:
    int last_x = -1;
    int last_y = -1;
    GamePhase current_phase = PHASE_CHANCE;
    int player_scores[2] = {0, 0};
    int holding_meeples[2] = {7, 7};
    int currentPlayer = 0;
    int current_tile_in_hand = 0;

    Carcassonne();
    int currentTileType() const;
    Carcassonne clone() const;

    Placement getPlacement(int x, int y) const { return board.board[y][x]; }
    int getTotalRemaining() const { return deck.total_remaining; }
    void WriteMeepleMap(int player, float *span) const;

    void getAvailableDraws(ChanceBranch *out, int &count) const;
    void drawTile(int type_id);
    void getLegalTileMoves(TileMove *out, int &count) const;
    void placeTile(int x, int y, int rot);
    FixedVector<int, 6> getLegalMeepleMoves() const;
    void placeMeeple(int pos);
};
