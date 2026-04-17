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
    int token_id = 0;
};

class Feature {
  public:
    EdgeType type = NONE;
    std::bitset<73> tile_mask;
    uint8_t opens = 0;
    uint8_t meeple_mask[2] = {};

    Feature() = default;
    Feature(EdgeType type, int id);

    Feature operator+(const Feature &other) const;

    static int countMeeples(uint8_t mask);

    int meepleCount(int player) const;
    bool hasMeeples() const;
    int getTileCount() const;
    int getScore() const;
};

class Carcassonne {
  private:
    bool frontier[BOARD_SIZE][BOARD_SIZE] = {};
    FixedVector<std::pair<uint8_t, uint8_t>, MAX_FRONTIER_CELLS> frontier_cells;
    DisjointSet<Feature, std::plus<Feature>, EDGE_SLOT_COUNT> featureMap;
    FixedVector<MonasteryTracker, 6> active_monasteries;
    uint8_t free_meeple_ids[2][7] = {};
    uint8_t free_meeple_count[2] = {7, 7};

    int edgeIndex(int tile_id, int side) const;
    bool isInside(int x, int y) const;
    void addFrontierCell(int x, int y);
    void removeFrontierCell(int x, int y);
    int count3x3(int x, int y) const;
    bool canPlaceTileAt(int x, int y, const Tile &tile) const;
    bool hasValidMove(int tile_id) const;
    int acquireMeepleToken(int player);
    void releaseMeepleToken(int player, int token_id);
    void releaseFeatureMeeples(Feature &feature);
    void settleCompletedFeatures(int x, int y);
    void settleCompletedMonasteries();
    void settleScore(int x, int y);
    void resolveEndGameScore();
    void resolveNoMoreDraws();
    int consumeType(int type_id);
    void initializeTypeCounts();
    void placeTileOnBoard(int tile_id, int x, int y, int rot);

  public:
    Placement board[BOARD_SIZE][BOARD_SIZE] = {};
    EdgeType edge[BOARD_SIZE][BOARD_SIZE][4] = {};
    MeepleTokenState meeple_tokens[2][7] = {};
    int total_remaining = 0;
    int current_tile_in_hand = 0;
    int type_counts[CANONICAL_TILE_TYPE_COUNT + 1] = {};
    int last_x = -1;
    int last_y = -1;
    GamePhase current_phase = PHASE_CHANCE;
    bool is_game_over = false;
    int holding_meeples[2] = {7, 7};
    int player_scores[2] = {0, 0};
    int currentPlayer = 0;
    

    Carcassonne();
    int currentTileType() const;
    float terminalValue() const;
    Carcassonne clone() const;

    void getAvailableDraws(ChanceBranch *out, int &count) const;
    void drawTile(int type_id);
    void getLegalTileMoves(TileMove *out, int &count) const;
    void placeTile(int x, int y, int rot);
    FixedVector<int, 6> getLegalMeepleMoves() const;
    void placeMeeple(int pos);
};
