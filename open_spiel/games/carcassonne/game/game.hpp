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

// Tiles go only inside the view: VIEW_SIZE x VIEW_SIZE cells centred on the
// tiles placed so far (Carcassonne::view_x0, view_y0), which is what the
// players see and name their moves by. Following the tiles, 21 holds 99.96% of
// random base games and 98.5% with the river (greedy games: 100%, 99.7%); kept
// on the start tile it held 95.5% and 28% (tools/diag_board_free). The board
// only holds wherever the view can wander: 20 cells each way from the start
// tile on its centre cell.
constexpr int VIEW_SIZE = 21;
constexpr int BOARD_SIZE = 41;
static_assert(BOARD_SIZE % 2 == 1, "the start tile sits on the centre cell");
static_assert(VIEW_SIZE % 2 == 1 && VIEW_SIZE <= BOARD_SIZE, "the view has a centre cell and fits on the board");
constexpr int TOTAL_TILE_COUNT = PHYSICAL_TILE_COUNT;
constexpr int MAX_FRONTIER_CELLS = TOTAL_TILE_COUNT * 2 + 2;
constexpr int EDGE_SLOT_COUNT = TOTAL_TILE_COUNT * 4;
constexpr int FIELD_SLOT_COUNT = TOTAL_TILE_COUNT * MAX_TILE_FIELDS;
// A field scores this for every completed city it borders.
constexpr int FIELD_POINTS_PER_CITY = 3;
// ... and this for a majority holder whose pig is on it (Traders & Builders).
constexpr int PIG_FIELD_POINTS_PER_CITY = 4;
// Each player has 7 meeples, and with the Inns & Cathedrals rules a big meeple.
constexpr int MEEPLES_PER_PLAYER = 7;
// At most every meeple of both players stands as a farmer at once.
constexpr int MAX_FARMERS = 2 * (MEEPLES_PER_PLAYER + 1);

// Where a meeple goes on the tile just placed. A feature is named by its
// lowest side on that tile and a field by its lowest half-edge, so each move
// has one name.
constexpr int MEEPLE_POS_SKIP = -1;
constexpr int MEEPLE_POS_MONASTERY = 4;                                    // 0..3: the feature on that side
constexpr int MEEPLE_POS_FIELD = 5;                                        // 5..12: a farmer, by half-edge
constexpr int MEEPLE_POS_INNER_FIELD = MEEPLE_POS_FIELD + HALF_EDGE_COUNT; // 13: on the tile's inner field
// 14..27: the big meeple on spot pos - 14, one of 0..13 above.
constexpr int MEEPLE_POS_BIG = MEEPLE_POS_INNER_FIELD + 1;
// 28..31: the builder on the feature on side pos - 28 (roads and cities only).
constexpr int MEEPLE_POS_BUILDER = MEEPLE_POS_BIG + MEEPLE_POS_BIG;
// 32..39: the pig on the field of half-edge pos - 32. Never on an inner field:
// that field is new with its tile, so it cannot hold a farmer of the player yet.
constexpr int MEEPLE_POS_PIG = MEEPLE_POS_BUILDER + 4;
constexpr int MEEPLE_POS_COUNT = MEEPLE_POS_PIG + HALF_EDGE_COUNT - MEEPLE_POS_SKIP; // positions -1 .. 39
using MeepleMoves = FixedVector<int, MEEPLE_POS_COUNT>;

constexpr bool isBigMeeplePos(int pos) { return pos >= MEEPLE_POS_BIG && pos < MEEPLE_POS_BUILDER; }
constexpr bool isBuilderPos(int pos) { return pos >= MEEPLE_POS_BUILDER && pos < MEEPLE_POS_PIG; }
constexpr bool isPigPos(int pos) { return pos >= MEEPLE_POS_PIG; }
// Where a meeple move puts its piece, whichever piece: -1 .. 13 (a pig's is
// the farmer spot of its field).
constexpr int meepleSpot(int pos) {
    return isPigPos(pos)         ? MEEPLE_POS_FIELD + pos - MEEPLE_POS_PIG
           : isBuilderPos(pos)   ? pos - MEEPLE_POS_BUILDER
           : isBigMeeplePos(pos) ? pos - MEEPLE_POS_BIG
                                 : pos;
}

// PHASE_DRAGON: the dragon moves, one step a decision, after the meeple phase
// of a dragon tile and before that turn is scored. PHASE_SPOT: the second step
// of a choice the meeple phase made by cell (Carcassonne::chooseCell), a spot
// on that cell.
enum GamePhase {
    PHASE_CHANCE = 0,
    PHASE_TILE = 1,
    PHASE_MEEPLE = 2,
    PHASE_TERMINAL = 3,
    PHASE_DRAGON = 4,
    PHASE_SPOT = 5
};

// The choices the meeple phase makes by cell (The Princess & the Dragon):
// through a magic portal, a meeple onto a tile placed before; with the
// princess, a knight off the city she continues.
enum SpotChoice : uint8_t { SPOT_PORTAL = 0, SPOT_PRINCESS = 1 };

// The dragon moves up to this many tiles each time a dragon tile is placed.
constexpr int DRAGON_STEPS = 6;

// A piece standing on the board. Its spot is meepleSpot() of the move that put
// it there: a side 0..3 for a road or city (the builder's too), 4 the
// monastery, MEEPLE_POS_FIELD + half-edge a field (the pig's too), or the inner
// field.
//
// Its spots are where it stands on its tile, one bit per spot 0..13: every
// side (or half-edge) of the tile that its road, city or field had there when
// it was placed, or the monastery, or the inner field. The spot is the lowest
// of them. Taken when placed and never changed: a road or city that later
// joins another part of the tile does not move the piece. So two pieces on one
// tile never share a spot (the later one went on a feature with no piece), and
// the spots turn with the board, as the set is no choice of one side.
enum PieceKind : uint8_t { PIECE_MEEPLE = 0, PIECE_BIG_MEEPLE = 1, PIECE_BUILDER = 2, PIECE_PIG = 3 };
struct Piece {
    int8_t x = -1;
    int8_t y = -1;
    uint8_t tile_id = 0;
    int8_t spot = -1;
    uint8_t owner = 0;
    PieceKind kind = PIECE_MEEPLE;
    uint16_t spots = 0;
};
// Board cells, such as the targets of a choice by cell.
using Cells = FixedVector<std::pair<int8_t, int8_t>, TOTAL_TILE_COUNT>;
// Every meeple, big meeple, builder and pig of both players.
constexpr int MAX_PIECES = 2 * (MEEPLES_PER_PLAYER + 3);

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
    bool big = false; // the owner's meeple here is the big one
};

class Feature {
  public:
    EdgeType type = NONE;
    TileMask tile_mask;
    uint8_t opens = 0;
    // Each player's strength for the majority: a meeple counts 1, the big
    // meeple 2.
    uint8_t meeple_count[2] = {};
    // 1 if that player's big meeple is on the feature (it is in meeple_count too).
    uint8_t big_meeples[2] = {};
    // 1 if that player's builder is on the feature. Not a follower: not in
    // meeple_count, no part in the majority.
    uint8_t builders[2] = {};
    // Shields on the city, one per city piece that carries MARK_SHIELD.
    uint8_t shields = 0;
    // With the Inns & Cathedrals rules: inns on the road (MARK_INN) and
    // cathedrals in the city, one per piece. Left at 0 without the rules.
    uint8_t inns = 0;
    uint8_t cathedrals = 0;
    // With the Traders & Builders rules: the goods symbols in the city, by
    // kind (GOODS_MARKS_BY_KIND), until its completion hands them out.
    uint8_t goods[GOODS_KINDS] = {};

    Feature() = default;
    Feature(EdgeType type, int id);

    Feature operator+(const Feature &other) const;

    bool hasMeeples() const;
    int getTileCount() const;
    // What the feature scores now: once closed, its completion score; while
    // open, what the end of the game would give it, 0 for a road with an inn
    // or a city with a cathedral.
    int getScore() const;
    // The same without inns and cathedrals: a tile a point, a shield one more,
    // a closed city doubled.
    int getBaseScore() const;
};

class BoardModule {
  public:
    Placement board[BOARD_SIZE][BOARD_SIZE] = {};
    EdgeType edge[BOARD_SIZE][BOARD_SIZE][4] = {};
    int count3x3(int x, int y) const;
    bool canPlaceTileAt(int x, int y, const Tile &tile) const;
    void placeTileOnBoard(int tile_id, int x, int y, int rot, const Tile &tile);
};

// How many times placeTileOnBoard found a joined feature with fewer than two
// open edges and had to close it instead (see FeatureModule.cpp). Zero in every
// test so far; a long self-play run hit it.
long long OpensUnderflowCount();

class FeatureModule {
    void settleCompletedFeatures(int tile_id, int side, int *player_scores, int *holding_meeples,
                                 int *holding_big_meeples, int *holding_builders);

  public:
    DisjointSet<Feature, std::plus<Feature>, EDGE_SLOT_COUNT> featureMap;
    // Whether placed tiles bring their inns and cathedrals (the Inns &
    // Cathedrals rules), and their goods (Traders & Builders). Set before the
    // first tile.
    bool inns_cathedrals = false;
    bool goods_rules = false;
    FeatureModule();
    int edgeIndex(int tile_id, int side) const;
    void resolveEndGameScore(int *player_scores);
    void placeTileOnBoard(int tile_id, int x, int y, int rot, const Tile &tile, const BoardModule &board);
    // The sides (0..3) a meeple can go on: one per feature with no meeples.
    void getLegalMeepleMoves(MeepleMoves &ret, int x, int y, const BoardModule &board, const Tile &tile) const;
    void placeMeeple(int x, int y, int side, int player, bool big, const BoardModule &board);
    // The sides (0..3) `player`'s builder can go on: one per feature that
    // holds one of their followers, as MEEPLE_POS_BUILDER + side.
    void getLegalBuilderMoves(MeepleMoves &ret, int x, int y, const BoardModule &board, const Tile &tile,
                              int player) const;
    void placeBuilder(int x, int y, int side, int player, const BoardModule &board);
    // Whether a city or road of tile `tile_id` belongs to a feature that holds
    // `player`'s builder.
    bool hasBuilderOf(int tile_id, const Tile &tile, int player) const;
    // Hands the goods of every city of tile `tile_id` that is complete to
    // `tokens` (by kind) and clears them, so each city gives them once.
    void collectGoods(int tile_id, const Tile &tile, int *tokens);
    // Scores the features of the tile at (x, y) that are complete and gives
    // back their meeples and builders, each to its own supply.
    void settleAfterPlaceMeeple(int x, int y, const BoardModule &board, int *player_scores, int *holding_meeples,
                                int *holding_big_meeples, int *holding_builders);
    // Adds each feature that holds meeples to its majority holders, as the
    // end-game scoring and turn-end settlement would.
    void accumulatePendingScore(int *pending) const;
};

class Field {
  public:
    // For each piece of city the field borders, one of its edge slots
    // (FeatureModule::edgeIndex); featureMap finds the whole city from it.
    std::bitset<EDGE_SLOT_COUNT> city_edges;
    TileMask tile_mask;
    // Each player's strength for the majority, as Feature::meeple_count.
    uint8_t farmer_count[2] = {};
    // 1 if that player's pig is on the field. Not a farmer: not in
    // farmer_count, no part in the majority.
    uint8_t pigs[2] = {};

    Field operator+(const Field &other) const;

    bool hasFarmers() const;
    int getTileCount() const;
};

// The cities a field borders, each counted once.
struct CityCounts {
    int completed = 0;
    int open = 0;
};

// Farmers stay on their field for the whole game and score only at the end.
class FieldModule {
  public:
    // Slot (tile_id - 1) * MAX_TILE_FIELDS + local field of that tile.
    DisjointSet<Field, std::plus<Field>, FIELD_SLOT_COUNT> fieldMap;
    // The slot of every farmer on the board, so scoring visits only those fields.
    FixedVector<int16_t, MAX_FARMERS> farmed_slots;
    // Farmers each player has on the board, the big meeple counted as one; they
    // never come back, unless the dragon eats them.
    uint8_t farmers_placed[2] = {};
    // 1 once that player's big meeple is a farmer.
    uint8_t big_farmers[2] = {};
    FieldModule();
    int fieldIndex(int tile_id, int local) const;
    void placeTileOnBoard(int tile_id, int x, int y, const Tile &tile, const BoardModule &board,
                          const FeatureModule &features);
    void getLegalFarmerMoves(MeepleMoves &ret, int tile_id, const Tile &tile) const;
    void placeFarmer(int tile_id, const Tile &tile, int pos, int player, bool big = false);
    // Undoes placeFarmer: the dragon took that farmer home.
    void removeFarmer(int tile_id, const Tile &tile, int pos, int player, bool big);
    // The fields `player`'s pig can go on: one per field of a half-edge of the
    // tile that holds one of their farmers, as MEEPLE_POS_PIG + half-edge.
    void getLegalPigMoves(MeepleMoves &ret, int tile_id, const Tile &tile, int player) const;
    // On the field of half-edge `half_edge`, which already holds a farmer of
    // `player`, so its slot is in farmed_slots. Pigs stay till the end.
    void placePig(int tile_id, const Tile &tile, int half_edge, int player);
    // For each half-edge of the tile, the lowest half-edge of the tile in the
    // same field (-1 on city sides).
    void getHalfEdgeGroups(int tile_id, const Tile &tile, int8_t groups[HALF_EDGE_COUNT]) const;
    CityCounts adjacentCities(const Field &field, const FeatureModule &features) const;
    // Adds each field that holds farmers to its majority holders, as the
    // end-game scoring would: 3 a completed city it borders, 4 with the
    // holder's pig on it.
    void accumulateScore(int *scores, const FeatureModule &features) const;
};

class MonasteryModule {
  public:
    // One per monastery tile in the table: enough for any deck.
    FixedVector<MonasteryTracker, MONASTERY_TILE_COUNT> active_monasteries;
    void placeTileOnBoard(int tile_id, int x, int y, int rot);
    void resolveEndGameScore(int *player_scores);
    void placeMeeple(int x, int y, int player, bool big, const BoardModule &board);
    void settleCompletedMonasteries(int *player_scores, int *holding_meeples, int *holding_big_meeples);
    void accumulatePendingScore(int *pending) const;
    // Owner of the claimed monastery at (x, y), or -1.
    int ownerAt(int x, int y) const;
    // The same, but only if the owner's meeple there is the big one.
    int bigMeepleOwnerAt(int x, int y) const;
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
    // The size of this game's deck, the start tile included.
    int initial_total = 0;
    int type_counts[CANONICAL_TILE_TYPE_COUNT + 1] = {};
    // River rules: every river tile is drawn before the others, the lake last,
    // and the base start tile is left out (the spring replaces it).
    bool river_first = false;
    // The dragon rules before the first volcano: a dragon tile drawn now is set
    // aside and shuffled back once the dragon is in play, which is the same as
    // not drawing dragon tiles until then.
    bool hold_dragon_tiles = false;
    int consumeType(int type_id);
    // Deals every tile of the expansions in `expansions` (expansionBit() mask);
    // the other types are never drawn.
    void initializeTypeCounts(uint32_t expansions);
    void getAvailableDraws(ChanceBranch *out, int &count) const;
    // The tiles that can still be drawn: all those left but the held dragon
    // tiles. When none can, the game is over, even with dragon tiles set aside.
    int drawableRemaining() const;
};

class LogModule {
  public:
    int tile_x[PHYSICAL_TILE_COUNT + 1], tile_y[PHYSICAL_TILE_COUNT + 1];
    LogModule() {
        fill(tile_x, tile_x + PHYSICAL_TILE_COUNT + 1, -1);
        fill(tile_y, tile_y + PHYSICAL_TILE_COUNT + 1, -1);
    }
    void placeTileOnBoard(int tile_id, int x, int y, int rot) {
        tile_x[tile_id] = x;
        tile_y[tile_id] = y;
    }
    void getMeepleMap(const FeatureModule &features, const MonasteryModule &monasteries, int player, float *span) const {
        int opponent = 1 - player;
        for (int i = 1; i <= PHYSICAL_TILE_COUNT; i++) {
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
    FieldModule fields;
    MonasteryModule monasteries;
    FrontierModule frontier;
    BoardModule board;
    DeckModule deck;
    LogModule logs;

    void placeTileOnBoard(int tile_id, int x, int y, int rot);
    // Adds the tile just placed at (x, y) to the bounding box and centres the
    // view on it.
    void updateView(int x, int y);
    // Follows the river onto a river tile just placed.
    void advanceRiver(int x, int y, const Tile &tile);
    bool isLegalPlacement(int tile_id, int x, int y, int rot) const;
    bool hasValidMove(int tile_id) const;
    // The goods' end-game points as of now: GOODS_POINTS to whoever holds the
    // most tokens of a kind, ties included, if anyone holds one.
    void accumulateGoodsScore(int *scores) const;
    void resolveEndGameScore();
    void resolveNoMoreDraws();
    // The end of a turn, once its pieces are down and the dragon has moved:
    // scores what the last tile completed, then passes the turn on.
    void finishTurn();
    // Drops the records of the pieces that the turn's scoring sent home.
    void forgetSettledPieces();
    // The dragon takes every piece on the tile at (x, y) back to its owner's
    // supply (sendHome).
    void eatPiecesAt(int x, int y);
    // Sends piece `index` home and drops its record; returns the piece.
    Piece removePiece(int index);
    // The same, and with a meeple a builder or pig whose owner has no follower
    // left on its road, city or field.
    void sendHome(int index);
    bool dragonCanEnter(int x, int y) const;
    // Puts down the piece of meeple move `pos` on the tile at (x, y) for the
    // current player.
    void putPiece(int x, int y, int pos);
    // The spots (Piece::spots) of meeple spot `spot` on the tile at (x, y): the
    // sides or half-edges of the tile in that road, city or field now.
    uint16_t spotsOf(int x, int y, int spot) const;
    // After the meeple phase's move, whatever it was: the dragon moves if the
    // tile was a dragon tile, else the turn ends.
    void endPiecePhase();
    // The city of the last tile's princess (princess rules), as a feature root,
    // or -1.
    int princessCityRoot() const;
    // The pieces a choice by cell can pick on (x, y): spots for a portal,
    // indices into `pieces` (knights) for the princess.
    MeepleMoves portalMovesAt(int x, int y) const;
    FixedVector<int, MAX_PIECES> princessKnightsAt(int x, int y) const;

  public:
    int last_x = -1;
    int last_y = -1;
    // The bounding box of the tiles placed so far, and the view: the VIEW_SIZE x
    // VIEW_SIZE cells from (view_x0, view_y0), centred on that box (see
    // updateView). Tiles go only inside the view, so it always holds them all.
    int tiles_x0 = BOARD_SIZE, tiles_x1 = -1, tiles_y0 = BOARD_SIZE, tiles_y1 = -1;
    int view_x0 = 0, view_y0 = 0;
    bool inView(int x, int y) const {
        return x >= view_x0 && x < view_x0 + VIEW_SIZE && y >= view_y0 && y < view_y0 + VIEW_SIZE;
    }
    GamePhase current_phase = PHASE_CHANCE;
    int player_scores[2] = {0, 0};
    int holding_meeples[2] = {MEEPLES_PER_PLAYER, MEEPLES_PER_PLAYER};
    // The big meeple in hand: 1 or 0 with the big meeple rules, else 0.
    int holding_big_meeples[2] = {0, 0};
    // The builder in hand: 1 or 0 with the builder rules, else 0.
    int holding_builders[2] = {0, 0};
    // The pig in hand: 1 or 0 with the pig rules, else 0.
    int holding_pigs[2] = {0, 0};
    int currentPlayer = 0;
    int current_tile_in_hand = 0;
    int completed_turns = 0;
    int max_turns = 0;
    // The expansionBit() mask of the expansions this game deals, the base
    // included.
    uint32_t expansions = BASE_ONLY;
    // Those of them whose rules this game plays (a part of RULED_EXPANSIONS).
    uint32_t rules = 0;

    // Inns & Cathedrals: each player also has a big meeple, placed instead of a
    // meeple. It counts as two for the majority and scores no more; it comes
    // back like any meeple, and as a farmer stays till the end. A road with an
    // inn scores 2 a tile once closed, a city with a cathedral 3 a tile and a
    // shield; left open at the end, either scores nothing (Feature::getScore).
    bool big_meeple_rules = false;

    // Traders & Builders: each player also has a builder, placed instead of a
    // meeple on a road or city of the tile just placed that already holds one
    // of their followers. It is no follower: no part in the majority, and it
    // comes back when that road or city is completed. Whenever the player
    // places a tile that extends it, they place one more tile after it (always;
    // never a third).
    bool builder_rules = false;
    bool builder_extra_tile = false;   // the tile just placed extends the current player's builder
    bool builder_second_tile = false;  // the current player is on the second tile of a double turn
    // Traders & Builders: each player also has a pig, placed instead of a
    // meeple on a field of the tile just placed that already holds one of
    // their farmers. It is no farmer: no part in the majority. With it on the
    // field, a majority holder scores 4 a completed city instead of 3. It
    // stays till the end, as farmers do.
    bool pig_rules = false;
    // Traders & Builders: whoever places the tile that completes a city takes
    // its goods as tokens (FeatureModule::collectGoods), knights or none; at
    // the end each kind's most tokens score GOODS_POINTS.
    bool goods_rules = false;
    int goods_tokens[2][GOODS_KINDS] = {};

    // River rules (on whenever the river tiles are dealt). The river is laid
    // first, from the spring at the centre to the lake, each tile continuing it,
    // and it may not turn the same way twice in a row, straights in between or
    // not: it flows in at most two directions and never back past itself.
    bool river_rules = false;
    int river_x = -1;           // the empty cell the river flows into; -1 once the lake closes it
    int river_y = -1;
    int river_heading = -1;     // the side the river leaves its last tile by: 0 N, 1 E, 2 S, 3 W
    int river_last_turn = 0;    // its last bend, as (out - in heading) % 4: 1 clockwise, 3 anticlockwise; 0 none yet
    int river_tiles_placed = 0; // river tiles on the board, the spring included

    // The Princess & the Dragon: the dragon. The first volcano
    // brings it into play and each volcano after that takes it there; until
    // then no dragon tile is drawn (DeckModule::hold_dragon_tiles). After the
    // meeple phase of a dragon tile it moves up to DRAGON_STEPS tiles, one step
    // a decision, the player who placed the tile first and the two players in
    // turn: to a tile next to it, never back to one it visited this move. Each
    // tile it enters loses all its pieces, which go home; then the turn is
    // scored. No piece goes on a volcano tile.
    bool dragon_rules = false;
    int dragon_x = -1; // -1 until the first volcano
    int dragon_y = -1;
    int dragon_steps = 0;          // taken in the current move
    int dragon_turn_player = -1;   // whose turn the dragon moves in; -1 outside PHASE_DRAGON
    FixedVector<std::pair<int8_t, int8_t>, DRAGON_STEPS + 1> dragon_visited;  // this move, start included

    // The Princess & the Dragon, with the dragon. Magic portal: the turn a
    // portal tile is placed, a meeple or big meeple may go on any tile placed
    // before instead of this one, on a road, city, field or monastery that has
    // no piece and is not complete (the tile just placed may have completed it),
    // never on the dragon's tile. Princess: when a princess tile's city holds
    // knights, the player may send one of them home, either player's, instead
    // of placing a piece. Both are chosen by cell, then if the cell offers
    // more than one choice, by spot (PHASE_SPOT, chooseSpot).
    bool portal_rules = false;
    bool princess_rules = false;
    SpotChoice spot_choice = SPOT_PORTAL;  // in PHASE_SPOT, which choice
    int spot_x = -1;                       // and its cell
    int spot_y = -1;

    // Every piece on the board, where it stands.
    FixedVector<Piece, MAX_PIECES> pieces;

    explicit Carcassonne(int max_turns = 0);
    // Starts with the start tile turned by start_rotation quarter turns: the
    // whole game rotated about the centre. Used to test board-rotation symmetry.
    // The deck holds the base tiles and those of the expansions in
    // `expansions` (an expansionBit() mask). Of those, the ones in `rules` play
    // their rules too, and the river always does; by default every one that has
    // rules.
    Carcassonne(int max_turns, int start_rotation, uint32_t expansions = BASE_ONLY,
                uint32_t rules = RULED_EXPANSIONS);
    int currentTileType() const;
    Carcassonne clone() const;

    Placement getPlacement(int x, int y) const { return board.board[y][x]; }
    int getTotalRemaining() const { return deck.total_remaining; }
    // The size of this game's deck, the start tile included.
    int getDeckSize() const { return deck.initial_total; }
    int getRemainingTypeCount(int type_id) const { return deck.type_counts[type_id]; }
    void WriteMeepleMap(int player, float *span) const;

    // Read-only views for the observation tensor.
    const Feature &featureAt(int tile_id, int side) const {
        return features.featureMap.getSetData(features.edgeIndex(tile_id, side));
    }
    // The field local field `local` of a tile belongs to, named by its slot
    // (the same for every part of one field), and that field.
    int fieldRoot(int tile_id, int local) const { return fields.fieldMap.find(fields.fieldIndex(tile_id, local)); }
    const Field &fieldAtRoot(int root) const { return fields.fieldMap.getSetData(root); }
    CityCounts citiesNextTo(const Field &field) const { return fields.adjacentCities(field, features); }
    // Meeples a player has locked up as farmers for the rest of the game.
    int farmersOnBoard(int player) const { return fields.farmers_placed[player]; }
    bool isFrontier(int x, int y) const { return frontier.frontier[y][x]; }
    int coverage3x3(int x, int y) const { return board.count3x3(x, y); }
    int monasteryOwner(int x, int y) const { return monasteries.ownerAt(x, y); }
    // The owner of the monastery at (x, y) if it holds their big meeple, or -1.
    int monasteryBigMeepleOwner(int x, int y) const { return monasteries.bigMeepleOwnerAt(x, y); }
    // 1 once the player's big meeple is a farmer.
    int bigFarmers(int player) const { return fields.big_farmers[player]; }

    // Points each player still adds if the game ended now: features and
    // monasteries that hold meeples, including features the last tile closed,
    // which are settled when the current turn ends, fields and the goods'
    // end-game points. Zero once terminal.
    void getPendingScore(int pending[2]) const;
    // The same, by settling and end-game scoring a copy. Slow; for tests.
    void getPendingScoreByResolving(int pending[2]) const;
    // The part of getPendingScore() that fields score.
    void getPendingFieldScore(int pending[2]) const;

    void getAvailableDraws(ChanceBranch *out, int &count) const;
    void drawTile(int type_id);
    void getLegalTileMoves(TileMove *out, int &count) const;
    void placeTile(int x, int y, int rot);
    // Skip, then every free spot with a meeple if the player has one, then the
    // same spots with the big meeple (spot + MEEPLE_POS_BIG) if they have it,
    // then the builder's sides (MEEPLE_POS_BUILDER + side) and the pig's
    // half-edges (MEEPLE_POS_PIG + half-edge) if they have them.
    MeepleMoves getLegalMeepleMoves() const;
    // How the spots of the focus cell group, which is what maps the spot moves
    // of the current decision under board rotation: for each side, the lowest
    // side of the tile that names the same choice, -1 where none does; then the
    // same for each half-edge. The meeple phase and a portal's spots name a
    // feature or field by its lowest side or half-edge on the tile; the
    // princess's name a piece by its spot (the lowest of its spots).
    void getFocusSpotGroups(int8_t sides[4], int8_t half_edges[HALF_EDGE_COUNT]) const;
    // Then the dragon moves if the tile was a dragon tile, else the turn ends.
    void placeMeeple(int pos);

    // The cells the meeple phase can choose for a magic portal (the last tile
    // is one; every tile before it with a spot portalMovesAt offers) and for
    // the princess (the last tile is one; every tile with a knight in her
    // city). Empty in other phases.
    Cells getLegalPortalCells() const;
    Cells getLegalPrincessCells() const;
    // Does the choice on that cell, if it offers one move; else waits for it
    // in PHASE_SPOT.
    void chooseCell(SpotChoice choice, int x, int y);
    // PHASE_SPOT: a portal's meeple moves on the cell (a spot, or a spot +
    // MEEPLE_POS_BIG), or the princess's knights there, each by its spot.
    MeepleMoves getLegalSpotMoves() const;
    void chooseSpot(int pos);

    bool dragonInPlay() const { return dragon_x >= 0; }
    // The sides (0 N, 1 E, 2 S, 3 W) the dragon can step to in PHASE_DRAGON.
    FixedVector<int, 4> getLegalDragonMoves() const;
    // One step for the current player; the next step is the other player's.
    void moveDragon(int side);
    // The cell the current decision is about: the dragon's in PHASE_DRAGON,
    // the chosen cell in PHASE_SPOT, else the tile last placed.
    int focusX() const {
        return current_phase == PHASE_DRAGON ? dragon_x : current_phase == PHASE_SPOT ? spot_x : last_x;
    }
    int focusY() const {
        return current_phase == PHASE_DRAGON ? dragon_y : current_phase == PHASE_SPOT ? spot_y : last_y;
    }
};
