#ifndef OPEN_SPIEL_GAMES_CARCASSONNE_CARCASSONNE_H_
#define OPEN_SPIEL_GAMES_CARCASSONNE_CARCASSONNE_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

#include "game/game.hpp"

namespace open_spiel {
namespace carcassonne {

inline constexpr int kNumPlayers = 2;
inline constexpr int kChanceActionCount = CANONICAL_TILE_TYPE_COUNT;
// Tile actions and the observation's cells are those of the view (VIEW_SIZE
// across), which follows the tiles: (y * VIEW_SIZE + x) * 4 + rot places a tile
// on cell (x, y) of the view. CarcassonneState::TileAction and TileActionMove
// convert to and from board cells.
inline constexpr int kTileActionCount = VIEW_SIZE * VIEW_SIZE * 4;
// One action per meeple position, -1 (skip) to MEEPLE_POS_COUNT - 2: skip,
// the 14 spots with a meeple, the same 14 with the big meeple, the builder on
// sides 0-3, then the pig on the fields of half-edges 0-7.
inline constexpr int kMeepleActionCount = MEEPLE_POS_COUNT;
inline constexpr int kMeepleActionOffset = kTileActionCount;
inline constexpr int kNumDistinctPlayerActions = kTileActionCount + kMeepleActionCount;
// The conv policy head of alpha_zero_torch (model.cc) reads at most
// kMaxExtraActions = 64 actions beyond the tile placements; with more it
// silently falls back to the dense head.
static_assert(kMeepleActionCount <= 64, "alpha_zero_torch's conv policy head would not fit");

// Observation: spatial planes for what is on the board, then one plane that
// is not spatial. Its first kGlobalFeatures cells hold a vector of board-wide
// quantities (scores, the deck, the tile in hand, the phase); broadcasting
// each of them would fill a whole plane with one repeated value.
inline constexpr int kTerrainTypes = 4; // grass, city, road, river
inline constexpr int kNumSidePairs = 6;
inline constexpr int kLegalPlacementPlanes = 4;

// Tiles on the board.
inline constexpr int kOccupiedPlane = 0;
inline constexpr int kNorthTerrainPlane = 1; // 4 terrains per side, then the next side
inline constexpr int kEastTerrainPlane = kNorthTerrainPlane + kTerrainTypes;
inline constexpr int kSouthTerrainPlane = kEastTerrainPlane + kTerrainTypes;
inline constexpr int kWestTerrainPlane = kSouthTerrainPlane + kTerrainTypes;
// One per side: the city on that side carries a shield on this tile.
inline constexpr int kShieldPlane = kWestTerrainPlane + kTerrainTypes;
inline constexpr int kMonasteryPlane = kShieldPlane + 4;
// Pairs of non-grass sides (river included) a tile joins by itself, in the
// order N-E, N-S, N-W, E-S, E-W, S-W. This tells CGGC tiles with one city from
// those with two.
inline constexpr int kSideLinkPlane = kMonasteryPlane + 1;
// Where the tile in hand can go.
inline constexpr int kFrontierPlane = kSideLinkPlane + kNumSidePairs;
inline constexpr int kLegalPlacementPlane = kFrontierPlane + 1; // one per rotation
inline constexpr int kLastPlacedPlane = kLegalPlacementPlane + kLegalPlacementPlanes;
// The feature each city or road side of a tile belongs to, one plane per side
// for each quantity. Summing them along a feature needs the whole feature in view,
// which the convolutions cannot do, so they are computed here.
inline constexpr int kFeatureOpensPlane = kLastPlacedPlane + 1;         // min(opens, 6) / 6
inline constexpr int kFeatureScorePlane = kFeatureOpensPlane + 4;       // getBaseScore() / 30: no inn or cathedral
inline constexpr int kFeatureMyMeeplesPlane = kFeatureScorePlane + 4;   // strength (big meeple 2) / 7
inline constexpr int kFeatureOpponentMeeplesPlane = kFeatureMyMeeplesPlane + 4;
inline constexpr int kFeatureSignedScorePlane = kFeatureOpponentMeeplesPlane + 4; // +-getScore() / 30
// Monasteries: tiles around it / 9, and +1 mine, -1 the opponent's.
inline constexpr int kMonasteryCoveragePlane = kFeatureSignedScorePlane + 4;
inline constexpr int kMonasteryOwnerPlane = kMonasteryCoveragePlane + 1;
// The field each half-edge of a tile belongs to (see FieldLayout), one plane
// per half-edge for each quantity; 0 on city sides.
inline constexpr int kFieldMyFarmersPlane = kMonasteryOwnerPlane + 1;                 // strength / 7
inline constexpr int kFieldOpponentFarmersPlane = kFieldMyFarmersPlane + HALF_EDGE_COUNT;
inline constexpr int kFieldScorePlane = kFieldOpponentFarmersPlane + HALF_EDGE_COUNT; // 3 * completed cities / 30
inline constexpr int kFieldSizePlane = kFieldScorePlane + HALF_EDGE_COUNT;            // tiles / 60
inline constexpr int kFieldOpenCitiesPlane = kFieldSizePlane + HALF_EDGE_COUNT;       // open cities next to it / 10
// The same for a tile's inner field, which touches no half-edge.
inline constexpr int kInnerFieldMyFarmersPlane = kFieldOpenCitiesPlane + HALF_EDGE_COUNT;
inline constexpr int kInnerFieldOpponentFarmersPlane = kInnerFieldMyFarmersPlane + 1;
inline constexpr int kInnerFieldScorePlane = kInnerFieldOpponentFarmersPlane + 1;
inline constexpr int kInnerFieldSizePlane = kInnerFieldScorePlane + 1;
inline constexpr int kInnerFieldOpenCitiesPlane = kInnerFieldSizePlane + 1;
// The big meeple (Inns & Cathedrals), which counts 2 in the strength planes
// above but comes back as one piece: 1 where the feature on that side holds my
// / the opponent's big meeple, one plane per side.
inline constexpr int kFeatureMyBigMeeplePlane = kInnerFieldOpenCitiesPlane + 1;
inline constexpr int kFeatureOpponentBigMeeplePlane = kFeatureMyBigMeeplePlane + 4;
// +1 where the meeple on a monastery is my big meeple, -1 the opponent's.
inline constexpr int kMonasteryBigMeeplePlane = kFeatureOpponentBigMeeplePlane + 4;
// 1 where the feature on that side has an inn (a road) or a cathedral (a city),
// with the Inns & Cathedrals rules: closed it scores 2x / 3x, left open 0. The
// score plane leaves them out, the signed score plane counts them. One plane
// per side.
inline constexpr int kFeatureInnCathedralPlane = kMonasteryBigMeeplePlane + 1;
// The builder (Traders & Builders), which is no follower: 1 where the feature
// on that side holds my / the opponent's builder, one plane per side.
inline constexpr int kFeatureMyBuilderPlane = kFeatureInnCathedralPlane + 4;
inline constexpr int kFeatureOpponentBuilderPlane = kFeatureMyBuilderPlane + 4;
// The pig (Traders & Builders), which is no farmer: 1 where the field of that
// half-edge holds my / the opponent's pig, one plane per half-edge. Never on an
// inner field.
inline constexpr int kFieldMyPigPlane = kFeatureOpponentBuilderPlane + 4;
inline constexpr int kFieldOpponentPigPlane = kFieldMyPigPlane + HALF_EDGE_COUNT;
// Goods (Traders & Builders) in the city on that side, not yet handed out:
// wine, wheat, cloth (GOODS_MARKS_BY_KIND), a block of four sides each, / the
// goods of that kind in the table (9, 6, 5). Whoever places the tile that
// completes the city takes them.
inline constexpr int kFeatureGoodsPlane = kFieldOpponentPigPlane + HALF_EDGE_COUNT;
inline constexpr int kSpatialPlanes = kFeatureGoodsPlane + 4 * GOODS_KINDS;
inline constexpr int kGlobalFeaturePlane = kSpatialPlanes;
inline constexpr int kObservationPlanes = kGlobalFeaturePlane + 1;
static_assert(kLastPlacedPlane == 33);
static_assert(kSpatialPlanes == 150);

// Offsets in the global vector, all from the observing player's side.
inline constexpr int kGlobalMyScore = 0;           // clip(/100)
inline constexpr int kGlobalOpponentScore = 1;
inline constexpr int kGlobalScoreDiff = 2;         // clip(diff / 70)
inline constexpr int kGlobalMyPending = 3;         // clip(/70), see getPendingScore(); goods' points included
inline constexpr int kGlobalOpponentPending = 4;
inline constexpr int kGlobalStaticDiff = 5;        // banked + pending diff: clip(/3), clip(/10), clip(/60)
inline constexpr int kStaticDiffScales = 3;
inline constexpr int kGlobalMyMeeples = kGlobalStaticDiff + kStaticDiffScales; // / 7
inline constexpr int kGlobalOpponentMeeples = kGlobalMyMeeples + 1;
inline constexpr int kGlobalRemainingTiles = kGlobalOpponentMeeples + 1;      // / deck size (72 in the base game)
inline constexpr int kGlobalCompletedTurns = kGlobalRemainingTiles + 1;       // / (deck size - 1)
// Every type of every expansion has a slot, whether this game deals it or not.
inline constexpr int kGlobalRemainingByType = kGlobalCompletedTurns + 1;      // left / count in the table; 0 if not dealt
inline constexpr int kGlobalTileInHand = kGlobalRemainingByType + CANONICAL_TILE_TYPE_COUNT; // one-hot
inline constexpr int kGlobalTilePhase = kGlobalTileInHand + CANONICAL_TILE_TYPE_COUNT;
inline constexpr int kGlobalMeeplePhase = kGlobalTilePhase + 1;
// Legal meeple moves in action order: skip, sides 0-3, monastery, half-edges
// 0-7, inner field, then those 14 with the big meeple, then the builder on
// sides 0-3, then the pig on half-edges 0-7.
inline constexpr int kGlobalLegalMeeple = kGlobalMeeplePhase + 1;
inline constexpr int kGlobalLegalPlacements = kGlobalLegalMeeple + kMeepleActionCount; // / 100
inline constexpr int kGlobalIsPlayer0 = kGlobalLegalPlacements + 1;
// The part of the pending scores that fields score, clip(/40).
inline constexpr int kGlobalMyFieldPending = kGlobalIsPlayer0 + 1;
inline constexpr int kGlobalOpponentFieldPending = kGlobalMyFieldPending + 1;
// Meeples on the board as farmers, the big meeple not included, / 7. They
// never come back, so with the meeples held they also give how many are out on
// features and will return.
inline constexpr int kGlobalMyFarmers = kGlobalOpponentFieldPending + 1;
inline constexpr int kGlobalOpponentFarmers = kGlobalMyFarmers + 1;
// Each expansion other than the base, in Expansion order (inns_cathedrals,
// traders_builders, river, princess_dragon), as two cells: its tiles are dealt,
// its rules are on. off = (0, 0), tiles = (1, 0), on = (1, 1).
inline constexpr int kGlobalExpansionModes = kGlobalOpponentFarmers + 1;
inline constexpr int kGlobalExpansionCells = 2 * (EXPANSION_COUNT - 1);
// River tiles on the board, the spring included, / 12; 0 without the river.
inline constexpr int kGlobalRiverTiles = kGlobalExpansionModes + kGlobalExpansionCells;
// The big meeple in hand, 1 or 0; 0 without its rules.
inline constexpr int kGlobalMyBigMeeple = kGlobalRiverTiles + 1;
inline constexpr int kGlobalOpponentBigMeeple = kGlobalMyBigMeeple + 1;
// The big meeple is a farmer, 1 or 0: it never comes back. Neither this nor in
// hand, it is on a feature or a monastery, where the big meeple planes show it.
inline constexpr int kGlobalMyBigFarmer = kGlobalOpponentBigMeeple + 1;
inline constexpr int kGlobalOpponentBigFarmer = kGlobalMyBigFarmer + 1;
// The builder in hand, 1 or 0; 0 without its rules. Not in hand, the builder
// planes show where it is.
inline constexpr int kGlobalMyBuilder = kGlobalOpponentBigFarmer + 1;
inline constexpr int kGlobalOpponentBuilder = kGlobalMyBuilder + 1;
// The builder's double turn, for whoever is to play: the tile just placed
// extends their builder, so they place another after it; they are on that
// other tile, after which there is no third.
inline constexpr int kGlobalBuilderExtraTile = kGlobalOpponentBuilder + 1;
inline constexpr int kGlobalBuilderSecondTile = kGlobalBuilderExtraTile + 1;
// The pig in hand, 1 or 0; 0 without its rules. Not in hand, the pig planes
// show its field.
inline constexpr int kGlobalMyPig = kGlobalBuilderSecondTile + 1;
inline constexpr int kGlobalOpponentPig = kGlobalMyPig + 1;
// Goods tokens held, by kind (wine, wheat, cloth), / the goods of that kind in
// the table; 0 without their rules.
inline constexpr int kGlobalMyGoods = kGlobalOpponentPig + 1;
inline constexpr int kGlobalOpponentGoods = kGlobalMyGoods + GOODS_KINDS;
inline constexpr int kGlobalFeatures = kGlobalOpponentGoods + GOODS_KINDS;
static_assert(kGlobalFeatures == 86 + 2 * CANONICAL_TILE_TYPE_COUNT);
static_assert(kGlobalFeatures <= VIEW_SIZE * VIEW_SIZE);
inline constexpr int kObservationTensorSize = kObservationPlanes * VIEW_SIZE * VIEW_SIZE;

class CarcassonneGame;

class CarcassonneState : public State {
  public:
    explicit CarcassonneState(std::shared_ptr<const Game> game);
    // `expansions`: the expansionBit() mask of expansions whose tiles are dealt;
    // `rules`: those of them that play their rules (see ::Carcassonne).
    CarcassonneState(std::shared_ptr<const Game> game, int max_turns, uint32_t expansions = BASE_ONLY,
                     uint32_t rules = RULED_EXPANSIONS);
    CarcassonneState(std::shared_ptr<const Game> game, const ::Carcassonne &game_state);
    CarcassonneState(const CarcassonneState &) = default;

    Player CurrentPlayer() const override;
    std::string ActionToString(Player player, Action action) const override;
    std::string ToString() const override;
    bool IsTerminal() const override;
    std::vector<double> Returns() const override;
    std::string ObservationString(Player player) const override;
    void ObservationTensor(Player player, absl::Span<float> values) const override;
    std::unique_ptr<State> Clone() const override;
    ActionsAndProbs ChanceOutcomes() const override;
    std::vector<Action> LegalActions() const override;

    const ::Carcassonne &UnderlyingState() const { return game_state_; }

    // The tile action of this state that places the tile in hand on board cell
    // (x, y), which must be in the view, turned rot quarter turns; and back.
    // The view follows the tiles, so the same action is another board cell in
    // another state.
    Action TileAction(int x, int y, int rot) const;
    TileMove TileActionMove(Action action) const;

  protected:
    void DoApplyAction(Action action) override;

  private:
    ::Carcassonne game_state_;
};

class CarcassonneGame : public Game {
  public:
    explicit CarcassonneGame(const GameParameters &params);

    int NumDistinctActions() const override { return kNumDistinctPlayerActions; }
    std::unique_ptr<State> NewInitialState() const override {
        return std::unique_ptr<State>(new CarcassonneState(shared_from_this(), max_turns_, expansions_, rules_));
    }
    // Every type of every expansion, so the shapes do not depend on the options.
    int MaxChanceOutcomes() const override { return kChanceActionCount; }
    int NumPlayers() const override { return kNumPlayers; }
    double MinUtility() const override { return -1; }
    absl::optional<double> UtilitySum() const override { return 0; }
    double MaxUtility() const override { return 1; }
    std::vector<int> ObservationTensorShape() const override { return {kObservationPlanes, VIEW_SIZE, VIEW_SIZE}; }
    int MaxGameLength() const override {
        const int deck_size = deckSizeOf(expansions_);
        return max_turns_ > 0 ? (deck_size - 1) + 2 * max_turns_ : (deck_size - 1) * 3;
    }
    int MaxChanceNodesInHistory() const override { return deckSizeOf(expansions_) - 1; }

    // The expansionBit() mask of the expansions whose tiles this game deals,
    // the base game included.
    uint32_t Expansions() const { return expansions_; }
    // Those of them dealt "on": their rules are played too.
    uint32_t Rules() const { return rules_; }

  private:
    int max_turns_ = 0;
    uint32_t expansions_ = BASE_ONLY;
    uint32_t rules_ = 0;
};

// Board rotation, for training-data augmentation. Rules, deck and the square
// board are symmetric under turning the whole position by k quarter turns
// clockwise about the centre cell, and the view turns with it: the rotated
// position has the same value, its observation is a fixed rearrangement of
// planes and cells (about the view's centre cell), and every legal action maps
// to exactly one legal action.
inline constexpr int kNumBoardRotations = 4;

// Which sides of the just-placed tile belong to the same feature (see
// Carcassonne::getLastTileSideGroups), then which half-edges belong to the
// same field (getLastTileFieldGroups); all -1 outside the meeple phase. Meeple
// actions name a feature by its lowest side and a field by its lowest
// half-edge, which a rotation can change, and the observation alone does not
// say which sides share a feature.
inline constexpr int kFieldGroupOffset = 4;
using SideGroups = std::array<int8_t, kFieldGroupOffset + HALF_EDGE_COUNT>;
inline constexpr SideGroups kNoSideGroups = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

SideGroups GetSideGroups(const CarcassonneState &state);

// A player action legal in a position -> the same move in that position
// rotated by k quarter turns. `groups` must come from the original position.
Action RotateAction(Action action, int k, const SideGroups &groups);

// SideGroups of a position -> SideGroups of that position rotated by k.
SideGroups RotateSideGroups(const SideGroups &groups, int k);

// Observation of a position -> observation of that position rotated by k
// quarter turns. `groups` must come from the original position.
void RotateObservation(absl::Span<const float> observation, int k, const SideGroups &groups,
                       absl::Span<float> rotated);

// Every value of observation plane `plane` is n / ObservationPlaneDenominator(plane)
// for an integer n in [-128, 127] (the denominator is 1 for the 0/1 planes), or
// 0 for the global vector, whose values are not. Lets a replay buffer keep the
// planes as int8 without losing anything (see alpha_zero_torch's
// observation_codec.h).
float ObservationPlaneDenominator(int plane);

} // namespace carcassonne
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_CARCASSONNE_CARCASSONNE_H_
