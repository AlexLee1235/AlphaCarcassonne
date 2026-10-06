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
inline constexpr int kTileActionCount = BOARD_SIZE * BOARD_SIZE * 4;
// One action per meeple position, -1 (skip) to MEEPLE_POS_COUNT - 2.
inline constexpr int kMeepleActionCount = MEEPLE_POS_COUNT;
inline constexpr int kMeepleActionOffset = kTileActionCount;
inline constexpr int kNumDistinctPlayerActions = kTileActionCount + kMeepleActionCount;
// The conv policy head of alpha_zero_torch (model.cc) reads at most
// kMaxExtraActions = 16 actions beyond the tile placements; with more it
// silently falls back to the dense head.
static_assert(kMeepleActionCount <= 16, "alpha_zero_torch's conv policy head would not fit");

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
inline constexpr int kFeatureScorePlane = kFeatureOpensPlane + 4;       // getScore() / 30
inline constexpr int kFeatureMyMeeplesPlane = kFeatureScorePlane + 4;   // count / 7
inline constexpr int kFeatureOpponentMeeplesPlane = kFeatureMyMeeplesPlane + 4;
inline constexpr int kFeatureSignedScorePlane = kFeatureOpponentMeeplesPlane + 4; // +-getScore() / 30
// Monasteries: tiles around it / 9, and +1 mine, -1 the opponent's.
inline constexpr int kMonasteryCoveragePlane = kFeatureSignedScorePlane + 4;
inline constexpr int kMonasteryOwnerPlane = kMonasteryCoveragePlane + 1;
// The field each half-edge of a tile belongs to (see FieldLayout), one plane
// per half-edge for each quantity; 0 on city sides.
inline constexpr int kFieldMyFarmersPlane = kMonasteryOwnerPlane + 1;                 // count / 7
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
inline constexpr int kSpatialPlanes = kInnerFieldOpenCitiesPlane + 1;
inline constexpr int kGlobalFeaturePlane = kSpatialPlanes;
inline constexpr int kObservationPlanes = kGlobalFeaturePlane + 1;
static_assert(kLastPlacedPlane == 33);
static_assert(kSpatialPlanes == 101);

// Offsets in the global vector, all from the observing player's side.
inline constexpr int kGlobalMyScore = 0;           // clip(/100)
inline constexpr int kGlobalOpponentScore = 1;
inline constexpr int kGlobalScoreDiff = 2;         // clip(diff / 70)
inline constexpr int kGlobalMyPending = 3;         // clip(/70), see getPendingScore()
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
// 0-7, inner field.
inline constexpr int kGlobalLegalMeeple = kGlobalMeeplePhase + 1;
inline constexpr int kGlobalLegalPlacements = kGlobalLegalMeeple + kMeepleActionCount; // / 100
inline constexpr int kGlobalIsPlayer0 = kGlobalLegalPlacements + 1;
// The part of the pending scores that fields score, clip(/40).
inline constexpr int kGlobalMyFieldPending = kGlobalIsPlayer0 + 1;
inline constexpr int kGlobalOpponentFieldPending = kGlobalMyFieldPending + 1;
// Farmers on the board, / 7. They never come back, so with the meeples held
// they also give how many are out on features and will return.
inline constexpr int kGlobalMyFarmers = kGlobalOpponentFieldPending + 1;
inline constexpr int kGlobalOpponentFarmers = kGlobalMyFarmers + 1;
inline constexpr int kGlobalFeatures = kGlobalOpponentFarmers + 1;
static_assert(kGlobalFeatures == 35 + 2 * CANONICAL_TILE_TYPE_COUNT);
static_assert(kGlobalFeatures <= BOARD_SIZE * BOARD_SIZE);
inline constexpr int kObservationTensorSize = kObservationPlanes * BOARD_SIZE * BOARD_SIZE;

class CarcassonneGame;

class CarcassonneState : public State {
  public:
    explicit CarcassonneState(std::shared_ptr<const Game> game);
    // `expansions`: the expansionBit() mask of expansions whose tiles are dealt.
    CarcassonneState(std::shared_ptr<const Game> game, int max_turns, uint32_t expansions = BASE_ONLY);
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
        return std::unique_ptr<State>(new CarcassonneState(shared_from_this(), max_turns_, expansions_));
    }
    // Every type of every expansion, so the shapes do not depend on the options.
    int MaxChanceOutcomes() const override { return kChanceActionCount; }
    int NumPlayers() const override { return kNumPlayers; }
    double MinUtility() const override { return -1; }
    absl::optional<double> UtilitySum() const override { return 0; }
    double MaxUtility() const override { return 1; }
    std::vector<int> ObservationTensorShape() const override { return {kObservationPlanes, BOARD_SIZE, BOARD_SIZE}; }
    int MaxGameLength() const override {
        const int deck_size = deckSizeOf(expansions_);
        return max_turns_ > 0 ? (deck_size - 1) + 2 * max_turns_ : (deck_size - 1) * 3;
    }
    int MaxChanceNodesInHistory() const override { return deckSizeOf(expansions_) - 1; }

    // The expansionBit() mask of the expansions whose tiles this game deals,
    // the base game included.
    uint32_t Expansions() const { return expansions_; }

  private:
    int max_turns_ = 0;
    uint32_t expansions_ = BASE_ONLY;
};

// Board rotation, for training-data augmentation. Rules, deck and the square
// board are symmetric under turning the whole position by k quarter turns
// clockwise about the centre cell: the rotated position has the same value, its
// observation is a fixed rearrangement of planes and cells, and every legal
// action maps to exactly one legal action.
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
