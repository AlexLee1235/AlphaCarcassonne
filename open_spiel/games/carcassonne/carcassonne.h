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
inline constexpr int kMeepleActionCount = 6;
inline constexpr int kMeepleActionOffset = kTileActionCount;
inline constexpr int kNumDistinctPlayerActions = kTileActionCount + kMeepleActionCount;

// Observation: spatial planes for what is on the board, then one plane that
// is not spatial. Its first kGlobalFeatures cells hold a vector of board-wide
// quantities (scores, the deck, the tile in hand, the phase); broadcasting
// each of them would fill a whole plane with one repeated value.
inline constexpr int kTerrainTypes = 3; // grass, city, road
inline constexpr int kNumSidePairs = 6;
inline constexpr int kLegalPlacementPlanes = 4;

// Tiles on the board.
inline constexpr int kOccupiedPlane = 0;
inline constexpr int kNorthTerrainPlane = 1; // 3 terrains per side, then the next side
inline constexpr int kEastTerrainPlane = kNorthTerrainPlane + kTerrainTypes;
inline constexpr int kSouthTerrainPlane = kEastTerrainPlane + kTerrainTypes;
inline constexpr int kWestTerrainPlane = kSouthTerrainPlane + kTerrainTypes;
inline constexpr int kShieldPlane = kWestTerrainPlane + kTerrainTypes;
inline constexpr int kMonasteryPlane = kShieldPlane + 1;
// Pairs of non-grass sides a tile joins by itself, in the order N-E, N-S, N-W,
// E-S, E-W, S-W. This tells CGGC tiles with one city from those with two.
inline constexpr int kSideLinkPlane = kMonasteryPlane + 1;
// Where the tile in hand can go.
inline constexpr int kFrontierPlane = kSideLinkPlane + kNumSidePairs;
inline constexpr int kLegalPlacementPlane = kFrontierPlane + 1; // one per rotation
inline constexpr int kLastPlacedPlane = kLegalPlacementPlane + kLegalPlacementPlanes;
// The feature each non-grass side of a tile belongs to, one plane per side for
// each quantity. Summing them along a feature needs the whole feature in view,
// which the convolutions cannot do, so they are computed here.
inline constexpr int kFeatureOpensPlane = kLastPlacedPlane + 1;         // min(opens, 6) / 6
inline constexpr int kFeatureScorePlane = kFeatureOpensPlane + 4;       // getScore() / 12
inline constexpr int kFeatureMyMeeplesPlane = kFeatureScorePlane + 4;   // count / 7
inline constexpr int kFeatureOpponentMeeplesPlane = kFeatureMyMeeplesPlane + 4;
inline constexpr int kFeatureSignedScorePlane = kFeatureOpponentMeeplesPlane + 4; // +-getScore() / 12
// Monasteries: tiles around it / 9, and +1 mine, -1 the opponent's.
inline constexpr int kMonasteryCoveragePlane = kFeatureSignedScorePlane + 4;
inline constexpr int kMonasteryOwnerPlane = kMonasteryCoveragePlane + 1;
inline constexpr int kSpatialPlanes = kMonasteryOwnerPlane + 1;
inline constexpr int kGlobalFeaturePlane = kSpatialPlanes;
inline constexpr int kObservationPlanes = kGlobalFeaturePlane + 1;
static_assert(kLastPlacedPlane == 26);
static_assert(kSpatialPlanes == 49);

// Offsets in the global vector, all from the observing player's side.
inline constexpr int kGlobalMyScore = 0;           // / 40
inline constexpr int kGlobalOpponentScore = 1;
inline constexpr int kGlobalScoreDiff = 2;         // clip(diff / 20)
inline constexpr int kGlobalMyPending = 3;         // / 20, see getPendingScore()
inline constexpr int kGlobalOpponentPending = 4;
inline constexpr int kGlobalStaticDiff = 5;        // banked + pending diff: clip(/3), clip(/10), clip(/30)
inline constexpr int kStaticDiffScales = 3;
inline constexpr int kGlobalMyMeeples = kGlobalStaticDiff + kStaticDiffScales; // / 7
inline constexpr int kGlobalOpponentMeeples = kGlobalMyMeeples + 1;
inline constexpr int kGlobalRemainingTiles = kGlobalOpponentMeeples + 1;      // / 72
inline constexpr int kGlobalCompletedTurns = kGlobalRemainingTiles + 1;       // / 36
inline constexpr int kGlobalRemainingByType = kGlobalCompletedTurns + 1;      // left / initial count
inline constexpr int kGlobalTileInHand = kGlobalRemainingByType + CANONICAL_TILE_TYPE_COUNT; // one-hot
inline constexpr int kGlobalTilePhase = kGlobalTileInHand + CANONICAL_TILE_TYPE_COUNT;
inline constexpr int kGlobalMeeplePhase = kGlobalTilePhase + 1;
// Legal meeple moves in action order: skip, sides 0-3, monastery.
inline constexpr int kGlobalLegalMeeple = kGlobalMeeplePhase + 1;
inline constexpr int kGlobalLegalPlacements = kGlobalLegalMeeple + kMeepleActionCount; // / 100
inline constexpr int kGlobalIsPlayer0 = kGlobalLegalPlacements + 1;
inline constexpr int kGlobalFeatures = kGlobalIsPlayer0 + 1;
static_assert(kGlobalFeatures == 70);
static_assert(kGlobalFeatures <= BOARD_SIZE * BOARD_SIZE);
inline constexpr int kObservationTensorSize = kObservationPlanes * BOARD_SIZE * BOARD_SIZE;

class CarcassonneGame;

class CarcassonneState : public State {
  public:
    explicit CarcassonneState(std::shared_ptr<const Game> game);
    CarcassonneState(std::shared_ptr<const Game> game, int max_turns);
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
        return std::unique_ptr<State>(new CarcassonneState(shared_from_this(), max_turns_));
    }
    int MaxChanceOutcomes() const override { return kChanceActionCount; }
    int NumPlayers() const override { return kNumPlayers; }
    double MinUtility() const override { return -1; }
    absl::optional<double> UtilitySum() const override { return 0; }
    double MaxUtility() const override { return 1; }
    std::vector<int> ObservationTensorShape() const override { return {kObservationPlanes, BOARD_SIZE, BOARD_SIZE}; }
    int MaxGameLength() const override {
        return max_turns_ > 0 ? (PHYSICAL_TILE_COUNT - 1) + 2 * max_turns_
                              : (PHYSICAL_TILE_COUNT - 1) * 3;
    }
    int MaxChanceNodesInHistory() const override { return PHYSICAL_TILE_COUNT - 1; }

  private:
    int max_turns_ = 0;
};

// Board rotation, for training-data augmentation. Rules, deck and the square
// board are symmetric under turning the whole position by k quarter turns
// clockwise about the centre cell: the rotated position has the same value, its
// observation is a fixed rearrangement of planes and cells, and every legal
// action maps to exactly one legal action.
inline constexpr int kNumBoardRotations = 4;

// Which sides of the just-placed tile belong to the same feature (see
// Carcassonne::getLastTileSideGroups); all -1 outside the meeple phase. Meeple
// actions 0-3 name a feature by its lowest side, which a rotation can change,
// and the observation alone does not say which sides share a feature.
using SideGroups = std::array<int8_t, 4>;
inline constexpr SideGroups kNoSideGroups = {-1, -1, -1, -1};

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

} // namespace carcassonne
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_CARCASSONNE_CARCASSONNE_H_
