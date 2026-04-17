#ifndef OPEN_SPIEL_GAMES_CARCASSONNE_CARCASSONNE_H_
#define OPEN_SPIEL_GAMES_CARCASSONNE_CARCASSONNE_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

#include "carcassonne/game.hpp"

namespace open_spiel {
namespace carcassonne {

inline constexpr int kNumPlayers = 2;
inline constexpr int kChanceActionCount = CANONICAL_TILE_TYPE_COUNT;
inline constexpr int kTileActionCount = BOARD_SIZE * BOARD_SIZE * 4;
inline constexpr int kMeepleActionCount = 6;
inline constexpr int kMeepleActionOffset = kTileActionCount;
inline constexpr int kNumDistinctPlayerActions =
    kTileActionCount + kMeepleActionCount;

inline constexpr int kTerrainTypes = 3;  // grass, city, road
inline constexpr int kBoardFeaturePlanes = 4 * kTerrainTypes + 3;
inline constexpr int kMeepleFeaturePlanes = 10;
inline constexpr int kCurrentTileFeaturePlanes = 4 * kTerrainTypes + 2;
inline constexpr int kGlobalFeaturePlanes = 5;
inline constexpr int kObservationPlanes =
    kBoardFeaturePlanes + kMeepleFeaturePlanes +
    kCurrentTileFeaturePlanes + kGlobalFeaturePlanes;
inline constexpr int kNorthTerrainPlane = 0;
inline constexpr int kEastTerrainPlane = 3;
inline constexpr int kSouthTerrainPlane = 6;
inline constexpr int kWestTerrainPlane = 9;
inline constexpr int kShieldPlane = 12;
inline constexpr int kMonasteryPlane = 13;
inline constexpr int kLastPlacedPlane = 14;
inline constexpr int kMyMeeplePlane = 15;
inline constexpr int kOpponentMeeplePlane = 20;
inline constexpr int kCurrentTileNorthPlane = 25;
inline constexpr int kCurrentTileEastPlane = 28;
inline constexpr int kCurrentTileSouthPlane = 31;
inline constexpr int kCurrentTileWestPlane = 34;
inline constexpr int kCurrentTileShieldPlane = 37;
inline constexpr int kCurrentTileMonasteryPlane = 38;
inline constexpr int kMyHoldingMeeplesPlane = 39;
inline constexpr int kOpponentHoldingMeeplesPlane = 40;
inline constexpr int kRemainingTilesPlane = 41;
inline constexpr int kScoreDiffPlane = 42;
inline constexpr int kIsMeeplePhasePlane = 43;
inline constexpr int kObservationTensorSize =
    kObservationPlanes * BOARD_SIZE * BOARD_SIZE;

class CarcassonneGame;

class CarcassonneState : public State {
 public:
  explicit CarcassonneState(std::shared_ptr<const Game> game);
  CarcassonneState(const CarcassonneState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  ActionsAndProbs ChanceOutcomes() const override;
  std::vector<Action> LegalActions() const override;

  const ::Carcassonne& UnderlyingState() const { return game_state_; }

 protected:
  void DoApplyAction(Action action) override;

 private:
  ::Carcassonne game_state_;
};

class CarcassonneGame : public Game {
 public:
  explicit CarcassonneGame(const GameParameters& params);

  int NumDistinctActions() const override {
    return kNumDistinctPlayerActions;
  }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new CarcassonneState(shared_from_this()));
  }
  int MaxChanceOutcomes() const override { return kChanceActionCount; }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kObservationPlanes, BOARD_SIZE, BOARD_SIZE};
  }
  int MaxGameLength() const override {
    return (PHYSICAL_TILE_COUNT - 1) * 2;
  }
  int MaxChanceNodesInHistory() const override {
    return PHYSICAL_TILE_COUNT - 1;
  }
};

}  // namespace carcassonne
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CARCASSONNE_CARCASSONNE_H_
