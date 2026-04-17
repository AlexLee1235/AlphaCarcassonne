#include "open_spiel/games/carcassonne/carcassonne.h"

#include <cmath>
#include <memory>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace carcassonne {
namespace {

namespace testing = open_spiel::testing;

float PlaneValue(const std::vector<float>& tensor, int plane, int x, int y) {
  return tensor[(plane * BOARD_SIZE + y) * BOARD_SIZE + x];
}

int TestTerrainIndex(EdgeType edge_type) {
  switch (edge_type) {
    case GRASS:
      return 0;
    case CITY:
      return 1;
    case ROAD:
      return 2;
    case NONE:
      break;
  }
  SpielFatalError("Unexpected edge type in test.");
}

void CheckBroadcastPlane(const std::vector<float>& tensor, int plane,
                         float expected) {
  for (int y = 0; y < BOARD_SIZE; ++y) {
    for (int x = 0; x < BOARD_SIZE; ++x) {
      SPIEL_CHECK_TRUE(
          std::abs(PlaneValue(tensor, plane, x, y) - expected) < 1e-6f);
    }
  }
}

void AdvanceUntilMeeplePlaced(std::unique_ptr<State>* state) {
  constexpr Action kSkipMeepleAction = kMeepleActionOffset;
  for (int step = 0; step < 20; ++step) {
    while ((*state)->IsChanceNode()) {
      (*state)->ApplyAction((*state)->LegalActions()[0]);
    }
    (*state)->ApplyAction((*state)->LegalActions()[0]);
    const std::vector<Action> meeple_actions = (*state)->LegalActions();
    for (Action action : meeple_actions) {
      if (action != kSkipMeepleAction) {
        (*state)->ApplyAction(action);
        return;
      }
    }
    (*state)->ApplyAction(kSkipMeepleAction);
  }
  SpielFatalError("Failed to reach a state with a placed meeple.");
}

void ObservationTensorSmokeTest() {
  std::shared_ptr<const Game> game = LoadGame("carcassonne");
  std::unique_ptr<State> state = game->NewInitialState();
  const std::vector<int> shape = game->ObservationTensorShape();

  SPIEL_CHECK_EQ(shape.size(), 3);
  SPIEL_CHECK_EQ(shape[0], kObservationPlanes);
  SPIEL_CHECK_EQ(shape[1], BOARD_SIZE);
  SPIEL_CHECK_EQ(shape[2], BOARD_SIZE);

  SPIEL_CHECK_EQ(state->ObservationTensor(0).size(), kObservationTensorSize);
  SPIEL_CHECK_EQ(state->ObservationTensor(1).size(), kObservationTensorSize);

  const int center = BOARD_SIZE / 2;
  std::vector<float> initial_obs = state->ObservationTensor(0);
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kNorthTerrainPlane + 1, center, center),
                 1.0f);
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kEastTerrainPlane + 2, center, center),
                 1.0f);
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kSouthTerrainPlane + 0, center, center),
                 1.0f);
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kWestTerrainPlane + 2, center, center),
                 1.0f);
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kLastPlacedPlane, center, center), 1.0f);
  CheckBroadcastPlane(initial_obs, kMyHoldingMeeplesPlane, 1.0f);
  CheckBroadcastPlane(initial_obs, kOpponentHoldingMeeplesPlane, 1.0f);
  CheckBroadcastPlane(initial_obs, kRemainingTilesPlane, 71.0f / 72.0f);
  CheckBroadcastPlane(initial_obs, kScoreDiffPlane, 0.0f);
  CheckBroadcastPlane(initial_obs, kIsMeeplePhasePlane, 0.0f);

  while (state->IsChanceNode()) {
    state->ApplyAction(state->LegalActions()[0]);
  }
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_EQ(state->ObservationTensor(0).size(), kObservationTensorSize);
  auto* chance_done = dynamic_cast<CarcassonneState*>(state.get());
  SPIEL_CHECK_TRUE(chance_done != nullptr);
  const Tile& hand_tile =
      full_deck[chance_done->UnderlyingState().current_tile_in_hand][0];
  std::vector<float> tile_phase_obs = state->ObservationTensor(0);
  SPIEL_CHECK_EQ(PlaneValue(tile_phase_obs,
                            kCurrentTileNorthPlane + TestTerrainIndex(hand_tile.edge[0]),
                            0, 0),
                 1.0f);
  SPIEL_CHECK_EQ(PlaneValue(tile_phase_obs,
                            kCurrentTileEastPlane + TestTerrainIndex(hand_tile.edge[1]),
                            0, 0),
                 1.0f);
  SPIEL_CHECK_EQ(PlaneValue(tile_phase_obs,
                            kCurrentTileSouthPlane + TestTerrainIndex(hand_tile.edge[2]),
                            0, 0),
                 1.0f);
  SPIEL_CHECK_EQ(PlaneValue(tile_phase_obs,
                            kCurrentTileWestPlane + TestTerrainIndex(hand_tile.edge[3]),
                            0, 0),
                 1.0f);
  CheckBroadcastPlane(tile_phase_obs, kIsMeeplePhasePlane, 0.0f);

  state->ApplyAction(state->LegalActions()[0]);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_EQ(state->ObservationTensor(0).size(), kObservationTensorSize);
  std::vector<float> meeple_phase_obs = state->ObservationTensor(0);
  CheckBroadcastPlane(meeple_phase_obs, kCurrentTileNorthPlane, 0.0f);
  CheckBroadcastPlane(meeple_phase_obs, kCurrentTileEastPlane, 0.0f);
  CheckBroadcastPlane(meeple_phase_obs, kCurrentTileSouthPlane, 0.0f);
  CheckBroadcastPlane(meeple_phase_obs, kCurrentTileWestPlane, 0.0f);
  CheckBroadcastPlane(meeple_phase_obs, kCurrentTileShieldPlane, 0.0f);
  CheckBroadcastPlane(meeple_phase_obs, kCurrentTileMonasteryPlane, 0.0f);
  CheckBroadcastPlane(meeple_phase_obs, kIsMeeplePhasePlane, 1.0f);

  state->ApplyAction(state->LegalActions()[0]);
  SPIEL_CHECK_EQ(state->ObservationTensor(0).size(), kObservationTensorSize);
  SPIEL_CHECK_EQ(state->ObservationTensor(1).size(), kObservationTensorSize);
}

void RelativePerspectiveTest() {
  std::shared_ptr<const Game> game = LoadGame("carcassonne");
  std::unique_ptr<State> state = game->NewInitialState();
  AdvanceUntilMeeplePlaced(&state);

  auto* carcassonne_state = dynamic_cast<CarcassonneState*>(state.get());
  SPIEL_CHECK_TRUE(carcassonne_state != nullptr);
  const ::Carcassonne& core = carcassonne_state->UnderlyingState();
  std::vector<float> obs0 = state->ObservationTensor(0);
  std::vector<float> obs1 = state->ObservationTensor(1);

  bool found = false;
  for (int owner = 0; owner < 2 && !found; ++owner) {
    for (int token_id = 0; token_id < 7 && !found; ++token_id) {
      const MeepleTokenState& token = core.meeple_tokens[owner][token_id];
      if (!token.active) {
        continue;
      }
      const int my_plane_for_owner = kMyMeeplePlane + token.pos;
      const int opp_plane_for_owner = kOpponentMeeplePlane + token.pos;

      if (owner == 0) {
        SPIEL_CHECK_EQ(PlaneValue(obs0, my_plane_for_owner, token.x, token.y),
                       1.0f);
        SPIEL_CHECK_EQ(PlaneValue(obs1, opp_plane_for_owner, token.x, token.y),
                       1.0f);
      } else {
        SPIEL_CHECK_EQ(PlaneValue(obs0, opp_plane_for_owner, token.x, token.y),
                       1.0f);
        SPIEL_CHECK_EQ(PlaneValue(obs1, my_plane_for_owner, token.x, token.y),
                       1.0f);
      }
      found = true;
    }
  }
  SPIEL_CHECK_TRUE(found);

  CheckBroadcastPlane(obs0, kMyHoldingMeeplesPlane,
                      core.holding_meeples[0] / 7.0f);
  CheckBroadcastPlane(obs0, kOpponentHoldingMeeplesPlane,
                      core.holding_meeples[1] / 7.0f);
  CheckBroadcastPlane(obs1, kMyHoldingMeeplesPlane,
                      core.holding_meeples[1] / 7.0f);
  CheckBroadcastPlane(obs1, kOpponentHoldingMeeplesPlane,
                      core.holding_meeples[0] / 7.0f);

  const float score_diff =
      std::tanh((core.player_scores[0] - core.player_scores[1]) / 30.0f);
  CheckBroadcastPlane(obs0, kScoreDiffPlane, score_diff);
  CheckBroadcastPlane(obs1, kScoreDiffPlane, -score_diff);
}

void ReturnsMatchScoresTest() {
  absl::BitGen gen;
  std::shared_ptr<const Game> game = LoadGame("carcassonne");

  for (int sim = 0; sim < 5; ++sim) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      Action action = state->IsChanceNode()
                          ? SampleAction(state->ChanceOutcomes(), gen).first
                          : state->LegalActions()[0];
      state->ApplyAction(action);
    }

    auto* carcassonne_state = dynamic_cast<CarcassonneState*>(state.get());
    SPIEL_CHECK_TRUE(carcassonne_state != nullptr);

    const auto returns = state->Returns();
    SPIEL_CHECK_EQ(returns.size(), 2);
    SPIEL_CHECK_EQ(returns[0] + returns[1], 0);

    const ::Carcassonne& core = carcassonne_state->UnderlyingState();
    if (core.player_scores[0] > core.player_scores[1]) {
      SPIEL_CHECK_EQ(returns[0], 1);
      SPIEL_CHECK_EQ(returns[1], -1);
    } else if (core.player_scores[0] < core.player_scores[1]) {
      SPIEL_CHECK_EQ(returns[0], -1);
      SPIEL_CHECK_EQ(returns[1], 1);
    } else {
      SPIEL_CHECK_EQ(returns[0], 0);
      SPIEL_CHECK_EQ(returns[1], 0);
    }

    SPIEL_CHECK_EQ(state->ObservationTensor(0).size(), kObservationTensorSize);
    SPIEL_CHECK_EQ(state->ObservationTensor(1).size(), kObservationTensorSize);
  }
}

void BasicCarcassonneTests() {
  testing::LoadGameTest("carcassonne");
  testing::ChanceOutcomesTest(*LoadGame("carcassonne"));
  testing::RandomSimTest(*LoadGame("carcassonne"), 50);
  ObservationTensorSmokeTest();
  RelativePerspectiveTest();
  ReturnsMatchScoresTest();
}

}  // namespace
}  // namespace carcassonne
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::carcassonne::BasicCarcassonneTests();
}
