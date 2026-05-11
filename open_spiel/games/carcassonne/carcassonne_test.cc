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

void CheckZeroPlanes(const std::vector<float>& tensor, int first_plane,
                     int plane_count) {
  for (int plane = first_plane; plane < first_plane + plane_count; ++plane) {
    CheckBroadcastPlane(tensor, plane, 0.0f);
  }
}

void CheckPlanesEqual(const std::vector<float>& left, int left_plane,
                      const std::vector<float>& right, int right_plane) {
  for (int y = 0; y < BOARD_SIZE; ++y) {
    for (int x = 0; x < BOARD_SIZE; ++x) {
      SPIEL_CHECK_TRUE(std::abs(PlaneValue(left, left_plane, x, y) -
                                PlaneValue(right, right_plane, x, y)) < 1e-6f);
    }
  }
}

void DecodeTileActionForTest(Action action, int* x, int* y, int* rot) {
  *rot = action % 4;
  action /= 4;
  *x = action % BOARD_SIZE;
  *y = action / BOARD_SIZE;
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
  SPIEL_CHECK_EQ(shape[0], 51);
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
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kCityConnectivityPlane, center, center),
                 0.0f);
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kLastPlacedPlane, center, center), 1.0f);
  CheckZeroPlanes(initial_obs, kLegalPlacementPlane, kLegalPlacementPlanes);
  CheckBroadcastPlane(initial_obs, kMyHoldingMeeplesPlane, 1.0f);
  CheckBroadcastPlane(initial_obs, kOpponentHoldingMeeplesPlane, 1.0f);
  CheckBroadcastPlane(initial_obs, kRemainingTilesPlane, 71.0f / 72.0f);
  CheckBroadcastPlane(initial_obs, kScoreDiffPlane, 0.0f);
  CheckBroadcastPlane(initial_obs, kIsMeeplePhasePlane, 0.0f);
  CheckBroadcastPlane(initial_obs, kCurrentPlayerIsPlayer0Plane, 1.0f);

  while (state->IsChanceNode()) {
    state->ApplyAction(state->LegalActions()[0]);
  }
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_EQ(state->ObservationTensor(0).size(), kObservationTensorSize);
  auto* chance_done = dynamic_cast<CarcassonneState*>(state.get());
  SPIEL_CHECK_TRUE(chance_done != nullptr);
  const int hand_tile_id = chance_done->UnderlyingState().current_tile_in_hand;
  const Tile& hand_tile =
      full_deck[hand_tile_id][0];
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
  SPIEL_CHECK_EQ(PlaneValue(tile_phase_obs, kCurrentTileShieldPlane, 0, 0),
                 hand_tile.shield ? 1.0f : 0.0f);
  SPIEL_CHECK_EQ(PlaneValue(tile_phase_obs, kCurrentTileMonasteryPlane, 0, 0),
                 hand_tile.monastery ? 1.0f : 0.0f);
  const bool hand_has_city_connectivity =
      PHYSICAL_TO_CANONICAL_TYPE[hand_tile_id] == 14 ||
      PHYSICAL_TO_CANONICAL_TYPE[hand_tile_id] == 15;
  CheckBroadcastPlane(tile_phase_obs, kCurrentTileCityConnectivityPlane,
                      hand_has_city_connectivity ? 1.0f : 0.0f);
  CheckBroadcastPlane(tile_phase_obs, kCurrentPlayerIsPlayer0Plane, 1.0f);
  for (Action action : state->LegalActions()) {
    int tile_x;
    int tile_y;
    int rot;
    DecodeTileActionForTest(action, &tile_x, &tile_y, &rot);
    SPIEL_CHECK_EQ(PlaneValue(tile_phase_obs, kLegalPlacementPlane + rot, tile_x, tile_y),
                   1.0f);
  }
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
  CheckBroadcastPlane(meeple_phase_obs, kCurrentTileCityConnectivityPlane, 0.0f);
  CheckZeroPlanes(meeple_phase_obs, kLegalPlacementPlane, kLegalPlacementPlanes);
  CheckBroadcastPlane(meeple_phase_obs, kIsMeeplePhasePlane, 1.0f);
  CheckBroadcastPlane(meeple_phase_obs, kCurrentPlayerIsPlayer0Plane, 1.0f);

  state->ApplyAction(state->LegalActions()[0]);
  std::vector<float> player1_chance_obs = state->ObservationTensor(1);
  CheckBroadcastPlane(player1_chance_obs, kCurrentPlayerIsPlayer0Plane, 0.0f);
  while (state->IsChanceNode()) {
    state->ApplyAction(state->LegalActions()[0]);
  }
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  std::vector<float> player1_tile_phase_obs = state->ObservationTensor(1);
  CheckBroadcastPlane(player1_tile_phase_obs, kCurrentPlayerIsPlayer0Plane,
                      0.0f);
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

  for (int offset = 0; offset < 5; ++offset) {
    CheckPlanesEqual(obs0, kMyMeeplePlane + offset, obs1, kOpponentMeeplePlane + offset);
    CheckPlanesEqual(obs0, kOpponentMeeplePlane + offset, obs1, kMyMeeplePlane + offset);
  }

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

  const float current_player_is_player0 =
      core.currentPlayer == 0 ? 1.0f : 0.0f;
  CheckBroadcastPlane(obs0, kCurrentPlayerIsPlayer0Plane,
                      current_player_is_player0);
  CheckBroadcastPlane(obs1, kCurrentPlayerIsPlayer0Plane,
                      current_player_is_player0);
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

void ShortGameMaxTurnsTest() {
  std::shared_ptr<const Game> game = LoadGame("carcassonne(max_turns=10)");
  SPIEL_CHECK_EQ(game->MaxGameLength(), (PHYSICAL_TILE_COUNT - 1) + 20);

  std::unique_ptr<State> state = game->NewInitialState();
  int actions_applied = 0;
  while (!state->IsTerminal()) {
    SPIEL_CHECK_LT(actions_applied, game->MaxGameLength());
    const std::vector<Action> legal_actions = state->LegalActions();
    SPIEL_CHECK_FALSE(legal_actions.empty());
    state->ApplyAction(legal_actions[0]);
    actions_applied++;
  }

  auto* carcassonne_state = dynamic_cast<CarcassonneState*>(state.get());
  SPIEL_CHECK_TRUE(carcassonne_state != nullptr);
  const ::Carcassonne& core = carcassonne_state->UnderlyingState();
  SPIEL_CHECK_EQ(core.current_phase, PHASE_TERMINAL);
  SPIEL_CHECK_TRUE(core.completed_turns == 10 || core.getTotalRemaining() == 0);
  SPIEL_CHECK_LE(core.completed_turns, 10);

  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns.size(), 2);
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
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
}

void BasicCarcassonneTests() {
  testing::LoadGameTest("carcassonne");
  testing::LoadGameTest("carcassonne(max_turns=10)");
  testing::ChanceOutcomesTest(*LoadGame("carcassonne"));
  testing::RandomSimTest(*LoadGame("carcassonne"), 50);
  ObservationTensorSmokeTest();
  RelativePerspectiveTest();
  ReturnsMatchScoresTest();
  ShortGameMaxTurnsTest();
}

}  // namespace
}  // namespace carcassonne
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::carcassonne::BasicCarcassonneTests();
}
