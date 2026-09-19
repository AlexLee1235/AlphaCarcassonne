#include "open_spiel/games/carcassonne/carcassonne.h"
#include "open_spiel/games/carcassonne/carcassonne_test_utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
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

float GlobalValue(const std::vector<float>& tensor, int index) {
  return tensor[kGlobalFeaturePlane * BOARD_SIZE * BOARD_SIZE + index];
}

bool Near(float left, float right) { return std::abs(left - right) < 1e-6f; }

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

void CheckConstantPlane(const std::vector<float>& tensor, int plane,
                        float expected) {
  for (int y = 0; y < BOARD_SIZE; ++y) {
    for (int x = 0; x < BOARD_SIZE; ++x) {
      SPIEL_CHECK_TRUE(Near(PlaneValue(tensor, plane, x, y), expected));
    }
  }
}

void CheckZeroPlanes(const std::vector<float>& tensor, int first_plane,
                     int plane_count) {
  for (int plane = first_plane; plane < first_plane + plane_count; ++plane) {
    CheckConstantPlane(tensor, plane, 0.0f);
  }
}

// `sign` * left == right on every cell.
void CheckPlanesEqual(const std::vector<float>& left, int left_plane,
                      const std::vector<float>& right, int right_plane,
                      float sign = 1.0f) {
  for (int y = 0; y < BOARD_SIZE; ++y) {
    for (int x = 0; x < BOARD_SIZE; ++x) {
      SPIEL_CHECK_TRUE(Near(sign * PlaneValue(left, left_plane, x, y),
                            PlaneValue(right, right_plane, x, y)));
    }
  }
}

float PlaneSum(const std::vector<float>& tensor, int plane) {
  float sum = 0.0f;
  for (int y = 0; y < BOARD_SIZE; ++y) {
    for (int x = 0; x < BOARD_SIZE; ++x) {
      sum += PlaneValue(tensor, plane, x, y);
    }
  }
  return sum;
}

void DecodeTileActionForTest(Action action, int* x, int* y, int* rot) {
  *rot = action % 4;
  action /= 4;
  *x = action % BOARD_SIZE;
  *y = action / BOARD_SIZE;
}

int DecodeMeepleActionForTest(Action action) {
  return action - kMeepleActionOffset - 1;
}

void CheckLegalMeepleGlobals(const std::vector<float>& tensor,
                             const std::vector<Action>& legal_actions) {
  float expected[kMeepleActionCount] = {};
  for (Action action : legal_actions) {
    if (action >= kMeepleActionOffset) {
      expected[action - kMeepleActionOffset] = 1.0f;
    }
  }
  for (int i = 0; i < kMeepleActionCount; ++i) {
    SPIEL_CHECK_EQ(GlobalValue(tensor, kGlobalLegalMeeple + i), expected[i]);
  }
}

void CheckRemainingByType(const std::vector<float>& tensor,
                          const ::Carcassonne& core) {
  for (int type_id = 1; type_id <= CANONICAL_TILE_TYPE_COUNT; ++type_id) {
    const int initial_count = tile_type_tables.draw_count_by_type[type_id];
    SPIEL_CHECK_GT(initial_count, 0);
    const float expected =
        static_cast<float>(core.getRemainingTypeCount(type_id)) / initial_count;
    SPIEL_CHECK_TRUE(Near(
        GlobalValue(tensor, kGlobalRemainingByType + type_id - 1), expected));
  }
}

void CheckTileInHand(const std::vector<float>& tensor, int type_id) {
  for (int type = 1; type <= CANONICAL_TILE_TYPE_COUNT; ++type) {
    SPIEL_CHECK_EQ(GlobalValue(tensor, kGlobalTileInHand + type - 1),
                   type == type_id ? 1.0f : 0.0f);
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
  SPIEL_CHECK_EQ(shape[0], 50);
  SPIEL_CHECK_EQ(shape[0], kObservationPlanes);
  SPIEL_CHECK_EQ(shape[1], BOARD_SIZE);
  SPIEL_CHECK_EQ(shape[2], BOARD_SIZE);
  SPIEL_CHECK_EQ(game->NumDistinctActions(), 4 * BOARD_SIZE * BOARD_SIZE + 6);

  SPIEL_CHECK_EQ(state->ObservationTensor(0).size(), kObservationTensorSize);
  SPIEL_CHECK_EQ(state->ObservationTensor(1).size(), kObservationTensorSize);

  // The start tile, type 20 (city north, road east-west, grass south), alone
  // on the centre cell.
  const int c = BOARD_SIZE / 2;
  std::vector<float> initial_obs = state->ObservationTensor(0);
  auto* initial_state = dynamic_cast<CarcassonneState*>(state.get());
  SPIEL_CHECK_TRUE(initial_state != nullptr);
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kOccupiedPlane, c, c), 1.0f);
  SPIEL_CHECK_EQ(PlaneSum(initial_obs, kOccupiedPlane), 1.0f);
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kNorthTerrainPlane + 1, c, c), 1.0f);
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kEastTerrainPlane + 2, c, c), 1.0f);
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kSouthTerrainPlane + 0, c, c), 1.0f);
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kWestTerrainPlane + 2, c, c), 1.0f);
  // Side pairs N-E, N-S, N-W, E-S, E-W, S-W: only the road joins E and W.
  for (int pair = 0; pair < kNumSidePairs; ++pair) {
    SPIEL_CHECK_EQ(PlaneValue(initial_obs, kSideLinkPlane + pair, c, c),
                   pair == 4 ? 1.0f : 0.0f);
  }
  SPIEL_CHECK_EQ(PlaneSum(initial_obs, kFrontierPlane), 4.0f);
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kFrontierPlane, c, c - 1), 1.0f);
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kFrontierPlane, c + 1, c), 1.0f);
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kFrontierPlane, c, c + 1), 1.0f);
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kFrontierPlane, c - 1, c), 1.0f);
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kLastPlacedPlane, c, c), 1.0f);
  // The city has one open edge, the road two; each is worth 1 so far.
  SPIEL_CHECK_TRUE(
      Near(PlaneValue(initial_obs, kFeatureOpensPlane + 0, c, c), 1.0f / 6));
  SPIEL_CHECK_TRUE(
      Near(PlaneValue(initial_obs, kFeatureOpensPlane + 1, c, c), 2.0f / 6));
  SPIEL_CHECK_TRUE(
      Near(PlaneValue(initial_obs, kFeatureOpensPlane + 3, c, c), 2.0f / 6));
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kFeatureOpensPlane + 2, c, c), 0.0f);
  for (int side : {0, 1, 3}) {
    SPIEL_CHECK_TRUE(Near(PlaneValue(initial_obs, kFeatureScorePlane + side, c, c),
                          1.0f / 12));
  }
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kFeatureScorePlane + 2, c, c), 0.0f);
  CheckZeroPlanes(initial_obs, kFeatureMyMeeplesPlane, 12);
  CheckZeroPlanes(initial_obs, kLegalPlacementPlane, kLegalPlacementPlanes);
  CheckZeroPlanes(initial_obs, kMonasteryCoveragePlane, 2);
  CheckRemainingByType(initial_obs, initial_state->UnderlyingState());
  for (int i : {kGlobalMyScore, kGlobalOpponentScore, kGlobalScoreDiff,
                kGlobalMyPending, kGlobalOpponentPending, kGlobalStaticDiff,
                kGlobalStaticDiff + 1, kGlobalStaticDiff + 2,
                kGlobalCompletedTurns, kGlobalTilePhase, kGlobalMeeplePhase,
                kGlobalLegalPlacements}) {
    SPIEL_CHECK_EQ(GlobalValue(initial_obs, i), 0.0f);
  }
  SPIEL_CHECK_EQ(GlobalValue(initial_obs, kGlobalMyMeeples), 1.0f);
  SPIEL_CHECK_EQ(GlobalValue(initial_obs, kGlobalOpponentMeeples), 1.0f);
  SPIEL_CHECK_TRUE(
      Near(GlobalValue(initial_obs, kGlobalRemainingTiles), 71.0f / 72.0f));
  SPIEL_CHECK_EQ(GlobalValue(initial_obs, kGlobalIsPlayer0), 1.0f);
  CheckTileInHand(initial_obs, 0);
  CheckLegalMeepleGlobals(initial_obs, {});
  // Only the vector's own cells are used in its plane.
  for (int i = kGlobalFeatures; i < BOARD_SIZE * BOARD_SIZE; ++i) {
    SPIEL_CHECK_EQ(GlobalValue(initial_obs, i), 0.0f);
  }

  while (state->IsChanceNode()) {
    state->ApplyAction(state->LegalActions()[0]);
  }
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  auto* chance_done = dynamic_cast<CarcassonneState*>(state.get());
  SPIEL_CHECK_TRUE(chance_done != nullptr);
  std::vector<float> tile_phase_obs = state->ObservationTensor(0);
  CheckTileInHand(tile_phase_obs, chance_done->UnderlyingState().currentTileType());
  SPIEL_CHECK_EQ(GlobalValue(tile_phase_obs, kGlobalTilePhase), 1.0f);
  SPIEL_CHECK_EQ(GlobalValue(tile_phase_obs, kGlobalMeeplePhase), 0.0f);
  CheckLegalMeepleGlobals(tile_phase_obs, {});
  CheckRemainingByType(tile_phase_obs, chance_done->UnderlyingState());
  const std::vector<Action> tile_actions = state->LegalActions();
  float legal_cells = 0.0f;
  for (int rot = 0; rot < kLegalPlacementPlanes; ++rot) {
    legal_cells += PlaneSum(tile_phase_obs, kLegalPlacementPlane + rot);
  }
  SPIEL_CHECK_EQ(legal_cells, static_cast<float>(tile_actions.size()));
  for (Action action : tile_actions) {
    int tile_x;
    int tile_y;
    int rot;
    DecodeTileActionForTest(action, &tile_x, &tile_y, &rot);
    SPIEL_CHECK_EQ(PlaneValue(tile_phase_obs, kLegalPlacementPlane + rot, tile_x, tile_y),
                   1.0f);
  }
  SPIEL_CHECK_TRUE(Near(GlobalValue(tile_phase_obs, kGlobalLegalPlacements),
                        tile_actions.size() / 100.0f));

  int placed_x;
  int placed_y;
  int placed_rot;
  DecodeTileActionForTest(tile_actions[0], &placed_x, &placed_y, &placed_rot);
  state->ApplyAction(tile_actions[0]);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  std::vector<float> meeple_phase_obs = state->ObservationTensor(0);
  CheckTileInHand(meeple_phase_obs, 0);
  SPIEL_CHECK_EQ(GlobalValue(meeple_phase_obs, kGlobalTilePhase), 0.0f);
  SPIEL_CHECK_EQ(GlobalValue(meeple_phase_obs, kGlobalMeeplePhase), 1.0f);
  SPIEL_CHECK_EQ(GlobalValue(meeple_phase_obs, kGlobalLegalPlacements), 0.0f);
  CheckZeroPlanes(meeple_phase_obs, kLegalPlacementPlane, kLegalPlacementPlanes);
  CheckLegalMeepleGlobals(meeple_phase_obs, state->LegalActions());
  SPIEL_CHECK_EQ(PlaneSum(meeple_phase_obs, kLastPlacedPlane), 1.0f);
  SPIEL_CHECK_EQ(PlaneValue(meeple_phase_obs, kLastPlacedPlane, placed_x, placed_y),
                 1.0f);
  SPIEL_CHECK_EQ(PlaneSum(meeple_phase_obs, kOccupiedPlane), 2.0f);
  SPIEL_CHECK_EQ(PlaneValue(meeple_phase_obs, kFrontierPlane, placed_x, placed_y),
                 0.0f);
  SPIEL_CHECK_EQ(GlobalValue(meeple_phase_obs, kGlobalIsPlayer0), 1.0f);

  state->ApplyAction(state->LegalActions()[0]);
  std::vector<float> player1_chance_obs = state->ObservationTensor(1);
  SPIEL_CHECK_EQ(GlobalValue(player1_chance_obs, kGlobalIsPlayer0), 0.0f);
  SPIEL_CHECK_TRUE(Near(GlobalValue(player1_chance_obs, kGlobalCompletedTurns),
                        1.0f / 36.0f));
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

  // Player 0 has a meeple out; each player sees it on their own side
  float player0_meeples = 0.0f;
  for (int side = 0; side < 4; ++side) {
    player0_meeples += PlaneSum(obs0, kFeatureMyMeeplesPlane + side);
    CheckPlanesEqual(obs0, kFeatureMyMeeplesPlane + side, obs1,
                     kFeatureOpponentMeeplesPlane + side);
    CheckPlanesEqual(obs0, kFeatureOpponentMeeplesPlane + side, obs1,
                     kFeatureMyMeeplesPlane + side);
    CheckPlanesEqual(obs0, kFeatureSignedScorePlane + side, obs1,
                     kFeatureSignedScorePlane + side, -1.0f);
    CheckPlanesEqual(obs0, kFeatureScorePlane + side, obs1,
                     kFeatureScorePlane + side);
    CheckPlanesEqual(obs0, kFeatureOpensPlane + side, obs1,
                     kFeatureOpensPlane + side);
  }
  // (or on a monastery, which has its own owner plane).
  SPIEL_CHECK_GT(player0_meeples + PlaneSum(obs0, kMonasteryOwnerPlane), 0.0f);
  CheckPlanesEqual(obs0, kMonasteryOwnerPlane, obs1, kMonasteryOwnerPlane,
                   -1.0f);

  int pending[2];
  core.getPendingScore(pending);
  const std::vector<float>* views[2] = {&obs0, &obs1};
  for (Player player = 0; player < kNumPlayers; ++player) {
    const std::vector<float>& obs = *views[player];
    const int opponent = 1 - player;
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalMyScore),
                          core.player_scores[player] / 40.0f));
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalOpponentScore),
                          core.player_scores[opponent] / 40.0f));
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalMyPending),
                          pending[player] / 20.0f));
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalOpponentPending),
                          pending[opponent] / 20.0f));
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalMyMeeples),
                          core.holding_meeples[player] / 7.0f));
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalOpponentMeeples),
                          core.holding_meeples[opponent] / 7.0f));
    const int diff = core.player_scores[player] - core.player_scores[opponent] +
                     pending[player] - pending[opponent];
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalStaticDiff),
                          std::max(-1.0f, std::min(1.0f, diff / 3.0f))));
  }
  for (int i : {kGlobalScoreDiff, kGlobalStaticDiff, kGlobalStaticDiff + 1,
                kGlobalStaticDiff + 2}) {
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs0, i), -GlobalValue(obs1, i)));
  }
  // Player 0 claimed a feature, so something is at stake for them.
  SPIEL_CHECK_GT(pending[0], 0);

  const float current_player_is_player0 =
      core.currentPlayer == 0 ? 1.0f : 0.0f;
  SPIEL_CHECK_EQ(GlobalValue(obs0, kGlobalIsPlayer0), current_player_is_player0);
  SPIEL_CHECK_EQ(GlobalValue(obs1, kGlobalIsPlayer0), current_player_is_player0);
}

// Pending points are counted without copying the game; check them against
// settling and end-game scoring a copy, at every state of random games. That
// includes the meeple phase right after a tile closes a feature that holds
// meeples: it is settled when the turn ends whatever the move, so it counts.
void PendingScoreTest() {
  std::mt19937 rng(20260919);
  std::shared_ptr<const Game> game = LoadGame("carcassonne");
  int closed_but_unsettled = 0;
  for (int sim = 0; sim < 100; ++sim) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (true) {
      const ::Carcassonne& core =
          dynamic_cast<const CarcassonneState&>(*state).UnderlyingState();
      int fast[2];
      int slow[2];
      core.getPendingScore(fast);
      core.getPendingScoreByResolving(slow);
      SPIEL_CHECK_EQ(fast[0], slow[0]);
      SPIEL_CHECK_EQ(fast[1], slow[1]);
      if (state->IsTerminal()) {
        SPIEL_CHECK_EQ(fast[0], 0);
        SPIEL_CHECK_EQ(fast[1], 0);
        break;
      }
      if (core.current_phase == PHASE_MEEPLE) {
        const Placement placement = core.getPlacement(core.last_x, core.last_y);
        const Tile& tile = full_deck[placement.id][placement.rotation];
        for (int side = 0; side < 4; ++side) {
          const Feature& feature = core.featureAt(placement.id, side);
          if (tile.edge[side] != GRASS && feature.opens == 0 &&
              feature.hasMeeples()) {
            ++closed_but_unsettled;
            break;
          }
        }
      }
      const std::vector<Action> legal = state->LegalActions();
      state->ApplyAction(
          state->IsChanceNode()
              ? SampleAction(state->ChanceOutcomes(), rng).first
              : legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)]);
    }
  }
  std::cout << "PendingScoreTest: " << closed_but_unsettled
            << " meeple-phase states with a closed, unsettled feature"
            << std::endl;
  SPIEL_CHECK_GT(closed_but_unsettled, 0);
}

// A feature where both players hold the same number of meeples is not an
// unclaimed one: both score it, and nobody can add a meeple. The two meeple
// planes keep that apart, which a single difference plane would not.
void TiedFeatureTest() {
  std::mt19937 rng(20260920);
  std::shared_ptr<const Game> game = LoadGame("carcassonne");
  int tied_sides = 0;
  for (int sim = 0; sim < 200 && tied_sides == 0; ++sim) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (!state->IsChanceNode()) {
        const ::Carcassonne& core =
            dynamic_cast<const CarcassonneState&>(*state).UnderlyingState();
        const std::vector<float> obs = state->ObservationTensor(0);
        for (int y = 0; y < BOARD_SIZE; ++y) {
          for (int x = 0; x < BOARD_SIZE; ++x) {
            const Placement placement = core.getPlacement(x, y);
            if (placement.id == 0) continue;
            const Tile& tile = full_deck[placement.id][placement.rotation];
            for (int side = 0; side < 4; ++side) {
              if (tile.edge[side] == GRASS) continue;
              const Feature& feature = core.featureAt(placement.id, side);
              const int count = feature.meeple_count[0];
              if (count == 0 || feature.meeple_count[1] != count) continue;
              ++tied_sides;
              SPIEL_CHECK_TRUE(Near(
                  PlaneValue(obs, kFeatureMyMeeplesPlane + side, x, y), count / 7.0f));
              SPIEL_CHECK_TRUE(Near(
                  PlaneValue(obs, kFeatureOpponentMeeplesPlane + side, x, y),
                  count / 7.0f));
              // SPIEL_CHECK_EQ's own locals are called x and y.
              SPIEL_CHECK_TRUE(
                  PlaneValue(obs, kFeatureSignedScorePlane + side, x, y) == 0.0f);
            }
          }
        }
      }
      const std::vector<Action> legal = state->LegalActions();
      state->ApplyAction(
          state->IsChanceNode()
              ? SampleAction(state->ChanceOutcomes(), rng).first
              : legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)]);
    }
  }
  std::cout << "TiedFeatureTest: " << tied_sides << " tied feature sides seen"
            << std::endl;
  SPIEL_CHECK_GT(tied_sides, 0);
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

void LastUnplaceableTileTest() {
  auto state = LastUnplaceableTileState(LoadGame("carcassonne"));
  const auto before = state->ToString();
  auto clone = state->Clone();
  clone->ApplyAction(2);
  SPIEL_CHECK_TRUE(clone->IsTerminal());
  SPIEL_CHECK_EQ(clone->CurrentPlayer(), kTerminalPlayerId);
  SPIEL_CHECK_TRUE(clone->LegalActions().empty());
  SPIEL_CHECK_EQ(clone->Returns(), (std::vector<double>{1.0, -1.0}));
  const auto& core = dynamic_cast<const CarcassonneState&>(*clone).UnderlyingState();
  SPIEL_CHECK_EQ(core.completed_turns, 70);
  SPIEL_CHECK_EQ(core.getTotalRemaining(), 0);
  SPIEL_CHECK_EQ(core.current_tile_in_hand, 0);
  SPIEL_CHECK_EQ(core.player_scores[0], 57);
  SPIEL_CHECK_EQ(core.player_scores[1], 34);
  SPIEL_CHECK_EQ(clone->ObservationTensor(0).size(), kObservationTensorSize);
  SPIEL_CHECK_EQ(clone->ObservationTensor(1).size(), kObservationTensorSize);
  SPIEL_CHECK_EQ(state->ToString(), before);
  state->ApplyAction(2);
  SPIEL_CHECK_EQ(state->ToString(), clone->ToString());
}

// Plays a game (following `history`, then random moves) and, alongside it, the
// same game on a board turned by k quarter turns: start tile turned, every move
// turned with RotateAction. At every decision the twin's own observations,
// legal moves and side groups must equal the rotation of the original's, and
// both games must end with the same scores. Returns how many legal meeple moves
// were renamed differently from a plain side shift (the lowest side of the
// feature changed), so the caller can check that case was exercised.
int CheckRotatedTwin(absl::Span<const Action> history, int k,
                     std::mt19937* rng) {
  std::shared_ptr<const Game> game = LoadGame("carcassonne");
  CarcassonneState state(game);
  CarcassonneState twin(game, ::Carcassonne(/*max_turns=*/0,
                                            /*start_rotation=*/k));
  // CarcassonneState's override hides State::ObservationTensor(Player).
  const State& state_view = state;
  const State& twin_view = twin;
  int renamed_meeple_moves = 0;
  for (int step = 0; !state.IsTerminal(); ++step) {
    SPIEL_CHECK_FALSE(twin.IsTerminal());
    const std::vector<Action> legal = state.LegalActions();
    Action action;
    if (step < static_cast<int>(history.size())) {
      action = history[step];
    } else if (state.IsChanceNode()) {
      action = SampleAction(state.ChanceOutcomes(), *rng).first;
    } else {
      action = legal[std::uniform_int_distribution<int>(
          0, legal.size() - 1)(*rng)];
    }

    if (state.IsChanceNode()) {
      SPIEL_CHECK_TRUE(twin.IsChanceNode());
      SPIEL_CHECK_EQ(legal, twin.LegalActions());
      state.ApplyAction(action);
      twin.ApplyAction(action);
      continue;
    }

    SPIEL_CHECK_EQ(state.CurrentPlayer(), twin.CurrentPlayer());
    const SideGroups groups = GetSideGroups(state);
    const SideGroups rotated_groups = RotateSideGroups(groups, k);
    SPIEL_CHECK_TRUE(rotated_groups == GetSideGroups(twin));
    SPIEL_CHECK_TRUE(RotateSideGroups(rotated_groups, 4 - k) == groups);

    for (Player player = 0; player < kNumPlayers; ++player) {
      const std::vector<float> observation =
          state_view.ObservationTensor(player);
      std::vector<float> rotated(kObservationTensorSize);
      RotateObservation(observation, k, groups, absl::MakeSpan(rotated));
      SPIEL_CHECK_TRUE(rotated == twin_view.ObservationTensor(player));
      std::vector<float> back(kObservationTensorSize);
      RotateObservation(rotated, 4 - k, rotated_groups, absl::MakeSpan(back));
      SPIEL_CHECK_TRUE(back == observation);
    }

    std::vector<Action> rotated_legal;
    for (Action legal_action : legal) {
      const Action rotated_action = RotateAction(legal_action, k, groups);
      SPIEL_CHECK_EQ(RotateAction(rotated_action, 4 - k, rotated_groups),
                     legal_action);
      rotated_legal.push_back(rotated_action);
      const int pos = DecodeMeepleActionForTest(legal_action);
      if (legal_action >= kMeepleActionOffset && pos >= 0 && pos < 4 &&
          DecodeMeepleActionForTest(rotated_action) != (pos + k) % 4) {
        ++renamed_meeple_moves;
      }
    }
    std::sort(rotated_legal.begin(), rotated_legal.end());
    SPIEL_CHECK_EQ(rotated_legal, twin.LegalActions());

    twin.ApplyAction(RotateAction(action, k, groups));
    state.ApplyAction(action);
  }

  SPIEL_CHECK_TRUE(twin.IsTerminal());
  SPIEL_CHECK_EQ(state.Returns(), twin.Returns());
  const ::Carcassonne& core = state.UnderlyingState();
  const ::Carcassonne& twin_core = twin.UnderlyingState();
  for (Player player = 0; player < kNumPlayers; ++player) {
    SPIEL_CHECK_EQ(core.player_scores[player], twin_core.player_scores[player]);
    SPIEL_CHECK_EQ(core.holding_meeples[player],
                   twin_core.holding_meeples[player]);
  }
  return renamed_meeple_moves;
}

void RotationEquivarianceTest() {
  std::mt19937 rng(20260915);
  int renamed_meeple_moves = 0;
  for (int k = 1; k < kNumBoardRotations; ++k) {
    renamed_meeple_moves +=
        CheckRotatedTwin(kLastUnplaceableTileHistory, k, &rng);
    for (int game = 0; game < 20; ++game) {
      renamed_meeple_moves += CheckRotatedTwin({}, k, &rng);
    }
  }
  std::cout << "RotationEquivarianceTest: " << renamed_meeple_moves
            << " meeple moves renamed beyond a side shift" << std::endl;
  SPIEL_CHECK_GT(renamed_meeple_moves, 0);
}

void BasicCarcassonneTests() {
  testing::LoadGameTest("carcassonne");
  testing::LoadGameTest("carcassonne(max_turns=10)");
  testing::ChanceOutcomesTest(*LoadGame("carcassonne"));
  testing::RandomSimTest(*LoadGame("carcassonne"), 50);
  ObservationTensorSmokeTest();
  RelativePerspectiveTest();
  PendingScoreTest();
  TiedFeatureTest();
  ReturnsMatchScoresTest();
  ShortGameMaxTurnsTest();
  LastUnplaceableTileTest();
  RotationEquivarianceTest();
}

}  // namespace
}  // namespace carcassonne
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::carcassonne::BasicCarcassonneTests();
}
