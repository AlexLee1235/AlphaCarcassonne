#include "open_spiel/games/carcassonne/carcassonne.h"
#include "open_spiel/games/carcassonne/carcassonne_test_utils.h"

#include <algorithm>
#include <array>
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
  SPIEL_CHECK_EQ(shape[0], 95);
  SPIEL_CHECK_EQ(shape[0], kObservationPlanes);
  SPIEL_CHECK_EQ(shape[1], BOARD_SIZE);
  SPIEL_CHECK_EQ(shape[2], BOARD_SIZE);
  SPIEL_CHECK_EQ(game->NumDistinctActions(), 4 * BOARD_SIZE * BOARD_SIZE + 15);

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
                          1.0f / 30));
  }
  SPIEL_CHECK_EQ(PlaneValue(initial_obs, kFeatureScorePlane + 2, c, c), 0.0f);
  CheckZeroPlanes(initial_obs, kFeatureMyMeeplesPlane, 12);
  CheckZeroPlanes(initial_obs, kLegalPlacementPlane, kLegalPlacementPlanes);
  CheckZeroPlanes(initial_obs, kMonasteryCoveragePlane, 2);
  // Its fields, north (half-edges 2, 7) and south (3-6) of the road, hold no
  // farmers and border no completed city. Each is one tile; the north one
  // borders the open city.
  CheckZeroPlanes(initial_obs, kFieldMyFarmersPlane,
                  kFieldSizePlane - kFieldMyFarmersPlane);
  for (int half_edge = 0; half_edge < HALF_EDGE_COUNT; ++half_edge) {
    const bool on_field = half_edge >= 2;
    const bool north = half_edge == 2 || half_edge == 7;
    SPIEL_CHECK_TRUE(Near(PlaneSum(initial_obs, kFieldSizePlane + half_edge),
                          on_field ? 1.0f / 60 : 0.0f));
    SPIEL_CHECK_TRUE(Near(PlaneValue(initial_obs, kFieldSizePlane + half_edge, c, c),
                          on_field ? 1.0f / 60 : 0.0f));
    SPIEL_CHECK_TRUE(Near(PlaneSum(initial_obs, kFieldOpenCitiesPlane + half_edge),
                          north ? 1.0f / 10 : 0.0f));
    SPIEL_CHECK_TRUE(
        Near(PlaneValue(initial_obs, kFieldOpenCitiesPlane + half_edge, c, c),
             north ? 1.0f / 10 : 0.0f));
  }
  CheckZeroPlanes(initial_obs, kInnerFieldMyFarmersPlane,
                  kInnerFieldOpenCitiesPlane + 1 - kInnerFieldMyFarmersPlane);
  CheckRemainingByType(initial_obs, initial_state->UnderlyingState());
  for (int i : {kGlobalMyScore, kGlobalOpponentScore, kGlobalScoreDiff,
                kGlobalMyPending, kGlobalOpponentPending, kGlobalStaticDiff,
                kGlobalStaticDiff + 1, kGlobalStaticDiff + 2,
                kGlobalCompletedTurns, kGlobalTilePhase, kGlobalMeeplePhase,
                kGlobalLegalPlacements, kGlobalMyFieldPending,
                kGlobalOpponentFieldPending, kGlobalMyFarmers,
                kGlobalOpponentFarmers}) {
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
                        1.0f / 71.0f));
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
  for (int half_edge = 0; half_edge < HALF_EDGE_COUNT; ++half_edge) {
    CheckPlanesEqual(obs0, kFieldMyFarmersPlane + half_edge, obs1,
                     kFieldOpponentFarmersPlane + half_edge);
    CheckPlanesEqual(obs0, kFieldOpponentFarmersPlane + half_edge, obs1,
                     kFieldMyFarmersPlane + half_edge);
    for (int plane : {kFieldScorePlane, kFieldSizePlane, kFieldOpenCitiesPlane}) {
      CheckPlanesEqual(obs0, plane + half_edge, obs1, plane + half_edge);
    }
  }
  CheckPlanesEqual(obs0, kInnerFieldMyFarmersPlane, obs1,
                   kInnerFieldOpponentFarmersPlane);
  for (int plane : {kInnerFieldScorePlane, kInnerFieldSizePlane,
                    kInnerFieldOpenCitiesPlane}) {
    CheckPlanesEqual(obs0, plane, obs1, plane);
  }

  int pending[2];
  core.getPendingScore(pending);
  int field_pending[2];
  core.getPendingFieldScore(field_pending);
  const std::vector<float>* views[2] = {&obs0, &obs1};
  for (Player player = 0; player < kNumPlayers; ++player) {
    const std::vector<float>& obs = *views[player];
    const int opponent = 1 - player;
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalMyScore),
                          std::min(1.0f, core.player_scores[player] / 100.0f)));
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalOpponentScore),
                          std::min(1.0f, core.player_scores[opponent] / 100.0f)));
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalMyPending),
                          std::min(1.0f, pending[player] / 70.0f)));
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalOpponentPending),
                          std::min(1.0f, pending[opponent] / 70.0f)));
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalMyFieldPending),
                          std::min(1.0f, field_pending[player] / 40.0f)));
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalOpponentFieldPending),
                          std::min(1.0f, field_pending[opponent] / 40.0f)));
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalMyMeeples),
                          core.holding_meeples[player] / 7.0f));
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalOpponentMeeples),
                          core.holding_meeples[opponent] / 7.0f));
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalMyFarmers),
                          core.farmersOnBoard(player) / 7.0f));
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalOpponentFarmers),
                          core.farmersOnBoard(opponent) / 7.0f));
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

// Fields renamed in order of first appearance (half-edges, then the inner one),
// followed by the field count and each field's city sides: equal for two
// tiles exactly when their field layouts are.
std::vector<int> CanonicalFieldLayout(const Tile& tile) {
  std::array<int, MAX_TILE_FIELDS> renamed;
  renamed.fill(-1);
  int next = 0;
  std::vector<int> layout;
  for (int half_edge = 0; half_edge < HALF_EDGE_COUNT; ++half_edge) {
    const int field = tile.field[half_edge];
    if (field != -1 && renamed[field] == -1) renamed[field] = next++;
    layout.push_back(field == -1 ? -1 : renamed[field]);
  }
  if (tile.innerField() != -1) renamed[tile.innerField()] = next++;
  layout.push_back(tile.field_count);
  std::array<int, MAX_TILE_FIELDS> city_sides{};
  for (int field = 0; field < tile.field_count; ++field) {
    city_sides[renamed[field]] = tile.field_city_sides[field];
  }
  layout.insert(layout.end(), city_sides.begin(), city_sides.end());
  return layout;
}

// What the observation shows of a tile's shape: its terrain and which pairs
// of non-grass sides it joins.
std::vector<int> TileLook(const Tile& tile) {
  std::vector<int> look(tile.edge, tile.edge + 4);
  for (int a = 0; a < 4; ++a) {
    for (int b = a + 1; b < 4; ++b) {
      look.push_back(tile.edge[a] != GRASS && tile.edge[b] != GRASS &&
                     tile.link[a] == tile.link[b]);
    }
  }
  return look;
}

// Every base tile, in every rotation, keeps the rules FieldLayout states.
void FieldLayoutTest() {
  std::vector<Tile> all_rotations;
  for (const TileBlueprint& blueprint : base_deck) {
    Tile tile = blueprint.tile;
    for (int rot = 0; rot < 4; ++rot, tile = tile.rotate()) {
      all_rotations.push_back(tile);
      SPIEL_CHECK_LE(tile.field_count, MAX_TILE_FIELDS);
      // Fields 0 .. edge_fields - 1 are on half-edges; at most one more, the
      // inner field, is on none.
      int edge_fields = 0;
      for (int half_edge = 0; half_edge < HALF_EDGE_COUNT; ++half_edge) {
        edge_fields = std::max(edge_fields, tile.field[half_edge] + 1);
      }
      SPIEL_CHECK_LE(edge_fields, tile.field_count);
      SPIEL_CHECK_LE(tile.field_count, edge_fields + 1);
      std::array<bool, MAX_TILE_FIELDS> on_half_edge{};
      for (int side = 0; side < 4; ++side) {
        const int a = tile.field[2 * side];
        const int b = tile.field[2 * side + 1];
        if (tile.edge[side] == CITY) {
          SPIEL_CHECK_EQ(a, -1);
          SPIEL_CHECK_EQ(b, -1);
          continue;
        }
        SPIEL_CHECK_GE(std::min(a, b), 0);
        SPIEL_CHECK_LT(std::max(a, b), tile.field_count);
        on_half_edge[a] = on_half_edge[b] = true;
        if (tile.edge[side] == GRASS) SPIEL_CHECK_EQ(a, b);
        // A road splits its side between two fields, unless it ends at a
        // monastery on this tile (type 2).
        if (tile.edge[side] == ROAD) SPIEL_CHECK_EQ(a == b, tile.monastery);
      }
      for (int field = 0; field < edge_fields; ++field) {
        SPIEL_CHECK_TRUE(on_half_edge[field]);
      }
      for (int field = 0; field < MAX_TILE_FIELDS; ++field) {
        const int sides = tile.field_city_sides[field];
        if (field >= tile.field_count) {
          SPIEL_CHECK_EQ(sides, 0);
          continue;
        }
        for (int side = 0; side < 4; ++side) {
          if (!(sides & (1 << side))) continue;
          SPIEL_CHECK_EQ(tile.edge[side], CITY);
          // Every side of a city the field borders.
          for (int other = 0; other < 4; ++other) {
            if (tile.edge[other] == CITY && tile.link[other] == tile.link[side]) {
              SPIEL_CHECK_TRUE(sides & (1 << other));
            }
          }
        }
      }
    }
    // Four quarter turns are the identity.
    SPIEL_CHECK_TRUE(CanonicalFieldLayout(tile) ==
                     CanonicalFieldLayout(blueprint.tile));
    SPIEL_CHECK_TRUE(std::equal(tile.field, tile.field + HALF_EDGE_COUNT,
                                blueprint.tile.field));
  }

  const auto first_of_type = [](int type) -> const Tile& {
    return full_deck[tile_type_tables.draw_physical_ids_by_type[type][0]][0];
  };
  SPIEL_CHECK_EQ(first_of_type(24).field_count, 4);  // RRRR
  SPIEL_CHECK_EQ(first_of_type(3).field_count, 0);   // CCCC
  SPIEL_CHECK_EQ(first_of_type(2).field[4], first_of_type(2).field[5]);

  // The observation has no field planes for a tile's own layout, so the
  // layout must follow from what it does show.
  for (const Tile& a : all_rotations) {
    for (const Tile& b : all_rotations) {
      if (TileLook(a) == TileLook(b)) {
        SPIEL_CHECK_TRUE(CanonicalFieldLayout(a) == CanonicalFieldLayout(b));
      }
    }
  }
}

// Draws `type`, places it at (tile_x, tile_y) turned `rot` and plays meeple
// position `pos`, checking each step is legal.
void PlayTurn(::Carcassonne* game, int type, int tile_x, int tile_y, int rot,
              int pos) {
  SPIEL_CHECK_EQ(game->current_phase, PHASE_CHANCE);
  game->drawTile(type);
  SPIEL_CHECK_EQ(game->current_phase, PHASE_TILE);
  std::array<TileMove, kTileActionCount> moves{};
  int count = 0;
  game->getLegalTileMoves(moves.data(), count);
  bool legal = false;
  for (int i = 0; i < count; ++i) {
    legal = legal || (moves[i].x == tile_x && moves[i].y == tile_y &&
                      moves[i].rot == rot);
  }
  SPIEL_CHECK_TRUE(legal);
  game->placeTile(tile_x, tile_y, rot);
  const MeepleMoves meeple_moves = game->getLegalMeepleMoves();
  SPIEL_CHECK_TRUE(std::find(meeple_moves.begin(), meeple_moves.end(), pos) !=
                   meeple_moves.end());
  game->placeMeeple(pos);
}

void CheckFieldPending(const ::Carcassonne& game, int player0, int player1) {
  int pending[2];
  game.getPendingFieldScore(pending);
  SPIEL_CHECK_EQ(pending[0], player0);
  SPIEL_CHECK_EQ(pending[1], player1);
}

// The field that local field `local` of the tile at (tile_x, tile_y) belongs
// to: how many tiles it spans, and the open and completed cities next to it.
void CheckField(const ::Carcassonne& game, int tile_x, int tile_y, int local,
                int tiles, int open_cities, int completed_cities) {
  const Placement placement = game.getPlacement(tile_x, tile_y);
  const Field& field = game.fieldAtRoot(game.fieldRoot(placement.id, local));
  const CityCounts cities = game.citiesNextTo(field);
  SPIEL_CHECK_EQ(field.getTileCount(), tiles);
  SPIEL_CHECK_EQ(cities.open, open_cities);
  SPIEL_CHECK_EQ(cities.completed, completed_cities);
}

// Fields built by hand, scored 3 per completed city next to them for whoever
// has the most farmers there. Local field 0 of the start tile is its north
// field, 1 its south field.
void FieldScoringTest() {
  const int c = BOARD_SIZE / 2;
  // The start tile (type 20) at (c, c): city north, road east-west, grass
  // south. Its north field borders the city, its south field nothing.
  {
    ::Carcassonne game(/*max_turns=*/8);
    // P0: another type 20 east of it; a farmer on the north field (half-edge
    // 2), next to both open cities.
    PlayTurn(&game, 20, c + 1, c, 0, MEEPLE_POS_FIELD + 2);
    CheckField(game, c, c, 0, /*tiles=*/2, /*open=*/2, /*completed=*/0);
    // P1: a CGGG south, city facing south; a farmer on the south field.
    PlayTurn(&game, 16, c, c + 1, 2, MEEPLE_POS_FIELD + 0);
    // P0: a T junction east; a farmer on its south-east corner, a field of
    // its own between two roads.
    PlayTurn(&game, 23, c + 2, c, 0, MEEPLE_POS_FIELD + 3);
    CheckFieldPending(game, 0, 0);
    CheckField(game, c, c, 0, 3, 2, 0);
    CheckField(game, c, c, 1, 4, 1, 0);
    CheckField(game, c + 2, c, 1, 1, 0, 0);
    // P1: closes the start tile's city; a farmer on the field north of it.
    PlayTurn(&game, 16, c, c - 1, 2, MEEPLE_POS_FIELD + 0);
    CheckFieldPending(game, 3, 3);
    CheckField(game, c, c, 0, 3, 1, 1);
    CheckField(game, c, c - 1, 0, 1, 0, 1);
    // P0: closes the second city, which the same two fields border.
    PlayTurn(&game, 16, c + 1, c - 1, 2, MEEPLE_POS_SKIP);
    CheckFieldPending(game, 6, 6);
    CheckField(game, c, c, 0, 3, 0, 2);
    CheckField(game, c, c - 1, 0, 2, 0, 2);
    // P1: a monastery ends the junction's south road, joining P1's south
    // field with P0's corner: tied, but next to an open city only. The
    // junction tile has a piece in both and counts once.
    PlayTurn(&game, 2, c + 2, c + 1, 2, MEEPLE_POS_SKIP);
    CheckFieldPending(game, 6, 6);
    CheckField(game, c, c, 1, 5, 1, 0);
    // P0: closes the south city; the tied field scores for both.
    PlayTurn(&game, 16, c, c + 2, 0, MEEPLE_POS_SKIP);
    CheckFieldPending(game, 9, 9);
    CheckField(game, c, c, 1, 5, 0, 1);
    int pending[2];
    int resolved[2];
    game.getPendingScore(pending);
    game.getPendingScoreByResolving(resolved);
    SPIEL_CHECK_EQ(pending[0], resolved[0]);
    SPIEL_CHECK_EQ(pending[1], resolved[1]);
    // P1: a monastery ends the junction's east road and joins everything
    // south of the two cities: two P0 farmers to one, next to all three
    // cities. P1 keeps the field north of them.
    PlayTurn(&game, 2, c + 3, c, 1, MEEPLE_POS_SKIP);
    SPIEL_CHECK_EQ(game.current_phase, PHASE_TERMINAL);
    SPIEL_CHECK_EQ(game.player_scores[0], 9);
    SPIEL_CHECK_EQ(game.player_scores[1], 6);
    // Ten pieces on six tiles: the start tile, its neighbour and the junction
    // each have pieces on both sides of the road.
    CheckField(game, c, c, 0, 6, 0, 3);
    // Farmers never come back: two each, and nothing else is out.
    SPIEL_CHECK_EQ(game.holding_meeples[0], 5);
    SPIEL_CHECK_EQ(game.holding_meeples[1], 5);
    SPIEL_CHECK_EQ(game.farmersOnBoard(0), 2);
    SPIEL_CHECK_EQ(game.farmersOnBoard(1), 2);
  }
  // A field next to two separate cities on one tile scores each of them.
  {
    ::Carcassonne game(/*max_turns=*/3);
    // P0: a CGCG south of the start tile, cities east and west; a farmer on
    // its field.
    PlayTurn(&game, 15, c, c + 1, 1, MEEPLE_POS_FIELD + 0);
    CheckFieldPending(game, 0, 0);
    CheckField(game, c, c + 1, 0, 2, 2, 0);
    PlayTurn(&game, 16, c + 1, c + 1, 3, MEEPLE_POS_SKIP);  // closes the east city
    CheckFieldPending(game, 3, 0);
    CheckField(game, c, c + 1, 0, 2, 1, 1);
    PlayTurn(&game, 16, c - 1, c + 1, 1, MEEPLE_POS_SKIP);  // and the west one
    SPIEL_CHECK_EQ(game.current_phase, PHASE_TERMINAL);
    SPIEL_CHECK_EQ(game.player_scores[0], 6);
    SPIEL_CHECK_EQ(game.player_scores[1], 0);
    CheckField(game, c, c + 1, 0, 2, 0, 2);
  }
}

// Grass ringed by four separate cities is one field on no half-edge, next to
// all four. No base tile is like that, so this drives the modules directly.
void InnerFieldTest() {
  const Tile four_cities(CITY, CITY, CITY, CITY, 0, 1, 2, 3,
                         {{-1, -1, -1, -1, -1, -1, -1, -1},
                          1,
                          {SIDE_N | SIDE_E | SIDE_S | SIDE_W}});
  Tile rotated = four_cities;
  for (int k = 0; k < kNumBoardRotations; ++k, rotated = rotated.rotate()) {
    SPIEL_CHECK_EQ(rotated.innerField(), 0);
    SPIEL_CHECK_EQ(rotated.field_count, 1);
    SPIEL_CHECK_EQ(rotated.field_city_sides[0], 0xF);
  }

  // Borrow the id of a tile without a shield that this test does not place.
  // FieldModule never looks the borrowed id up in full_deck: a neighbour only
  // does that across a side that is not a city, and this tile has none.
  const int id = tile_type_tables.draw_physical_ids_by_type[1][0];
  SPIEL_CHECK_FALSE(SHIELD_MASK[id]);
  auto board = std::make_unique<BoardModule>();
  auto features = std::make_unique<FeatureModule>();
  auto fields = std::make_unique<FieldModule>();
  auto place = [&](int tile_id, int tile_x, int tile_y, int rot,
                   const Tile& tile) {
    SPIEL_CHECK_TRUE(board->canPlaceTileAt(tile_x, tile_y, tile));
    board->placeTileOnBoard(tile_id, tile_x, tile_y, rot, tile);
    features->placeTileOnBoard(tile_id, tile_x, tile_y, rot, tile, *board);
    fields->placeTileOnBoard(tile_id, tile_x, tile_y, tile, *board, *features);
  };
  auto check_scores = [&](int player0, int player1) {
    int scores[2] = {0, 0};
    fields->accumulateScore(scores, *features);
    SPIEL_CHECK_EQ(scores[0], player0);
    SPIEL_CHECK_EQ(scores[1], player1);
  };
  auto check_cities = [&](int open_cities, int completed_cities) {
    const Field& field = fields->fieldMap.getSetData(fields->fieldIndex(id, 0));
    const CityCounts cities = fields->adjacentCities(field, *features);
    SPIEL_CHECK_EQ(field.getTileCount(), 1);
    SPIEL_CHECK_EQ(cities.open, open_cities);
    SPIEL_CHECK_EQ(cities.completed, completed_cities);
  };

  const int c = BOARD_SIZE / 2;
  place(id, c, c, 0, four_cities);
  MeepleMoves moves;
  fields->getLegalFarmerMoves(moves, id, four_cities);
  SPIEL_CHECK_EQ(moves.size(), 1);
  SPIEL_CHECK_EQ(moves[0], MEEPLE_POS_INNER_FIELD);
  fields->placeFarmer(id, four_cities, MEEPLE_POS_INNER_FIELD, 0);
  MeepleMoves taken;
  fields->getLegalFarmerMoves(taken, id, four_cities);
  SPIEL_CHECK_EQ(taken.size(), 0);
  check_scores(0, 0);
  check_cities(4, 0);

  // Close each city with a CGGG whose city faces the centre: N, E, S, W.
  const auto& caps = tile_type_tables.draw_physical_ids_by_type[16];
  const int cap_cells[4][2] = {{c, c - 1}, {c + 1, c}, {c, c + 1}, {c - 1, c}};
  const int cap_rotations[4] = {2, 3, 0, 1};
  for (int i = 0; i < 4; ++i) {
    place(caps[i], cap_cells[i][0], cap_cells[i][1], cap_rotations[i],
          full_deck[caps[i]][cap_rotations[i]]);
    check_scores(FIELD_POINTS_PER_CITY * (i + 1), 0);
    check_cities(3 - i, i + 1);
  }

  // The majority takes it all; a tie scores for both. (A second farmer on a
  // field is not a legal move; this only checks the count.)
  fields->placeFarmer(id, four_cities, MEEPLE_POS_INNER_FIELD, 1);
  check_scores(12, 12);
  fields->placeFarmer(id, four_cities, MEEPLE_POS_INNER_FIELD, 1);
  check_scores(0, 12);

  // The inner field is the same field whichever way the board turns.
  const Action inner_action = kMeepleActionOffset + MEEPLE_POS_INNER_FIELD + 1;
  for (int k = 0; k < kNumBoardRotations; ++k) {
    SPIEL_CHECK_EQ(RotateAction(inner_action, k, kNoSideGroups), inner_action);
  }
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
// feature changed), so the caller can check that case was exercised, and adds
// the same count for farmer moves (half-edge shifts) to `renamed_farmer_moves`.
int CheckRotatedTwin(absl::Span<const Action> history, int k,
                     std::mt19937* rng, int* renamed_farmer_moves) {
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
      const int half_edge = pos - MEEPLE_POS_FIELD;
      if (legal_action >= kMeepleActionOffset && pos >= MEEPLE_POS_FIELD &&
          pos < MEEPLE_POS_INNER_FIELD &&
          DecodeMeepleActionForTest(rotated_action) - MEEPLE_POS_FIELD !=
              (half_edge + 2 * k) % HALF_EDGE_COUNT) {
        ++*renamed_farmer_moves;
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
  int renamed_farmer_moves = 0;
  for (int k = 1; k < kNumBoardRotations; ++k) {
    renamed_meeple_moves += CheckRotatedTwin(kLastUnplaceableTileHistory, k,
                                             &rng, &renamed_farmer_moves);
    for (int game = 0; game < 20; ++game) {
      renamed_meeple_moves +=
          CheckRotatedTwin({}, k, &rng, &renamed_farmer_moves);
    }
  }
  std::cout << "RotationEquivarianceTest: " << renamed_meeple_moves
            << " meeple moves renamed beyond a side shift, "
            << renamed_farmer_moves
            << " farmer moves beyond a half-edge shift" << std::endl;
  SPIEL_CHECK_GT(renamed_meeple_moves, 0);
  SPIEL_CHECK_GT(renamed_farmer_moves, 0);
}

// Every spatial observation value is n / ObservationPlaneDenominator(plane)
// for an int8 n, and the global plane is 0 past the global vector: what lets
// alpha_zero_torch keep observations as int8 (observation_codec.h).
void ObservationDenominatorTest() {
  constexpr int kPlaneSize = BOARD_SIZE * BOARD_SIZE;
  for (int plane = 0; plane < kSpatialPlanes; ++plane) {
    SPIEL_CHECK_GT(ObservationPlaneDenominator(plane), 0.0f);
  }
  SPIEL_CHECK_EQ(ObservationPlaneDenominator(kGlobalFeaturePlane), 0.0f);

  std::shared_ptr<const Game> game = LoadGame("carcassonne");
  std::mt19937 rng(20261005);
  for (int g = 0; g < 10; ++g) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        state->ApplyAction(SampleAction(state->ChanceOutcomes(), rng).first);
        continue;
      }
      for (Player player = 0; player < kNumPlayers; ++player) {
        const std::vector<float> observation = state->ObservationTensor(player);
        for (int plane = 0; plane < kSpatialPlanes; ++plane) {
          const float denominator = ObservationPlaneDenominator(plane);
          for (int cell = 0; cell < kPlaneSize; ++cell) {
            const float value = observation[plane * kPlaneSize + cell];
            const long n = std::lround(value * denominator);
            SPIEL_CHECK_GE(n, -128);
            SPIEL_CHECK_LE(n, 127);
            SPIEL_CHECK_EQ(static_cast<float>(n) / denominator, value);
          }
        }
        for (int cell = kGlobalFeatures; cell < kPlaneSize; ++cell) {
          SPIEL_CHECK_EQ(observation[kGlobalFeaturePlane * kPlaneSize + cell],
                         0.0f);
        }
      }
      const std::vector<Action> legal = state->LegalActions();
      state->ApplyAction(
          legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)]);
    }
  }
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
  FieldLayoutTest();
  FieldScoringTest();
  InnerFieldTest();
  ReturnsMatchScoresTest();
  ShortGameMaxTurnsTest();
  LastUnplaceableTileTest();
  RotationEquivarianceTest();
  ObservationDenominatorTest();
}

}  // namespace
}  // namespace carcassonne
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::carcassonne::BasicCarcassonneTests();
}
