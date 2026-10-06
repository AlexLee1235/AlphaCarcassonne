#include "open_spiel/games/carcassonne/carcassonne.h"
#include "open_spiel/games/carcassonne/carcassonne_test_utils.h"
#include "open_spiel/games/carcassonne/game/tile_check.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
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

// Every expansion's tiles, with the rules of those that have them (the river,
// Inns & Cathedrals, the builder).
constexpr const char* kAllExpansionsGame =
    "carcassonne(inns_cathedrals=on,traders_builders=on,river=on,"
    "princess_dragon=tiles)";
constexpr const char* kRiverGame = "carcassonne(river=on)";
constexpr const char* kInnsCathedralsGame = "carcassonne(inns_cathedrals=on)";
constexpr const char* kTradersBuildersGame = "carcassonne(traders_builders=on)";

int TestTerrainIndex(EdgeType edge_type) {
  switch (edge_type) {
    case GRASS:
      return 0;
    case CITY:
      return 1;
    case ROAD:
      return 2;
    case RIVER:
      return 3;
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
  SPIEL_CHECK_EQ(shape[0], 123);
  SPIEL_CHECK_EQ(shape[0], kObservationPlanes);
  SPIEL_CHECK_EQ(shape[1], BOARD_SIZE);
  SPIEL_CHECK_EQ(shape[2], BOARD_SIZE);
  SPIEL_CHECK_EQ(game->NumDistinctActions(), 4 * BOARD_SIZE * BOARD_SIZE + 33);

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
  CheckZeroPlanes(initial_obs, kShieldPlane, 4);
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
  // No big meeple, inn, cathedral or builder in the base game.
  CheckZeroPlanes(initial_obs, kFeatureMyBigMeeplePlane,
                  kSpatialPlanes - kFeatureMyBigMeeplePlane);
  CheckRemainingByType(initial_obs, initial_state->UnderlyingState());
  for (int i : {kGlobalMyScore, kGlobalOpponentScore, kGlobalScoreDiff,
                kGlobalMyPending, kGlobalOpponentPending, kGlobalStaticDiff,
                kGlobalStaticDiff + 1, kGlobalStaticDiff + 2,
                kGlobalCompletedTurns, kGlobalTilePhase, kGlobalMeeplePhase,
                kGlobalLegalPlacements, kGlobalMyFieldPending,
                kGlobalOpponentFieldPending, kGlobalMyFarmers,
                kGlobalOpponentFarmers, kGlobalRiverTiles, kGlobalMyBigMeeple,
                kGlobalOpponentBigMeeple, kGlobalMyBigFarmer,
                kGlobalOpponentBigFarmer, kGlobalMyBuilder,
                kGlobalOpponentBuilder, kGlobalBuilderExtraTile,
                kGlobalBuilderSecondTile}) {
    SPIEL_CHECK_EQ(GlobalValue(initial_obs, i), 0.0f);
  }
  for (int cell = 0; cell < kGlobalExpansionCells; ++cell) {
    SPIEL_CHECK_EQ(GlobalValue(initial_obs, kGlobalExpansionModes + cell), 0.0f);
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
                          (core.farmersOnBoard(player) - core.bigFarmers(player)) / 7.0f));
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalOpponentFarmers),
                          (core.farmersOnBoard(opponent) - core.bigFarmers(opponent)) / 7.0f));
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
          if (isFeatureEdge(tile.edge[side]) && feature.opens == 0 &&
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
              if (!isFeatureEdge(tile.edge[side])) continue;
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

// Pairs of types that look the same to the observation but have different
// field layouts (tile_check::TileLookConflicts). The observation has no planes
// for a tile's own field layout, so a network cannot tell them apart
// (docs/carcassonne_field_observation.md §2.2). A type goes here only once it
// has been looked at, with the reason it is accepted.
// A conflict is accepted when one of its types is listed here: the tile whose
// fields differ from tiles that look the same. Telling them apart needs
// candidate A of that doc.
const std::vector<int> kHiddenFieldTypes = {
    // A city wall runs into a corner of the tile and cuts off the grass on
    // either side of it, where the lookalikes' grass goes round the corner.
    42,  // a city on one side only, walls corner to corner (vs caps: 16, 82, 91)
    62,  // corner city, road from its gate; wall into the corner beside the road
    65,  // the same, the other way round
    67,  // corner city, roads from its gate on both free sides; wall between them
    94,  // corner city, wall into the opposite corner (vs plain corner cities)
    95,  // two cities, a diagonal strip of grass between them (vs type 38)
    // The roads end in the grass short of the city, so they do not cut the
    // grass, where in 45 and 72 they reach the city gates.
    76,
};

void PrintAll(const char* what, const std::vector<std::string>& messages) {
  for (const std::string& message : messages) {
    std::cerr << what << ": " << message << std::endl;
  }
}

// Every tile of the table, in every rotation, keeps the rules FieldLayout
// states (tile_check::CheckTile), and each expansion section that is started
// holds the whole box.
void TileTableTest() {
  PrintAll("tile table", tile_check::CheckTileTable());
  SPIEL_CHECK_TRUE(tile_check::CheckTileTable().empty());
  PrintAll("tile table", tile_check::IncompleteExpansions());
  SPIEL_CHECK_TRUE(tile_check::IncompleteExpansions().empty());

  for (const TileBlueprint& blueprint : all_tiles) {
    Tile tile = blueprint.tile;
    for (int rot = 0; rot < 4; ++rot, tile = tile.rotate()) {
      SPIEL_CHECK_TRUE(tile_check::CheckTile(tile).empty());
    }
    // Four quarter turns are the identity.
    SPIEL_CHECK_TRUE(tile_check::CanonicalFieldLayout(tile) ==
                     tile_check::CanonicalFieldLayout(blueprint.tile));
    SPIEL_CHECK_TRUE(std::equal(tile.field, tile.field + HALF_EDGE_COUNT,
                                blueprint.tile.field));
  }

  // In the base deck a road splits its side between two fields, unless it
  // ends at a monastery on this tile (type 2).
  for (int row = 0; row < BASE_TILE_TYPE_COUNT; ++row) {
    const Tile& tile = all_tiles[row].tile;
    for (int side = 0; side < 4; ++side) {
      if (tile.edge[side] == ROAD) {
        SPIEL_CHECK_EQ(tile.field[2 * side] == tile.field[2 * side + 1],
                       tile.monastery);
      }
    }
  }

  const auto first_of_type = [](int type) -> const Tile& {
    return full_deck[tile_type_tables.draw_physical_ids_by_type[type][0]][0];
  };
  SPIEL_CHECK_EQ(first_of_type(24).field_count, 4);  // RRRR
  SPIEL_CHECK_EQ(first_of_type(3).field_count, 0);   // CCCC
  SPIEL_CHECK_EQ(first_of_type(2).field[4], first_of_type(2).field[5]);

  // The observation has no field planes for a tile's own layout, so the
  // layout must follow from what it does show, but for the listed tiles.
  const auto hidden = [](int type) {
    return std::find(kHiddenFieldTypes.begin(), kHiddenFieldTypes.end(), type) !=
           kHiddenFieldTypes.end();
  };
  for (const std::pair<int, int>& conflict : tile_check::TileLookConflicts()) {
    const bool accepted = hidden(conflict.first) || hidden(conflict.second);
    if (!accepted) {
      std::cerr << "tile table: types " << conflict.first << " and "
                << conflict.second
                << " look the same to the observation but have different "
                   "fields; see kHiddenFieldTypes"
                << std::endl;
    }
    SPIEL_CHECK_TRUE(accepted);
  }
}

// The checker itself: it must catch the mistakes it is there for.
void TileCheckTest() {
  // Fine: a straight river, the same shape as the straight road.
  SPIEL_CHECK_TRUE(tile_check::CheckTile(
                       Tile(RIVER, GRASS, RIVER, GRASS, 0, 1, 0, 2,
                            {{0, 1, 1, 1, 1, 0, 0, 0}, 2, {}}))
                       .empty());
  // Fine: a lake ends the river, so the grass runs round it.
  SPIEL_CHECK_TRUE(tile_check::CheckTile(
                       Tile(RIVER, GRASS, GRASS, GRASS, 0, 1, 2, 3,
                            {{0, 0, 0, 0, 0, 0, 0, 0}, 1, {}}))
                       .empty());
  const auto fails = [](const Tile& tile) {
    return !tile_check::CheckTile(tile).empty();
  };
  // A city and a road sharing a link.
  SPIEL_CHECK_TRUE(fails(Tile(CITY, ROAD, GRASS, ROAD, 0, 0, 2, 0,
                              {{-1, -1, 0, 1, 1, 1, 1, 0}, 2, {SIDE_N}})));
  // Two grass sides sharing a link.
  SPIEL_CHECK_TRUE(fails(Tile(GRASS, GRASS, GRASS, GRASS, 0, 0, 2, 3,
                              {{0, 0, 0, 0, 0, 0, 0, 0}, 1, {}}, {},
                              TILE_MONASTERY)));
  // A city half-edge with a field.
  SPIEL_CHECK_TRUE(fails(Tile(CITY, GRASS, GRASS, GRASS, 0, 1, 2, 3,
                              {{0, -1, 0, 0, 0, 0, 0, 0}, 1, {SIDE_N}})));
  // A river going on to the far side, but one field on both its halves.
  SPIEL_CHECK_TRUE(fails(Tile(RIVER, GRASS, RIVER, GRASS, 0, 1, 0, 2,
                              {{0, 0, 0, 0, 0, 0, 0, 0}, 1, {}})));
  // A field listing only one side of the city it borders.
  SPIEL_CHECK_TRUE(fails(Tile(CITY, CITY, GRASS, CITY, 0, 0, 1, 0,
                              {{-1, -1, -1, -1, 0, 0, -1, -1}, 1, {SIDE_N}})));
  // Two inner fields.
  SPIEL_CHECK_TRUE(fails(Tile(CITY, CITY, CITY, CITY, 0, 1, 2, 3,
                              {{-1, -1, -1, -1, -1, -1, -1, -1}, 2, {0xF, 0xF}})));

  // Marks. Fine: a shield on one of two cities, goods on the other, an inn
  // on a road, and marks for the whole tile.
  const FieldLayout two_cities = {{-1, -1, 0, 0, -1, -1, 0, 0}, 1, {SIDE_N | SIDE_S}};
  SPIEL_CHECK_TRUE(tile_check::CheckTile(
                       Tile(CITY, GRASS, CITY, GRASS, 0, 1, 2, 3, two_cities,
                            {MARK_SHIELD, 0, MARK_WINE, 0}, TILE_DRAGON))
                       .empty());
  SPIEL_CHECK_TRUE(tile_check::CheckTile(
                       Tile(ROAD, GRASS, ROAD, GRASS, 0, 1, 0, 2,
                            {{0, 1, 1, 1, 1, 0, 0, 0}, 2, {}}, {MARK_INN, 0, 0, 0},
                            TILE_VOLCANO | TILE_PORTAL))
                       .empty());
  // A shield on a road.
  SPIEL_CHECK_TRUE(fails(Tile(ROAD, GRASS, ROAD, GRASS, 0, 1, 0, 2,
                              {{0, 1, 1, 1, 1, 0, 0, 0}, 2, {}}, {MARK_SHIELD, 0, 0, 0})));
  // An inn on a city.
  SPIEL_CHECK_TRUE(fails(Tile(CITY, GRASS, CITY, GRASS, 0, 1, 2, 3, two_cities,
                              {MARK_INN, 0, 0, 0})));
  // A mark on grass.
  SPIEL_CHECK_TRUE(fails(Tile(CITY, GRASS, CITY, GRASS, 0, 1, 2, 3, two_cities,
                              {0, MARK_SHIELD, 0, 0})));
  // Two kinds of goods in one city, written on two of its sides.
  SPIEL_CHECK_TRUE(fails(Tile(CITY, CITY, GRASS, CITY, 0, 0, 1, 0,
                              {{-1, -1, -1, -1, 0, 0, -1, -1}, 1, {SIDE_N | SIDE_E | SIDE_W}},
                              {MARK_WINE, MARK_CLOTH, 0, 0})));
  // An unknown tile mark.
  SPIEL_CHECK_TRUE(fails(Tile(GRASS, GRASS, GRASS, GRASS, 0, 1, 2, 3,
                              {{0, 0, 0, 0, 0, 0, 0, 0}, 1, {}}, {}, 1 << 7)));
}

// A shield belongs to one city of its tile: on a tile with two cities only
// the city with the shield scores it, whichever way the tile is turned.
void ShieldPerCityTest() {
  // A city north and a separate one south, the shield on the north one.
  const Tile two_cities(CITY, GRASS, CITY, GRASS, 0, 1, 2, 3,
                        {{-1, -1, 0, 0, -1, -1, 0, 0}, 1, {SIDE_N | SIDE_S}},
                        {MARK_SHIELD, 0, 0, 0});
  SPIEL_CHECK_TRUE(tile_check::CheckTile(two_cities).empty());
  const int c = BOARD_SIZE / 2;
  const auto& caps = tile_type_tables.draw_physical_ids_by_type[16];  // CGGG
  for (int k = 0; k < kNumBoardRotations; ++k) {
    Tile turned = two_cities;
    for (int i = 0; i < k; ++i) turned = turned.rotate();
    const int shield_side = k % 4;          // where north went
    const int plain_side = (k + 2) % 4;     // where south went
    SPIEL_CHECK_TRUE(turned.featureMarks(shield_side) & MARK_SHIELD);
    SPIEL_CHECK_FALSE(turned.featureMarks(plain_side) & MARK_SHIELD);

    // Borrowed id of a tile this test does not otherwise place. FeatureModule
    // never looks it up in full_deck; FieldModule is not used.
    const int id = tile_type_tables.draw_physical_ids_by_type[1][0];
    auto board = std::make_unique<BoardModule>();
    auto features = std::make_unique<FeatureModule>();
    board->placeTileOnBoard(id, c, c, k, turned);
    features->placeTileOnBoard(id, c, c, k, turned, *board);
    auto score = [&](int tile_id, int side) {
      return features->featureMap.getSetData(features->edgeIndex(tile_id, side))
          .getScore();
    };
    SPIEL_CHECK_EQ(score(id, shield_side), 2);  // one tile, one shield
    SPIEL_CHECK_EQ(score(id, plain_side), 1);

    // Close the shielded city with a CGGG whose city faces it.
    const int dx[4] = {0, 1, 0, -1};
    const int dy[4] = {-1, 0, 1, 0};
    const int cap_x = c + dx[shield_side];
    const int cap_y = c + dy[shield_side];
    const int cap_rot = (shield_side + 2) % 4;  // its city (north) turned to face back
    const Tile& cap = full_deck[caps[0]][cap_rot];
    SPIEL_CHECK_TRUE(board->canPlaceTileAt(cap_x, cap_y, cap));
    board->placeTileOnBoard(caps[0], cap_x, cap_y, cap_rot, cap);
    features->placeTileOnBoard(caps[0], cap_x, cap_y, cap_rot, cap, *board);
    SPIEL_CHECK_EQ(score(id, shield_side), (2 + 1) * 2);
    SPIEL_CHECK_EQ(score(id, plain_side), 1);
  }
}

// Draws `type` and places it at (tile_x, tile_y) turned `rot`, checking that
// is legal.
void PlaceTile(::Carcassonne* game, int type, int tile_x, int tile_y, int rot) {
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
}

// The same, then plays meeple position `pos`, checking it is legal.
void PlayTurn(::Carcassonne* game, int type, int tile_x, int tile_y, int rot,
              int pos) {
  PlaceTile(game, type, tile_x, tile_y, rot);
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

  // Borrow the id of a tile this test does not place. FieldModule never looks
  // the borrowed id up in full_deck: a neighbour only does that across a side
  // that is not a city, and this tile has none.
  const int id = tile_type_tables.draw_physical_ids_by_type[1][0];
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
  // A big meeple counts as two farmers, one piece.
  fields->placeFarmer(id, four_cities, MEEPLE_POS_INNER_FIELD, 0, /*big=*/true);
  check_scores(12, 0);
  fields->placeFarmer(id, four_cities, MEEPLE_POS_INNER_FIELD, 1);
  check_scores(12, 12);
  SPIEL_CHECK_EQ(fields->farmers_placed[0], 2);
  SPIEL_CHECK_EQ(fields->big_farmers[0], 1);
  SPIEL_CHECK_EQ(fields->big_farmers[1], 0);

  // The inner field is the same field whichever way the board turns.
  const Action inner_action = kMeepleActionOffset + MEEPLE_POS_INNER_FIELD + 1;
  for (int k = 0; k < kNumBoardRotations; ++k) {
    SPIEL_CHECK_EQ(RotateAction(inner_action, k, kNoSideGroups), inner_action);
  }
}

// The meeple moves offered: skip first, then each free spot with a meeple if
// the player has one and with the big meeple if they have it.
void CheckBigMeepleMoves(const ::Carcassonne& core) {
  const MeepleMoves moves = core.getLegalMeepleMoves();
  SPIEL_CHECK_GE(moves.size(), 1);
  SPIEL_CHECK_EQ(moves[0], MEEPLE_POS_SKIP);
  std::vector<int> spots;
  std::vector<int> big_spots;
  for (int i = 1; i < moves.size(); ++i) {
    if (isBuilderPos(moves[i])) continue;  // see CheckBuilders
    (isBigMeeplePos(moves[i]) ? big_spots : spots).push_back(meepleSpot(moves[i]));
  }
  const int player = core.currentPlayer;
  const bool meeple = core.holding_meeples[player] > 0;
  const bool big = core.holding_big_meeples[player] > 0;
  if (!meeple) SPIEL_CHECK_TRUE(spots.empty());
  if (!big) SPIEL_CHECK_TRUE(big_spots.empty());
  if (meeple && big) SPIEL_CHECK_TRUE(spots == big_spots);
}

// Every piece is in hand, on a feature, on a monastery or a farmer: 7 meeples
// and, with the rules, one big meeple each. The observation shows where the
// big ones are.
void CheckBigMeeples(const State& state) {
  const ::Carcassonne& core =
      dynamic_cast<const CarcassonneState&>(state).UnderlyingState();
  const std::vector<float> obs0 = state.ObservationTensor(0);
  const std::vector<float> obs1 = state.ObservationTensor(1);
  int meeples[2] = {core.holding_meeples[0], core.holding_meeples[1]};
  int bigs[2] = {core.holding_big_meeples[0], core.holding_big_meeples[1]};
  std::vector<const Feature*> seen;
  // SPIEL_CHECK_EQ's own locals are called x and y.
  for (int ty = 0; ty < BOARD_SIZE; ++ty) {
    for (int tx = 0; tx < BOARD_SIZE; ++tx) {
      const Placement placement = core.getPlacement(tx, ty);
      if (placement.id == 0) continue;
      const Tile& tile = full_deck[placement.id][placement.rotation];
      for (int side = 0; side < 4; ++side) {
        if (!isFeatureEdge(tile.edge[side])) continue;
        const Feature& feature = core.featureAt(placement.id, side);
        for (Player player = 0; player < kNumPlayers; ++player) {
          const std::vector<float>& obs = player == 0 ? obs0 : obs1;
          const int opponent = 1 - player;
          SPIEL_CHECK_EQ(PlaneValue(obs, kFeatureMyBigMeeplePlane + side, tx, ty),
                         static_cast<float>(feature.big_meeples[player]));
          SPIEL_CHECK_EQ(
              PlaneValue(obs, kFeatureOpponentBigMeeplePlane + side, tx, ty),
              static_cast<float>(feature.big_meeples[opponent]));
        }
        if (std::find(seen.begin(), seen.end(), &feature) != seen.end()) continue;
        seen.push_back(&feature);
        for (Player player = 0; player < kNumPlayers; ++player) {
          SPIEL_CHECK_LE(feature.big_meeples[player], 1);
          const int pieces =
              feature.meeple_count[player] - 2 * feature.big_meeples[player];
          SPIEL_CHECK_GE(pieces, 0);
          meeples[player] += pieces;
          bigs[player] += feature.big_meeples[player];
        }
      }
      if (tile.monastery) {
        const int owner = core.monasteryOwner(tx, ty);
        const int big_owner = core.monasteryBigMeepleOwner(tx, ty);
        if (owner == -1) {
          SPIEL_CHECK_EQ(big_owner, -1);
        } else {
          SPIEL_CHECK_TRUE(big_owner == -1 || big_owner == owner);
          ++(big_owner == owner ? bigs : meeples)[owner];
        }
        const float mine = big_owner == -1 ? 0.0f : (big_owner == 0 ? 1.0f : -1.0f);
        SPIEL_CHECK_EQ(PlaneValue(obs0, kMonasteryBigMeeplePlane, tx, ty), mine);
        SPIEL_CHECK_EQ(PlaneValue(obs1, kMonasteryBigMeeplePlane, tx, ty), -mine);
      }
    }
  }
  for (Player player = 0; player < kNumPlayers; ++player) {
    const int farmers = core.farmersOnBoard(player) - core.bigFarmers(player);
    meeples[player] += farmers;
    bigs[player] += core.bigFarmers(player);
    SPIEL_CHECK_EQ(meeples[player], MEEPLES_PER_PLAYER);
    SPIEL_CHECK_EQ(bigs[player], core.big_meeple_rules ? 1 : 0);
    // The global vector places every piece short of naming its feature:
    // meeples in hand and farmers (the rest are out on features), the big
    // meeple in hand or a farmer (else the big meeple planes show it).
    const int opponent = 1 - player;
    const std::vector<float>& obs = player == 0 ? obs0 : obs1;
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalMyFarmers), farmers / 7.0f));
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalMyBigMeeple),
                   static_cast<float>(core.holding_big_meeples[player]));
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalOpponentBigMeeple),
                   static_cast<float>(core.holding_big_meeples[opponent]));
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalMyBigFarmer),
                   static_cast<float>(core.bigFarmers(player)));
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalOpponentBigFarmer),
                   static_cast<float>(core.bigFarmers(opponent)));
  }
  int fast[2];
  int slow[2];
  core.getPendingScore(fast);
  core.getPendingScoreByResolving(slow);
  SPIEL_CHECK_EQ(fast[0], slow[0]);
  SPIEL_CHECK_EQ(fast[1], slow[1]);
  if (core.current_phase == PHASE_MEEPLE) {
    CheckBigMeepleMoves(core);
  }
}

// Inns & Cathedrals' big meeple: placed instead of a meeple, it counts as two
// for the majority, scores no more, and comes back to its own supply.
void BigMeepleTest() {
  const uint32_t inns = BASE_ONLY | expansionBit(EXP_INNS_CATHEDRALS);
  SPIEL_CHECK_FALSE(::Carcassonne().big_meeple_rules);
  SPIEL_CHECK_FALSE(::Carcassonne(0, START_TILE_ROTATION, inns, /*rules=*/0u)
                        .big_meeple_rules);

  // A road: P0's big meeple against one P1 meeple. The start tile (type 20)
  // at the centre has a road east-west.
  const int c = BOARD_SIZE / 2;
  std::shared_ptr<const Game> inns_game = LoadGame(kInnsCathedralsGame);
  ::Carcassonne game(/*max_turns=*/0, START_TILE_ROTATION, inns);
  SPIEL_CHECK_TRUE(game.big_meeple_rules);
  SPIEL_CHECK_EQ(game.holding_big_meeples[0], 1);
  SPIEL_CHECK_EQ(game.holding_big_meeples[1], 1);
  // P0: a bend west of the start tile turns its road south. Each free spot is
  // offered with either meeple; P0 puts the big one on the road.
  PlaceTile(&game, 22, c - 1, c, 3);
  const MeepleMoves first_moves = game.getLegalMeepleMoves();
  CheckBigMeepleMoves(game);
  SPIEL_CHECK_GT(first_moves.size(), 1);
  SPIEL_CHECK_EQ((first_moves.size() - 1) % 2, 0);
  SPIEL_CHECK_TRUE(std::find(first_moves.begin(), first_moves.end(),
                             MEEPLE_POS_BIG + 1) != first_moves.end());
  game.placeMeeple(MEEPLE_POS_BIG + 1);
  SPIEL_CHECK_EQ(game.holding_meeples[0], MEEPLES_PER_PLAYER);
  SPIEL_CHECK_EQ(game.holding_big_meeples[0], 0);
  const int bend = game.getPlacement(c - 1, c).id;
  SPIEL_CHECK_EQ(game.featureAt(bend, 1).meeple_count[0], 2);
  SPIEL_CHECK_EQ(game.featureAt(bend, 1).big_meeples[0], 1);
  // P1: a bend south of the start tile, its own road; a meeple on it.
  PlayTurn(&game, 22, c, c + 1, 0, 2);
  // P0: a bend joins the two roads. P0's big meeple is out, so only meeple
  // moves are offered.
  PlaceTile(&game, 22, c - 1, c + 1, 2);
  CheckBigMeepleMoves(game);
  for (int move : game.getLegalMeepleMoves()) {
    SPIEL_CHECK_FALSE(isBigMeeplePos(move));
  }
  game.placeMeeple(MEEPLE_POS_SKIP);
  const Feature& road = game.featureAt(bend, 1);
  SPIEL_CHECK_EQ(road.meeple_count[0], 2);
  SPIEL_CHECK_EQ(road.meeple_count[1], 1);
  SPIEL_CHECK_EQ(road.getTileCount(), 4);
  // Two against one: the road is P0's alone.
  int pending[2];
  game.getPendingScore(pending);
  SPIEL_CHECK_EQ(pending[0], 4);
  SPIEL_CHECK_EQ(pending[1], 0);
  {
    CarcassonneState view(inns_game, game);
    const State& state = view;
    const std::vector<float> obs0 = state.ObservationTensor(0);
    const std::vector<float> obs1 = state.ObservationTensor(1);
    // The start tile's west side is on the road, its east side too.
    for (int side : {1, 3}) {
      SPIEL_CHECK_EQ(PlaneValue(obs0, kFeatureMyBigMeeplePlane + side, c, c), 1.0f);
      SPIEL_CHECK_EQ(PlaneValue(obs1, kFeatureOpponentBigMeeplePlane + side, c, c),
                     1.0f);
      SPIEL_CHECK_TRUE(
          Near(PlaneValue(obs0, kFeatureMyMeeplesPlane + side, c, c), 2.0f / 7));
      SPIEL_CHECK_TRUE(Near(
          PlaneValue(obs0, kFeatureOpponentMeeplesPlane + side, c, c), 1.0f / 7));
      SPIEL_CHECK_GT(PlaneValue(obs0, kFeatureSignedScorePlane + side, c, c), 0.0f);
    }
    SPIEL_CHECK_EQ(GlobalValue(obs0, kGlobalMyBigMeeple), 0.0f);
    SPIEL_CHECK_EQ(GlobalValue(obs0, kGlobalOpponentBigMeeple), 1.0f);
    CheckBigMeeples(state);
  }
  // P1: a monastery ends the road east of the start tile; P0 another south of
  // P1's bend, which completes it: six tiles for P0. Each meeple goes back to
  // its own supply.
  PlayTurn(&game, 2, c + 1, c, 1, MEEPLE_POS_SKIP);
  PlayTurn(&game, 2, c, c + 2, 2, MEEPLE_POS_SKIP);
  SPIEL_CHECK_EQ(game.player_scores[0], 6);
  SPIEL_CHECK_EQ(game.player_scores[1], 0);
  for (Player player = 0; player < kNumPlayers; ++player) {
    SPIEL_CHECK_EQ(game.holding_meeples[player], MEEPLES_PER_PLAYER);
    SPIEL_CHECK_EQ(game.holding_big_meeples[player], 1);
  }
  // With only the big meeple left, only it is offered; with nothing, skip.
  game.holding_meeples[1] = 0;
  game.drawTile(21);
  std::array<TileMove, kTileActionCount> tile_moves{};
  int tile_move_count = 0;
  game.getLegalTileMoves(tile_moves.data(), tile_move_count);
  SPIEL_CHECK_GT(tile_move_count, 0);
  game.placeTile(tile_moves[0].x, tile_moves[0].y, tile_moves[0].rot);
  const MeepleMoves big_only = game.getLegalMeepleMoves();
  SPIEL_CHECK_GT(big_only.size(), 1);
  for (int i = 1; i < big_only.size(); ++i) {
    SPIEL_CHECK_TRUE(isBigMeeplePos(big_only[i]));
  }
  game.holding_big_meeples[1] = 0;
  SPIEL_CHECK_EQ(game.getLegalMeepleMoves().size(), 1);

  // A big meeple on a completed monastery goes back to the big supply.
  MonasteryModule monasteries;
  monasteries.active_monasteries.push_back({c, c, 9, 1, true});
  monasteries.active_monasteries.push_back({c + 2, c, 9, 0, false});
  SPIEL_CHECK_EQ(monasteries.bigMeepleOwnerAt(c, c), 1);
  SPIEL_CHECK_EQ(monasteries.bigMeepleOwnerAt(c + 2, c), -1);
  SPIEL_CHECK_EQ(monasteries.ownerAt(c + 2, c), 0);
  int scores[2] = {0, 0};
  int holding[2] = {0, 0};
  int holding_big[2] = {0, 0};
  monasteries.settleCompletedMonasteries(scores, holding, holding_big);
  SPIEL_CHECK_EQ(scores[0], 9);
  SPIEL_CHECK_EQ(scores[1], 9);
  SPIEL_CHECK_EQ(holding[0], 1);
  SPIEL_CHECK_EQ(holding[1], 0);
  SPIEL_CHECK_EQ(holding_big[0], 0);
  SPIEL_CHECK_EQ(holding_big[1], 1);
  SPIEL_CHECK_EQ(monasteries.active_monasteries.size(), 0);

  // Random games: with "on" every piece is accounted for at every state;
  // dealt as "tiles" there is no big meeple.
  std::mt19937 rng(20261007);
  int big_placed = 0;
  int big_returned = 0;
  int big_farmers = 0;
  for (const char* game_string : {kInnsCathedralsGame, kAllExpansionsGame,
                                  "carcassonne(inns_cathedrals=tiles)"}) {
    std::shared_ptr<const Game> random_game = LoadGame(game_string);
    for (int sim = 0; sim < 40; ++sim) {
      std::unique_ptr<State> state = random_game->NewInitialState();
      const ::Carcassonne& core =
          dynamic_cast<const CarcassonneState&>(*state).UnderlyingState();
      while (true) {
        if (!state->IsChanceNode()) {
          CheckBigMeeples(*state);
        }
        if (state->IsTerminal()) break;
        const std::vector<Action> legal = state->LegalActions();
        const Action action =
            state->IsChanceNode()
                ? SampleAction(state->ChanceOutcomes(), rng).first
                : legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)];
        const int held[2] = {core.holding_big_meeples[0],
                             core.holding_big_meeples[1]};
        if (core.current_phase == PHASE_MEEPLE &&
            isBigMeeplePos(DecodeMeepleActionForTest(action))) {
          ++big_placed;
          big_farmers += meepleSpot(DecodeMeepleActionForTest(action)) >=
                         MEEPLE_POS_FIELD;
        }
        state->ApplyAction(action);
        for (Player player = 0; player < kNumPlayers; ++player) {
          big_returned += core.holding_big_meeples[player] > held[player];
        }
      }
    }
  }
  std::cout << "BigMeepleTest: " << big_placed << " big meeples placed ("
            << big_farmers << " as farmers), " << big_returned << " returned"
            << std::endl;
  SPIEL_CHECK_GT(big_placed, 0);
  SPIEL_CHECK_GT(big_farmers, 0);
  SPIEL_CHECK_GT(big_returned, 0);
}

// The inn / cathedral planes match the features on every side: the flag, the
// score without them and the signed score with them. Without the rules no
// feature has either. Counts the sides seen with an inn and with a cathedral.
void CheckInnCathedralPlanes(const State& state, int* inn_sides,
                             int* cathedral_sides) {
  const ::Carcassonne& core =
      dynamic_cast<const CarcassonneState&>(state).UnderlyingState();
  const std::vector<float> obs = state.ObservationTensor(0);
  // SPIEL_CHECK_EQ's own locals are called x and y.
  for (int ty = 0; ty < BOARD_SIZE; ++ty) {
    for (int tx = 0; tx < BOARD_SIZE; ++tx) {
      const Placement placement = core.getPlacement(tx, ty);
      if (placement.id == 0) continue;
      const Tile& tile = full_deck[placement.id][placement.rotation];
      for (int side = 0; side < 4; ++side) {
        const float flag =
            PlaneValue(obs, kFeatureInnCathedralPlane + side, tx, ty);
        if (!isFeatureEdge(tile.edge[side])) {
          SPIEL_CHECK_EQ(flag, 0.0f);
          continue;
        }
        const Feature& feature = core.featureAt(placement.id, side);
        const int inns = feature.inns;
        const int cathedrals = feature.cathedrals;
        if (!core.big_meeple_rules) {
          SPIEL_CHECK_EQ(inns, 0);
          SPIEL_CHECK_EQ(cathedrals, 0);
        }
        // Inns line roads, cathedrals stand in cities.
        if (feature.type == ROAD) SPIEL_CHECK_EQ(cathedrals, 0);
        if (feature.type == CITY) SPIEL_CHECK_EQ(inns, 0);
        *inn_sides += inns > 0;
        *cathedral_sides += cathedrals > 0;
        const bool marked = inns > 0 || cathedrals > 0;
        SPIEL_CHECK_EQ(flag, marked ? 1.0f : 0.0f);
        SPIEL_CHECK_TRUE(
            Near(PlaneValue(obs, kFeatureScorePlane + side, tx, ty),
                 std::min(feature.getBaseScore() / 30.0f, 1.0f)));
        if (!marked) {
          SPIEL_CHECK_EQ(feature.getScore(), feature.getBaseScore());
        } else if (feature.opens > 0) {
          // Left open it would score nothing.
          SPIEL_CHECK_EQ(feature.getScore(), 0);
          SPIEL_CHECK_EQ(
              PlaneValue(obs, kFeatureSignedScorePlane + side, tx, ty), 0.0f);
        }
      }
    }
  }
}

// Inns & Cathedrals: a road with an inn scores 2 a tile once closed, a city
// with a cathedral 3 a tile and a shield; left open at the end, nothing.
// Dealt as "tiles" they score as plain roads and cities.
void InnCathedralTest() {
  const uint32_t inns = BASE_ONLY | expansionBit(EXP_INNS_CATHEDRALS);
  std::shared_ptr<const Game> inns_game = LoadGame(kInnsCathedralsGame);
  // The start tile (type 20) at the centre: a city north, a road east-west.
  const int c = BOARD_SIZE / 2;
  for (bool ruled : {true, false}) {
    const uint32_t rules = ruled ? RULED_EXPANSIONS : 0u;

    // The road through the start tile with an inn on it (type 48, a straight
    // road east of it) and a monastery at each end: four tiles.
    {
      ::Carcassonne game(/*max_turns=*/0, START_TILE_ROTATION, inns, rules);
      SPIEL_CHECK_EQ(game.big_meeple_rules, ruled);
      PlayTurn(&game, 48, c + 1, c, 0, 1);               // P0, on the road
      PlayTurn(&game, 2, c + 2, c, 1, MEEPLE_POS_SKIP);  // P1 ends it east
      const Feature& road = game.featureAt(game.getPlacement(c, c).id, 1);
      SPIEL_CHECK_EQ(static_cast<int>(road.inns), ruled ? 1 : 0);
      SPIEL_CHECK_EQ(road.getBaseScore(), 3);
      SPIEL_CHECK_EQ(road.getScore(), ruled ? 0 : 3);
      int pending[2];
      game.getPendingScore(pending);
      SPIEL_CHECK_EQ(pending[0], ruled ? 0 : 3);
      {
        CarcassonneState view(inns_game, game);
        const State& state = view;
        const std::vector<float> obs = state.ObservationTensor(0);
        for (int side : {1, 3}) {
          SPIEL_CHECK_EQ(PlaneValue(obs, kFeatureInnCathedralPlane + side, c, c),
                         ruled ? 1.0f : 0.0f);
          SPIEL_CHECK_TRUE(
              Near(PlaneValue(obs, kFeatureScorePlane + side, c, c), 3.0f / 30));
          SPIEL_CHECK_TRUE(
              Near(PlaneValue(obs, kFeatureSignedScorePlane + side, c, c),
                   ruled ? 0.0f : 3.0f / 30));
        }
        // The city has no cathedral.
        SPIEL_CHECK_EQ(PlaneValue(obs, kFeatureInnCathedralPlane + 0, c, c), 0.0f);
      }
      PlayTurn(&game, 2, c - 1, c, 3, MEEPLE_POS_SKIP);  // P0 ends it west
      SPIEL_CHECK_EQ(game.player_scores[0], ruled ? 8 : 4);
      SPIEL_CHECK_EQ(game.player_scores[1], 0);
    }
    // The same road left open at the end: two tiles.
    {
      ::Carcassonne game(/*max_turns=*/1, START_TILE_ROTATION, inns, rules);
      PlayTurn(&game, 48, c + 1, c, 0, 1);
      SPIEL_CHECK_EQ(game.current_phase, PHASE_TERMINAL);
      SPIEL_CHECK_EQ(game.player_scores[0], ruled ? 0 : 2);
    }

    // The start tile's city with the cathedral north of it, closed by three
    // CGGG (type 16): five tiles, no shield.
    {
      ::Carcassonne game(/*max_turns=*/0, START_TILE_ROTATION, inns, rules);
      PlayTurn(&game, CATHEDRAL_TYPE, c, c - 1, 0, 0);  // P0, in the city
      const Feature& city = game.featureAt(game.getPlacement(c, c).id, 0);
      SPIEL_CHECK_EQ(static_cast<int>(city.cathedrals), ruled ? 1 : 0);
      SPIEL_CHECK_EQ(static_cast<int>(city.opens), 3);
      SPIEL_CHECK_EQ(city.getBaseScore(), 2);
      SPIEL_CHECK_EQ(city.getScore(), ruled ? 0 : 2);
      {
        CarcassonneState view(inns_game, game);
        const State& state = view;
        const std::vector<float> obs = state.ObservationTensor(0);
        SPIEL_CHECK_EQ(PlaneValue(obs, kFeatureInnCathedralPlane + 0, c, c),
                       ruled ? 1.0f : 0.0f);
        for (int side = 0; side < 4; ++side) {
          SPIEL_CHECK_EQ(
              PlaneValue(obs, kFeatureInnCathedralPlane + side, c, c - 1),
              ruled ? 1.0f : 0.0f);
        }
        // The road has no inn.
        SPIEL_CHECK_EQ(PlaneValue(obs, kFeatureInnCathedralPlane + 1, c, c), 0.0f);
      }
      PlayTurn(&game, 16, c, c - 2, 2, MEEPLE_POS_SKIP);      // P1, north
      PlayTurn(&game, 16, c + 1, c - 1, 3, MEEPLE_POS_SKIP);  // P0, east
      PlayTurn(&game, 16, c - 1, c - 1, 1, MEEPLE_POS_SKIP);  // P1, west: closed
      SPIEL_CHECK_EQ(game.player_scores[0], ruled ? 15 : 10);
      SPIEL_CHECK_EQ(game.player_scores[1], 0);
    }
    // The same city left open at the end: four tiles.
    {
      ::Carcassonne game(/*max_turns=*/3, START_TILE_ROTATION, inns, rules);
      PlayTurn(&game, CATHEDRAL_TYPE, c, c - 1, 0, 0);
      PlayTurn(&game, 16, c, c - 2, 2, MEEPLE_POS_SKIP);
      PlayTurn(&game, 16, c + 1, c - 1, 3, MEEPLE_POS_SKIP);
      SPIEL_CHECK_EQ(game.current_phase, PHASE_TERMINAL);
      SPIEL_CHECK_EQ(game.player_scores[0], ruled ? 0 : 4);
    }
  }

  // Random games: the planes at every state, and no inn or cathedral counts
  // when dealt as "tiles".
  std::mt19937 rng(20261008);
  int inn_sides = 0;
  int cathedral_sides = 0;
  for (const char* game_string : {kInnsCathedralsGame, kAllExpansionsGame,
                                  "carcassonne(inns_cathedrals=tiles)"}) {
    std::shared_ptr<const Game> random_game = LoadGame(game_string);
    for (int sim = 0; sim < 20; ++sim) {
      std::unique_ptr<State> state = random_game->NewInitialState();
      while (true) {
        if (!state->IsChanceNode()) {
          CheckInnCathedralPlanes(*state, &inn_sides, &cathedral_sides);
        }
        if (state->IsTerminal()) break;
        const std::vector<Action> legal = state->LegalActions();
        state->ApplyAction(
            state->IsChanceNode()
                ? SampleAction(state->ChanceOutcomes(), rng).first
                : legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)]);
      }
    }
  }
  std::cout << "InnCathedralTest: " << inn_sides << " inn road sides, "
            << cathedral_sides << " cathedral city sides seen" << std::endl;
  SPIEL_CHECK_GT(inn_sides, 0);
  SPIEL_CHECK_GT(cathedral_sides, 0);
}

// Every builder is in hand or on a feature that holds one of its owner's
// followers; the observation shows where, and the builder is offered exactly on
// the features of the tile just placed that hold such a follower.
void CheckBuilders(const State& state) {
  const ::Carcassonne& core =
      dynamic_cast<const CarcassonneState&>(state).UnderlyingState();
  const std::vector<float> obs0 = state.ObservationTensor(0);
  const std::vector<float> obs1 = state.ObservationTensor(1);
  int builders[2] = {core.holding_builders[0], core.holding_builders[1]};
  std::vector<const Feature*> seen;
  // SPIEL_CHECK_EQ's own locals are called x and y.
  for (int ty = 0; ty < BOARD_SIZE; ++ty) {
    for (int tx = 0; tx < BOARD_SIZE; ++tx) {
      const Placement placement = core.getPlacement(tx, ty);
      if (placement.id == 0) continue;
      const Tile& tile = full_deck[placement.id][placement.rotation];
      for (int side = 0; side < 4; ++side) {
        if (!isFeatureEdge(tile.edge[side])) continue;
        const Feature& feature = core.featureAt(placement.id, side);
        for (Player player = 0; player < kNumPlayers; ++player) {
          const std::vector<float>& obs = player == 0 ? obs0 : obs1;
          SPIEL_CHECK_EQ(PlaneValue(obs, kFeatureMyBuilderPlane + side, tx, ty),
                         static_cast<float>(feature.builders[player]));
          SPIEL_CHECK_EQ(
              PlaneValue(obs, kFeatureOpponentBuilderPlane + side, tx, ty),
              static_cast<float>(feature.builders[1 - player]));
        }
        if (std::find(seen.begin(), seen.end(), &feature) != seen.end()) continue;
        seen.push_back(&feature);
        for (Player player = 0; player < kNumPlayers; ++player) {
          const int on_feature = feature.builders[player];
          SPIEL_CHECK_LE(on_feature, 1);
          // It stays only with one of its owner's followers.
          if (on_feature > 0) {
            SPIEL_CHECK_GT(static_cast<int>(feature.meeple_count[player]), 0);
          }
          builders[player] += on_feature;
        }
      }
    }
  }
  for (Player player = 0; player < kNumPlayers; ++player) {
    SPIEL_CHECK_EQ(builders[player], core.builder_rules ? 1 : 0);
    const std::vector<float>& obs = player == 0 ? obs0 : obs1;
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalMyBuilder),
                   static_cast<float>(core.holding_builders[player]));
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalOpponentBuilder),
                   static_cast<float>(core.holding_builders[1 - player]));
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalBuilderExtraTile),
                   core.builder_extra_tile ? 1.0f : 0.0f);
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalBuilderSecondTile),
                   core.builder_second_tile ? 1.0f : 0.0f);
  }
  SPIEL_CHECK_FALSE(core.builder_extra_tile && core.builder_second_tile);
  if (core.current_phase != PHASE_MEEPLE) return;
  const int player = core.currentPlayer;
  const Placement last = core.getPlacement(core.last_x, core.last_y);
  const Tile& last_tile = full_deck[last.id][last.rotation];
  std::vector<int> expected;
  std::vector<const Feature*> last_features;
  for (int side = 0; side < 4; ++side) {
    if (!isFeatureEdge(last_tile.edge[side])) continue;
    const Feature& feature = core.featureAt(last.id, side);
    if (std::find(last_features.begin(), last_features.end(), &feature) !=
        last_features.end()) {
      continue;  // named by its lowest side
    }
    last_features.push_back(&feature);
    if (core.holding_builders[player] > 0 && feature.meeple_count[player] > 0) {
      expected.push_back(MEEPLE_POS_BUILDER + side);
    }
  }
  std::vector<int> offered;
  for (int move : core.getLegalMeepleMoves()) {
    if (isBuilderPos(move)) offered.push_back(move);
  }
  SPIEL_CHECK_TRUE(offered == expected);
}

// Traders & Builders' builder: it joins one of its owner's followers, and each
// tile its owner adds to that road or city brings one more tile after it, never
// a third. It does not count for the majority and comes back with the
// followers.
void BuilderTest() {
  const uint32_t traders = BASE_ONLY | expansionBit(EXP_TRADERS_BUILDERS);
  SPIEL_CHECK_FALSE(::Carcassonne().builder_rules);
  SPIEL_CHECK_FALSE(::Carcassonne(0, START_TILE_ROTATION, traders, /*rules=*/0u)
                        .builder_rules);
  auto builder_moves = [](const ::Carcassonne& core) {
    std::vector<int> moves;
    for (int move : core.getLegalMeepleMoves()) {
      if (isBuilderPos(move)) moves.push_back(move);
    }
    return moves;
  };
  auto play_meeple = [](::Carcassonne* core, int pos) {
    const MeepleMoves moves = core->getLegalMeepleMoves();
    SPIEL_CHECK_TRUE(std::find(moves.begin(), moves.end(), pos) != moves.end());
    core->placeMeeple(pos);
  };

  // The start tile (type 20) at the centre has a road east-west, and so has
  // type 21 turned once.
  const int c = BOARD_SIZE / 2;
  std::shared_ptr<const Game> traders_game = LoadGame(kTradersBuildersGame);
  ::Carcassonne game(/*max_turns=*/0, START_TILE_ROTATION, traders);
  SPIEL_CHECK_TRUE(game.builder_rules);
  SPIEL_CHECK_EQ(game.holding_builders[0], 1);
  SPIEL_CHECK_EQ(game.holding_builders[1], 1);
  // P0: a meeple on the road. It held no follower of P0's yet: no builder.
  PlaceTile(&game, 21, c + 1, c, 1);
  SPIEL_CHECK_TRUE(builder_moves(game).empty());
  play_meeple(&game, 1);
  // P1 closes the start tile's city.
  PlayTurn(&game, 16, c, c - 1, 2, MEEPLE_POS_SKIP);
  // P0 extends the road: the builder can go on it, a meeple cannot.
  PlaceTile(&game, 21, c + 2, c, 1);
  SPIEL_CHECK_FALSE(game.builder_extra_tile);
  SPIEL_CHECK_TRUE((builder_moves(game) == std::vector<int>{MEEPLE_POS_BUILDER + 1}));
  {
    const MeepleMoves moves = game.getLegalMeepleMoves();
    SPIEL_CHECK_TRUE(std::find(moves.begin(), moves.end(), 1) == moves.end());
  }
  play_meeple(&game, MEEPLE_POS_BUILDER + 1);
  const int start_id = game.getPlacement(c, c).id;
  SPIEL_CHECK_EQ(game.holding_builders[0], 0);
  SPIEL_CHECK_EQ(static_cast<int>(game.featureAt(start_id, 1).builders[0]), 1);
  // It is no follower: the strength is still the one meeple's.
  SPIEL_CHECK_EQ(static_cast<int>(game.featureAt(start_id, 1).meeple_count[0]), 1);
  // Placed this turn, it gave no extra tile.
  SPIEL_CHECK_EQ(game.currentPlayer, 1);
  // P1 extends P0's road: only its owner's tiles count.
  PlayTurn(&game, 21, c - 1, c, 1, MEEPLE_POS_SKIP);
  SPIEL_CHECK_EQ(game.currentPlayer, 0);
  // P0 extends it: one more tile.
  PlaceTile(&game, 21, c + 3, c, 1);
  SPIEL_CHECK_TRUE(game.builder_extra_tile);
  play_meeple(&game, MEEPLE_POS_SKIP);
  SPIEL_CHECK_EQ(game.current_phase, PHASE_CHANCE);
  SPIEL_CHECK_EQ(game.currentPlayer, 0);
  SPIEL_CHECK_TRUE(game.builder_second_tile);
  {
    CarcassonneState view(traders_game, game);
    const State& state = view;
    const std::vector<float> obs = state.ObservationTensor(0);
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalBuilderSecondTile), 1.0f);
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalBuilderExtraTile), 0.0f);
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalMyBuilder), 0.0f);
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalOpponentBuilder), 1.0f);
    for (int side : {1, 3}) {
      SPIEL_CHECK_EQ(PlaneValue(obs, kFeatureMyBuilderPlane + side, c, c), 1.0f);
      SPIEL_CHECK_EQ(PlaneValue(obs, kFeatureOpponentBuilderPlane + side, c, c),
                     0.0f);
    }
    SPIEL_CHECK_TRUE(state.ToString().find("builder_second_tile") !=
                     std::string::npos);
    CheckBuilders(state);
  }
  // The second tile extends it again, but there is no third.
  PlaceTile(&game, 21, c + 4, c, 1);
  SPIEL_CHECK_FALSE(game.builder_extra_tile);
  play_meeple(&game, MEEPLE_POS_SKIP);
  SPIEL_CHECK_EQ(game.currentPlayer, 1);
  SPIEL_CHECK_FALSE(game.builder_second_tile);
  // P1 ends the road east; P0 ends it west and completes it: eight tiles, the
  // builder adds nothing, and it comes back with the meeple. The tile still
  // extended the road, so P0 places one more.
  PlayTurn(&game, 2, c + 5, c, 1, MEEPLE_POS_SKIP);
  PlaceTile(&game, 2, c - 2, c, 3);
  SPIEL_CHECK_TRUE(game.builder_extra_tile);
  play_meeple(&game, MEEPLE_POS_SKIP);
  SPIEL_CHECK_EQ(game.player_scores[0], 8);
  SPIEL_CHECK_EQ(game.player_scores[1], 0);
  SPIEL_CHECK_EQ(game.holding_meeples[0], MEEPLES_PER_PLAYER);
  SPIEL_CHECK_EQ(game.holding_builders[0], 1);
  SPIEL_CHECK_EQ(game.currentPlayer, 0);
  SPIEL_CHECK_TRUE(game.builder_second_tile);

  // Random games: every state checked; a meeple move keeps the same player
  // exactly when its tile extended their builder and the game goes on.
  // Dealt as "tiles" there is no builder.
  std::mt19937 rng(20261009);
  int builders_placed = 0;
  int builders_returned = 0;
  int double_turns = 0;
  for (const char* game_string : {kTradersBuildersGame, kAllExpansionsGame,
                                  "carcassonne(traders_builders=tiles)"}) {
    std::shared_ptr<const Game> random_game = LoadGame(game_string);
    for (int sim = 0; sim < 20; ++sim) {
      std::unique_ptr<State> state = random_game->NewInitialState();
      const ::Carcassonne& core =
          dynamic_cast<const CarcassonneState&>(*state).UnderlyingState();
      while (true) {
        if (!state->IsChanceNode()) {
          CheckBuilders(*state);
        }
        if (state->IsTerminal()) break;
        const std::vector<Action> legal = state->LegalActions();
        const Action action =
            state->IsChanceNode()
                ? SampleAction(state->ChanceOutcomes(), rng).first
                : legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)];
        const bool meeple_phase = core.current_phase == PHASE_MEEPLE;
        const bool extra_tile = core.builder_extra_tile;
        const int mover = core.currentPlayer;
        const int held[2] = {core.holding_builders[0], core.holding_builders[1]};
        if (meeple_phase && isBuilderPos(DecodeMeepleActionForTest(action))) {
          ++builders_placed;
        }
        state->ApplyAction(action);
        for (Player player = 0; player < kNumPlayers; ++player) {
          builders_returned += core.holding_builders[player] > held[player];
        }
        if (meeple_phase && !state->IsTerminal()) {
          SPIEL_CHECK_EQ(core.currentPlayer, extra_tile ? mover : 1 - mover);
          SPIEL_CHECK_EQ(core.builder_second_tile, extra_tile);
          double_turns += extra_tile;
        }
      }
    }
  }
  std::cout << "BuilderTest: " << builders_placed << " builders placed, "
            << builders_returned << " returned, " << double_turns
            << " double turns" << std::endl;
  SPIEL_CHECK_GT(builders_placed, 0);
  SPIEL_CHECK_GT(builders_returned, 0);
  SPIEL_CHECK_GT(double_turns, 0);
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
  SPIEL_CHECK_EQ(game->MaxGameLength(), (tileCountIn(BASE_ONLY) - 1) + 20);

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
// the same count for farmer moves (half-edge shifts) to `renamed_farmer_moves`
// and the legal big meeple moves rotated to `big_meeple_moves`. The big meeple
// names its spots like the meeple, so both count the same way.
int CheckRotatedTwin(absl::Span<const Action> history, int k,
                     std::mt19937* rng, int* renamed_farmer_moves,
                     int* big_meeple_moves,
                     const std::string& game_string = "carcassonne") {
  std::shared_ptr<const Game> game = LoadGame(game_string);
  const auto& carcassonne_game = dynamic_cast<const CarcassonneGame&>(*game);
  const uint32_t expansions = carcassonne_game.Expansions();
  const uint32_t rules = carcassonne_game.Rules();
  CarcassonneState state(game, /*max_turns=*/0, expansions, rules);
  CarcassonneState twin(game, ::Carcassonne(/*max_turns=*/0,
                                            /*start_rotation=*/k, expansions,
                                            rules));
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
      if (legal_action < kMeepleActionOffset) continue;
      const int full_pos = DecodeMeepleActionForTest(legal_action);
      const int rotated_pos = DecodeMeepleActionForTest(rotated_action);
      // A big meeple or builder move stays one, on the rotated spot.
      SPIEL_CHECK_EQ(isBigMeeplePos(rotated_pos), isBigMeeplePos(full_pos));
      SPIEL_CHECK_EQ(isBuilderPos(rotated_pos), isBuilderPos(full_pos));
      if (isBigMeeplePos(full_pos)) ++*big_meeple_moves;
      const int pos = meepleSpot(full_pos);
      const int rotated_spot = meepleSpot(rotated_pos);
      if (pos >= 0 && pos < 4 && rotated_spot != (pos + k) % 4) {
        ++renamed_meeple_moves;
      }
      const int half_edge = pos - MEEPLE_POS_FIELD;
      if (pos >= MEEPLE_POS_FIELD && pos < MEEPLE_POS_INNER_FIELD &&
          rotated_spot - MEEPLE_POS_FIELD !=
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
    SPIEL_CHECK_EQ(core.holding_big_meeples[player],
                   twin_core.holding_big_meeples[player]);
  }
  return renamed_meeple_moves;
}

void RotationEquivarianceTest() {
  std::mt19937 rng(20260915);
  int renamed_meeple_moves = 0;
  int renamed_farmer_moves = 0;
  int big_meeple_moves = 0;
  for (int k = 1; k < kNumBoardRotations; ++k) {
    renamed_meeple_moves +=
        CheckRotatedTwin(kLastUnplaceableTileHistory, k, &rng,
                         &renamed_farmer_moves, &big_meeple_moves);
    for (int game = 0; game < 20; ++game) {
      renamed_meeple_moves += CheckRotatedTwin({}, k, &rng, &renamed_farmer_moves,
                                               &big_meeple_moves);
    }
    for (int game = 0; game < 3; ++game) {
      for (const char* game_string : {kAllExpansionsGame, kRiverGame,
                                      kInnsCathedralsGame,
                                      kTradersBuildersGame}) {
        renamed_meeple_moves +=
            CheckRotatedTwin({}, k, &rng, &renamed_farmer_moves,
                             &big_meeple_moves, game_string);
      }
    }
  }
  std::cout << "RotationEquivarianceTest: " << renamed_meeple_moves
            << " meeple moves renamed beyond a side shift, "
            << renamed_farmer_moves
            << " farmer moves beyond a half-edge shift, " << big_meeple_moves
            << " big meeple moves" << std::endl;
  SPIEL_CHECK_GT(renamed_meeple_moves, 0);
  SPIEL_CHECK_GT(renamed_farmer_moves, 0);
  SPIEL_CHECK_GT(big_meeple_moves, 0);
}

// Each expansion option deals that box's tiles and no others, "on" with its
// rules; the shapes stay the same whatever is on.
void ExpansionOptionsTest() {
  std::shared_ptr<const Game> base = LoadGame("carcassonne");
  SPIEL_CHECK_EQ(dynamic_cast<const CarcassonneGame&>(*base).Expansions(),
                 BASE_ONLY);
  SPIEL_CHECK_EQ(dynamic_cast<const CarcassonneGame&>(*base).Rules(), 0u);
  // Every expansion has a mode beyond off; the river only "on".
  std::vector<std::pair<int, std::string>> options = {{EXP_BASE, ""}};
  for (int expansion = 1; expansion < EXPANSION_COUNT; ++expansion) {
    const uint32_t bit = expansionBit(static_cast<Expansion>(expansion));
    if (expansion != EXP_RIVER) options.push_back({expansion, "tiles"});
    if (bit & RULED_EXPANSIONS) options.push_back({expansion, "on"});
  }
  SPIEL_CHECK_EQ(static_cast<int>(options.size()), 1 + (EXPANSION_COUNT - 1) + 2);
  for (const auto& [expansion, mode] : options) {
    uint32_t mask = BASE_ONLY;
    std::string game_string = "carcassonne";
    const bool on = mode == "on";
    if (expansion != EXP_BASE) {
      mask |= expansionBit(static_cast<Expansion>(expansion));
      game_string = absl::StrCat("carcassonne(", EXPANSION_NAMES[expansion],
                                 "=", mode, ")");
    }
    std::shared_ptr<const Game> game = LoadGame(game_string);
    SPIEL_CHECK_EQ(dynamic_cast<const CarcassonneGame&>(*game).Expansions(),
                   mask);
    SPIEL_CHECK_EQ(dynamic_cast<const CarcassonneGame&>(*game).Rules(),
                   on ? mask & ~BASE_ONLY : 0u);
    SPIEL_CHECK_EQ(game->ObservationTensorShape(),
                   base->ObservationTensorShape());
    SPIEL_CHECK_EQ(game->NumDistinctActions(), base->NumDistinctActions());
    SPIEL_CHECK_EQ(game->MaxChanceOutcomes(), CANONICAL_TILE_TYPE_COUNT);
    SPIEL_CHECK_EQ(game->MaxChanceNodesInHistory(), deckSizeOf(mask) - 1);

    std::unique_ptr<State> state = game->NewInitialState();
    const ::Carcassonne& core =
        dynamic_cast<const CarcassonneState&>(*state).UnderlyingState();
    SPIEL_CHECK_EQ(core.getDeckSize(), deckSizeOf(mask));
    SPIEL_CHECK_EQ(core.getTotalRemaining(), deckSizeOf(mask) - 1);
    double total = 0.0;
    for (const auto& [action, probability] : state->ChanceOutcomes()) {
      const TileBlueprint& blueprint = all_tiles[action];  // type action + 1
      SPIEL_CHECK_TRUE(mask & expansionBit(blueprint.expansion));
      total += probability;
    }
    SPIEL_CHECK_TRUE(Near(total, 1.0));
    // Remaining tiles count against this game's deck.
    const std::vector<float> obs = state->ObservationTensor(0);
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalRemainingTiles),
                          (deckSizeOf(mask) - 1.0f) / deckSizeOf(mask)));
    // Each expansion's mode: off (0, 0), tiles (1, 0) or on (1, 1).
    for (int other = 1; other < EXPANSION_COUNT; ++other) {
      const int cell = kGlobalExpansionModes + 2 * (other - 1);
      const bool dealt = other == expansion;
      SPIEL_CHECK_EQ(GlobalValue(obs, cell), dealt ? 1.0f : 0.0f);
      SPIEL_CHECK_EQ(GlobalValue(obs, cell + 1), dealt && on ? 1.0f : 0.0f);
    }
    // The spring is on the board.
    SPIEL_CHECK_TRUE(Near(GlobalValue(obs, kGlobalRiverTiles),
                          expansion == EXP_RIVER ? 1.0f / 12 : 0.0f));
    // Inns & Cathedrals "on" gives each player a big meeple.
    const bool big = expansion == EXP_INNS_CATHEDRALS && on;
    SPIEL_CHECK_EQ(core.big_meeple_rules, big);
    for (Player player = 0; player < kNumPlayers; ++player) {
      SPIEL_CHECK_EQ(core.holding_big_meeples[player], big ? 1 : 0);
    }
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalMyBigMeeple), big ? 1.0f : 0.0f);
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalOpponentBigMeeple),
                   big ? 1.0f : 0.0f);
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalMyBigFarmer), 0.0f);
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalOpponentBigFarmer), 0.0f);
    // Traders & Builders "on" gives each player a builder.
    const bool builder = expansion == EXP_TRADERS_BUILDERS && on;
    SPIEL_CHECK_EQ(core.builder_rules, builder);
    for (Player player = 0; player < kNumPlayers; ++player) {
      SPIEL_CHECK_EQ(core.holding_builders[player], builder ? 1 : 0);
    }
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalMyBuilder), builder ? 1.0f : 0.0f);
    SPIEL_CHECK_EQ(GlobalValue(obs, kGlobalOpponentBuilder),
                   builder ? 1.0f : 0.0f);
  }
}

// A river meets only river, splits fields like a road and takes no meeple.
// The table may hold no river tiles yet, so these are built here and driven
// through the modules directly.
void RiverTileTest() {
  SPIEL_CHECK_FALSE(isFeatureEdge(RIVER));
  SPIEL_CHECK_FALSE(isFeatureEdge(GRASS));
  SPIEL_CHECK_TRUE(isFeatureEdge(CITY));
  SPIEL_CHECK_TRUE(isFeatureEdge(ROAD));

  // A road crossing a north-south river on a bridge: four corner fields.
  const Tile bridge(RIVER, ROAD, RIVER, ROAD, 0, 1, 0, 1,
                    {{0, 1, 1, 2, 2, 3, 3, 0}, 4, {}});
  const Tile grass_south(GRASS, GRASS, GRASS, GRASS, 0, 1, 2, 3,
                         {{0, 0, 0, 0, 0, 0, 0, 0}, 1, {}});
  const Tile river_south(GRASS, GRASS, RIVER, GRASS, 0, 1, 2, 3,
                         {{0, 0, 0, 0, 0, 0, 0, 0}, 1, {}});
  SPIEL_CHECK_TRUE(tile_check::CheckTile(bridge).empty());
  SPIEL_CHECK_TRUE(tile_check::CheckTile(river_south).empty());

  // Borrowed ids of tiles this test does not otherwise place. Nothing here
  // looks them up in full_deck: FieldModule is not used.
  const int bridge_id = tile_type_tables.draw_physical_ids_by_type[1][0];
  const int road_id = tile_type_tables.draw_physical_ids_by_type[1][1];
  auto board = std::make_unique<BoardModule>();
  auto features = std::make_unique<FeatureModule>();
  const int c = BOARD_SIZE / 2;
  SPIEL_CHECK_TRUE(board->canPlaceTileAt(c, c, bridge));
  board->placeTileOnBoard(bridge_id, c, c, 0, bridge);
  features->placeTileOnBoard(bridge_id, c, c, 0, bridge, *board);

  // North of the bridge: only a tile with river on its south side fits.
  SPIEL_CHECK_FALSE(board->canPlaceTileAt(c, c - 1, grass_south));
  SPIEL_CHECK_TRUE(board->canPlaceTileAt(c, c - 1, river_south));
  // East: the road must meet a road; turned so a river faces it, it does not.
  SPIEL_CHECK_TRUE(board->canPlaceTileAt(c + 1, c, bridge));
  SPIEL_CHECK_FALSE(board->canPlaceTileAt(c + 1, c, bridge.rotate()));

  // The only meeple spot is the road, named by its lowest side, east.
  MeepleMoves moves;
  features->getLegalMeepleMoves(moves, c, c, *board, bridge);
  SPIEL_CHECK_EQ(moves.size(), 1);
  SPIEL_CHECK_EQ(moves[0], 1);

  // A straight road continuing east joins the bridge's road; the river is no
  // feature, so nothing joins across it.
  const Tile road_ew = Tile(ROAD, GRASS, ROAD, GRASS, 0, 1, 0, 2,
                            {{0, 1, 1, 1, 1, 0, 0, 0}, 2, {}})
                           .rotate();
  SPIEL_CHECK_TRUE(board->canPlaceTileAt(c + 1, c, road_ew));
  board->placeTileOnBoard(road_id, c + 1, c, 1, road_ew);
  features->placeTileOnBoard(road_id, c + 1, c, 1, road_ew, *board);
  const Feature& road = features->featureMap.getSetData(
      features->edgeIndex(bridge_id, 1));
  SPIEL_CHECK_EQ(road.type, ROAD);
  SPIEL_CHECK_EQ(road.getTileCount(), 2);
  SPIEL_CHECK_EQ(road.opens, 2);
  int scores[2] = {0, 0};
  features->resolveEndGameScore(scores);
  SPIEL_CHECK_EQ(scores[0], 0);
  SPIEL_CHECK_EQ(scores[1], 0);
}

bool IsRiverType(int type) {
  return all_tiles[type - 1].expansion == EXP_RIVER;
}

std::vector<TileMove> LegalTileMoves(const ::Carcassonne& game) {
  std::array<TileMove, kTileActionCount> moves{};
  int count = 0;
  game.getLegalTileMoves(moves.data(), count);
  return std::vector<TileMove>(moves.begin(), moves.begin() + count);
}

// The turn the river makes on `tile` (as turned) flowing in from where it
// ends now, counted as Carcassonne::river_last_turn counts it; 0 straight on.
int RiverTurn(const ::Carcassonne& game, const Tile& tile) {
  for (int side = 0; side < 4; ++side) {
    if (tile.edge[side] == RIVER && side != (game.river_heading + 2) % 4) {
      return (side - game.river_heading + 4) % 4;
    }
  }
  return 0;
}

// Draws river tile `type` and returns the turn the river makes on each legal
// placement, by rotation (-1 where it may not go), checking they are all at
// the river's end.
std::array<int, 4> DrawRiverTile(::Carcassonne* game, int type) {
  const int end_x = game->river_x;
  const int end_y = game->river_y;
  game->drawTile(type);
  SPIEL_CHECK_EQ(game->current_phase, PHASE_TILE);
  std::array<int, 4> turns = {-1, -1, -1, -1};
  for (const TileMove& move : LegalTileMoves(*game)) {
    SPIEL_CHECK_EQ(static_cast<int>(move.x), end_x);
    SPIEL_CHECK_EQ(static_cast<int>(move.y), end_y);
    turns[move.rot] =
        RiverTurn(*game, full_deck[game->current_tile_in_hand][move.rot]);
  }
  return turns;
}

// With the river on, the spring starts the game, the river tiles are drawn
// first and the lake last, each continuing the river without turning the same
// way twice in a row.
void RiverRulesTest() {
  const int c = BOARD_SIZE / 2;
  const int dx[4] = {0, 1, 0, -1};
  const int dy[4] = {-1, 0, 1, 0};

  // The spring flows south from the centre; the base start tile is left out.
  std::shared_ptr<const Game> game = LoadGame(kRiverGame);
  std::unique_ptr<State> initial = game->NewInitialState();
  const ::Carcassonne& start =
      dynamic_cast<const CarcassonneState&>(*initial).UnderlyingState();
  SPIEL_CHECK_TRUE(start.river_rules);
  SPIEL_CHECK_EQ(PHYSICAL_TO_CANONICAL_TYPE[start.getPlacement(c, c).id],
                 RIVER_SPRING_TYPE);
  SPIEL_CHECK_EQ(start.river_heading, 2);
  SPIEL_CHECK_EQ(start.river_x, c);
  SPIEL_CHECK_EQ(start.river_y, c + 1);
  SPIEL_CHECK_EQ(start.getDeckSize(), 72 + 12 - 1);
  SPIEL_CHECK_EQ(start.getDeckSize(), deckSizeOf(expansionBit(EXP_RIVER)));
  SPIEL_CHECK_EQ(start.getTotalRemaining(), start.getDeckSize() - 1);
  SPIEL_CHECK_EQ(start.getRemainingTypeCount(START_TILE_TYPE),
                 all_tiles[START_TILE_TYPE - 1].count - 1);
  SPIEL_CHECK_EQ(game->MaxChanceNodesInHistory(), start.getDeckSize() - 1);
  // First come the river tiles other than the spring and the lake.
  const ActionsAndProbs first_draws = initial->ChanceOutcomes();
  SPIEL_CHECK_EQ(first_draws.size(), 8);
  for (const auto& [action, probability] : first_draws) {
    const int type = action + 1;
    SPIEL_CHECK_TRUE(IsRiverType(type));
    SPIEL_CHECK_NE(type, RIVER_LAKE_TYPE);
    SPIEL_CHECK_TRUE(Near(probability, all_tiles[type - 1].count / 10.0));
  }
  SPIEL_CHECK_FALSE(::Carcassonne().river_rules);

  // Bends: types 27, 30 (two of them) and 32. Turned rot 2 a type 30 flows in
  // from the north and out west (clockwise), turned rot 3 out east.
  ::Carcassonne river(/*max_turns=*/0, START_TILE_ROTATION,
                      BASE_ONLY | expansionBit(EXP_RIVER));
  SPIEL_CHECK_TRUE((DrawRiverTile(&river, 30) ==
                    std::array<int, 4>{-1, -1, 1, 3}));
  river.placeTile(c, c + 1, 2);
  river.placeMeeple(MEEPLE_POS_SKIP);
  SPIEL_CHECK_EQ(river.river_last_turn, 1);
  SPIEL_CHECK_EQ(river.river_heading, 3);
  // Type 27 could take it north (rot 1, clockwise again) or south (rot 2);
  // only south is left.
  SPIEL_CHECK_TRUE((DrawRiverTile(&river, 27) ==
                    std::array<int, 4>{-1, -1, 3, -1}));
  river.placeTile(c - 1, c + 1, 2);
  river.placeMeeple(MEEPLE_POS_SKIP);
  SPIEL_CHECK_EQ(river.river_last_turn, 3);
  SPIEL_CHECK_EQ(river.river_heading, 2);
  // A straight goes either way round and does not reset the last turn...
  SPIEL_CHECK_TRUE((DrawRiverTile(&river, 34) ==
                    std::array<int, 4>{0, -1, 0, -1}));
  river.placeTile(c - 1, c + 2, 0);
  river.placeMeeple(MEEPLE_POS_SKIP);
  SPIEL_CHECK_EQ(river.river_last_turn, 3);
  // ... so the next bend has to turn clockwise.
  SPIEL_CHECK_TRUE((DrawRiverTile(&river, 30) ==
                    std::array<int, 4>{-1, -1, 1, -1}));

  // Random games, the river alone and with every other expansion's tiles.
  std::mt19937 rng(20261006);
  int same_way_bends = 0;  // placements only the turn rule ruled out
  int discarded = 0;       // river tiles with nowhere to go
  int placed = 0;          // river tiles placed, the springs included
  int games = 0;
  for (const char* game_string : {kRiverGame, kAllExpansionsGame}) {
    std::shared_ptr<const Game> river_game = LoadGame(game_string);
    for (int sim = 0; sim < 100; ++sim, ++games) {
      std::unique_ptr<State> state = river_game->NewInitialState();
      const ::Carcassonne& core =
          dynamic_cast<const CarcassonneState&>(*state).UnderlyingState();
      int river_placed = 1;  // the spring
      while (!state->IsTerminal()) {
        if (state->IsChanceNode()) {
          int river_left = 0;  // other than the lake
          for (int type = 1; type <= CANONICAL_TILE_TYPE_COUNT; ++type) {
            if (IsRiverType(type) && type != RIVER_LAKE_TYPE) {
              river_left += core.getRemainingTypeCount(type);
            }
          }
          const bool lake_left =
              core.getRemainingTypeCount(RIVER_LAKE_TYPE) > 0;
          const ActionsAndProbs outcomes = state->ChanceOutcomes();
          for (const auto& [action, probability] : outcomes) {
            const int type = action + 1;
            if (river_left > 0) {
              SPIEL_CHECK_TRUE(IsRiverType(type));
              SPIEL_CHECK_NE(type, RIVER_LAKE_TYPE);
            } else if (lake_left) {
              SPIEL_CHECK_EQ(type, RIVER_LAKE_TYPE);
            } else {
              SPIEL_CHECK_FALSE(IsRiverType(type));
            }
          }
          const Action draw = SampleAction(outcomes, rng).first;
          state->ApplyAction(draw);
          if (IsRiverType(draw + 1) && core.current_phase != PHASE_TILE) {
            ++discarded;
          }
          continue;
        }
        const std::vector<Action> legal = state->LegalActions();
        const Action action =
            legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)];
        if (core.current_phase == PHASE_TILE &&
            IsRiverType(core.currentTileType())) {
          // The observation counts the river tiles on the board.
          const std::vector<float> obs = state->ObservationTensor(0);
          SPIEL_CHECK_EQ(core.river_tiles_placed, river_placed);
          SPIEL_CHECK_TRUE(
              Near(GlobalValue(obs, kGlobalRiverTiles), river_placed / 12.0f));
          for (int expansion = 1; expansion < EXPANSION_COUNT; ++expansion) {
            const int cell = kGlobalExpansionModes + 2 * (expansion - 1);
            const bool dealt =
                (core.expansions & expansionBit(static_cast<Expansion>(expansion))) != 0;
            SPIEL_CHECK_EQ(GlobalValue(obs, cell), dealt ? 1.0f : 0.0f);
            // Both games deal the river, Inns & Cathedrals and Traders &
            // Builders, if at all, "on".
            const bool ruled = dealt && (expansion == EXP_RIVER ||
                                         expansion == EXP_INNS_CATHEDRALS ||
                                         expansion == EXP_TRADERS_BUILDERS);
            SPIEL_CHECK_EQ(GlobalValue(obs, cell + 1), ruled ? 1.0f : 0.0f);
          }
          const int in_side = (core.river_heading + 2) % 4;
          for (Action legal_action : legal) {
            int tile_x, tile_y, rot;
            DecodeTileActionForTest(legal_action, &tile_x, &tile_y, &rot);
            SPIEL_CHECK_EQ(tile_x, core.river_x);
            SPIEL_CHECK_EQ(tile_y, core.river_y);
            const int turn =
                RiverTurn(core, full_deck[core.current_tile_in_hand][rot]);
            SPIEL_CHECK_TRUE(turn == 0 || turn != core.river_last_turn);
          }
          for (int rot = 0; rot < 4; ++rot) {
            const Tile& tile = full_deck[core.current_tile_in_hand][rot];
            if (tile.edge[in_side] == RIVER && core.river_last_turn != 0 &&
                RiverTurn(core, tile) == core.river_last_turn) {
              ++same_way_bends;
            }
          }
          ++river_placed;
        }
        state->ApplyAction(action);
      }

      // The river is one path from the spring to the lake, through every
      // river tile placed.
      SPIEL_CHECK_EQ(core.river_x, -1);
      int walk_x = c;
      int walk_y = c;
      int heading = -1;
      int walked = 0;
      int last_type = 0;
      while (true) {
        const Placement placement = core.getPlacement(walk_x, walk_y);
        SPIEL_CHECK_NE(static_cast<int>(placement.id), 0);
        ++walked;
        last_type = PHYSICAL_TO_CANONICAL_TYPE[placement.id];
        const Tile& tile = full_deck[placement.id][placement.rotation];
        int out = -1;
        for (int side = 0; side < 4; ++side) {
          if (tile.edge[side] == RIVER &&
              (heading < 0 || side != (heading + 2) % 4)) {
            out = side;
          }
        }
        if (out < 0) break;
        heading = out;
        walk_x += dx[out];
        walk_y += dy[out];
      }
      SPIEL_CHECK_EQ(last_type, RIVER_LAKE_TYPE);
      SPIEL_CHECK_EQ(walked, river_placed);
      SPIEL_CHECK_EQ(core.river_tiles_placed, river_placed);
      SPIEL_CHECK_TRUE(Near(GlobalValue(state->ObservationTensor(0),
                                        kGlobalRiverTiles),
                            river_placed / 12.0f));
      placed += river_placed;
    }
  }
  std::cout << "RiverRulesTest: " << games << " games, " << same_way_bends
            << " same-way bends ruled out, " << discarded
            << " river tiles discarded" << std::endl;
  SPIEL_CHECK_GT(same_way_bends, 0);
  SPIEL_CHECK_EQ(placed + discarded,
                 games * tileCountIn(expansionBit(EXP_RIVER)));
}

// Random games dealing every expansion that is in the table: legal play, the
// right deck, and river sides never shown as features.
void ExpansionGamesTest() {
  std::mt19937 rng(20261004);
  std::shared_ptr<const Game> game = LoadGame(kAllExpansionsGame);
  testing::RandomSimTest(*game, 10);
  int river_sides = 0;
  int shield_sides = 0;
  for (int sim = 0; sim < 20; ++sim) {
    std::unique_ptr<State> state = game->NewInitialState();
    int tiles_drawn = 1;  // the start tile
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        ++tiles_drawn;
      } else {
        const ::Carcassonne& core =
            dynamic_cast<const CarcassonneState&>(*state).UnderlyingState();
        const std::vector<float> obs = state->ObservationTensor(0);
        for (int y = 0; y < BOARD_SIZE; ++y) {
          for (int x = 0; x < BOARD_SIZE; ++x) {
            const Placement placement = core.getPlacement(x, y);
            if (placement.id == 0) continue;
            const Tile& tile = full_deck[placement.id][placement.rotation];
            // SPIEL_CHECK_EQ's own locals are called x and y.
            for (int side = 0; side < 4; ++side) {
              const int terrain = kNorthTerrainPlane + side * kTerrainTypes;
              SPIEL_CHECK_TRUE(PlaneValue(obs, terrain + 3, x, y) ==
                               (tile.edge[side] == RIVER ? 1.0f : 0.0f));
              const bool shield = tile.edge[side] == CITY &&
                                  (tile.featureMarks(side) & MARK_SHIELD);
              shield_sides += shield ? 1 : 0;
              SPIEL_CHECK_TRUE(PlaneValue(obs, kShieldPlane + side, x, y) ==
                               (shield ? 1.0f : 0.0f));
              if (tile.edge[side] != RIVER) continue;
              ++river_sides;
              SPIEL_CHECK_TRUE(
                  PlaneValue(obs, kFeatureOpensPlane + side, x, y) == 0.0f);
              SPIEL_CHECK_TRUE(
                  PlaneValue(obs, kFeatureScorePlane + side, x, y) == 0.0f);
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
    const ::Carcassonne& core =
        dynamic_cast<const CarcassonneState&>(*state).UnderlyingState();
    SPIEL_CHECK_EQ(core.getDeckSize(), deckSizeOf(ALL_EXPANSIONS));
    SPIEL_CHECK_LE(tiles_drawn, core.getDeckSize());
  }
  std::cout << "ExpansionGamesTest: " << deckSizeOf(ALL_EXPANSIONS)
            << " tiles in the deck, " << river_sides << " river sides, "
            << shield_sides << " shielded city sides seen; opens underflows so far: "
            << OpensUnderflowCount() << std::endl;
  if (tileCountIn(expansionBit(EXP_RIVER)) > 0) {
    SPIEL_CHECK_GT(river_sides, 0);
  }
  SPIEL_CHECK_GT(shield_sides, 0);
  SPIEL_CHECK_EQ(OpensUnderflowCount(), 0);
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

  std::mt19937 rng(20261005);
  for (int g = 0; g < 20; ++g) {
    // Half of them with the big meeple, which counts 2 in the meeple planes.
    std::shared_ptr<const Game> game =
        LoadGame(g % 2 == 0 ? "carcassonne" : kInnsCathedralsGame);
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
  testing::LoadGameTest(kAllExpansionsGame);
  testing::LoadGameTest(kRiverGame);
  testing::LoadGameTest(kInnsCathedralsGame);
  testing::LoadGameTest("carcassonne(inns_cathedrals=tiles)");
  testing::LoadGameTest(kTradersBuildersGame);
  testing::ChanceOutcomesTest(*LoadGame("carcassonne"));
  testing::ChanceOutcomesTest(*LoadGame(kAllExpansionsGame));
  testing::ChanceOutcomesTest(*LoadGame(kRiverGame));
  testing::ChanceOutcomesTest(*LoadGame(kInnsCathedralsGame));
  testing::ChanceOutcomesTest(*LoadGame(kTradersBuildersGame));
  testing::RandomSimTest(*LoadGame("carcassonne"), 50);
  testing::RandomSimTest(*LoadGame(kRiverGame), 20);
  testing::RandomSimTest(*LoadGame(kInnsCathedralsGame), 20);
  testing::RandomSimTest(*LoadGame(kTradersBuildersGame), 20);
  ObservationTensorSmokeTest();
  RelativePerspectiveTest();
  PendingScoreTest();
  TiedFeatureTest();
  TileTableTest();
  TileCheckTest();
  ShieldPerCityTest();
  ExpansionOptionsTest();
  RiverTileTest();
  RiverRulesTest();
  ExpansionGamesTest();
  FieldScoringTest();
  InnerFieldTest();
  BigMeepleTest();
  InnCathedralTest();
  BuilderTest();
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
