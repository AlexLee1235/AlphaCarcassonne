#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/games/carcassonne/carcassonne.h"
#include "open_spiel/spiel.h"

ABSL_FLAG(std::string, game, "carcassonne", "Game string to load.");
ABSL_FLAG(int, seed, 1, "Random seed.");
ABSL_FLAG(int, turns, 3, "Number of complete tile+meeple turns to play.");
ABSL_FLAG(int, observe_player, -1,
          "Observation player. -1 means current decision player.");
ABSL_FLAG(bool, sample_chance_to_decision, true,
          "After the requested turns, sample chance nodes until a decision node.");
ABSL_FLAG(bool, print_zero_planes, false, "Print planes that are all zero.");
ABSL_FLAG(bool, full_broadcast_planes, false,
          "Print full matrices for broadcast planes instead of one value.");

namespace open_spiel {
namespace carcassonne {
namespace {

float PlaneValue(const std::vector<float>& tensor, int plane, int x, int y) {
  return tensor[(plane * BOARD_SIZE + y) * BOARD_SIZE + x];
}

bool IsBroadcastPlane(const std::vector<float>& tensor, int plane) {
  const float first = PlaneValue(tensor, plane, 0, 0);
  for (int y = 0; y < BOARD_SIZE; ++y) {
    for (int x = 0; x < BOARD_SIZE; ++x) {
      if (std::abs(PlaneValue(tensor, plane, x, y) - first) > 1e-6f) {
        return false;
      }
    }
  }
  return true;
}

bool IsZeroPlane(const std::vector<float>& tensor, int plane) {
  for (int y = 0; y < BOARD_SIZE; ++y) {
    for (int x = 0; x < BOARD_SIZE; ++x) {
      if (std::abs(PlaneValue(tensor, plane, x, y)) > 1e-6f) {
        return false;
      }
    }
  }
  return true;
}

std::string TerrainName(int terrain) {
  switch (terrain) {
    case 0:
      return "grass";
    case 1:
      return "city";
    case 2:
      return "road";
    case 3:
      return "river";
    default:
      return "unknown";
  }
}

void AddTerrainNames(std::vector<std::string>* names, int base,
                     const std::string& prefix) {
  for (int terrain = 0; terrain < kTerrainTypes; ++terrain) {
    (*names)[base + terrain] = prefix + "_" + TerrainName(terrain);
  }
}

std::vector<std::string> PlaneNames() {
  std::vector<std::string> names(kObservationPlanes);
  for (int plane = 0; plane < kObservationPlanes; ++plane) {
    names[plane] = "plane_" + std::to_string(plane);
  }

  names[kOccupiedPlane] = "occupied";
  AddTerrainNames(&names, kNorthTerrainPlane, "board_north");
  AddTerrainNames(&names, kEastTerrainPlane, "board_east");
  AddTerrainNames(&names, kSouthTerrainPlane, "board_south");
  AddTerrainNames(&names, kWestTerrainPlane, "board_west");
  for (int side = 0; side < 4; ++side) {
    names[kShieldPlane + side] = "board_shield_side_" + std::to_string(side);
  }
  names[kMonasteryPlane] = "board_monastery";
  const char* side_pairs[kNumSidePairs] = {"N-E", "N-S", "N-W",
                                           "E-S", "E-W", "S-W"};
  for (int pair = 0; pair < kNumSidePairs; ++pair) {
    names[kSideLinkPlane + pair] = std::string("side_link_") + side_pairs[pair];
  }

  names[kFrontierPlane] = "frontier";
  for (int rot = 0; rot < kLegalPlacementPlanes; ++rot) {
    names[kLegalPlacementPlane + rot] =
        "legal_tile_placement_rot_" + std::to_string(rot);
  }
  names[kLastPlacedPlane] = "last_placed_tile";

  for (int side = 0; side < 4; ++side) {
    const std::string suffix = "_side_" + std::to_string(side);
    names[kFeatureOpensPlane + side] = "feature_opens" + suffix;
    names[kFeatureScorePlane + side] = "feature_score" + suffix;
    names[kFeatureMyMeeplesPlane + side] = "feature_my_meeples" + suffix;
    names[kFeatureOpponentMeeplesPlane + side] =
        "feature_opponent_meeples" + suffix;
    names[kFeatureSignedScorePlane + side] = "feature_signed_score" + suffix;
  }
  names[kMonasteryCoveragePlane] = "monastery_coverage";
  names[kMonasteryOwnerPlane] = "monastery_owner";
  for (int half_edge = 0; half_edge < HALF_EDGE_COUNT; ++half_edge) {
    const std::string suffix = "_half_edge_" + std::to_string(half_edge);
    names[kFieldMyFarmersPlane + half_edge] = "field_my_farmers" + suffix;
    names[kFieldOpponentFarmersPlane + half_edge] =
        "field_opponent_farmers" + suffix;
    names[kFieldScorePlane + half_edge] = "field_score" + suffix;
    names[kFieldSizePlane + half_edge] = "field_size" + suffix;
    names[kFieldOpenCitiesPlane + half_edge] = "field_open_cities" + suffix;
  }
  for (int half_edge = 0; half_edge < HALF_EDGE_COUNT; ++half_edge) {
    const std::string suffix = "_half_edge_" + std::to_string(half_edge);
    names[kFieldMyPigPlane + half_edge] = "field_my_pig" + suffix;
    names[kFieldOpponentPigPlane + half_edge] = "field_opponent_pig" + suffix;
  }
  names[kInnerFieldMyFarmersPlane] = "inner_field_my_farmers";
  names[kInnerFieldOpponentFarmersPlane] = "inner_field_opponent_farmers";
  names[kInnerFieldScorePlane] = "inner_field_score";
  names[kInnerFieldSizePlane] = "inner_field_size";
  names[kInnerFieldOpenCitiesPlane] = "inner_field_open_cities";
  for (int side = 0; side < 4; ++side) {
    const std::string suffix = "_side_" + std::to_string(side);
    names[kFeatureMyBigMeeplePlane + side] = "feature_my_big_meeple" + suffix;
    names[kFeatureOpponentBigMeeplePlane + side] =
        "feature_opponent_big_meeple" + suffix;
  }
  names[kMonasteryBigMeeplePlane] = "monastery_big_meeple";
  for (int side = 0; side < 4; ++side) {
    const std::string suffix = "_side_" + std::to_string(side);
    names[kFeatureInnCathedralPlane + side] = "feature_inn_or_cathedral" + suffix;
    names[kFeatureMyBuilderPlane + side] = "feature_my_builder" + suffix;
    names[kFeatureOpponentBuilderPlane + side] = "feature_opponent_builder" + suffix;
  }
  names[kGlobalFeaturePlane] = "global_vector";
  return names;
}

std::vector<std::string> GlobalFeatureNames() {
  std::vector<std::string> names(kGlobalFeatures);
  names[kGlobalMyScore] = "my_score";
  names[kGlobalOpponentScore] = "opponent_score";
  names[kGlobalScoreDiff] = "score_diff";
  names[kGlobalMyPending] = "my_pending";
  names[kGlobalOpponentPending] = "opponent_pending";
  const char* scales[kStaticDiffScales] = {"3", "10", "60"};
  for (int scale = 0; scale < kStaticDiffScales; ++scale) {
    names[kGlobalStaticDiff + scale] = std::string("static_diff_/") + scales[scale];
  }
  names[kGlobalMyMeeples] = "my_holding_meeples";
  names[kGlobalOpponentMeeples] = "opponent_holding_meeples";
  names[kGlobalRemainingTiles] = "remaining_tiles";
  names[kGlobalCompletedTurns] = "completed_turns";
  for (int type_id = 1; type_id <= CANONICAL_TILE_TYPE_COUNT; ++type_id) {
    names[kGlobalRemainingByType + type_id - 1] =
        "remaining_type_" + std::to_string(type_id);
    names[kGlobalTileInHand + type_id - 1] =
        "tile_in_hand_type_" + std::to_string(type_id);
  }
  names[kGlobalTilePhase] = "is_tile_phase";
  names[kGlobalMeeplePhase] = "is_meeple_phase";
  // Skip, then the spots with the meeple and with the big meeple, then the
  // builder's sides and the pig's half-edges.
  const char* spots[MEEPLE_POS_BIG] = {
      "edge_0",  "edge_1",  "edge_2",  "edge_3",  "monastery",
      "field_0", "field_1", "field_2", "field_3", "field_4",
      "field_5", "field_6", "field_7", "inner_field"};
  names[kGlobalLegalMeeple] = "legal_meeple_skip";
  for (int spot = 0; spot < MEEPLE_POS_BIG; ++spot) {
    names[kGlobalLegalMeeple + 1 + spot] =
        std::string("legal_meeple_") + spots[spot];
    names[kGlobalLegalMeeple + 1 + MEEPLE_POS_BIG + spot] =
        std::string("legal_big_meeple_") + spots[spot];
  }
  for (int side = 0; side < 4; ++side) {
    names[kGlobalLegalMeeple + 1 + MEEPLE_POS_BUILDER + side] =
        std::string("legal_builder_") + spots[side];
  }
  for (int half_edge = 0; half_edge < HALF_EDGE_COUNT; ++half_edge) {
    names[kGlobalLegalMeeple + 1 + MEEPLE_POS_PIG + half_edge] =
        std::string("legal_pig_") + spots[MEEPLE_POS_FIELD + half_edge];
  }
  names[kGlobalLegalPlacements] = "legal_placements";
  names[kGlobalIsPlayer0] = "current_player_is_player0";
  names[kGlobalMyFieldPending] = "my_field_pending";
  names[kGlobalOpponentFieldPending] = "opponent_field_pending";
  names[kGlobalMyFarmers] = "my_farmers";
  names[kGlobalOpponentFarmers] = "opponent_farmers";
  for (int expansion = 1; expansion < EXPANSION_COUNT; ++expansion) {
    const int cell = kGlobalExpansionModes + 2 * (expansion - 1);
    names[cell] = std::string(EXPANSION_NAMES[expansion]) + "_tiles";
    names[cell + 1] = std::string(EXPANSION_NAMES[expansion]) + "_rules";
  }
  names[kGlobalRiverTiles] = "river_tiles_placed";
  names[kGlobalMyBigMeeple] = "my_holding_big_meeple";
  names[kGlobalOpponentBigMeeple] = "opponent_holding_big_meeple";
  names[kGlobalMyBigFarmer] = "my_big_farmer";
  names[kGlobalOpponentBigFarmer] = "opponent_big_farmer";
  names[kGlobalMyBuilder] = "my_holding_builder";
  names[kGlobalOpponentBuilder] = "opponent_holding_builder";
  names[kGlobalBuilderExtraTile] = "builder_extra_tile";
  names[kGlobalBuilderSecondTile] = "builder_second_tile";
  names[kGlobalMyPig] = "my_holding_pig";
  names[kGlobalOpponentPig] = "opponent_holding_pig";
  return names;
}

// The global vector is not a picture of the board: print it by name.
void PrintGlobalFeatures(const std::vector<float>& tensor, bool print_zeros) {
  std::cout << "\n--- plane " << kGlobalFeaturePlane
            << ": global_vector (first " << kGlobalFeatures << " cells) ---\n";
  const std::vector<std::string> names = GlobalFeatureNames();
  const int offset = kGlobalFeaturePlane * BOARD_SIZE * BOARD_SIZE;
  for (int i = 0; i < kGlobalFeatures; ++i) {
    const float value = tensor[offset + i];
    if (!print_zeros && std::abs(value) < 1e-6f) continue;
    std::cout << std::setw(3) << i << " " << std::left << std::setw(28)
              << names[i] << std::right << std::fixed << std::setprecision(3)
              << value << "\n";
  }
}

void PrintPlane(const std::vector<float>& tensor, int plane,
                const std::string& name, bool full_broadcast) {
  std::cout << "\n--- plane " << plane << ": " << name << " ---\n";
  if (!full_broadcast && IsBroadcastPlane(tensor, plane)) {
    std::cout << "broadcast value: " << std::fixed << std::setprecision(3)
              << PlaneValue(tensor, plane, 0, 0) << "\n";
    return;
  }

  for (int y = 0; y < BOARD_SIZE; ++y) {
    std::cout << "y=" << std::setw(2) << y << " ";
    for (int x = 0; x < BOARD_SIZE; ++x) {
      const float value = PlaneValue(tensor, plane, x, y);
      if (std::abs(value) < 1e-6f) {
        std::cout << "   .";
      } else if (std::abs(value - 1.0f) < 1e-6f) {
        std::cout << "   1";
      } else {
        std::cout << " " << std::setw(3) << std::fixed
                  << std::setprecision(1) << value;
      }
    }
    std::cout << "\n";
  }
}

Action SampleChanceAction(const State& state, std::mt19937* rng) {
  return SampleAction(state.ChanceOutcomes(), *rng).first;
}

Action SamplePlayerAction(const State& state, std::mt19937* rng) {
  std::vector<Action> actions = state.LegalActions();
  absl::uniform_int_distribution<> dist(0, actions.size() - 1);
  return actions[dist(*rng)];
}

int Main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string game_string = absl::GetFlag(FLAGS_game);
  const int seed = absl::GetFlag(FLAGS_seed);
  const int requested_turns = absl::GetFlag(FLAGS_turns);
  const int observe_player_flag = absl::GetFlag(FLAGS_observe_player);
  const bool sample_chance_to_decision =
      absl::GetFlag(FLAGS_sample_chance_to_decision);
  const bool print_zero_planes = absl::GetFlag(FLAGS_print_zero_planes);
  const bool full_broadcast_planes =
      absl::GetFlag(FLAGS_full_broadcast_planes);

  std::mt19937 rng(seed);
  std::shared_ptr<const Game> game = LoadGame(game_string);
  std::unique_ptr<State> state = game->NewInitialState();

  std::cout << "game: " << game_string << "\n";
  std::cout << "seed: " << seed << "\n";
  std::cout << "requested complete turns: " << requested_turns << "\n";
  std::cout << "observation shape: [" << kObservationPlanes << ", "
            << BOARD_SIZE << ", " << BOARD_SIZE << "]\n";

  int turns_completed = 0;
  int actions_applied = 0;
  while (!state->IsTerminal() && turns_completed < requested_turns) {
    if (state->IsChanceNode()) {
      Action action = SampleChanceAction(*state, &rng);
      std::cout << "chance: " << state->ActionToString(kChancePlayerId, action)
                << "\n";
      state->ApplyAction(action);
      continue;
    }

    const auto* carcassonne_state =
        dynamic_cast<const CarcassonneState*>(state.get());
    SPIEL_CHECK_TRUE(carcassonne_state != nullptr);
    const bool completes_turn =
        carcassonne_state->UnderlyingState().current_phase == PHASE_MEEPLE;
    const Player player = state->CurrentPlayer();
    Action action = SamplePlayerAction(*state, &rng);
    std::cout << "player " << player << ": "
              << state->ActionToString(player, action) << "\n";
    state->ApplyAction(action);
    actions_applied++;
    if (completes_turn) {
      turns_completed++;
      std::cout << "completed turns: " << turns_completed << "\n";
    }
  }

  while (!state->IsTerminal() && sample_chance_to_decision &&
         state->IsChanceNode()) {
    Action action = SampleChanceAction(*state, &rng);
    std::cout << "post-turn chance: "
              << state->ActionToString(kChancePlayerId, action) << "\n";
    state->ApplyAction(action);
  }

  int observe_player = observe_player_flag;
  if (observe_player == -1) {
    observe_player = state->IsTerminal() || state->IsChanceNode()
                         ? 0
                         : state->CurrentPlayer();
  }
  SPIEL_CHECK_GE(observe_player, 0);
  SPIEL_CHECK_LT(observe_player, kNumPlayers);

  std::cout << "\n=== final state ===\n";
  std::cout << state->ToString() << "\n";
  std::cout << "is_terminal: " << state->IsTerminal() << "\n";
  std::cout << "current_player: " << state->CurrentPlayer() << "\n";
  std::cout << "actions_applied: " << actions_applied << "\n";
  std::cout << "turns_completed: " << turns_completed << "\n";
  std::cout << "observe_player: " << observe_player << "\n";

  std::vector<float> observation = state->ObservationTensor(observe_player);
  std::vector<std::string> names = PlaneNames();

  std::cout << "\n=== observation tensor ===\n";
  for (int plane = 0; plane < kObservationPlanes; ++plane) {
    if (plane == kGlobalFeaturePlane) {
      PrintGlobalFeatures(observation, print_zero_planes);
      continue;
    }
    if (!print_zero_planes && IsZeroPlane(observation, plane)) {
      continue;
    }
    PrintPlane(observation, plane, names[plane], full_broadcast_planes);
  }
  std::cout << "\nprinted zero planes: " << print_zero_planes << "\n";
  std::cout << "use --print_zero_planes=true to include all-zero planes\n";
  return 0;
}

}  // namespace
}  // namespace carcassonne
}  // namespace open_spiel

int main(int argc, char** argv) {
  return open_spiel::carcassonne::Main(argc, argv);
}
