#include "open_spiel/games/carcassonne/carcassonne.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace carcassonne {
namespace {

constexpr float kMeepleNormalization = 7.0f;
constexpr float kRemainingNormalization = TOTAL_TILE_COUNT;
constexpr float kScoreDiffNormalization = 30.0f;

const GameType kGameType{/*short_name=*/"carcassonne",
                         /*long_name=*/"Carcassonne",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/kNumPlayers,
                         /*min_num_players=*/kNumPlayers,
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/{{"max_turns", GameParameter(0)}}};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
    return std::shared_ptr<const Game>(new CarcassonneGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

Action EncodeChanceAction(int type_id) {
    SPIEL_CHECK_GE(type_id, 1);
    SPIEL_CHECK_LE(type_id, CANONICAL_TILE_TYPE_COUNT);
    return type_id - 1;
}

int DecodeChanceAction(Action action) {
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, kChanceActionCount);
    return action + 1;
}

Action EncodeTileAction(int board_x, int board_y, int rotation) {
    SPIEL_CHECK_GE(board_x, 0);
    SPIEL_CHECK_LT(board_x, BOARD_SIZE);
    SPIEL_CHECK_GE(board_y, 0);
    SPIEL_CHECK_LT(board_y, BOARD_SIZE);
    SPIEL_CHECK_GE(rotation, 0);
    SPIEL_CHECK_LT(rotation, 4);
    return ((board_y * BOARD_SIZE) + board_x) * 4 + rotation;
}

void DecodeTileAction(Action action, int *x, int *y, int *rot) {
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, kTileActionCount);
    *rot = action % 4;
    action /= 4;
    *x = action % BOARD_SIZE;
    *y = action / BOARD_SIZE;
}

Action EncodeMeepleAction(int pos) {
    SPIEL_CHECK_GE(pos, -1);
    SPIEL_CHECK_LE(pos, 4);
    return kMeepleActionOffset + pos + 1;
}

int DecodeMeepleAction(Action action) {
    SPIEL_CHECK_GE(action, kMeepleActionOffset);
    SPIEL_CHECK_LT(action, kNumDistinctPlayerActions);
    return action - kMeepleActionOffset - 1;
}

std::string PhaseToString(GamePhase phase) {
    switch (phase) {
    case PHASE_CHANCE:
        return "chance";
    case PHASE_TILE:
        return "tile";
    case PHASE_MEEPLE:
        return "meeple";
    case PHASE_TERMINAL:
        return "terminal";
    }
    return "unknown";
}

int PhaseIndex(GamePhase phase) {
    switch (phase) {
    case PHASE_CHANCE:
        return 0;
    case PHASE_TILE:
        return 1;
    case PHASE_MEEPLE:
        return 2;
    case PHASE_TERMINAL:
        return 3;
    }
    return 3;
}

int TerrainIndex(EdgeType edge_type) {
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
    SpielFatalError("Unexpected edge type for terrain plane.");
}

void SetPlaneValue(absl::Span<float> values, int plane, int x, int y, float value) {
    const int index = (plane * BOARD_SIZE + y) * BOARD_SIZE + x;
    values[index] = value;
}

void BroadcastPlane(absl::Span<float> values, int plane, float value) {
    const int plane_offset = plane * BOARD_SIZE * BOARD_SIZE;
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
        values[plane_offset + i] = value;
    }
}

void SetTileTerrainPlanes(absl::Span<float> values, const Tile &tile, int north_plane, int east_plane, int south_plane,
                          int west_plane, int x, int y) {
    SetPlaneValue(values, north_plane + TerrainIndex(tile.edge[0]), x, y, 1.0f);
    SetPlaneValue(values, east_plane + TerrainIndex(tile.edge[1]), x, y, 1.0f);
    SetPlaneValue(values, south_plane + TerrainIndex(tile.edge[2]), x, y, 1.0f);
    SetPlaneValue(values, west_plane + TerrainIndex(tile.edge[3]), x, y, 1.0f);
}

bool HasCityConnectivityPlane(int canonical_type) { return canonical_type == 14 || canonical_type == 15; }

} // namespace

CarcassonneState::CarcassonneState(std::shared_ptr<const Game> game) : State(std::move(game)), game_state_() {}

CarcassonneState::CarcassonneState(std::shared_ptr<const Game> game, int max_turns)
    : State(std::move(game)), game_state_(max_turns) {}

CarcassonneState::CarcassonneState(std::shared_ptr<const Game> game, const ::Carcassonne &game_state)
    : State(std::move(game)), game_state_(game_state) {}

Player CarcassonneState::CurrentPlayer() const {
    if (IsTerminal()) {
        return kTerminalPlayerId;
    }
    if (game_state_.current_phase == PHASE_CHANCE) {
        return kChancePlayerId;
    }
    return game_state_.currentPlayer;
}

std::string CarcassonneState::ActionToString(Player player, Action action) const {
    if (player == kChancePlayerId) {
        return absl::StrCat("draw_type(", DecodeChanceAction(action), ")");
    }

    if (action < kTileActionCount) {
        int x;
        int y;
        int rot;
        DecodeTileAction(action, &x, &y, &rot);
        return absl::StrCat("place_tile(x=", x, ", y=", y, ", rot=", rot, ")");
    }

    const int meeple_pos = DecodeMeepleAction(action);
    if (meeple_pos == -1) {
        return "place_meeple(skip)";
    }
    if (meeple_pos == 4) {
        return "place_meeple(monastery)";
    }
    return absl::StrCat("place_meeple(edge=", meeple_pos, ")");
}

std::string CarcassonneState::ToString() const {
    std::vector<std::string> board_rows;
    board_rows.reserve(BOARD_SIZE);
    for (int y = 0; y < BOARD_SIZE; ++y) {
        std::vector<std::string> row;
        row.reserve(BOARD_SIZE);
        for (int x = 0; x < BOARD_SIZE; ++x) {
            const Placement &placement = game_state_.getPlacement(x,y);
            if (placement.id == 0) {
                row.push_back(".");
            } else {
                row.push_back(absl::StrCat(PHYSICAL_TO_CANONICAL_TYPE[placement.id]));
            }
        }
        board_rows.push_back(absl::StrJoin(row, " "));
    }

    return absl::StrCat("phase=", PhaseToString(game_state_.current_phase), " current_player=", game_state_.currentPlayer,
                        " current_tile_type=", game_state_.currentTileType(), " remaining=", game_state_.getTotalRemaining(),
                        " scores=[", game_state_.player_scores[0], ", ", game_state_.player_scores[1], "] holding=[",
                        game_state_.holding_meeples[0], ", ", game_state_.holding_meeples[1], "]\n",
                        absl::StrJoin(board_rows, "\n"));
}

bool CarcassonneState::IsTerminal() const { return game_state_.current_phase == PHASE_TERMINAL; }

std::vector<double> CarcassonneState::Returns() const {
    if (!IsTerminal()) {
        return {0.0, 0.0};
    }

    if (game_state_.player_scores[0] > game_state_.player_scores[1]) {
        return {1.0, -1.0};
    }
    if (game_state_.player_scores[0] < game_state_.player_scores[1]) {
        return {-1.0, 1.0};
    }
    return {0.0, 0.0};
}

std::string CarcassonneState::ObservationString(Player player) const {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, kNumPlayers);
    return ToString();
}

void CarcassonneState::ObservationTensor(Player player, absl::Span<float> values) const {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, kNumPlayers);
    SPIEL_CHECK_EQ(values.size(), kObservationTensorSize);
    std::fill(values.begin(), values.end(), 0.0f);
    for (int y = 0; y < BOARD_SIZE; ++y) {
        for (int x = 0; x < BOARD_SIZE; ++x) {
            const Placement &placement = game_state_.getPlacement(x,y);
            if (placement.id == 0) {
                continue;
            }
            const Tile &tile = full_deck[placement.id][placement.rotation];
            SetTileTerrainPlanes(values, tile, kNorthTerrainPlane, kEastTerrainPlane, kSouthTerrainPlane, kWestTerrainPlane, x,
                                 y);
            if (tile.shield) {
                SetPlaneValue(values, kShieldPlane, x, y, 1.0f);
            }
            if (tile.monastery) {
                SetPlaneValue(values, kMonasteryPlane, x, y, 1.0f);
            }
            if (HasCityConnectivityPlane(PHYSICAL_TO_CANONICAL_TYPE[placement.id])) {
                SetPlaneValue(values, kCityConnectivityPlane, x, y, 1.0f);
            }
            if (game_state_.last_x == x && game_state_.last_y == y) {
                SetPlaneValue(values, kLastPlacedPlane, x, y, 1.0f);
            }
        }
    }

    game_state_.WriteMeepleMap(player, values.data() + kMyMeeplePlane * BOARD_SIZE * BOARD_SIZE);

    if (game_state_.current_tile_in_hand != 0) {
        const Tile &current_tile = full_deck[game_state_.current_tile_in_hand][0];
        const bool has_current_city_connectivity =
            HasCityConnectivityPlane(PHYSICAL_TO_CANONICAL_TYPE[game_state_.current_tile_in_hand]);
        for (int y = 0; y < BOARD_SIZE; ++y) {
            for (int x = 0; x < BOARD_SIZE; ++x) {
                SetTileTerrainPlanes(values, current_tile, kCurrentTileNorthPlane, kCurrentTileEastPlane, kCurrentTileSouthPlane,
                                     kCurrentTileWestPlane, x, y);
                if (current_tile.shield) {
                    SetPlaneValue(values, kCurrentTileShieldPlane, x, y, 1.0f);
                }
                if (current_tile.monastery) {
                    SetPlaneValue(values, kCurrentTileMonasteryPlane, x, y, 1.0f);
                }
                if (has_current_city_connectivity) {
                    SetPlaneValue(values, kCurrentTileCityConnectivityPlane, x, y, 1.0f);
                }
            }
        }
    }

    if (game_state_.current_phase == PHASE_TILE && game_state_.current_tile_in_hand != 0) {
        std::array<TileMove, kTileActionCount> tile_moves{};
        int count = 0;
        game_state_.getLegalTileMoves(tile_moves.data(), count);
        for (int i = 0; i < count; ++i) {
            SetPlaneValue(values, kLegalPlacementPlane + tile_moves[i].rot, tile_moves[i].x, tile_moves[i].y, 1.0f);
        }
    }

    if (game_state_.current_phase == PHASE_MEEPLE) {
        FixedVector<int, kMeepleActionCount> meeple_moves = game_state_.getLegalMeepleMoves();
        for (int i = 0; i < meeple_moves.size(); ++i) {
            const int pos = meeple_moves[i];
            if (pos >= 0 && pos < kLegalMeeplePlanes) {
                BroadcastPlane(values, kLegalMeeplePlane + pos, 1.0f);
            }
        }
    }

    for (int type_id = 1; type_id <= CANONICAL_TILE_TYPE_COUNT; ++type_id) {
        const int initial_count = tile_type_tables.draw_count_by_type[type_id];
        const float remaining = static_cast<float>(game_state_.getRemainingTypeCount(type_id));
        BroadcastPlane(values, kRemainingTileTypePlane + type_id - 1, remaining / initial_count);
    }

    const int opponent = 1 - player;
    BroadcastPlane(values, kMyHoldingMeeplesPlane, game_state_.holding_meeples[player] / kMeepleNormalization);
    BroadcastPlane(values, kOpponentHoldingMeeplesPlane, game_state_.holding_meeples[opponent] / kMeepleNormalization);
    BroadcastPlane(values, kRemainingTilesPlane, game_state_.getTotalRemaining() / kRemainingNormalization);

    const float score_diff = static_cast<float>(game_state_.player_scores[player] - game_state_.player_scores[opponent]);
    BroadcastPlane(values, kScoreDiffPlane, std::tanh(score_diff / kScoreDiffNormalization));
    BroadcastPlane(values, kIsMeeplePhasePlane, game_state_.current_phase == PHASE_MEEPLE ? 1.0f : 0.0f);
    BroadcastPlane(values, kCurrentPlayerIsPlayer0Plane, game_state_.currentPlayer == 0 ? 1.0f : 0.0f);

    if (game_state_.current_phase != PHASE_MEEPLE) {
        SPIEL_CHECK_EQ(values[kIsMeeplePhasePlane * BOARD_SIZE * BOARD_SIZE], 0.0f);
    }
}

std::unique_ptr<State> CarcassonneState::Clone() const { return std::unique_ptr<State>(new CarcassonneState(*this)); }

ActionsAndProbs CarcassonneState::ChanceOutcomes() const {
    SPIEL_CHECK_TRUE(CurrentPlayer() == kChancePlayerId);
    std::array<ChanceBranch, CANONICAL_TILE_TYPE_COUNT> draws{};
    int count = 0;
    game_state_.getAvailableDraws(draws.data(), count);

    ActionsAndProbs outcomes;
    outcomes.reserve(count);
    for (int i = 0; i < count; ++i) {
        outcomes.push_back({EncodeChanceAction(draws[i].type_id), draws[i].probability});
    }
    return outcomes;
}

std::vector<Action> CarcassonneState::LegalActions() const {
    if (IsTerminal()) {
        return {};
    }

    if (CurrentPlayer() == kChancePlayerId) {
        std::vector<Action> actions;
        std::array<ChanceBranch, CANONICAL_TILE_TYPE_COUNT> draws{};
        int count = 0;
        game_state_.getAvailableDraws(draws.data(), count);
        actions.reserve(count);
        for (int i = 0; i < count; ++i) {
            actions.push_back(EncodeChanceAction(draws[i].type_id));
        }
        return actions;
    }

    if (game_state_.current_phase == PHASE_TILE) {
        std::vector<Action> actions;
        std::array<TileMove, kTileActionCount> tile_moves{};
        int count = 0;
        game_state_.getLegalTileMoves(tile_moves.data(), count);
        actions.reserve(count);
        for (int i = 0; i < count; ++i) {
            actions.push_back(EncodeTileAction(tile_moves[i].x, tile_moves[i].y, tile_moves[i].rot));
        }
        std::sort(actions.begin(), actions.end());
        return actions;
    }

    SPIEL_CHECK_EQ(game_state_.current_phase, PHASE_MEEPLE);
    std::vector<Action> actions;
    FixedVector<int, 6> meeple_moves = game_state_.getLegalMeepleMoves();
    actions.reserve(meeple_moves.size());
    for (int i = 0; i < meeple_moves.size(); ++i) {
        actions.push_back(EncodeMeepleAction(meeple_moves[i]));
    }
    std::sort(actions.begin(), actions.end());
    return actions;
}

void CarcassonneState::DoApplyAction(Action action) {
    if (game_state_.current_phase == PHASE_CHANCE) {
        game_state_.drawTile(DecodeChanceAction(action));
        return;
    }

    if (game_state_.current_phase == PHASE_TILE) {
        int x;
        int y;
        int rot;
        DecodeTileAction(action, &x, &y, &rot);
        game_state_.placeTile(x, y, rot);
        return;
    }

    SPIEL_CHECK_EQ(game_state_.current_phase, PHASE_MEEPLE);
    game_state_.placeMeeple(DecodeMeepleAction(action));
}

CarcassonneGame::CarcassonneGame(const GameParameters &params)
    : Game(kGameType, params), max_turns_(ParameterValue<int>("max_turns")) {
    SPIEL_CHECK_GE(max_turns_, 0);
}

} // namespace carcassonne
} // namespace open_spiel
