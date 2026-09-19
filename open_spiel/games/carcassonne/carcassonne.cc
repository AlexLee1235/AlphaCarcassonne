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
constexpr float kScoreNormalization = 40.0f;
constexpr float kScoreDiffNormalization = 20.0f;
constexpr float kPendingNormalization = 20.0f;
constexpr std::array<float, kStaticDiffScales> kStaticDiffNormalizations = {3.0f, 10.0f, 30.0f};
constexpr float kTurnNormalization = 36.0f;
constexpr float kLegalPlacementNormalization = 100.0f;
constexpr int kMaxOpens = 6;
constexpr float kFeatureScoreNormalization = 12.0f;
constexpr float kMonasteryCoverageNormalization = 9.0f;

// The side pairs of the side-link planes, in plane order.
constexpr std::array<std::array<int, 2>, kNumSidePairs> kSidePairs = {{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};

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

void SetTileTerrainPlanes(absl::Span<float> values, const Tile &tile, int x, int y) {
    for (int side = 0; side < 4; ++side) {
        SetPlaneValue(values, kNorthTerrainPlane + side * kTerrainTypes + TerrainIndex(tile.edge[side]), x, y, 1.0f);
    }
}

float Clip(float value) { return std::max(-1.0f, std::min(1.0f, value)); }

int SidePairIndex(int a, int b) {
    for (int pair = 0; pair < kNumSidePairs; ++pair) {
        if ((kSidePairs[pair][0] == a && kSidePairs[pair][1] == b) ||
            (kSidePairs[pair][0] == b && kSidePairs[pair][1] == a)) {
            return pair;
        }
    }
    SpielFatalError("Not a pair of distinct sides.");
}

// One quarter turn clockwise maps (x, y) to (N-1-y, x): north (y-1) becomes
// east (x+1), matching Tile::rotate().
void RotateCell(int k, int *x, int *y) {
    for (int i = 0; i < k; ++i) {
        const int old_x = *x;
        *x = BOARD_SIZE - 1 - *y;
        *y = old_x;
    }
}

// Planes that come in one-per-side (or one-per-rotation) blocks follow the
// rotation, and so do the side pairs; every other plane keeps its index and
// only its cells move.
int RotatePlane(int plane, int k) {
    if (plane >= kNorthTerrainPlane && plane < kShieldPlane) {
        const int offset = plane - kNorthTerrainPlane;
        return kNorthTerrainPlane + ((offset / kTerrainTypes + k) % 4) * kTerrainTypes + offset % kTerrainTypes;
    }
    if (plane >= kSideLinkPlane && plane < kSideLinkPlane + kNumSidePairs) {
        const std::array<int, 2> &pair = kSidePairs[plane - kSideLinkPlane];
        return kSideLinkPlane + SidePairIndex((pair[0] + k) % 4, (pair[1] + k) % 4);
    }
    for (int first : {kLegalPlacementPlane, kFeatureOpensPlane, kFeatureScorePlane, kFeatureMyMeeplesPlane,
                      kFeatureOpponentMeeplesPlane, kFeatureSignedScorePlane}) {
        if (plane >= first && plane < first + 4) {
            return first + (plane - first + k) % 4;
        }
    }
    return plane;
}

// For each k, the source index of every observation value.
const std::array<std::vector<int>, kNumBoardRotations> &RotationSourceIndices() {
    static const std::array<std::vector<int>, kNumBoardRotations> tables = [] {
        std::array<std::vector<int>, kNumBoardRotations> result;
        for (int k = 0; k < kNumBoardRotations; ++k) {
            result[k].resize(kObservationTensorSize);
            for (int plane = 0; plane < kObservationPlanes; ++plane) {
                for (int y = 0; y < BOARD_SIZE; ++y) {
                    for (int x = 0; x < BOARD_SIZE; ++x) {
                        int rx = x;
                        int ry = y;
                        // The global vector is not a picture of the board.
                        if (plane != kGlobalFeaturePlane) {
                            RotateCell(k, &rx, &ry);
                        }
                        result[k][(RotatePlane(plane, k) * BOARD_SIZE + ry) * BOARD_SIZE + rx] =
                            (plane * BOARD_SIZE + y) * BOARD_SIZE + x;
                    }
                }
            }
        }
        return result;
    }();
    return tables;
}

// Meeple side `side` names a feature by its lowest side; after rotation the
// feature is named by the lowest of its rotated sides.
int RotateMeepleSide(int side, int k, const SideGroups &groups) {
    SPIEL_CHECK_NE(groups[side], -1);
    int rotated = 4;
    for (int s = 0; s < 4; ++s) {
        if (groups[s] == groups[side]) {
            rotated = std::min(rotated, (s + k) % 4);
        }
    }
    return rotated;
}

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
    const int opponent = 1 - player;

    for (int y = 0; y < BOARD_SIZE; ++y) {
        for (int x = 0; x < BOARD_SIZE; ++x) {
            if (game_state_.isFrontier(x, y)) {
                SetPlaneValue(values, kFrontierPlane, x, y, 1.0f);
            }
            const Placement &placement = game_state_.getPlacement(x, y);
            if (placement.id == 0) {
                continue;
            }
            const Tile &tile = full_deck[placement.id][placement.rotation];
            SetPlaneValue(values, kOccupiedPlane, x, y, 1.0f);
            SetTileTerrainPlanes(values, tile, x, y);
            if (tile.shield) {
                SetPlaneValue(values, kShieldPlane, x, y, 1.0f);
            }
            if (tile.monastery) {
                SetPlaneValue(values, kMonasteryPlane, x, y, 1.0f);
                SetPlaneValue(values, kMonasteryCoveragePlane, x, y,
                              game_state_.coverage3x3(x, y) / kMonasteryCoverageNormalization);
                const int owner = game_state_.monasteryOwner(x, y);
                if (owner != -1) {
                    SetPlaneValue(values, kMonasteryOwnerPlane, x, y, owner == player ? 1.0f : -1.0f);
                }
            }
            for (int pair = 0; pair < kNumSidePairs; ++pair) {
                const int a = kSidePairs[pair][0];
                const int b = kSidePairs[pair][1];
                if (tile.edge[a] != GRASS && tile.edge[b] != GRASS && tile.link[a] == tile.link[b]) {
                    SetPlaneValue(values, kSideLinkPlane + pair, x, y, 1.0f);
                }
            }
            if (game_state_.last_x == x && game_state_.last_y == y) {
                SetPlaneValue(values, kLastPlacedPlane, x, y, 1.0f);
            }
            for (int side = 0; side < 4; ++side) {
                if (tile.edge[side] == GRASS) {
                    continue;
                }
                const Feature &feature = game_state_.featureAt(placement.id, side);
                const float score = static_cast<float>(feature.getScore());
                const int mine = feature.meeple_count[player];
                const int theirs = feature.meeple_count[opponent];
                // Equal meeples score for both, which leaves the difference alone.
                const float holder = mine > theirs ? 1.0f : (theirs > mine ? -1.0f : 0.0f);
                SetPlaneValue(values, kFeatureOpensPlane + side, x, y,
                              std::min<int>(feature.opens, kMaxOpens) / static_cast<float>(kMaxOpens));
                SetPlaneValue(values, kFeatureScorePlane + side, x, y, std::min(score / kFeatureScoreNormalization, 1.0f));
                SetPlaneValue(values, kFeatureMyMeeplesPlane + side, x, y, mine / kMeepleNormalization);
                SetPlaneValue(values, kFeatureOpponentMeeplesPlane + side, x, y, theirs / kMeepleNormalization);
                SetPlaneValue(values, kFeatureSignedScorePlane + side, x, y,
                              Clip(holder * score / kFeatureScoreNormalization));
            }
        }
    }

    int legal_placements = 0;
    if (game_state_.current_phase == PHASE_TILE && game_state_.current_tile_in_hand != 0) {
        std::array<TileMove, kTileActionCount> tile_moves{};
        game_state_.getLegalTileMoves(tile_moves.data(), legal_placements);
        for (int i = 0; i < legal_placements; ++i) {
            SetPlaneValue(values, kLegalPlacementPlane + tile_moves[i].rot, tile_moves[i].x, tile_moves[i].y, 1.0f);
        }
    }

    absl::Span<float> global = values.subspan(kGlobalFeaturePlane * BOARD_SIZE * BOARD_SIZE, kGlobalFeatures);
    const int *scores = game_state_.player_scores;
    int pending[2];
    game_state_.getPendingScore(pending);
    const float score_diff = static_cast<float>(scores[player] - scores[opponent]);
    const float static_diff = score_diff + static_cast<float>(pending[player] - pending[opponent]);
    global[kGlobalMyScore] = scores[player] / kScoreNormalization;
    global[kGlobalOpponentScore] = scores[opponent] / kScoreNormalization;
    global[kGlobalScoreDiff] = Clip(score_diff / kScoreDiffNormalization);
    global[kGlobalMyPending] = pending[player] / kPendingNormalization;
    global[kGlobalOpponentPending] = pending[opponent] / kPendingNormalization;
    for (int scale = 0; scale < kStaticDiffScales; ++scale) {
        global[kGlobalStaticDiff + scale] = Clip(static_diff / kStaticDiffNormalizations[scale]);
    }
    global[kGlobalMyMeeples] = game_state_.holding_meeples[player] / kMeepleNormalization;
    global[kGlobalOpponentMeeples] = game_state_.holding_meeples[opponent] / kMeepleNormalization;
    global[kGlobalRemainingTiles] = game_state_.getTotalRemaining() / kRemainingNormalization;
    global[kGlobalCompletedTurns] = game_state_.completed_turns / kTurnNormalization;
    for (int type_id = 1; type_id <= CANONICAL_TILE_TYPE_COUNT; ++type_id) {
        const int initial_count = tile_type_tables.draw_count_by_type[type_id];
        global[kGlobalRemainingByType + type_id - 1] =
            static_cast<float>(game_state_.getRemainingTypeCount(type_id)) / initial_count;
    }
    const int type_in_hand = game_state_.currentTileType();
    if (type_in_hand != 0) {
        global[kGlobalTileInHand + type_in_hand - 1] = 1.0f;
    }
    global[kGlobalTilePhase] = game_state_.current_phase == PHASE_TILE ? 1.0f : 0.0f;
    global[kGlobalMeeplePhase] = game_state_.current_phase == PHASE_MEEPLE ? 1.0f : 0.0f;
    if (game_state_.current_phase == PHASE_MEEPLE) {
        FixedVector<int, kMeepleActionCount> meeple_moves = game_state_.getLegalMeepleMoves();
        for (int i = 0; i < meeple_moves.size(); ++i) {
            global[kGlobalLegalMeeple + meeple_moves[i] + 1] = 1.0f;
        }
    }
    global[kGlobalLegalPlacements] = legal_placements / kLegalPlacementNormalization;
    global[kGlobalIsPlayer0] = game_state_.currentPlayer == 0 ? 1.0f : 0.0f;
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

SideGroups GetSideGroups(const CarcassonneState &state) {
    SideGroups groups = kNoSideGroups;
    const ::Carcassonne &core = state.UnderlyingState();
    if (core.current_phase == PHASE_MEEPLE) {
        core.getLastTileSideGroups(groups.data());
    }
    return groups;
}

Action RotateAction(Action action, int k, const SideGroups &groups) {
    SPIEL_CHECK_GE(k, 0);
    SPIEL_CHECK_LT(k, kNumBoardRotations);
    if (action < kTileActionCount) {
        int x;
        int y;
        int rot;
        DecodeTileAction(action, &x, &y, &rot);
        RotateCell(k, &x, &y);
        return EncodeTileAction(x, y, (rot + k) % 4);
    }
    const int pos = DecodeMeepleAction(action);
    if (pos < 0 || pos > 3) {
        return action;  // Skip and monastery do not depend on orientation.
    }
    return EncodeMeepleAction(RotateMeepleSide(pos, k, groups));
}

SideGroups RotateSideGroups(const SideGroups &groups, int k) {
    SideGroups rotated = kNoSideGroups;
    for (int side = 0; side < 4; ++side) {
        if (groups[side] != -1) {
            rotated[(side + k) % 4] = static_cast<int8_t>(RotateMeepleSide(side, k, groups));
        }
    }
    return rotated;
}

void RotateObservation(absl::Span<const float> observation, int k, const SideGroups &groups,
                       absl::Span<float> rotated) {
    SPIEL_CHECK_GE(k, 0);
    SPIEL_CHECK_LT(k, kNumBoardRotations);
    SPIEL_CHECK_EQ(observation.size(), kObservationTensorSize);
    SPIEL_CHECK_EQ(rotated.size(), kObservationTensorSize);
    const std::vector<int> &source = RotationSourceIndices()[k];
    for (int i = 0; i < kObservationTensorSize; ++i) {
        rotated[i] = observation[source[i]];
    }
    // The legal meeple sides sit in the global vector, which the table leaves
    // in place; rename each legal side instead.
    const int legal_sides = kGlobalFeaturePlane * BOARD_SIZE * BOARD_SIZE + kGlobalLegalMeeple + 1;
    for (int side = 0; side < 4; ++side) {
        rotated[legal_sides + side] = 0.0f;
    }
    for (int side = 0; side < 4; ++side) {
        if (observation[legal_sides + side] != 0.0f) {
            rotated[legal_sides + RotateMeepleSide(side, k, groups)] = 1.0f;
        }
    }
}

} // namespace carcassonne
} // namespace open_spiel
