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

// Meeples in hand and on the board. A feature's strength (the big meeple counts
// 2) can reach 9 / 7.
constexpr float kMeepleNormalization = MEEPLES_PER_PLAYER;
// River tiles on the board count against the 12 river tiles.
constexpr float kRiverTileNormalization = tileCountIn(expansionBit(EXP_RIVER));
// Goods, in a city or as tokens, count against the goods of their kind in the
// table: 9 wine, 6 wheat, 5 cloth.
constexpr std::array<float, GOODS_KINDS> kGoodsNormalization = {
    static_cast<float>(goodsInTable(GOODS_MARKS_BY_KIND[0])), static_cast<float>(goodsInTable(GOODS_MARKS_BY_KIND[1])),
    static_cast<float>(goodsInTable(GOODS_MARKS_BY_KIND[2]))};
// The remaining tiles are counted against the deck this game deals, and the
// completed turns against the tiles drawn after the start tile: completed_turns
// counts the turns of both players, one per tile.
// The measured scales below give the p99 at the last decision as random /
// greedy / self-play games, and the largest value seen; section 4 of
// CLAUDE.md says how to measure them again. All are clipped.
// Points a player has scored: 20 / 85 / 93, max 120 (tools/diag_pending_scale,
// tools/diag_replay_scale).
constexpr float kScoreNormalization = 100.0f;
// The difference in them: 18 / 66 / 54, max 85.
constexpr float kScoreDiffNormalization = 70.0f;
// Pending points of a player: 33 / 62 / 69, max 76.
constexpr float kPendingNormalization = 70.0f;
// The fields' part of it: 15 / 36 / 36, max 45.
constexpr float kFieldPendingNormalization = 40.0f;
// Banked plus pending difference: 28 / 60 / 49, max 75. The /3 and /10 scales
// tell close games apart.
constexpr std::array<float, kStaticDiffScales> kStaticDiffNormalizations = {3.0f, 10.0f, 60.0f};
constexpr float kLegalPlacementNormalization = 100.0f;
constexpr int kMaxOpens = 6;
// The spatial planes, per side or half-edge (tools/diag_plane_scale). The
// observation codec needs their denominators to be integers up to 127.
// A feature's getScore(): 12 / 26 / 26, max 52 (the base game, where it is
// getBaseScore(); a cathedral's city scores 3x, not measured yet).
constexpr float kFeatureScoreNormalization = 30.0f;
constexpr float kMonasteryCoverageNormalization = 9.0f;
// 3 x the completed cities next to a field: 9 / 30 / 30, max 36.
constexpr float kFieldScoreNormalization = 30.0f;
// Tiles in a field: 30 / 56 / 60, max 65.
constexpr float kFieldSizeNormalization = 60.0f;
// Open cities next to a field: 10 / 9 / 8, max 19.
constexpr float kFieldOpenCitiesNormalization = 10.0f;

// The planes one field is written to: a half-edge's or an inner field's.
struct FieldPlanes {
    int my_farmers;
    int opponent_farmers;
    int score;
    int size;
    int open_cities;
};

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
                         // Each expansion: "off", "tiles" to deal its tiles
                         // without its rules, or, for those whose rules are
                         // played (RULED_EXPANSIONS), "on": tiles and rules.
                         // inns_cathedrals "on" brings the big meeple, inns
                         // and cathedrals; traders_builders "on" the builder,
                         // the pig and the goods; princess_dragon "on" the
                         // dragon (no princess, magic portal or fairy yet).
                         // The river is "off" or "on".
                         /*parameter_specification=*/
                         {{"max_turns", GameParameter(0)},
                          {EXPANSION_NAMES[EXP_INNS_CATHEDRALS], GameParameter(std::string("off"))},
                          {EXPANSION_NAMES[EXP_TRADERS_BUILDERS], GameParameter(std::string("off"))},
                          {EXPANSION_NAMES[EXP_RIVER], GameParameter(std::string("off"))},
                          {EXPANSION_NAMES[EXP_PRINCESS_DRAGON], GameParameter(std::string("off"))}}};

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

// Actions by cell of the view; CarcassonneState::CellAction and
// TileActionMove add the view's position on the board.
Action EncodeCellAction(int view_x, int view_y, int plane) {
    SPIEL_CHECK_GE(view_x, 0);
    SPIEL_CHECK_LT(view_x, VIEW_SIZE);
    SPIEL_CHECK_GE(view_y, 0);
    SPIEL_CHECK_LT(view_y, VIEW_SIZE);
    SPIEL_CHECK_GE(plane, 0);
    SPIEL_CHECK_LT(plane, kCellActionPlanes);
    return ((view_y * VIEW_SIZE) + view_x) * kCellActionPlanes + plane;
}

void DecodeCellAction(Action action, int *view_x, int *view_y, int *plane) {
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, kCellActionCount);
    *plane = action % kCellActionPlanes;
    action /= kCellActionPlanes;
    *view_x = action % VIEW_SIZE;
    *view_y = action / VIEW_SIZE;
}

Action EncodeMeepleAction(int pos) {
    SPIEL_CHECK_GE(pos, MEEPLE_POS_SKIP);
    SPIEL_CHECK_LT(pos, MEEPLE_POS_COUNT - 1);
    return kMeepleActionOffset + pos + 1;
}

int DecodeMeepleAction(Action action) {
    SPIEL_CHECK_GE(action, kMeepleActionOffset);
    SPIEL_CHECK_LT(action, kDragonActionOffset);
    return action - kMeepleActionOffset - 1;
}

// A step of the dragon to side 0 N, 1 E, 2 S, 3 W of its tile.
Action EncodeDragonAction(int side) {
    SPIEL_CHECK_GE(side, 0);
    SPIEL_CHECK_LT(side, kDragonActionCount);
    return kDragonActionOffset + side;
}

int DecodeDragonAction(Action action) {
    SPIEL_CHECK_GE(action, kDragonActionOffset);
    SPIEL_CHECK_LT(action, kNumDistinctPlayerActions);
    return action - kDragonActionOffset;
}

constexpr char kSideLetters[] = "NESW";

std::string PhaseToString(GamePhase phase) {
    switch (phase) {
    case PHASE_CHANCE:
        return "chance";
    case PHASE_TILE:
        return "tile";
    case PHASE_MEEPLE:
        return "meeple";
    case PHASE_DRAGON:
        return "dragon";
    case PHASE_SPOT:
        return "spot";
    case PHASE_TERMINAL:
        return "terminal";
    }
    return "unknown";
}

int TerrainIndex(EdgeType edge_type) {
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
    SpielFatalError("Unexpected edge type for terrain plane.");
}

// (x, y) is a cell of the view.
void SetPlaneValue(absl::Span<float> values, int plane, int x, int y, float value) {
    const int index = (plane * VIEW_SIZE + y) * VIEW_SIZE + x;
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

// One quarter turn clockwise maps view cell (x, y) to (N-1-y, x): north (y-1)
// becomes east (x+1), matching Tile::rotate().
void RotateCell(int k, int *x, int *y) {
    for (int i = 0; i < k; ++i) {
        const int old_x = *x;
        *x = VIEW_SIZE - 1 - *y;
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
    for (int first : {kShieldPlane, kLegalPlacementPlane, kFeatureOpensPlane, kFeatureScorePlane, kFeatureMyMeeplesPlane,
                      kFeatureOpponentMeeplesPlane, kFeatureSignedScorePlane, kFeatureMyBigMeeplePlane,
                      kFeatureOpponentBigMeeplePlane, kFeatureInnCathedralPlane, kFeatureMyBuilderPlane,
                      kFeatureOpponentBuilderPlane, kFeatureGoodsPlane, kFeatureGoodsPlane + 4,
                      kFeatureGoodsPlane + 8, kMyPiecePlane, kOpponentPiecePlane, kFairyPiecePlane}) {
        if (plane >= first && plane < first + 4) {
            return first + (plane - first + k) % 4;
        }
    }
    // A quarter turn moves each half-edge two places on.
    for (int first : {kFieldMyFarmersPlane, kFieldOpponentFarmersPlane, kFieldScorePlane, kFieldSizePlane,
                      kFieldOpenCitiesPlane, kFieldMyPigPlane, kFieldOpponentPigPlane,
                      kMyPiecePlane + MEEPLE_POS_FIELD, kOpponentPiecePlane + MEEPLE_POS_FIELD,
                      kFairyPiecePlane + MEEPLE_POS_FIELD}) {
        if (plane >= first && plane < first + HALF_EDGE_COUNT) {
            return first + (plane - first + 2 * k) % HALF_EDGE_COUNT;
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
                for (int y = 0; y < VIEW_SIZE; ++y) {
                    for (int x = 0; x < VIEW_SIZE; ++x) {
                        int rx = x;
                        int ry = y;
                        // The global vector is not a picture of the board.
                        if (plane != kGlobalFeaturePlane) {
                            RotateCell(k, &rx, &ry);
                        }
                        result[k][(RotatePlane(plane, k) * VIEW_SIZE + ry) * VIEW_SIZE + rx] =
                            (plane * VIEW_SIZE + y) * VIEW_SIZE + x;
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

// The same for a field named by its lowest half-edge.
int RotateFieldHalfEdge(int half_edge, int k, const SideGroups &groups) {
    const int8_t *field_groups = groups.data() + kFieldGroupOffset;
    SPIEL_CHECK_NE(field_groups[half_edge], -1);
    int rotated = HALF_EDGE_COUNT;
    for (int e = 0; e < HALF_EDGE_COUNT; ++e) {
        if (field_groups[e] == field_groups[half_edge]) {
            rotated = std::min(rotated, (e + 2 * k) % HALF_EDGE_COUNT);
        }
    }
    return rotated;
}

} // namespace

CarcassonneState::CarcassonneState(std::shared_ptr<const Game> game) : State(std::move(game)), game_state_() {}

CarcassonneState::CarcassonneState(std::shared_ptr<const Game> game, int max_turns, uint32_t expansions,
                                   uint32_t rules)
    : State(std::move(game)), game_state_(max_turns, START_TILE_ROTATION, expansions, rules) {}

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

Action CarcassonneState::CellAction(int x, int y, int plane) const {
    SPIEL_CHECK_TRUE(game_state_.inView(x, y));
    return EncodeCellAction(x - game_state_.view_x0, y - game_state_.view_y0, plane);
}

std::pair<int, int> CarcassonneState::CellActionCell(Action action) const {
    int x;
    int y;
    int plane;
    DecodeCellAction(action, &x, &y, &plane);
    return {game_state_.view_x0 + x, game_state_.view_y0 + y};
}

TileMove CarcassonneState::TileActionMove(Action action) const {
    SPIEL_CHECK_TRUE(IsTileAction(action));
    const auto [x, y] = CellActionCell(action);
    return {static_cast<uint8_t>(x), static_cast<uint8_t>(y), static_cast<uint8_t>(action % kCellActionPlanes)};
}

// The actor logs record games with these strings, tiles by board cell;
// tools/actor_log.hpp parses them to replay self-play games.
std::string CarcassonneState::ActionToString(Player player, Action action) const {
    if (player == kChancePlayerId) {
        return absl::StrCat("draw_type(", DecodeChanceAction(action), ")");
    }

    if (IsTileAction(action)) {
        const TileMove move = TileActionMove(action);
        return absl::StrCat("place_tile(x=", static_cast<int>(move.x), ", y=", static_cast<int>(move.y),
                            ", rot=", static_cast<int>(move.rot), ")");
    }
    if (action < kCellActionCount) {
        const auto [x, y] = CellActionCell(action);
        const int plane = action % kCellActionPlanes;
        return absl::StrCat(plane == kPortalCellPlane     ? "portal("
                            : plane == kPrincessCellPlane ? "princess("
                                                          : "move_fairy(",
                            "x=", x, ", y=", y, ")");
    }
    if (action >= kDragonActionOffset) {
        return absl::StrCat("move_dragon(dir=", std::string(1, kSideLetters[DecodeDragonAction(action)]), ")");
    }

    const int meeple_pos = DecodeMeepleAction(action);
    if (meeple_pos == MEEPLE_POS_SKIP) {
        return "place_meeple(skip)";
    }
    // The princess's knight or the fairy's meeple, by its spot on the cell
    // chosen.
    const bool spot_phase = game_state_.current_phase == PHASE_SPOT;
    const bool knight = spot_phase && game_state_.spot_choice == SPOT_PRINCESS;
    const bool fairy = spot_phase && game_state_.spot_choice == SPOT_FAIRY;
    const char *verb = knight                       ? "remove_knight("
                       : fairy                      ? "fairy_meeple("
                       : isPigPos(meeple_pos)       ? "place_pig("
                       : isBuilderPos(meeple_pos)   ? "place_builder("
                       : isBigMeeplePos(meeple_pos) ? "place_big_meeple("
                                                    : "place_meeple(";
    const int spot = meepleSpot(meeple_pos);
    if (spot == MEEPLE_POS_MONASTERY) {
        return absl::StrCat(verb, "monastery)");
    }
    if (spot == MEEPLE_POS_INNER_FIELD) {
        return absl::StrCat(verb, "inner_field)");
    }
    if (spot >= MEEPLE_POS_FIELD) {
        return absl::StrCat(verb, "field=", spot - MEEPLE_POS_FIELD, ")");
    }
    return absl::StrCat(verb, "edge=", spot, ")");
}

// The board as the view shows it, which holds every tile.
std::string CarcassonneState::ToString() const {
    std::vector<std::string> board_rows;
    board_rows.reserve(VIEW_SIZE);
    for (int y = game_state_.view_y0; y < game_state_.view_y0 + VIEW_SIZE; ++y) {
        std::vector<std::string> row;
        row.reserve(VIEW_SIZE);
        for (int x = game_state_.view_x0; x < game_state_.view_x0 + VIEW_SIZE; ++x) {
            const Placement &placement = game_state_.getPlacement(x, y);
            if (placement.id == 0) {
                row.push_back(".");
            } else {
                row.push_back(absl::StrCat(PHYSICAL_TO_CANONICAL_TYPE[placement.id]));
            }
        }
        board_rows.push_back(absl::StrJoin(row, " "));
    }

    // The expansions' pieces in hand and the builder's double turn.
    std::string expansion_pieces =
        game_state_.big_meeple_rules ? absl::StrCat(" big=[", game_state_.holding_big_meeples[0], ", ",
                                                    game_state_.holding_big_meeples[1], "]")
                                     : "";
    if (game_state_.builder_rules) {
        absl::StrAppend(&expansion_pieces, " builders=[", game_state_.holding_builders[0], ", ",
                        game_state_.holding_builders[1], "]",
                        game_state_.builder_extra_tile ? " builder_extra_tile" : "",
                        game_state_.builder_second_tile ? " builder_second_tile" : "");
    }
    if (game_state_.pig_rules) {
        absl::StrAppend(&expansion_pieces, " pigs=[", game_state_.holding_pigs[0], ", ", game_state_.holding_pigs[1],
                        "]");
    }
    if (game_state_.goods_rules) {
        // Wine, wheat, cloth tokens of each player.
        const int(*tokens)[GOODS_KINDS] = game_state_.goods_tokens;
        absl::StrAppend(&expansion_pieces, " goods=[", tokens[0][0], ", ", tokens[0][1], ", ", tokens[0][2], " / ",
                        tokens[1][0], ", ", tokens[1][1], ", ", tokens[1][2], "]");
    }
    if (game_state_.dragon_rules && game_state_.dragonInPlay()) {
        absl::StrAppend(&expansion_pieces, " dragon=(", game_state_.dragon_x, ", ", game_state_.dragon_y, ")");
        if (game_state_.current_phase == PHASE_DRAGON) {
            absl::StrAppend(&expansion_pieces, " dragon_steps=", game_state_.dragon_steps);
        }
    }
    if (game_state_.fairy_rules && game_state_.fairy_x >= 0) {
        // Its tile, and the player whose meeple it is next to (-1 for none).
        absl::StrAppend(&expansion_pieces, " fairy=(", game_state_.fairy_x, ", ", game_state_.fairy_y, ") of ",
                        game_state_.fairyOwner());
    }
    if (game_state_.current_phase == PHASE_SPOT) {
        absl::StrAppend(&expansion_pieces,
                        game_state_.spot_choice == SPOT_PORTAL     ? " portal_spot=("
                        : game_state_.spot_choice == SPOT_PRINCESS ? " princess_spot=("
                                                                   : " fairy_spot=(",
                        game_state_.spot_x, ", ", game_state_.spot_y, ")");
    }
    return absl::StrCat("phase=", PhaseToString(game_state_.current_phase), " current_player=", game_state_.currentPlayer,
                        " current_tile_type=", game_state_.currentTileType(), " remaining=", game_state_.getTotalRemaining(),
                        " scores=[", game_state_.player_scores[0], ", ", game_state_.player_scores[1], "] holding=[",
                        game_state_.holding_meeples[0], ", ", game_state_.holding_meeples[1], "]", expansion_pieces,
                        " view=(", game_state_.view_x0, ", ", game_state_.view_y0, ")\n",
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

    // Cities next to each field, by root slot; counted once per field rather
    // than once per half-edge.
    std::array<CityCounts, FIELD_SLOT_COUNT> field_cities;
    std::array<bool, FIELD_SLOT_COUNT> field_counted{};
    auto set_field_planes = [&](int root, const FieldPlanes &planes, int x, int y) {
        const Field &field = game_state_.fieldAtRoot(root);
        if (!field_counted[root]) {
            field_cities[root] = game_state_.citiesNextTo(field);
            field_counted[root] = true;
        }
        const CityCounts &cities = field_cities[root];
        SetPlaneValue(values, planes.my_farmers, x, y, field.farmer_count[player] / kMeepleNormalization);
        SetPlaneValue(values, planes.opponent_farmers, x, y, field.farmer_count[opponent] / kMeepleNormalization);
        SetPlaneValue(values, planes.score, x, y,
                      std::min(FIELD_POINTS_PER_CITY * cities.completed / kFieldScoreNormalization, 1.0f));
        SetPlaneValue(values, planes.size, x, y, std::min(field.getTileCount() / kFieldSizeNormalization, 1.0f));
        SetPlaneValue(values, planes.open_cities, x, y, std::min(cities.open / kFieldOpenCitiesNormalization, 1.0f));
    };

    // (bx, by) is the board cell of view cell (x, y).
    for (int y = 0; y < VIEW_SIZE; ++y) {
        for (int x = 0; x < VIEW_SIZE; ++x) {
            const int bx = game_state_.view_x0 + x;
            const int by = game_state_.view_y0 + y;
            if (game_state_.isFrontier(bx, by)) {
                SetPlaneValue(values, kFrontierPlane, x, y, 1.0f);
            }
            const Placement &placement = game_state_.getPlacement(bx, by);
            if (placement.id == 0) {
                continue;
            }
            const Tile &tile = full_deck[placement.id][placement.rotation];
            SetPlaneValue(values, kOccupiedPlane, x, y, 1.0f);
            SetTileTerrainPlanes(values, tile, x, y);
            for (int side = 0; side < 4; ++side) {
                if (tile.edge[side] == CITY && (tile.featureMarks(side) & MARK_SHIELD)) {
                    SetPlaneValue(values, kShieldPlane + side, x, y, 1.0f);
                }
            }
            if (tile.monastery) {
                SetPlaneValue(values, kMonasteryPlane, x, y, 1.0f);
                SetPlaneValue(values, kMonasteryCoveragePlane, x, y,
                              game_state_.coverage3x3(bx, by) / kMonasteryCoverageNormalization);
                const int owner = game_state_.monasteryOwner(bx, by);
                if (owner != -1) {
                    SetPlaneValue(values, kMonasteryOwnerPlane, x, y, owner == player ? 1.0f : -1.0f);
                }
                const int big_owner = game_state_.monasteryBigMeepleOwner(bx, by);
                if (big_owner != -1) {
                    SetPlaneValue(values, kMonasteryBigMeeplePlane, x, y, big_owner == player ? 1.0f : -1.0f);
                }
            }
            for (int pair = 0; pair < kNumSidePairs; ++pair) {
                const int a = kSidePairs[pair][0];
                const int b = kSidePairs[pair][1];
                if (tile.edge[a] != GRASS && tile.edge[b] != GRASS && tile.link[a] == tile.link[b]) {
                    SetPlaneValue(values, kSideLinkPlane + pair, x, y, 1.0f);
                }
            }
            if (game_state_.focusX() == bx && game_state_.focusY() == by) {
                SetPlaneValue(values, kLastPlacedPlane, x, y, 1.0f);
            }
            for (int side = 0; side < 4; ++side) {
                if (!isFeatureEdge(tile.edge[side])) {
                    continue;
                }
                const Feature &feature = game_state_.featureAt(placement.id, side);
                // The size without an inn or cathedral, which the flag plane
                // shows; the signed score is what the feature really adds.
                const float base_score = static_cast<float>(feature.getBaseScore());
                const float score = static_cast<float>(feature.getScore());
                const int mine = feature.meeple_count[player];
                const int theirs = feature.meeple_count[opponent];
                // Equal strengths score for both, which leaves the difference alone.
                const float holder = mine > theirs ? 1.0f : (theirs > mine ? -1.0f : 0.0f);
                SetPlaneValue(values, kFeatureOpensPlane + side, x, y,
                              std::min<int>(feature.opens, kMaxOpens) / static_cast<float>(kMaxOpens));
                SetPlaneValue(values, kFeatureScorePlane + side, x, y,
                              std::min(base_score / kFeatureScoreNormalization, 1.0f));
                SetPlaneValue(values, kFeatureMyMeeplesPlane + side, x, y, mine / kMeepleNormalization);
                SetPlaneValue(values, kFeatureOpponentMeeplesPlane + side, x, y, theirs / kMeepleNormalization);
                SetPlaneValue(values, kFeatureSignedScorePlane + side, x, y,
                              Clip(holder * score / kFeatureScoreNormalization));
                SetPlaneValue(values, kFeatureMyBigMeeplePlane + side, x, y, feature.big_meeples[player]);
                SetPlaneValue(values, kFeatureOpponentBigMeeplePlane + side, x, y, feature.big_meeples[opponent]);
                if (feature.inns > 0 || feature.cathedrals > 0) {
                    SetPlaneValue(values, kFeatureInnCathedralPlane + side, x, y, 1.0f);
                }
                for (int kind = 0; kind < GOODS_KINDS; ++kind) {
                    SetPlaneValue(values, kFeatureGoodsPlane + 4 * kind + side, x, y,
                                  feature.goods[kind] / kGoodsNormalization[kind]);
                }
                SetPlaneValue(values, kFeatureMyBuilderPlane + side, x, y, feature.builders[player]);
                SetPlaneValue(values, kFeatureOpponentBuilderPlane + side, x, y, feature.builders[opponent]);
            }
            for (int half_edge = 0; half_edge < HALF_EDGE_COUNT; ++half_edge) {
                if (tile.field[half_edge] == -1) {
                    continue;
                }
                const int root = game_state_.fieldRoot(placement.id, tile.field[half_edge]);
                set_field_planes(root,
                                 {kFieldMyFarmersPlane + half_edge, kFieldOpponentFarmersPlane + half_edge,
                                  kFieldScorePlane + half_edge, kFieldSizePlane + half_edge,
                                  kFieldOpenCitiesPlane + half_edge},
                                 x, y);
                const Field &field = game_state_.fieldAtRoot(root);
                SetPlaneValue(values, kFieldMyPigPlane + half_edge, x, y, field.pigs[player]);
                SetPlaneValue(values, kFieldOpponentPigPlane + half_edge, x, y, field.pigs[opponent]);
            }
            const int inner_field = tile.innerField();
            if (inner_field != -1) {
                set_field_planes(game_state_.fieldRoot(placement.id, inner_field),
                                 {kInnerFieldMyFarmersPlane, kInnerFieldOpponentFarmersPlane, kInnerFieldScorePlane,
                                  kInnerFieldSizePlane, kInnerFieldOpenCitiesPlane},
                                 x, y);
            }
        }
    }

    int legal_placements = 0;
    if (game_state_.current_phase == PHASE_TILE && game_state_.current_tile_in_hand != 0) {
        std::array<TileMove, kTileActionCount> tile_moves{};
        game_state_.getLegalTileMoves(tile_moves.data(), legal_placements);
        for (int i = 0; i < legal_placements; ++i) {
            SetPlaneValue(values, kLegalPlacementPlane + tile_moves[i].rot, tile_moves[i].x - game_state_.view_x0,
                          tile_moves[i].y - game_state_.view_y0, 1.0f);
        }
    }

    // The tile each piece stands on: its spots there, or for the builder and
    // the pig just the tile.
    for (const Piece &piece : game_state_.pieces) {
        const int x = piece.x - game_state_.view_x0;
        const int y = piece.y - game_state_.view_y0;
        const bool mine = piece.owner == player;
        if (piece.kind == PIECE_BUILDER || piece.kind == PIECE_PIG) {
            const int plane = piece.kind == PIECE_BUILDER ? (mine ? kMyBuilderTilePlane : kOpponentBuilderTilePlane)
                                                          : (mine ? kMyPigTilePlane : kOpponentPigTilePlane);
            SetPlaneValue(values, plane, x, y, 1.0f);
            continue;
        }
        const int first = mine ? kMyPiecePlane : kOpponentPiecePlane;
        const float strength = piece.kind == PIECE_BIG_MEEPLE ? 1.0f : 0.5f;
        for (int spot = 0; spot < kPieceSpotPlanes; ++spot) {
            if (piece.spots >> spot & 1) {
                SetPlaneValue(values, first + spot, x, y, strength);
            }
        }
    }

    // The cells the meeple phase can choose for a portal's meeple, the
    // princess's knight or the fairy.
    for (const auto &[bx, by] : game_state_.getLegalPortalCells()) {
        SetPlaneValue(values, kLegalPortalCellPlane, bx - game_state_.view_x0, by - game_state_.view_y0, 1.0f);
    }
    for (const auto &[bx, by] : game_state_.getLegalPrincessCells()) {
        SetPlaneValue(values, kLegalPrincessCellPlane, bx - game_state_.view_x0, by - game_state_.view_y0, 1.0f);
    }
    for (const auto &[bx, by] : game_state_.getLegalFairyCells()) {
        SetPlaneValue(values, kLegalFairyCellPlane, bx - game_state_.view_x0, by - game_state_.view_y0, 1.0f);
    }

    // The fairy, and the spots of the meeple it is next to.
    const int fairy_owner = game_state_.fairyOwner();
    if (game_state_.fairy_x >= 0) {
        const int fx = game_state_.fairy_x - game_state_.view_x0;
        const int fy = game_state_.fairy_y - game_state_.view_y0;
        SetPlaneValue(values, kFairyPlane, fx, fy, 1.0f);
        for (const Piece &piece : game_state_.pieces) {
            if (fairy_owner < 0 || piece.x != game_state_.fairy_x || piece.y != game_state_.fairy_y ||
                piece.spot != game_state_.fairy_spot || piece.kind == PIECE_BUILDER || piece.kind == PIECE_PIG) {
                continue;
            }
            for (int spot = 0; spot < kPieceSpotPlanes; ++spot) {
                if (piece.spots >> spot & 1) {
                    SetPlaneValue(values, kFairyPiecePlane + spot, fx, fy, 1.0f);
                }
            }
        }
    }

    const bool dragon_phase = game_state_.current_phase == PHASE_DRAGON;
    if (game_state_.dragonInPlay()) {
        SetPlaneValue(values, kDragonPlane, game_state_.dragon_x - game_state_.view_x0,
                      game_state_.dragon_y - game_state_.view_y0, 1.0f);
        for (int i = 0; dragon_phase && i < game_state_.dragon_visited.size(); ++i) {
            SetPlaneValue(values, kDragonVisitedPlane, game_state_.dragon_visited[i].first - game_state_.view_x0,
                          game_state_.dragon_visited[i].second - game_state_.view_y0, 1.0f);
        }
    }

    absl::Span<float> global = values.subspan(kGlobalFeaturePlane * VIEW_SIZE * VIEW_SIZE, kGlobalFeatures);
    const int *scores = game_state_.player_scores;
    int pending[2];
    game_state_.getPendingScore(pending);
    const float score_diff = static_cast<float>(scores[player] - scores[opponent]);
    const float static_diff = score_diff + static_cast<float>(pending[player] - pending[opponent]);
    global[kGlobalMyScore] = Clip(scores[player] / kScoreNormalization);
    global[kGlobalOpponentScore] = Clip(scores[opponent] / kScoreNormalization);
    global[kGlobalScoreDiff] = Clip(score_diff / kScoreDiffNormalization);
    global[kGlobalMyPending] = Clip(pending[player] / kPendingNormalization);
    global[kGlobalOpponentPending] = Clip(pending[opponent] / kPendingNormalization);
    for (int scale = 0; scale < kStaticDiffScales; ++scale) {
        global[kGlobalStaticDiff + scale] = Clip(static_diff / kStaticDiffNormalizations[scale]);
    }
    global[kGlobalMyMeeples] = game_state_.holding_meeples[player] / kMeepleNormalization;
    global[kGlobalOpponentMeeples] = game_state_.holding_meeples[opponent] / kMeepleNormalization;
    const float deck_size = static_cast<float>(game_state_.getDeckSize());
    global[kGlobalRemainingTiles] = game_state_.getTotalRemaining() / deck_size;
    global[kGlobalCompletedTurns] = game_state_.completed_turns / (deck_size - 1.0f);
    // Types this game does not deal have nothing left: 0.
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
    if (game_state_.current_phase == PHASE_MEEPLE || game_state_.current_phase == PHASE_SPOT) {
        const MeepleMoves meeple_moves = game_state_.current_phase == PHASE_MEEPLE ? game_state_.getLegalMeepleMoves()
                                                                                  : game_state_.getLegalSpotMoves();
        for (int i = 0; i < meeple_moves.size(); ++i) {
            global[kGlobalLegalMeeple + meeple_moves[i] + 1] = 1.0f;
        }
    }
    global[kGlobalLegalPlacements] = legal_placements / kLegalPlacementNormalization;
    global[kGlobalIsPlayer0] = game_state_.currentPlayer == 0 ? 1.0f : 0.0f;
    int field_pending[2];
    game_state_.getPendingFieldScore(field_pending);
    global[kGlobalMyFieldPending] = Clip(field_pending[player] / kFieldPendingNormalization);
    global[kGlobalOpponentFieldPending] = Clip(field_pending[opponent] / kFieldPendingNormalization);
    global[kGlobalMyFarmers] =
        (game_state_.farmersOnBoard(player) - game_state_.bigFarmers(player)) / kMeepleNormalization;
    global[kGlobalOpponentFarmers] =
        (game_state_.farmersOnBoard(opponent) - game_state_.bigFarmers(opponent)) / kMeepleNormalization;
    for (int expansion = 1; expansion < EXPANSION_COUNT; ++expansion) {
        const uint32_t bit = expansionBit(static_cast<Expansion>(expansion));
        global[kGlobalExpansionModes + 2 * (expansion - 1)] = (game_state_.expansions & bit) ? 1.0f : 0.0f;
        global[kGlobalExpansionModes + 2 * (expansion - 1) + 1] = (game_state_.rules & bit) ? 1.0f : 0.0f;
    }
    global[kGlobalRiverTiles] = game_state_.river_tiles_placed / kRiverTileNormalization;
    global[kGlobalMyBigMeeple] = game_state_.holding_big_meeples[player];
    global[kGlobalOpponentBigMeeple] = game_state_.holding_big_meeples[opponent];
    global[kGlobalMyBigFarmer] = game_state_.bigFarmers(player);
    global[kGlobalOpponentBigFarmer] = game_state_.bigFarmers(opponent);
    global[kGlobalMyBuilder] = game_state_.holding_builders[player];
    global[kGlobalOpponentBuilder] = game_state_.holding_builders[opponent];
    global[kGlobalBuilderExtraTile] = game_state_.builder_extra_tile ? 1.0f : 0.0f;
    global[kGlobalBuilderSecondTile] = game_state_.builder_second_tile ? 1.0f : 0.0f;
    global[kGlobalMyPig] = game_state_.holding_pigs[player];
    global[kGlobalOpponentPig] = game_state_.holding_pigs[opponent];
    for (int kind = 0; kind < GOODS_KINDS; ++kind) {
        global[kGlobalMyGoods + kind] = game_state_.goods_tokens[player][kind] / kGoodsNormalization[kind];
        global[kGlobalOpponentGoods + kind] = game_state_.goods_tokens[opponent][kind] / kGoodsNormalization[kind];
    }
    global[kGlobalDragonInPlay] = game_state_.dragonInPlay() ? 1.0f : 0.0f;
    global[kGlobalDragonPhase] = dragon_phase ? 1.0f : 0.0f;
    global[kGlobalDragonStepsLeft] =
        dragon_phase ? static_cast<float>(DRAGON_STEPS - game_state_.dragon_steps) / DRAGON_STEPS : 0.0f;
    const int turn_player = dragon_phase ? game_state_.dragon_turn_player : game_state_.currentPlayer;
    global[kGlobalMyTurn] = turn_player == player ? 1.0f : 0.0f;
    if (dragon_phase) {
        const FixedVector<int, 4> dragon_moves = game_state_.getLegalDragonMoves();
        for (int side : dragon_moves) {
            global[kGlobalLegalDragon + side] = 1.0f;
        }
    }
    if (game_state_.current_phase == PHASE_SPOT) {
        global[game_state_.spot_choice == SPOT_PORTAL     ? kGlobalSpotPortal
               : game_state_.spot_choice == SPOT_PRINCESS ? kGlobalSpotPrincess
                                                          : kGlobalSpotFairy] = 1.0f;
    }
    global[kGlobalFairyInSupply] = game_state_.fairy_rules && game_state_.fairy_x < 0 ? 1.0f : 0.0f;
    global[kGlobalMyFairy] = fairy_owner == player ? 1.0f : 0.0f;
    global[kGlobalOpponentFairy] = fairy_owner == opponent ? 1.0f : 0.0f;
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
            actions.push_back(TileAction(tile_moves[i].x, tile_moves[i].y, tile_moves[i].rot));
        }
        std::sort(actions.begin(), actions.end());
        return actions;
    }

    if (game_state_.current_phase == PHASE_DRAGON) {
        std::vector<Action> actions;
        for (int side : game_state_.getLegalDragonMoves()) {
            actions.push_back(EncodeDragonAction(side));
        }
        return actions;
    }

    std::vector<Action> actions;
    if (game_state_.current_phase == PHASE_SPOT) {
        for (int pos : game_state_.getLegalSpotMoves()) {
            actions.push_back(EncodeMeepleAction(pos));
        }
        std::sort(actions.begin(), actions.end());
        return actions;
    }

    SPIEL_CHECK_EQ(game_state_.current_phase, PHASE_MEEPLE);
    MeepleMoves meeple_moves = game_state_.getLegalMeepleMoves();
    actions.reserve(meeple_moves.size());
    for (int i = 0; i < meeple_moves.size(); ++i) {
        actions.push_back(EncodeMeepleAction(meeple_moves[i]));
    }
    for (const auto &[x, y] : game_state_.getLegalPortalCells()) {
        actions.push_back(CellAction(x, y, kPortalCellPlane));
    }
    for (const auto &[x, y] : game_state_.getLegalPrincessCells()) {
        actions.push_back(CellAction(x, y, kPrincessCellPlane));
    }
    for (const auto &[x, y] : game_state_.getLegalFairyCells()) {
        actions.push_back(CellAction(x, y, kFairyCellPlane));
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
        const TileMove move = TileActionMove(action);
        game_state_.placeTile(move.x, move.y, move.rot);
        return;
    }

    if (game_state_.current_phase == PHASE_DRAGON) {
        game_state_.moveDragon(DecodeDragonAction(action));
        return;
    }

    if (game_state_.current_phase == PHASE_SPOT) {
        game_state_.chooseSpot(DecodeMeepleAction(action));
        return;
    }

    SPIEL_CHECK_EQ(game_state_.current_phase, PHASE_MEEPLE);
    if (action < kCellActionCount) {
        const auto [x, y] = CellActionCell(action);
        const int plane = action % kCellActionPlanes;
        SPIEL_CHECK_GE(plane, kPortalCellPlane);
        game_state_.chooseCell(plane == kPortalCellPlane     ? SPOT_PORTAL
                               : plane == kPrincessCellPlane ? SPOT_PRINCESS
                                                             : SPOT_FAIRY,
                               x, y);
        return;
    }
    game_state_.placeMeeple(DecodeMeepleAction(action));
}

CarcassonneGame::CarcassonneGame(const GameParameters &params)
    : Game(kGameType, params), max_turns_(ParameterValue<int>("max_turns")) {
    SPIEL_CHECK_GE(max_turns_, 0);
    for (int expansion = 1; expansion < EXPANSION_COUNT; ++expansion) {
        const std::string mode = ParameterValue<std::string>(EXPANSION_NAMES[expansion]);
        const uint32_t bit = expansionBit(static_cast<Expansion>(expansion));
        // "tiles" deals the tiles alone, unless they need the rules; "on" adds
        // the rules, where the engine plays them.
        const bool tiles_mode = (RULES_REQUIRED_EXPANSIONS & bit) == 0;
        const bool on_mode = (RULED_EXPANSIONS & bit) != 0;
        if (mode == "tiles" && tiles_mode) {
            expansions_ |= bit;
        } else if (mode == "on" && on_mode) {
            expansions_ |= bit;
            rules_ |= bit;
        } else if (mode != "off") {
            SpielFatalError(absl::StrCat("carcassonne: ", EXPANSION_NAMES[expansion], "=", mode, "; expected off",
                                         tiles_mode ? ", tiles" : "", on_mode ? ", on" : ""));
        }
    }
}

SideGroups GetSideGroups(const CarcassonneState &state) {
    SideGroups groups = kNoSideGroups;
    state.UnderlyingState().getFocusSpotGroups(groups.data(), groups.data() + kFieldGroupOffset);
    return groups;
}

Action RotateAction(Action action, int k, const SideGroups &groups) {
    SPIEL_CHECK_GE(k, 0);
    SPIEL_CHECK_LT(k, kNumBoardRotations);
    if (action < kCellActionCount) {
        int x;
        int y;
        int plane;
        DecodeCellAction(action, &x, &y, &plane);
        RotateCell(k, &x, &y);
        // A tile's rotation turns too; the other planes do not depend on it.
        return EncodeCellAction(x, y, plane < 4 ? (plane + k) % 4 : plane);
    }
    if (action >= kDragonActionOffset) {
        return EncodeDragonAction((DecodeDragonAction(action) + k) % 4);
    }
    const int pos = DecodeMeepleAction(action);
    // The big meeple, the builder and the pig go on the same spots further on
    // (MEEPLE_POS_BIG, MEEPLE_POS_BUILDER, MEEPLE_POS_PIG for the half-edges).
    const int spot = meepleSpot(pos);
    const int offset = pos - spot;
    if (spot >= 0 && spot < 4) {
        return EncodeMeepleAction(offset + RotateMeepleSide(spot, k, groups));
    }
    if (spot >= MEEPLE_POS_FIELD && spot < MEEPLE_POS_INNER_FIELD) {
        return EncodeMeepleAction(offset + MEEPLE_POS_FIELD + RotateFieldHalfEdge(spot - MEEPLE_POS_FIELD, k, groups));
    }
    return action;  // Skip, monastery and inner fields do not depend on orientation.
}

SideGroups RotateSideGroups(const SideGroups &groups, int k) {
    SideGroups rotated = kNoSideGroups;
    for (int side = 0; side < 4; ++side) {
        if (groups[side] != -1) {
            rotated[(side + k) % 4] = static_cast<int8_t>(RotateMeepleSide(side, k, groups));
        }
    }
    for (int half_edge = 0; half_edge < HALF_EDGE_COUNT; ++half_edge) {
        if (groups[kFieldGroupOffset + half_edge] != -1) {
            rotated[kFieldGroupOffset + (half_edge + 2 * k) % HALF_EDGE_COUNT] =
                static_cast<int8_t>(RotateFieldHalfEdge(half_edge, k, groups));
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
    // The legal meeple sides and half-edges sit in the global vector, which the
    // table leaves in place; rename each legal one instead: the sides of the
    // meeple, the big meeple and the builder, and the half-edges of the
    // meeple, the big meeple and the pig.
    const int legal_moves = kGlobalFeaturePlane * VIEW_SIZE * VIEW_SIZE + kGlobalLegalMeeple + 1;
    for (int first_side : {0, MEEPLE_POS_BIG, MEEPLE_POS_BUILDER}) {
        const int legal_sides = legal_moves + first_side;
        for (int side = 0; side < 4; ++side) {
            rotated[legal_sides + side] = 0.0f;
        }
        for (int side = 0; side < 4; ++side) {
            if (observation[legal_sides + side] != 0.0f) {
                rotated[legal_sides + RotateMeepleSide(side, k, groups)] = 1.0f;
            }
        }
    }
    for (int first_half_edge : {MEEPLE_POS_FIELD, MEEPLE_POS_BIG + MEEPLE_POS_FIELD, MEEPLE_POS_PIG}) {
        const int legal_half_edges = legal_moves + first_half_edge;
        for (int half_edge = 0; half_edge < HALF_EDGE_COUNT; ++half_edge) {
            rotated[legal_half_edges + half_edge] = 0.0f;
        }
        for (int half_edge = 0; half_edge < HALF_EDGE_COUNT; ++half_edge) {
            if (observation[legal_half_edges + half_edge] != 0.0f) {
                rotated[legal_half_edges + RotateFieldHalfEdge(half_edge, k, groups)] = 1.0f;
            }
        }
    }
    // The dragon's legal steps turn with the board.
    const int legal_dragon = kGlobalFeaturePlane * VIEW_SIZE * VIEW_SIZE + kGlobalLegalDragon;
    for (int side = 0; side < kDragonActionCount; ++side) {
        rotated[legal_dragon + (side + k) % 4] = observation[legal_dragon + side];
    }
}

float ObservationPlaneDenominator(int plane) {
    SPIEL_CHECK_GE(plane, 0);
    SPIEL_CHECK_LT(plane, kObservationPlanes);
    auto in = [plane](int first, int count) { return plane >= first && plane < first + count; };
    // Each is the normalization ObservationTensor writes the plane with.
    // Everything up to the last-placed plane is 0/1, and so are the big meeple,
    // inn / cathedral, builder and pig planes, the builder's and pig's tiles,
    // the dragon's, the cells a portal, the princess or the fairy can choose,
    // and the fairy's; the monastery owners are +-1.
    if (plane <= kLastPlacedPlane || plane == kMonasteryOwnerPlane || in(kFeatureMyBigMeeplePlane, 4) ||
        in(kFeatureOpponentBigMeeplePlane, 4) || plane == kMonasteryBigMeeplePlane ||
        in(kFeatureInnCathedralPlane, 4) || in(kFeatureMyBuilderPlane, 4) || in(kFeatureOpponentBuilderPlane, 4) ||
        in(kFieldMyPigPlane, HALF_EDGE_COUNT) || in(kFieldOpponentPigPlane, HALF_EDGE_COUNT) ||
        in(kMyBuilderTilePlane, kSpatialPlanes - kMyBuilderTilePlane)) {
        return 1.0f;
    }
    // A piece's strength / 2.
    if (in(kMyPiecePlane, 2 * kPieceSpotPlanes)) {
        return 2.0f;
    }
    if (in(kFeatureOpensPlane, 4)) {
        return static_cast<float>(kMaxOpens);
    }
    if (in(kFeatureGoodsPlane, 4 * GOODS_KINDS)) {
        return kGoodsNormalization[(plane - kFeatureGoodsPlane) / 4];
    }
    if (in(kFeatureScorePlane, 4) || in(kFeatureSignedScorePlane, 4)) {
        return kFeatureScoreNormalization;
    }
    if (in(kFeatureMyMeeplesPlane, 4) || in(kFeatureOpponentMeeplesPlane, 4) ||
        in(kFieldMyFarmersPlane, HALF_EDGE_COUNT) || in(kFieldOpponentFarmersPlane, HALF_EDGE_COUNT) ||
        plane == kInnerFieldMyFarmersPlane || plane == kInnerFieldOpponentFarmersPlane) {
        return kMeepleNormalization;
    }
    if (plane == kMonasteryCoveragePlane) {
        return kMonasteryCoverageNormalization;
    }
    if (in(kFieldScorePlane, HALF_EDGE_COUNT) || plane == kInnerFieldScorePlane) {
        return kFieldScoreNormalization;
    }
    if (in(kFieldSizePlane, HALF_EDGE_COUNT) || plane == kInnerFieldSizePlane) {
        return kFieldSizeNormalization;
    }
    if (in(kFieldOpenCitiesPlane, HALF_EDGE_COUNT) || plane == kInnerFieldOpenCitiesPlane) {
        return kFieldOpenCitiesNormalization;
    }
    if (plane == kGlobalFeaturePlane) {
        return 0.0f;
    }
    SpielFatalError(absl::StrCat("No denominator for observation plane ", plane, "."));
}

} // namespace carcassonne
} // namespace open_spiel
