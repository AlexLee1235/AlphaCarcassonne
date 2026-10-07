#include <array>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "open_spiel/games/carcassonne/game/game.hpp"

namespace py = pybind11;

namespace {

constexpr int kMeepleMapPlanes = 10;
constexpr int kMeepleMapSize = kMeepleMapPlanes * BOARD_SIZE * BOARD_SIZE;
constexpr int kMaxTileMoves = BOARD_SIZE * BOARD_SIZE * 4;

std::vector<std::tuple<int, double>> GetAvailableDraws(const Carcassonne &game) {
    std::array<ChanceBranch, CANONICAL_TILE_TYPE_COUNT> draws{};
    int count = 0;
    game.getAvailableDraws(draws.data(), count);

    std::vector<std::tuple<int, double>> result;
    result.reserve(count);
    for (int i = 0; i < count; ++i) {
        result.emplace_back(draws[i].type_id, draws[i].probability);
    }
    return result;
}

std::vector<std::tuple<int, int, int>> GetLegalTileMoves(const Carcassonne &game) {
    std::array<TileMove, kMaxTileMoves> moves{};
    int count = 0;
    game.getLegalTileMoves(moves.data(), count);

    std::vector<std::tuple<int, int, int>> result;
    result.reserve(count);
    for (int i = 0; i < count; ++i) {
        result.emplace_back(static_cast<int>(moves[i].x), static_cast<int>(moves[i].y), static_cast<int>(moves[i].rot));
    }
    return result;
}

std::vector<int> GetLegalMeepleMoves(const Carcassonne &game) {
    MeepleMoves moves = game.getLegalMeepleMoves();

    std::vector<int> result;
    result.reserve(moves.size());
    for (int i = 0; i < moves.size(); ++i) {
        result.push_back(moves[i]);
    }
    return result;
}

std::vector<std::tuple<int, int, int, int>> GetPlacedTiles(const Carcassonne &game) {
    std::vector<std::tuple<int, int, int, int>> result;
    for (int y = 0; y < BOARD_SIZE; ++y) {
        for (int x = 0; x < BOARD_SIZE; ++x) {
            const Placement placement = game.getPlacement(x, y);
            if (placement.id == 0) {
                continue;
            }
            result.emplace_back(x, y, static_cast<int>(placement.id), static_cast<int>(placement.rotation));
        }
    }
    return result;
}

std::vector<std::tuple<int, int, int, int>> GetMeepleTokens(const Carcassonne &game) {
    std::array<float, kMeepleMapSize> meeple_map{};
    game.WriteMeepleMap(/*player=*/0, meeple_map.data());

    std::vector<std::tuple<int, int, int, int>> result;
    for (int y = 0; y < BOARD_SIZE; ++y) {
        for (int x = 0; x < BOARD_SIZE; ++x) {
            if (game.getPlacement(x, y).id == 0) {
                continue;
            }
            for (int plane = 0; plane < kMeepleMapPlanes; ++plane) {
                const int index = (plane * BOARD_SIZE + y) * BOARD_SIZE + x;
                if (meeple_map[index] <= 0.0f) {
                    continue;
                }
                const int player = plane < 5 ? 0 : 1;
                const int pos = plane < 5 ? plane : plane - 5;
                result.emplace_back(player, x, y, pos);
            }
        }
    }
    return result;
}

std::vector<int> PhysicalToCanonicalType() {
    return std::vector<int>(PHYSICAL_TO_CANONICAL_TYPE.begin(), PHYSICAL_TO_CANONICAL_TYPE.end());
}

std::vector<int> PlayerScores(const Carcassonne &game) {
    return {game.player_scores[0], game.player_scores[1]};
}

std::vector<int> HoldingMeeples(const Carcassonne &game) {
    return {game.holding_meeples[0], game.holding_meeples[1]};
}

} // namespace

PYBIND11_MODULE(_carcassonne_cpp, m) {
    m.doc() = "Small pybind11 bridge from the Flet UI to the current Carcassonne core engine.";

    m.attr("BOARD_SIZE") = BOARD_SIZE;
    m.attr("VIEW_SIZE") = VIEW_SIZE;
    m.attr("PHASE_CHANCE") = static_cast<int>(PHASE_CHANCE);
    m.attr("PHASE_TILE") = static_cast<int>(PHASE_TILE);
    m.attr("PHASE_MEEPLE") = static_cast<int>(PHASE_MEEPLE);
    m.attr("PHASE_TERMINAL") = static_cast<int>(PHASE_TERMINAL);
    m.attr("PHYSICAL_TO_CANONICAL_TYPE") = PhysicalToCanonicalType();
    m.attr("MEEPLE_POS_SKIP") = MEEPLE_POS_SKIP;
    m.attr("MEEPLE_POS_MONASTERY") = MEEPLE_POS_MONASTERY;
    m.attr("MEEPLE_POS_FIELD") = MEEPLE_POS_FIELD;
    m.attr("MEEPLE_POS_INNER_FIELD") = MEEPLE_POS_INNER_FIELD;
    m.attr("MEEPLE_POS_BIG") = MEEPLE_POS_BIG;
    m.attr("MEEPLE_POS_BUILDER") = MEEPLE_POS_BUILDER;
    m.attr("MEEPLE_POS_PIG") = MEEPLE_POS_PIG;
    m.attr("HALF_EDGE_COUNT") = HALF_EDGE_COUNT;
    // Expansions, as the game parameters name them (EXPANSION_NAMES[0] is the base).
    m.attr("EXPANSION_NAMES") = std::vector<std::string>(EXPANSION_NAMES, EXPANSION_NAMES + EXPANSION_COUNT);
    m.attr("BASE_ONLY") = BASE_ONLY;
    m.attr("RULED_EXPANSIONS") = RULED_EXPANSIONS;
    m.attr("RULES_REQUIRED_EXPANSIONS") = RULES_REQUIRED_EXPANSIONS;
    m.def("expansion_bit", [](int expansion) { return expansionBit(static_cast<Expansion>(expansion)); });

    py::class_<Carcassonne>(m, "Carcassonne")
        .def(py::init<>())
        // `expansions`: the expansion_bit() mask of the expansions whose tiles are
        // dealt, the base included; `rules`: those of them that play their rules.
        .def(py::init([](uint32_t expansions, uint32_t rules) {
                 return Carcassonne(0, START_TILE_ROTATION, expansions, rules);
             }),
             py::arg("expansions"), py::arg("rules"))
        .def_property_readonly("expansions", [](const Carcassonne &game) { return game.expansions; })
        .def_property_readonly("rules", [](const Carcassonne &game) { return game.rules; })
        .def_property_readonly("current_phase", [](const Carcassonne &game) { return static_cast<int>(game.current_phase); })
        .def_property_readonly("is_game_over",
                               [](const Carcassonne &game) { return game.current_phase == PHASE_TERMINAL; })
        .def_property_readonly("current_player", [](const Carcassonne &game) { return game.currentPlayer; })
        .def_property_readonly("current_tile_in_hand",
                               [](const Carcassonne &game) { return game.current_tile_in_hand; })
        .def_property_readonly("player_scores", &PlayerScores)
        .def_property_readonly("holding_meeples", &HoldingMeeples)
        // The first cell of the view, which follows the tiles: every tile and
        // every legal move is in the VIEW_SIZE x VIEW_SIZE cells from it.
        .def_property_readonly("view_origin",
                               [](const Carcassonne &game) { return std::make_tuple(game.view_x0, game.view_y0); })
        .def("get_available_draws", &GetAvailableDraws)
        .def("draw_tile", &Carcassonne::drawTile)
        .def("get_legal_tile_moves", &GetLegalTileMoves)
        .def("place_tile", &Carcassonne::placeTile)
        .def("get_legal_meeple_moves", &GetLegalMeepleMoves)
        .def("place_meeple", &Carcassonne::placeMeeple)
        .def("get_placed_tiles", &GetPlacedTiles)
        .def("get_meeple_tokens", &GetMeepleTokens);
}
