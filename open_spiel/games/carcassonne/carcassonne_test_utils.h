#ifndef OPEN_SPIEL_GAMES_CARCASSONNE_CARCASSONNE_TEST_UTILS_H_
#define OPEN_SPIEL_GAMES_CARCASSONNE_CARCASSONNE_TEST_UTILS_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "open_spiel/games/carcassonne/carcassonne.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace carcassonne {

// A legal full-length game with the all-city tile (chance action 2) left in
// the deck and no matching placement. The next draw must score and terminate
// directly, without another player decision. Fixed moves make this fixture
// independent of RNG/distribution implementations and legal-move ordering.
// Random games almost always leave the all-city tile somewhere to go, so this
// one held it back and placed every tile where it left it the fewest spots.
// Each turn: the chance action drawn, the board cell of the tile as cells
// from the start tile and its rotation, and the meeple position (-1 skips).
// Actions name cells of the view, which follows the tiles, so they depend on
// VIEW_SIZE and kCellActionPlanes: LastUnplaceableTileHistory() makes them.
struct FixtureTurn {
  Action draw;
  int dx, dy, rot;
  int meeple;
};
inline constexpr FixtureTurn kLastUnplaceableTileTurns[] = {
    {20, 1, 0, 3, -1}, {15, 0, -1, 2, 2}, {20, 0, -2, 3, 1}, {17, -1, 0, 0, -1}, {7, 1, -1, 1, 9},
    {9, 2, -1, 3, 0}, {1, 1, 1, 3, 4}, {20, 1, 2, 3, 8}, {0, -2, 0, 1, 4}, {21, -2, -1, 2, -1},
    {18, -1, -2, 2, 5}, {21, 0, -3, 1, 5}, {22, 2, 1, 1, 5}, {4, -3, 0, 3, -1}, {5, -3, 1, 1, 0},
    {17, -3, 2, 0, -1}, {14, -4, 0, 3, -1}, {3, -2, 1, 2, -1}, {15, -5, 0, 1, -1}, {0, -4, 2, 1, 5},
    {21, 3, -1, 0, -1}, {18, 3, 0, 3, 2}, {11, -4, -1, 2, -1}, {12, -5, -1, 0, 1}, {6, 1, -2, 1, -1},
    {21, -6, 0, 0, 2}, {19, 0, 2, 0, 0}, {20, 4, -1, 2, -1}, {8, -5, 1, 3, -1}, {21, 0, -4, 3, -1},
    {20, -4, 3, 3, -1}, {21, -4, 4, 0, -1}, {10, -1, 1, 3, -1}, {21, -5, -2, 2, -1}, {3, -5, 2, 3, -1},
    {21, -5, 4, 2, -1}, {13, -6, 2, 1, -1}, {14, 3, -2, 3, -1}, {15, 0, 3, 1, -1}, {6, -1, 2, 3, -1},
    {16, 1, -3, 2, -1}, {18, -3, 3, 2, -1}, {16, -6, -1, 1, -1}, {15, -1, 3, 0, -1}, {1, -6, 4, 0, -1},
    {8, -3, -1, 3, -1}, {20, 0, 4, 1, -1}, {7, 1, 3, 3, -1}, {10, -2, 2, 1, -1}, {3, -3, -2, 0, 0},
    {0, 2, 3, 1, 4}, {19, -3, -3, 2, -1}, {22, -4, 5, 3, -1}, {20, 3, -3, 3, -1}, {16, 1, 4, 0, -1},
    {19, 4, -2, 3, -1}, {12, -3, 4, 3, -1}, {0, -1, -4, 0, -1}, {9, 2, 0, 1, -1}, {21, -7, -1, 3, -1},
    {15, -5, 5, 3, -1}, {9, -2, -2, 0, -1}, {23, 1, -4, 3, -1}, {22, 2, -3, 2, -1}, {22, -2, -4, 1, -1},
    {13, -2, 4, 1, -1}, {17, 5, -1, 0, -1}, {7, 5, -2, 2, -1}, {14, 0, -5, 1, -1}, {20, 6, -1, 3, -1},
};

// kLastUnplaceableTileTurns as actions: a draw, a tile and a meeple move a
// turn. Each tile action names its cell in the view of the game so far.
inline std::vector<Action> LastUnplaceableTileHistory() {
  CarcassonneState state(LoadGame("carcassonne"));
  std::vector<Action> history;
  auto play = [&](Action action) {
    history.push_back(action);
    state.ApplyAction(action);
  };
  for (const FixtureTurn& turn : kLastUnplaceableTileTurns) {
    play(turn.draw);
    play(state.TileAction(BOARD_SIZE / 2 + turn.dx, BOARD_SIZE / 2 + turn.dy,
                          turn.rot));
    play(kMeepleActionOffset + turn.meeple + 1);
  }
  return history;
}

inline std::unique_ptr<State> LastUnplaceableTileState(
    std::shared_ptr<const Game> game) {
  auto state = std::make_unique<CarcassonneState>(std::move(game));
  for (Action action : LastUnplaceableTileHistory()) {
    SPIEL_CHECK_FALSE(state->IsTerminal());
    const auto legal = state->LegalActions();
    SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), action) != legal.end());
    state->ApplyAction(action);
  }
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  SPIEL_CHECK_EQ(state->UnderlyingState().completed_turns, 70);
  SPIEL_CHECK_EQ(state->UnderlyingState().getTotalRemaining(), 1);
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>{2});
  return state;
}

}  // namespace carcassonne
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CARCASSONNE_CARCASSONNE_TEST_UTILS_H_
