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
// directly, without another player decision. Fixed actions make this fixture
// independent of RNG/distribution implementations and legal-move ordering.
// Random games almost always leave the all-city tile somewhere to go, so this
// one held it back and placed every tile where it left it the fewest spots.
// Tile actions name cells of the view, which follows the tiles. The actions
// are numbered as when the game was found, four per cell (the rotations) and
// the meeple moves from 1764: LastUnplaceableTileHistory() renumbers them.
static_assert(VIEW_SIZE == 21, "kLastUnplaceableTileHistory is a game in a 21x21 view");
inline constexpr Action kLastUnplaceableTileHistory[] = {
    20,887,1764,15,798,1767,20,715,1766,17,960,1764,7,885,1774,
    9,891,1765,1,1055,1769,20,1055,1773,0,873,1769,21,790,1764,
    18,710,1770,21,629,1770,22,973,1770,4,871,1764,5,953,1765,
    17,1036,1764,14,867,1764,3,962,1764,15,865,1764,0,1037,1770,
    21,812,1764,18,899,1767,11,786,1764,12,780,1766,6,721,1764,
    21,860,1767,19,1052,1765,20,818,1764,8,951,1764,21,551,1764,
    20,1207,1764,21,1204,1764,10,967,1764,21,698,1764,3,1035,1764,
    21,1202,1764,13,1029,1764,14,731,1764,15,1137,1764,6,1051,1764,
    16,638,1764,18,1126,1764,16,777,1764,15,1132,1764,1,1196,1764,
    8,791,1764,20,1221,1764,7,1143,1764,10,1045,1764,3,704,1765,
    0,1145,1769,19,622,1764,22,1291,1764,20,647,1764,16,1224,1764,
    19,735,1764,12,1211,1764,0,544,1764,9,893,1764,21,775,1764,
    15,1287,1764,9,708,1764,23,555,1764,22,642,1764,22,541,1764,
    13,1213,1764,17,820,1764,7,738,1764,14,465,1764,20,827,1764};

// kLastUnplaceableTileHistory in today's action numbers: chance actions stay,
// tile actions keep their cell and rotation, meeple moves their position.
inline std::vector<Action> LastUnplaceableTileHistory() {
  constexpr int kOldCellPlanes = 4;
  constexpr Action kOldMeepleOffset = VIEW_SIZE * VIEW_SIZE * kOldCellPlanes;
  std::vector<Action> history;
  int step = 0;
  for (Action action : kLastUnplaceableTileHistory) {
    // Every turn is a draw, a tile and a meeple move.
    if (step++ % 3 == 0) {
      history.push_back(action);
    } else if (action < kOldMeepleOffset) {
      history.push_back(action / kOldCellPlanes * kCellActionPlanes + action % kOldCellPlanes);
    } else {
      history.push_back(action - kOldMeepleOffset + kMeepleActionOffset);
    }
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
