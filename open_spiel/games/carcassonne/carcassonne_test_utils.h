#ifndef OPEN_SPIEL_GAMES_CARCASSONNE_CARCASSONNE_TEST_UTILS_H_
#define OPEN_SPIEL_GAMES_CARCASSONNE_CARCASSONNE_TEST_UTILS_H_

#include <algorithm>
#include <memory>

#include "open_spiel/games/carcassonne/carcassonne.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace carcassonne {

// A legal full-length game with the all-city tile (chance action 2) left in
// the deck and no matching placement. The next draw must score and terminate
// directly, without another player decision. Fixed actions make this fixture
// independent of RNG/distribution implementations and legal-move ordering.
// Tile actions encode the board size; this game was found on 21x21.
static_assert(BOARD_SIZE == 21, "kLastUnplaceableTileHistory is a 21x21 game");
inline constexpr Action kLastUnplaceableTileHistory[] = {
    21,967,1766,18,798,1768,0,962,1764,20,1047,1764,17,1040,1764,
    13,1131,1768,0,1135,1769,6,1213,1764,21,1210,1765,22,1292,1764,
    8,959,1767,14,953,1768,7,950,1764,15,1032,1764,9,1297,1765,
    19,1303,1764,20,803,1766,14,875,1768,12,870,1764,18,1288,1764,
    3,788,1765,18,968,1766,4,864,1764,21,944,1764,21,1385,1765,
    15,785,1764,22,702,1764,21,1471,1766,10,1055,1764,8,861,1764,
    6,778,1764,20,858,1764,0,1464,1764,20,1117,1766,22,1377,1764,
    16,698,1767,23,712,1764,19,886,1764,17,1201,1764,14,1388,1764,
    16,1219,1764,20,852,1764,21,889,1764,20,614,1764,19,1125,1764,
    20,1552,1764,7,1207,1767,9,706,1765,15,1136,1764,1,628,1764,
    11,1141,1764,7,1224,1764,9,795,1764,3,1305,1764,1,1638,1769,
    21,1231,1764,21,1632,1764,12,1308,1764,3,783,1764,10,708,1764,
    5,1115,1764,20,1558,1765,15,1109,1764,0,1549,1764,15,610,1764,
    21,525,1764,22,1234,1764,17,1027,1764,13,1021,1764,16,1718,1764};

inline std::unique_ptr<State> LastUnplaceableTileState(
    std::shared_ptr<const Game> game) {
  auto state = std::make_unique<CarcassonneState>(std::move(game));
  for (Action action : kLastUnplaceableTileHistory) {
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
