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
inline constexpr Action kLastUnplaceableTileHistory[] = {
    20,511,902,7,390,902,5,394,900,16,452,900,22,459,902,
    14,569,902,18,399,903,9,565,903,0,629,900,9,507,900,
    6,572,903,20,634,900,8,502,902,9,514,900,15,579,900,
    22,460,903,8,496,900,10,438,900,6,494,902,21,691,902,
    3,490,900,16,485,901,3,549,900,3,552,900,22,447,901,
    7,608,900,10,560,900,11,558,900,23,335,904,12,604,900,
    12,600,900,13,440,900,15,382,900,21,275,902,14,543,902,
    15,547,900,4,662,902,22,623,900,21,639,902,0,429,900,
    7,720,900,0,581,905,21,681,900,18,678,900,20,781,900,
    16,667,900,13,737,900,15,743,900,18,697,900,0,732,900,
    20,586,900,14,703,900,20,761,900,21,465,900,1,369,900,
    19,707,900,21,277,900,1,711,900,19,842,900,17,519,900,
    20,481,900,15,423,900,20,713,900,17,717,900,17,846,900,
    20,218,900,21,222,900,21,653,900,19,777,900,21,625,900};

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
