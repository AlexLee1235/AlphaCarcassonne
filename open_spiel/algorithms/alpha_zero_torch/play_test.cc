#include "open_spiel/algorithms/alpha_zero_torch/alpha_zero.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "open_spiel/games/carcassonne/carcassonne_test_utils.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {
namespace {

class LastDrawGame : public carcassonne::CarcassonneGame {
 public:
  explicit LastDrawGame(bool already_terminal = false)
      : CarcassonneGame({}), already_terminal_(already_terminal) {}

  std::unique_ptr<State> NewInitialState() const override {
    auto state = carcassonne::LastUnplaceableTileState(shared_from_this());
    if (already_terminal_) state->ApplyAction(2);
    return state;
  }

 private:
  bool already_terminal_;
};

void ChanceTerminalTest() {
  NoopLogger logger;
  std::mt19937 rng(0);
  // No player can move here. Accessing a bot at all is a regression.
  std::vector<std::unique_ptr<MCTSBot>> bots(2);
  for (bool already_terminal : {false, true}) {
    auto game = std::make_shared<LastDrawGame>(already_terminal);
    const auto trajectory = PlayGame(&logger, 0, *game, &bots, &rng,
                                     1.0, 10, 2.0);
    SPIEL_CHECK_TRUE(trajectory.states.empty());
    SPIEL_CHECK_EQ(trajectory.returns, (std::vector<double>{1.0, -1.0}));
  }
}

void PlayerTerminalAndCutoffTest() {
  NoopLogger logger;
  auto game = LoadGame("carcassonne(max_turns=1)");
  for (bool cutoff : {false, true}) {
    std::mt19937 rng(0);
    auto evaluator = std::make_shared<RandomRolloutEvaluator>(1, 0);
    std::vector<std::unique_ptr<MCTSBot>> bots;
    for (int player = 0; player < 2; ++player) {
      bots.push_back(std::make_unique<MCTSBot>(
          *game, evaluator, 2.0, 16, 10, false, player, false,
          ChildSelectionPolicy::PUCT, 0.0, 0.0, true));
    }
    const auto trajectory = PlayGame(&logger, 0, *game, &bots, &rng,
                                     1.0, 10, cutoff ? -1.0 : 2.0);
    SPIEL_CHECK_EQ(trajectory.states.size(), cutoff ? 1 : 2);
    SPIEL_CHECK_EQ(trajectory.returns.size(), 2);
    SPIEL_CHECK_EQ(trajectory.returns[0] + trajectory.returns[1], 0.0);
    SPIEL_CHECK_TRUE(std::isfinite(trajectory.returns[0]));
    if (cutoff) {
      const auto& sample = trajectory.states.back();
      SPIEL_CHECK_EQ(trajectory.returns[sample.current_player], sample.value);
    }
    // The recorded side groups must agree with the legal meeple moves: each
    // legal meeple side is the lowest side of its feature, and each legal
    // farmer half-edge the lowest half-edge of its field.
    constexpr int kFields = carcassonne::kFieldGroupOffset;
    for (const auto& sample : trajectory.states) {
      const auto& groups = sample.symmetry_context;
      for (int side = 0; side < 4; ++side) {
        if (groups[side] != -1) SPIEL_CHECK_EQ(groups[groups[side]], groups[side]);
      }
      for (int half_edge = 0; half_edge < HALF_EDGE_COUNT; ++half_edge) {
        const int group = groups[kFields + half_edge];
        if (group != -1) SPIEL_CHECK_EQ(groups[kFields + group], group);
      }
      for (Action action : sample.legal_actions) {
        // The big meeple names the same spots.
        const int pos = meepleSpot(action - carcassonne::kMeepleActionOffset - 1);
        if (carcassonne::IsTileAction(action)) {
          SPIEL_CHECK_TRUE(groups == carcassonne::kNoSideGroups);
        } else if (action < carcassonne::kMeepleActionOffset) {
          continue;  // A cell chosen for a portal or the princess.
        } else if (pos >= 0 && pos < 4) {
          SPIEL_CHECK_EQ(groups[pos], pos);
        } else if (pos >= MEEPLE_POS_FIELD && pos < MEEPLE_POS_INNER_FIELD) {
          SPIEL_CHECK_EQ(groups[kFields + pos - MEEPLE_POS_FIELD],
                         pos - MEEPLE_POS_FIELD);
        }
      }
    }
  }
}

// temperature_drop counts decisions: before it the move is sampled from the
// visit counts, from it on it is the most visited one. The chance draws in
// between do not count; when they did, a base game stopped sampling after
// about two thirds as many decisions.
void TemperatureDropCountsDecisionsTest() {
  NoopLogger logger;
  auto game = LoadGame("carcassonne(max_turns=6)");
  // Decisions alternate tile, meeple: the last one sampled is the third tile.
  constexpr int kDrop = 5;
  int last_sampled_off_best = 0;
  for (int seed = 0; seed < 20; ++seed) {
    std::mt19937 rng(seed);
    auto evaluator = std::make_shared<RandomRolloutEvaluator>(1, seed);
    std::vector<std::unique_ptr<MCTSBot>> bots;
    for (int player = 0; player < 2; ++player) {
      bots.push_back(std::make_unique<MCTSBot>(
          *game, evaluator, 2.0, 16, 10, false, 2 * seed + player, false,
          ChildSelectionPolicy::PUCT, 0.0, 0.0, true));
    }
    const auto trajectory = PlayGame(&logger, 0, *game, &bots, &rng,
                                     1.0, kDrop, 2.0);
    SPIEL_CHECK_GT(trajectory.states.size(), kDrop);
    for (int i = 0; i < trajectory.states.size(); ++i) {
      const Trajectory::State& sample = trajectory.states[i];
      double best = 0.0;
      double chosen = 0.0;
      for (const auto& [action, probability] : sample.policy) {
        best = std::max(best, probability);
        if (action == sample.action) chosen = probability;
      }
      if (i >= kDrop) {
        SPIEL_CHECK_EQ(chosen, best);
      } else if (i == kDrop - 1 && chosen < best) {
        ++last_sampled_off_best;
      }
    }
  }
  SPIEL_CHECK_GT(last_sampled_off_best, 0);
}

}  // namespace
}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel

int main() {
  open_spiel::algorithms::torch_az::ChanceTerminalTest();
  open_spiel::algorithms::torch_az::PlayerTerminalAndCutoffTest();
  open_spiel::algorithms::torch_az::TemperatureDropCountsDecisionsTest();
}
