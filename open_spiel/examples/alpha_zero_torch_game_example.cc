// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <array>
#include <cstdio>
#include <future>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/synchronization/mutex.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/algorithms/alpha_zero_torch/device_manager.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/bots/human/human_bot.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(std::string, game, "tic_tac_toe", "The name of the game to play.");
ABSL_FLAG(std::string, player1, "az", "Who controls player1.");
ABSL_FLAG(std::string, player2, "random", "Who controls player2.");
ABSL_FLAG(std::string, az_path, "", "Path to AZ experiment.");
ABSL_FLAG(std::string, az_graph_def, "vpnet.pb", "AZ graph definition file name.");
ABSL_FLAG(std::string, az2_path, "", "Optional path to player2's AZ experiment. Falls back to --az_path.");
ABSL_FLAG(std::string, az2_graph_def, "", "Optional player2 AZ graph definition. Falls back to --az_graph_def.");
ABSL_FLAG(double, uct_c, 2, "UCT exploration constant.");
ABSL_FLAG(int, rollout_count, 10, "How many rollouts per evaluation.");
ABSL_FLAG(int, max_simulations, 10000, "How many simulations to run.");
ABSL_FLAG(int, num_games, 1, "How many games to play.");
ABSL_FLAG(int, num_workers, 1, "How many games to play in parallel.");
ABSL_FLAG(int, max_memory_mb, 1000, "The maximum memory used before cutting the search short.");
ABSL_FLAG(int, az_checkpoint, -1, "Checkpoint of AZ model.");
ABSL_FLAG(int, az2_checkpoint, -2, "Optional player2 AZ checkpoint. Falls back to --az_checkpoint.");
ABSL_FLAG(int, az_batch_size, 1, "Batch size of AZ inference.");
ABSL_FLAG(int, az_threads, 1, "Number of threads to run for AZ inference.");
ABSL_FLAG(int, az_cache_size, 16384, "Cache size of AZ algorithm.");
ABSL_FLAG(int, az_cache_shards, 1, "Cache shards of AZ algorithm.");
ABSL_FLAG(std::string, az_device, "/cpu:0", "Torch device for the AZ model, e.g. /cpu:0, cpu, cuda:0.");
ABSL_FLAG(std::string, az2_device, "", "Optional player2 AZ device. Falls back to --az_device.");
ABSL_FLAG(bool, az_value_is_current_player, false,
          "Interpret AZ value output as current-player value.");
ABSL_FLAG(bool, solve, true, "Whether to use MCTS-Solver.");
ABSL_FLAG(uint_fast32_t, seed, 0, "Seed for MCTS.");
ABSL_FLAG(bool, verbose, false, "Show the MCTS stats of possible moves.");
ABSL_FLAG(bool, quiet, false, "Hide per-action state traces and game actions.");

constexpr int kUnsetAz2Checkpoint = -2;

uint_fast32_t Seed() {
    uint_fast32_t seed = absl::GetFlag(FLAGS_seed);
    return seed != 0 ? seed : absl::ToUnixMicros(absl::Now());
}

uint_fast32_t SeedWithOffset(uint_fast32_t seed, uint_fast32_t offset) { return seed + 2654435761u * (offset + 1); }

class SplitEvaluator : public open_spiel::algorithms::Evaluator {
  public:
    SplitEvaluator(std::shared_ptr<open_spiel::algorithms::Evaluator> prior,
                   std::shared_ptr<open_spiel::algorithms::Evaluator> value)
        : prior_(std::move(prior)), value_(std::move(value)) {}

    std::vector<double> Evaluate(const open_spiel::State &state) override { return value_->Evaluate(state); }

    open_spiel::ActionsAndProbs Prior(const open_spiel::State &state) override { return prior_->Prior(state); }

  private:
    std::shared_ptr<open_spiel::algorithms::Evaluator> prior_;
    std::shared_ptr<open_spiel::algorithms::Evaluator> value_;
};

bool RequiresAZEvaluator(const std::string &type) {
    return type == "az" || type == "az_prior_rollout_value" || type == "network_prior_rollout_value" ||
           type == "uniform_prior_az_value";
}

struct AZSpec {
    std::string path;
    std::string graph_def;
    std::string device;
    int checkpoint;
};

AZSpec GetAZSpecForPlayer(open_spiel::Player player) {
    if (player == 1) {
        const std::string az2_path = absl::GetFlag(FLAGS_az2_path);
        const std::string az2_graph_def = absl::GetFlag(FLAGS_az2_graph_def);
        const std::string az2_device = absl::GetFlag(FLAGS_az2_device);
        const int az2_checkpoint = absl::GetFlag(FLAGS_az2_checkpoint);
        return AZSpec{az2_path.empty() ? absl::GetFlag(FLAGS_az_path) : az2_path,
                      az2_graph_def.empty() ? absl::GetFlag(FLAGS_az_graph_def) : az2_graph_def,
                      az2_device.empty() ? absl::GetFlag(FLAGS_az_device) : az2_device,
                      az2_checkpoint == kUnsetAz2Checkpoint ? absl::GetFlag(FLAGS_az_checkpoint) : az2_checkpoint};
    }
    return AZSpec{absl::GetFlag(FLAGS_az_path), absl::GetFlag(FLAGS_az_graph_def), absl::GetFlag(FLAGS_az_device),
                  absl::GetFlag(FLAGS_az_checkpoint)};
}

bool SameAZSpec(const AZSpec &left, const AZSpec &right) {
    return left.path == right.path && left.graph_def == right.graph_def && left.device == right.device &&
           left.checkpoint == right.checkpoint;
}

std::shared_ptr<open_spiel::algorithms::torch_az::VPNetEvaluator>
InitAZEvaluator(const open_spiel::Game &game, const AZSpec &spec,
                std::vector<std::unique_ptr<open_spiel::algorithms::torch_az::DeviceManager>> *device_managers) {
    if (spec.path.empty()) {
        open_spiel::SpielFatalError("AlphaZero path must be specified.");
    }

    auto device_manager = std::make_unique<open_spiel::algorithms::torch_az::DeviceManager>();
    device_manager->AddDevice(open_spiel::algorithms::torch_az::VPNetModel(game, spec.path, spec.graph_def, spec.device));
    device_manager->Get(0, 0)->LoadCheckpoint(spec.checkpoint);
    auto evaluator =
        std::make_shared<open_spiel::algorithms::torch_az::VPNetEvaluator>(device_manager.get(),
                                                                           /*batch_size=*/absl::GetFlag(FLAGS_az_batch_size),
                                                                           /*threads=*/absl::GetFlag(FLAGS_az_threads),
                                                                           /*cache_size=*/absl::GetFlag(FLAGS_az_cache_size),
                                                                           /*cache_shards=*/absl::GetFlag(FLAGS_az_cache_shards),
                                                                           /*batch_wait_ms=*/1,
                                                                           absl::GetFlag(FLAGS_az_value_is_current_player));
    device_managers->push_back(std::move(device_manager));
    return evaluator;
}

std::unique_ptr<open_spiel::Bot> InitBot(std::string type, const open_spiel::Game &game, open_spiel::Player player,
                                         std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator,
                                         std::shared_ptr<open_spiel::algorithms::torch_az::VPNetEvaluator> az_evaluator,
                                         uint_fast32_t seed) {
    if (type == "az") {
        if (az_evaluator == nullptr) {
            open_spiel::SpielFatalError("AlphaZero evaluator is not initialized.");
        }
        return std::make_unique<open_spiel::algorithms::MCTSBot>(
            game, std::move(az_evaluator), absl::GetFlag(FLAGS_uct_c), absl::GetFlag(FLAGS_max_simulations),
            absl::GetFlag(FLAGS_max_memory_mb), absl::GetFlag(FLAGS_solve), seed, absl::GetFlag(FLAGS_verbose),
            open_spiel::algorithms::ChildSelectionPolicy::PUCT, 0, 0,
            /*dont_return_chance_node=*/true);
    }
    if (type == "puct_mcts") {
        return std::make_unique<open_spiel::algorithms::MCTSBot>(
            game, std::move(evaluator), absl::GetFlag(FLAGS_uct_c), absl::GetFlag(FLAGS_max_simulations),
            absl::GetFlag(FLAGS_max_memory_mb), absl::GetFlag(FLAGS_solve), seed, absl::GetFlag(FLAGS_verbose),
            open_spiel::algorithms::ChildSelectionPolicy::PUCT, 0, 0,
            /*dont_return_chance_node=*/true);
    }
    if (type == "az_prior_rollout_value" || type == "network_prior_rollout_value") {
        if (az_evaluator == nullptr) {
            open_spiel::SpielFatalError("AlphaZero evaluator is not initialized.");
        }
        auto split_evaluator = std::make_shared<SplitEvaluator>(az_evaluator, evaluator);
        return std::make_unique<open_spiel::algorithms::MCTSBot>(
            game, std::move(split_evaluator), absl::GetFlag(FLAGS_uct_c), absl::GetFlag(FLAGS_max_simulations),
            absl::GetFlag(FLAGS_max_memory_mb), absl::GetFlag(FLAGS_solve), seed, absl::GetFlag(FLAGS_verbose),
            open_spiel::algorithms::ChildSelectionPolicy::PUCT, 0, 0,
            /*dont_return_chance_node=*/true);
    }
    if (type == "uniform_prior_az_value") {
        if (az_evaluator == nullptr) {
            open_spiel::SpielFatalError("AlphaZero evaluator is not initialized.");
        }
        auto split_evaluator = std::make_shared<SplitEvaluator>(evaluator, az_evaluator);
        return std::make_unique<open_spiel::algorithms::MCTSBot>(
            game, std::move(split_evaluator), absl::GetFlag(FLAGS_uct_c), absl::GetFlag(FLAGS_max_simulations),
            absl::GetFlag(FLAGS_max_memory_mb), absl::GetFlag(FLAGS_solve), seed, absl::GetFlag(FLAGS_verbose),
            open_spiel::algorithms::ChildSelectionPolicy::PUCT, 0, 0,
            /*dont_return_chance_node=*/true);
    }
    if (type == "human") {
        return std::make_unique<open_spiel::HumanBot>();
    }
    if (type == "mcts") {
        return std::make_unique<open_spiel::algorithms::MCTSBot>(
            game, std::move(evaluator), absl::GetFlag(FLAGS_uct_c), absl::GetFlag(FLAGS_max_simulations),
            absl::GetFlag(FLAGS_max_memory_mb), absl::GetFlag(FLAGS_solve), seed, absl::GetFlag(FLAGS_verbose),
            open_spiel::algorithms::ChildSelectionPolicy::UCT, 0, 0,
            /*dont_return_chance_node=*/true);
    }
    if (type == "random") {
        return open_spiel::MakeUniformRandomBot(player, seed);
    }

    open_spiel::SpielFatalError("Bad player type. Known types: az, az_prior_rollout_value, "
                                "network_prior_rollout_value, uniform_prior_az_value, puct_mcts, human, "
                                "mcts, random");
}

open_spiel::Action GetAction(const open_spiel::State &state, std::string action_str) {
    for (open_spiel::Action action : state.LegalActions()) {
        if (action_str == state.ActionToString(state.CurrentPlayer(), action))
            return action;
    }
    return open_spiel::kInvalidAction;
}

std::pair<std::vector<double>, std::vector<std::string>> PlayGame(const open_spiel::Game &game,
                                                                  std::vector<std::unique_ptr<open_spiel::Bot>> &bots,
                                                                  std::mt19937 &rng,
                                                                  const std::vector<std::string> &initial_actions) {
    bool quiet = absl::GetFlag(FLAGS_quiet);
    std::unique_ptr<open_spiel::State> state = game.NewInitialState();
    std::vector<std::string> history;

    if (!quiet)
        std::cerr << "Initial state:\n" << state << std::endl;

    // Play the initial actions (if there are any).
    for (const auto &action_str : initial_actions) {
        open_spiel::Player current_player = state->CurrentPlayer();
        open_spiel::Action action = GetAction(*state, action_str);

        if (action == open_spiel::kInvalidAction)
            open_spiel::SpielFatalError(absl::StrCat("Invalid action: ", action_str));

        history.push_back(action_str);
        state->ApplyAction(action);

        if (!quiet) {
            std::cerr << "Player " << current_player << " forced action: " << action_str << std::endl;
            std::cerr << "Next state:\n" << state->ToString() << std::endl;
        }
    }

    while (!state->IsTerminal()) {
        open_spiel::Player player = state->CurrentPlayer();

        open_spiel::Action action;
        if (state->IsChanceNode()) {
            // Chance node; sample one according to underlying distribution.
            open_spiel::ActionsAndProbs outcomes = state->ChanceOutcomes();
            action = open_spiel::SampleAction(outcomes, rng).first;
        } else {
            // The state must be a decision node, ask the right bot to make its
            // action.
            action = bots[player]->Step(*state);
        }
        if (!quiet)
            std::cerr << "Player " << player << " chose action: " << state->ActionToString(player, action) << std::endl;

        // Inform the other bot of the action performed.
        for (open_spiel::Player p = 0; p < bots.size(); ++p) {
            if (p != player) {
                bots[p]->InformAction(*state, player, action);
            }
        }

        // Update history and get the next state.
        history.push_back(state->ActionToString(player, action));
        state->ApplyAction(action);

        if (!quiet)
            std::cerr << "Next state:\n" << state->ToString() << std::endl;
    }

    if (!quiet) {
        std::cerr << "Returns: " << absl::StrJoin(state->Returns(), ", ") << std::endl;
        std::cerr << "Game actions: " << absl::StrJoin(history, ", ") << std::endl;
    }

    return {state->Returns(), history};
}

int main(int argc, char **argv) {
    std::vector<char *> positional_args = absl::ParseCommandLine(argc, argv);
    const uint_fast32_t base_seed = Seed();

    // Create the game.
    std::string game_name = absl::GetFlag(FLAGS_game);
    std::cerr << "Game: " << game_name << std::endl;
    std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame(game_name);

    // Ensure the game is AlphaZero-compatible and arguments are compatible.
    open_spiel::GameType game_type = game->GetType();
    if (game->NumPlayers() != 2)
        open_spiel::SpielFatalError("AlphaZero can only handle 2-player games.");
    if (game_type.reward_model != open_spiel::GameType::RewardModel::kTerminal)
        open_spiel::SpielFatalError("Game must have terminal rewards.");
    if (game_type.dynamics != open_spiel::GameType::Dynamics::kSequential)
        open_spiel::SpielFatalError("Game must have sequential turns.");
    const std::string player1 = absl::GetFlag(FLAGS_player1);
    const std::string player2 = absl::GetFlag(FLAGS_player2);
    const int num_games = absl::GetFlag(FLAGS_num_games);
    const int worker_count = num_games > 0 ? std::min(num_games, std::max(1, absl::GetFlag(FLAGS_num_workers))) : 0;
    if ((player1 == "human" || player2 == "human") && worker_count > 1) {
        open_spiel::SpielFatalError("Human players can only be used with --num_workers=1.");
    }

    const bool player1_needs_az = RequiresAZEvaluator(player1);
    const bool player2_needs_az = RequiresAZEvaluator(player2);

    std::vector<std::unique_ptr<open_spiel::algorithms::torch_az::DeviceManager>> device_managers;
    std::shared_ptr<open_spiel::algorithms::torch_az::VPNetEvaluator> az_evaluator1;
    std::shared_ptr<open_spiel::algorithms::torch_az::VPNetEvaluator> az_evaluator2;
    AZSpec az_spec1 = GetAZSpecForPlayer(0);
    AZSpec az_spec2 = GetAZSpecForPlayer(1);
    if (player1_needs_az) {
        az_evaluator1 = InitAZEvaluator(*game, az_spec1, &device_managers);
    }
    if (player2_needs_az) {
        if (player1_needs_az && SameAZSpec(az_spec1, az_spec2)) {
            az_evaluator2 = az_evaluator1;
        } else {
            az_evaluator2 = InitAZEvaluator(*game, az_spec2, &device_managers);
        }
    }

    std::vector<std::string> initial_actions;
    for (int i = 1; i < positional_args.size(); ++i) {
        initial_actions.push_back(positional_args[i]);
    }

    absl::Mutex results_mutex;
    std::map<std::string, int> histories;
    std::vector<double> overall_returns(2, 0);
    std::vector<int> overall_wins(2, 0);
    int overall_draws = 0;
    int completed_games = 0;

    std::vector<std::future<void>> futures;
    futures.reserve(worker_count);
    const int games_per_worker = worker_count > 0 ? num_games / worker_count : 0;
    const int extra_games = worker_count > 0 ? num_games % worker_count : 0;
    for (int worker = 0; worker < worker_count; ++worker) {
        const int worker_games = games_per_worker + (worker < extra_games ? 1 : 0);
        const int first_game = worker * games_per_worker + std::min(worker, extra_games);
        futures.push_back(std::async(std::launch::async, [&, worker, worker_games, first_game]() {
            std::shared_ptr<const open_spiel::Game> worker_game = open_spiel::LoadGame(game_name);
            for (int local_game = 0; local_game < worker_games; ++local_game) {
                const int game_num = first_game + local_game;

                const uint_fast32_t game_seed =
                    SeedWithOffset(base_seed, static_cast<uint_fast32_t>(game_num + 1000003 * worker));
                std::mt19937 rng(game_seed);
                auto evaluator = std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(
                    absl::GetFlag(FLAGS_rollout_count), SeedWithOffset(game_seed, 101));

                std::vector<std::unique_ptr<open_spiel::Bot>> bots;
                bots.push_back(InitBot(player1, *worker_game, 0, evaluator, az_evaluator1, SeedWithOffset(game_seed, 201)));
                bots.push_back(InitBot(player2, *worker_game, 1, evaluator, az_evaluator2, SeedWithOffset(game_seed, 301)));

                auto [returns, history] = PlayGame(*worker_game, bots, rng, initial_actions);

                {
                    absl::MutexLock lock(&results_mutex);
                    histories[absl::StrJoin(history, " ")] += 1;
                    bool has_winner = false;
                    for (int i = 0; i < returns.size(); ++i) {
                        double v = returns[i];
                        overall_returns[i] += v;
                        if (v > 0) {
                            overall_wins[i] += 1;
                            has_winner = true;
                        }
                    }
                    if (!has_winner) {
                        ++overall_draws;
                    }
                    ++completed_games;
                    std::cerr << "[game " << (game_num + 1) << " done " << completed_games << "/" << num_games
                              << " worker=" << worker << "] returns: " << absl::StrJoin(returns, ", ")
                              << " | cumulative wins: " << absl::StrJoin(overall_wins, ", ") << " losses: " << overall_wins[1]
                              << ", " << overall_wins[0] << " draws: " << overall_draws
                              << " returns: " << absl::StrJoin(overall_returns, ", ") << std::endl;
                }
            }
        }));
    }

    for (auto &future : futures) {
        future.get();
    }

    std::cerr << "Number of games played: " << completed_games << std::endl;
    std::cerr << "Number of distinct games played: " << histories.size() << std::endl;
    std::cerr << "Players: " << player1 << ", " << player2 << std::endl;
    std::cerr << "Overall wins: " << absl::StrJoin(overall_wins, ", ") << std::endl;
    std::cerr << "Overall losses: " << overall_wins[1] << ", " << overall_wins[0] << std::endl;
    std::cerr << "Overall draws: " << overall_draws << std::endl;
    std::cerr << "Overall returns: " << absl::StrJoin(overall_returns, ", ") << std::endl;

    return 0;
}
