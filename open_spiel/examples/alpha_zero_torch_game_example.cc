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
#include <cmath>
#include <cstdio>
#include <fstream>
#include <future>
#include <iomanip>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/synchronization/mutex.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/algorithms/alpha_zero_torch/device_manager.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/bots/human/human_bot.h"
#include "open_spiel/games/carcassonne/carcassonne.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/json.h"

ABSL_FLAG(std::string, game, "tic_tac_toe", "The name of the game to play.");
ABSL_FLAG(std::string, p1_type, "az", "Who controls player 1.");
ABSL_FLAG(std::string, p2_type, "random", "Who controls player 2.");
ABSL_FLAG(std::string, p1_az_path, "", "Path to player 1's AZ experiment.");
ABSL_FLAG(std::string, p2_az_path, "", "Path to player 2's AZ experiment.");
ABSL_FLAG(std::string, p1_az_graph_def, "vpnet.pb", "Player 1 AZ graph definition file name.");
ABSL_FLAG(std::string, p2_az_graph_def, "vpnet.pb", "Player 2 AZ graph definition file name.");
ABSL_FLAG(int, p1_az_checkpoint, -1, "Checkpoint of player 1's AZ model.");
ABSL_FLAG(int, p2_az_checkpoint, -1, "Checkpoint of player 2's AZ model.");
ABSL_FLAG(std::string, p1_az_device, "/cpu:0", "Torch device for player 1's AZ model, e.g. /cpu:0, cpu, cuda:0.");
ABSL_FLAG(std::string, p2_az_device, "/cpu:0", "Torch device for player 2's AZ model, e.g. /cpu:0, cpu, cuda:0.");
ABSL_FLAG(double, p1_uct_c, 2, "Player 1 UCT exploration constant.");
ABSL_FLAG(double, p2_uct_c, 2, "Player 2 UCT exploration constant.");
ABSL_FLAG(int, p1_max_simulations, 10000, "How many simulations player 1 runs per move.");
ABSL_FLAG(int, p2_max_simulations, 10000, "How many simulations player 2 runs per move.");
ABSL_FLAG(int, p1_max_memory_mb, 1000, "Player 1 maximum memory before cutting the search short.");
ABSL_FLAG(int, p2_max_memory_mb, 1000, "Player 2 maximum memory before cutting the search short.");
ABSL_FLAG(bool, p1_solve, true, "Whether player 1 uses MCTS-Solver.");
ABSL_FLAG(bool, p2_solve, true, "Whether player 2 uses MCTS-Solver.");
ABSL_FLAG(int, rollout_count, 10, "How many rollouts per evaluation.");
ABSL_FLAG(int, num_games, 2, "How many games to play. Must be even for paired same-deck matches.");
ABSL_FLAG(int, num_workers, 1, "How many games to play in parallel.");
ABSL_FLAG(int, az_batch_size, 0, "Batch size of AZ inference. 0 auto-scales from --num_workers.");
ABSL_FLAG(int, az_threads, 1, "Number of threads to run for AZ inference.");
ABSL_FLAG(int, az_cache_size, 16384, "Cache size of AZ algorithm.");
ABSL_FLAG(int, az_cache_shards, 1, "Cache shards of AZ algorithm.");
ABSL_FLAG(bool, az_value_is_current_player, false,
          "Interpret AZ value output as current-player value.");
ABSL_FLAG(uint_fast32_t, seed, 0, "Seed for MCTS.");
ABSL_FLAG(bool, verbose, false, "Show the MCTS stats of possible moves.");
ABSL_FLAG(bool, quiet, true, "Hide per-action state traces and game actions.");
ABSL_FLAG(std::string, log_dir, "", "Directory for per-worker JSONL game trajectory logs. Empty disables logging.");

uint_fast32_t Seed() {
    uint_fast32_t seed = absl::GetFlag(FLAGS_seed);
    return seed != 0 ? seed : absl::ToUnixMicros(absl::Now());
}

uint_fast32_t SeedWithOffset(uint_fast32_t seed, uint_fast32_t offset) { return seed + 2654435761u * (offset + 1); }

struct ConfidenceInterval {
    double low;
    double high;
};

struct GameResult {
    std::vector<std::string> history;
    std::vector<double> final_scores;
};

struct GameLogContext {
    std::ostream *out = nullptr;
    int worker = 0;
    int game_id = 0;
    int pair = 0;
    int leg = 0;
    std::string first;
};

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

ConfidenceInterval Wilson95ConfidenceInterval(int successes, int trials) {
    if (trials <= 0) {
        return {0.0, 0.0};
    }
    constexpr double z = 1.959963984540054;
    const double n = static_cast<double>(trials);
    const double p = static_cast<double>(successes) / n;
    const double z2 = z * z;
    const double denominator = 1.0 + z2 / n;
    const double center = (p + z2 / (2.0 * n)) / denominator;
    const double half_width = z * std::sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denominator;
    return {std::max(0.0, center - half_width), std::min(1.0, center + half_width)};
}

std::string FormatRate(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(4) << value;
    return out.str();
}

std::string FormatScoreValue(double value) {
    if (std::fabs(value - std::round(value)) < 1e-9) {
        return std::to_string(static_cast<long long>(std::llround(value)));
    }
    return FormatRate(value);
}

std::string FormatScoreVector(const std::vector<double> &values) {
    std::vector<std::string> parts;
    parts.reserve(values.size());
    for (double value : values) {
        parts.push_back(FormatScoreValue(value));
    }
    return absl::StrJoin(parts, ",");
}

std::string FormatConfidenceInterval(const ConfidenceInterval &interval) {
    return absl::StrCat("[", FormatRate(interval.low), ",", FormatRate(interval.high), "]");
}

std::vector<double> FinalScores(const open_spiel::State &state) {
    const auto *carcassonne_state = dynamic_cast<const open_spiel::carcassonne::CarcassonneState *>(&state);
    if (carcassonne_state != nullptr) {
        const auto &underlying = carcassonne_state->UnderlyingState();
        return {static_cast<double>(underlying.player_scores[0]), static_cast<double>(underlying.player_scores[1])};
    }
    return state.Returns();
}

int WinnerIndex(const std::vector<double> &scores) {
    if (scores.size() < 2) {
        return -1;
    }
    if (scores[0] > scores[1]) {
        return 0;
    }
    if (scores[1] > scores[0]) {
        return 1;
    }
    return -1;
}

std::string WinnerLabel(int winner) {
    if (winner == 0) {
        return "A";
    }
    if (winner == 1) {
        return "B";
    }
    return "draw";
}

std::string AgentLabel(int agent) {
    if (agent == 0) {
        return "A";
    }
    if (agent == 1) {
        return "B";
    }
    return "?";
}

open_spiel::json::Array JsonScoreArray(const std::vector<double> &values) {
    open_spiel::json::Array array;
    array.reserve(values.size());
    for (double value : values) {
        array.push_back(value);
    }
    return array;
}

open_spiel::json::Array JsonStringArray(const std::vector<std::string> &values) {
    open_spiel::json::Array array;
    array.reserve(values.size());
    for (const std::string &value : values) {
        array.push_back(value);
    }
    return array;
}

std::string JoinPath(const std::string &dir, const std::string &file) {
    if (dir.empty() || dir.back() == '/' || dir.back() == '\\') {
        return absl::StrCat(dir, file);
    }
    return absl::StrCat(dir, "/", file);
}

std::string WorkerLogPath(const std::string &log_dir, int worker) {
    std::ostringstream filename;
    filename << "worker_" << std::setw(4) << std::setfill('0') << worker << ".jsonl";
    return JoinPath(log_dir, filename.str());
}

std::string AgentForPlayer(open_spiel::Player player, const std::array<int, 2> &seat_to_agent) {
    if (player < 0) {
        return "chance";
    }
    return AgentLabel(seat_to_agent[player]);
}

void WriteStepLog(const GameLogContext *log_context, int step, const std::string &phase,
                  const open_spiel::State &state, open_spiel::Player player, open_spiel::Action action,
                  const std::string &action_string, const std::array<int, 2> &seat_to_agent) {
    if (log_context == nullptr || log_context->out == nullptr) {
        return;
    }
    const bool is_chance = state.IsChanceNode() || player < 0;
    const int seat = is_chance ? -1 : player;
    const std::string agent = is_chance ? "chance" : AgentForPlayer(player, seat_to_agent);
    const open_spiel::json::Object record({
        {"event", "step"},
        {"worker", log_context->worker},
        {"game_id", log_context->game_id},
        {"pair", log_context->pair},
        {"leg", log_context->leg},
        {"first", log_context->first},
        {"step", step},
        {"phase", phase},
        {"seat", seat},
        {"agent", agent},
        {"action_id", static_cast<int64_t>(action)},
        {"action", action_string},
        {"state", state.ToString()},
    });
    *log_context->out << open_spiel::json::ToString(record) << "\n";
}

void WriteGameEndLog(const GameLogContext *log_context, int winner, const std::vector<double> &final_scores,
                     const std::vector<std::string> &history) {
    if (log_context == nullptr || log_context->out == nullptr) {
        return;
    }
    const open_spiel::json::Object record({
        {"event", "game_end"},
        {"worker", log_context->worker},
        {"game_id", log_context->game_id},
        {"pair", log_context->pair},
        {"leg", log_context->leg},
        {"first", log_context->first},
        {"winner", WinnerLabel(winner)},
        {"final_score", JsonScoreArray(final_scores)},
        {"history", JsonStringArray(history)},
    });
    *log_context->out << open_spiel::json::ToString(record) << "\n";
    log_context->out->flush();
}

struct AZSpec {
    std::string path;
    std::string graph_def;
    std::string device;
    int checkpoint;
};

struct SearchSpec {
    double uct_c;
    int max_simulations;
    int max_memory_mb;
    bool solve;
};

struct PlayerSpec {
    std::string type;
    AZSpec az;
    SearchSpec search;
};

PlayerSpec GetPlayerSpec(open_spiel::Player player) {
    if (player == 1) {
        return PlayerSpec{absl::GetFlag(FLAGS_p2_type),
                          AZSpec{absl::GetFlag(FLAGS_p2_az_path), absl::GetFlag(FLAGS_p2_az_graph_def),
                                 absl::GetFlag(FLAGS_p2_az_device), absl::GetFlag(FLAGS_p2_az_checkpoint)},
                          SearchSpec{absl::GetFlag(FLAGS_p2_uct_c), absl::GetFlag(FLAGS_p2_max_simulations),
                                     absl::GetFlag(FLAGS_p2_max_memory_mb), absl::GetFlag(FLAGS_p2_solve)}};
    }
    return PlayerSpec{absl::GetFlag(FLAGS_p1_type),
                      AZSpec{absl::GetFlag(FLAGS_p1_az_path), absl::GetFlag(FLAGS_p1_az_graph_def),
                             absl::GetFlag(FLAGS_p1_az_device), absl::GetFlag(FLAGS_p1_az_checkpoint)},
                      SearchSpec{absl::GetFlag(FLAGS_p1_uct_c), absl::GetFlag(FLAGS_p1_max_simulations),
                                 absl::GetFlag(FLAGS_p1_max_memory_mb), absl::GetFlag(FLAGS_p1_solve)}};
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
    device_manager->Get(0, 0)->LoadCheckpointWeightsOnly(spec.checkpoint);
    const int requested_batch_size = absl::GetFlag(FLAGS_az_batch_size);
    const int batch_size =
        requested_batch_size > 0 ? requested_batch_size : std::max(1, std::min(64, absl::GetFlag(FLAGS_num_workers)));
    const int threads = batch_size > 1 ? std::max(1, absl::GetFlag(FLAGS_az_threads)) : 0;
    auto evaluator = std::make_shared<open_spiel::algorithms::torch_az::VPNetEvaluator>(
        device_manager.get(),
        /*batch_size=*/batch_size,
        /*threads=*/threads,
        /*cache_size=*/std::max(0, absl::GetFlag(FLAGS_az_cache_size)),
        /*cache_shards=*/std::max(1, absl::GetFlag(FLAGS_az_cache_shards)),
        /*batch_wait_ms=*/1,
        absl::GetFlag(FLAGS_az_value_is_current_player));
    device_managers->push_back(std::move(device_manager));
    return evaluator;
}

std::unique_ptr<open_spiel::Bot> InitBot(const PlayerSpec &spec, const open_spiel::Game &game, open_spiel::Player player,
                                         std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator,
                                         std::shared_ptr<open_spiel::algorithms::torch_az::VPNetEvaluator> az_evaluator,
                                         uint_fast32_t seed) {
    if (spec.type == "az") {
        if (az_evaluator == nullptr) {
            open_spiel::SpielFatalError("AlphaZero evaluator is not initialized.");
        }
        return std::make_unique<open_spiel::algorithms::MCTSBot>(
            game, std::move(az_evaluator), spec.search.uct_c, spec.search.max_simulations, spec.search.max_memory_mb,
            spec.search.solve, seed, absl::GetFlag(FLAGS_verbose),
            open_spiel::algorithms::ChildSelectionPolicy::PUCT, 0, 0,
            /*dont_return_chance_node=*/true);
    }
    if (spec.type == "puct_mcts") {
        return std::make_unique<open_spiel::algorithms::MCTSBot>(
            game, std::move(evaluator), spec.search.uct_c, spec.search.max_simulations, spec.search.max_memory_mb,
            spec.search.solve, seed, absl::GetFlag(FLAGS_verbose),
            open_spiel::algorithms::ChildSelectionPolicy::PUCT, 0, 0,
            /*dont_return_chance_node=*/true);
    }
    if (spec.type == "az_prior_rollout_value" || spec.type == "network_prior_rollout_value") {
        if (az_evaluator == nullptr) {
            open_spiel::SpielFatalError("AlphaZero evaluator is not initialized.");
        }
        auto split_evaluator = std::make_shared<SplitEvaluator>(az_evaluator, evaluator);
        return std::make_unique<open_spiel::algorithms::MCTSBot>(
            game, std::move(split_evaluator), spec.search.uct_c, spec.search.max_simulations,
            spec.search.max_memory_mb, spec.search.solve, seed, absl::GetFlag(FLAGS_verbose),
            open_spiel::algorithms::ChildSelectionPolicy::PUCT, 0, 0,
            /*dont_return_chance_node=*/true);
    }
    if (spec.type == "uniform_prior_az_value") {
        if (az_evaluator == nullptr) {
            open_spiel::SpielFatalError("AlphaZero evaluator is not initialized.");
        }
        auto split_evaluator = std::make_shared<SplitEvaluator>(evaluator, az_evaluator);
        return std::make_unique<open_spiel::algorithms::MCTSBot>(
            game, std::move(split_evaluator), spec.search.uct_c, spec.search.max_simulations,
            spec.search.max_memory_mb, spec.search.solve, seed, absl::GetFlag(FLAGS_verbose),
            open_spiel::algorithms::ChildSelectionPolicy::PUCT, 0, 0,
            /*dont_return_chance_node=*/true);
    }
    if (spec.type == "human") {
        return std::make_unique<open_spiel::HumanBot>();
    }
    if (spec.type == "mcts") {
        return std::make_unique<open_spiel::algorithms::MCTSBot>(
            game, std::move(evaluator), spec.search.uct_c, spec.search.max_simulations, spec.search.max_memory_mb,
            spec.search.solve, seed, absl::GetFlag(FLAGS_verbose),
            open_spiel::algorithms::ChildSelectionPolicy::UCT, 0, 0,
            /*dont_return_chance_node=*/true);
    }
    if (spec.type == "random") {
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

void ApplyInitialActions(open_spiel::State *state, const std::vector<std::string> &initial_actions,
                         std::vector<std::string> *history, bool quiet,
                         const std::array<int, 2> *seat_to_agent = nullptr,
                         const GameLogContext *log_context = nullptr, int *step = nullptr) {
    for (const auto &action_str : initial_actions) {
        open_spiel::Player current_player = state->CurrentPlayer();
        open_spiel::Action action = GetAction(*state, action_str);

        if (action == open_spiel::kInvalidAction)
            open_spiel::SpielFatalError(absl::StrCat("Invalid action: ", action_str));

        if (history != nullptr) {
            history->push_back(action_str);
        }
        if (seat_to_agent != nullptr && step != nullptr) {
            WriteStepLog(log_context, *step, "forced", *state, current_player, action,
                         state->ActionToString(current_player, action), *seat_to_agent);
            ++*step;
        }
        state->ApplyAction(action);

        if (!quiet) {
            std::cerr << "Player " << current_player << " forced action: " << action_str << std::endl;
            std::cerr << "Next state:\n" << state->ToString() << std::endl;
        }
    }
}

std::vector<open_spiel::Action> GenerateChanceSchedule(const open_spiel::Game &game,
                                                       const std::vector<std::string> &initial_actions,
                                                       std::mt19937 *rng) {
    std::unique_ptr<open_spiel::State> state = game.NewInitialState();
    ApplyInitialActions(state.get(), initial_actions, nullptr, /*quiet=*/true);

    const auto *carcassonne_state = dynamic_cast<const open_spiel::carcassonne::CarcassonneState *>(state.get());
    if (carcassonne_state == nullptr) {
        if (game.GetType().chance_mode == open_spiel::GameType::ChanceMode::kDeterministic) {
            return {};
        }
        open_spiel::SpielFatalError("Paired same-deck matches require the carcassonne game.");
    }

    const auto &underlying = carcassonne_state->UnderlyingState();
    int total_remaining = underlying.getTotalRemaining();
    std::vector<int> remaining_by_type(open_spiel::carcassonne::kChanceActionCount + 1, 0);
    for (int type_id = 1; type_id <= open_spiel::carcassonne::kChanceActionCount; ++type_id) {
        remaining_by_type[type_id] = underlying.getRemainingTypeCount(type_id);
    }

    std::vector<open_spiel::Action> schedule;
    schedule.reserve(total_remaining);
    while (total_remaining > 0) {
        const int sample = std::uniform_int_distribution<int>(1, total_remaining)(*rng);
        int cumulative = 0;
        int selected_type = 0;
        for (int type_id = 1; type_id <= open_spiel::carcassonne::kChanceActionCount; ++type_id) {
            cumulative += remaining_by_type[type_id];
            if (sample <= cumulative) {
                selected_type = type_id;
                break;
            }
        }
        if (selected_type == 0) {
            open_spiel::SpielFatalError("Failed to sample a Carcassonne chance action.");
        }
        schedule.push_back(selected_type - 1);
        --remaining_by_type[selected_type];
        --total_remaining;
    }
    return schedule;
}

std::vector<double> ScoresByAgent(const std::vector<double> &seat_scores, const std::array<int, 2> &seat_to_agent) {
    std::vector<double> agent_scores(2, 0.0);
    for (int seat = 0; seat < 2 && seat < seat_scores.size(); ++seat) {
        agent_scores[seat_to_agent[seat]] = seat_scores[seat];
    }
    return agent_scores;
}

GameResult PlayGame(const open_spiel::Game &game, std::vector<std::unique_ptr<open_spiel::Bot>> &bots,
                    const std::vector<std::string> &initial_actions,
                    const std::vector<open_spiel::Action> &chance_schedule,
                    const std::array<int, 2> &seat_to_agent, const GameLogContext *log_context) {
    bool quiet = absl::GetFlag(FLAGS_quiet);
    std::unique_ptr<open_spiel::State> state = game.NewInitialState();
    std::vector<std::string> history;
    int chance_index = 0;
    int step = 0;

    if (!quiet)
        std::cerr << "Initial state:\n" << state << std::endl;

    ApplyInitialActions(state.get(), initial_actions, &history, quiet, &seat_to_agent, log_context, &step);

    while (!state->IsTerminal()) {
        open_spiel::Player player = state->CurrentPlayer();

        open_spiel::Action action;
        std::string phase;
        if (state->IsChanceNode()) {
            if (chance_index >= chance_schedule.size()) {
                open_spiel::SpielFatalError("Fixed chance schedule ended before the game reached a terminal state.");
            }
            action = chance_schedule[chance_index++];
            phase = "chance";
            const std::vector<open_spiel::Action> legal_actions = state->LegalActions();
            if (std::find(legal_actions.begin(), legal_actions.end(), action) == legal_actions.end()) {
                open_spiel::SpielFatalError(absl::StrCat("Fixed chance action is illegal at index ", chance_index - 1,
                                                         ": ", state->ActionToString(player, action)));
            }
        } else {
            // The state must be a decision node, ask the right bot to make its
            // action.
            action = bots[player]->Step(*state);
            phase = "decision";
        }
        const std::string action_string = state->ActionToString(player, action);
        WriteStepLog(log_context, step, phase, *state, player, action, action_string, seat_to_agent);
        ++step;
        if (!quiet)
            std::cerr << "Player " << player << " chose action: " << action_string << std::endl;

        // Inform the other bot of the action performed.
        for (open_spiel::Player p = 0; p < bots.size(); ++p) {
            if (p != player) {
                bots[p]->InformAction(*state, player, action);
            }
        }

        // Update history and get the next state.
        history.push_back(action_string);
        state->ApplyAction(action);

        if (!quiet)
            std::cerr << "Next state:\n" << state->ToString() << std::endl;
    }

    if (!quiet) {
        std::cerr << "Game actions: " << absl::StrJoin(history, ", ") << std::endl;
    }

    std::vector<double> final_scores = ScoresByAgent(FinalScores(*state), seat_to_agent);
    WriteGameEndLog(log_context, WinnerIndex(final_scores), final_scores, history);
    return {history, final_scores};
}

int main(int argc, char **argv) {
    std::vector<char *> positional_args = absl::ParseCommandLine(argc, argv);
    const uint_fast32_t base_seed = Seed();

    // Create the game.
    std::string game_name = absl::GetFlag(FLAGS_game);
    const bool quiet = absl::GetFlag(FLAGS_quiet);
    if (!quiet) {
        std::cerr << "Game: " << game_name << std::endl;
    }
    std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame(game_name);

    // Ensure the game is AlphaZero-compatible and arguments are compatible.
    open_spiel::GameType game_type = game->GetType();
    if (game->NumPlayers() != 2)
        open_spiel::SpielFatalError("AlphaZero can only handle 2-player games.");
    if (game_type.reward_model != open_spiel::GameType::RewardModel::kTerminal)
        open_spiel::SpielFatalError("Game must have terminal rewards.");
    if (game_type.dynamics != open_spiel::GameType::Dynamics::kSequential)
        open_spiel::SpielFatalError("Game must have sequential turns.");
    const PlayerSpec p1_spec = GetPlayerSpec(0);
    const PlayerSpec p2_spec = GetPlayerSpec(1);
    const int num_games = absl::GetFlag(FLAGS_num_games);
    if (num_games < 0) {
        open_spiel::SpielFatalError("--num_games must be non-negative.");
    }
    if (num_games % 2 != 0) {
        open_spiel::SpielFatalError("--num_games must be even for paired same-deck matches.");
    }
    const int pair_count = num_games / 2;
    const int worker_count = pair_count > 0 ? std::min(pair_count, std::max(1, absl::GetFlag(FLAGS_num_workers))) : 0;
    if ((p1_spec.type == "human" || p2_spec.type == "human") && worker_count > 1) {
        open_spiel::SpielFatalError("Human players can only be used with --num_workers=1.");
    }

    const bool player1_needs_az = RequiresAZEvaluator(p1_spec.type);
    const bool player2_needs_az = RequiresAZEvaluator(p2_spec.type);

    std::vector<std::unique_ptr<open_spiel::algorithms::torch_az::DeviceManager>> device_managers;
    std::shared_ptr<open_spiel::algorithms::torch_az::VPNetEvaluator> az_evaluator1;
    std::shared_ptr<open_spiel::algorithms::torch_az::VPNetEvaluator> az_evaluator2;
    std::array<const PlayerSpec *, 2> agent_specs = {&p1_spec, &p2_spec};
    if (player1_needs_az) {
        az_evaluator1 = InitAZEvaluator(*game, p1_spec.az, &device_managers);
    }
    if (player2_needs_az) {
        if (player1_needs_az && SameAZSpec(p1_spec.az, p2_spec.az)) {
            az_evaluator2 = az_evaluator1;
        } else {
            az_evaluator2 = InitAZEvaluator(*game, p2_spec.az, &device_managers);
        }
    }
    std::array<std::shared_ptr<open_spiel::algorithms::torch_az::VPNetEvaluator>, 2> az_evaluators = {az_evaluator1,
                                                                                                     az_evaluator2};

    std::vector<std::string> initial_actions;
    for (int i = 1; i < positional_args.size(); ++i) {
        initial_actions.push_back(positional_args[i]);
    }
    const std::string log_dir = absl::GetFlag(FLAGS_log_dir);
    if (!log_dir.empty()) {
        if (open_spiel::file::Exists(log_dir)) {
            if (!open_spiel::file::IsDirectory(log_dir)) {
                open_spiel::SpielFatalError(absl::StrCat("--log_dir exists but is not a directory: ", log_dir));
            }
        } else if (!open_spiel::file::Mkdirs(log_dir)) {
            open_spiel::SpielFatalError(absl::StrCat("Failed to create --log_dir: ", log_dir));
        }
    }

    absl::Mutex results_mutex;
    std::map<std::string, int> histories;
    std::vector<int> overall_wins(2, 0);
    int overall_draws = 0;
    int completed_games = 0;

    std::vector<std::future<void>> futures;
    futures.reserve(worker_count);
    const int pairs_per_worker = worker_count > 0 ? pair_count / worker_count : 0;
    const int extra_pairs = worker_count > 0 ? pair_count % worker_count : 0;
    for (int worker = 0; worker < worker_count; ++worker) {
        const int worker_pairs = pairs_per_worker + (worker < extra_pairs ? 1 : 0);
        const int first_pair = worker * pairs_per_worker + std::min(worker, extra_pairs);
        futures.push_back(std::async(std::launch::async, [&, worker, worker_pairs, first_pair]() {
            std::shared_ptr<const open_spiel::Game> worker_game = open_spiel::LoadGame(game_name);
            std::ofstream worker_log;
            std::ostream *worker_log_stream = nullptr;
            if (!log_dir.empty()) {
                worker_log.open(WorkerLogPath(log_dir, worker), std::ios::out | std::ios::trunc);
                if (!worker_log) {
                    open_spiel::SpielFatalError(
                        absl::StrCat("Failed to open worker log: ", WorkerLogPath(log_dir, worker)));
                }
                worker_log_stream = &worker_log;
            }
            for (int local_pair = 0; local_pair < worker_pairs; ++local_pair) {
                const int pair_num = first_pair + local_pair;
                std::mt19937 deck_rng(SeedWithOffset(base_seed, static_cast<uint_fast32_t>(pair_num)));
                const std::vector<open_spiel::Action> chance_schedule =
                    GenerateChanceSchedule(*worker_game, initial_actions, &deck_rng);

                for (int leg = 0; leg < 2; ++leg) {
                    const int game_num = pair_num * 2 + leg;
                    const uint_fast32_t game_seed =
                        SeedWithOffset(base_seed, static_cast<uint_fast32_t>(game_num + 1000003));
                    const std::array<int, 2> seat_to_agent = leg == 0 ? std::array<int, 2>{0, 1}
                                                                      : std::array<int, 2>{1, 0};
                    auto evaluator = std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(
                        absl::GetFlag(FLAGS_rollout_count), SeedWithOffset(game_seed, 101));

                    std::vector<std::unique_ptr<open_spiel::Bot>> bots;
                    bots.push_back(InitBot(*agent_specs[seat_to_agent[0]], *worker_game, 0, evaluator,
                                           az_evaluators[seat_to_agent[0]], SeedWithOffset(game_seed, 201)));
                    bots.push_back(InitBot(*agent_specs[seat_to_agent[1]], *worker_game, 1, evaluator,
                                           az_evaluators[seat_to_agent[1]], SeedWithOffset(game_seed, 301)));

                    GameLogContext log_context{worker_log_stream, worker, game_num + 1, pair_num + 1, leg + 1,
                                               AgentLabel(seat_to_agent[0])};
                    GameResult result = PlayGame(*worker_game, bots, initial_actions, chance_schedule, seat_to_agent,
                                                 worker_log_stream == nullptr ? nullptr : &log_context);

                    {
                        absl::MutexLock lock(&results_mutex);
                        histories[absl::StrJoin(result.history, " ")] += 1;
                        const int winner = WinnerIndex(result.final_scores);
                        if (winner >= 0) {
                            overall_wins[winner] += 1;
                        } else {
                            ++overall_draws;
                        }
                        ++completed_games;
                        const ConfidenceInterval a_ci =
                            Wilson95ConfidenceInterval(overall_wins[0], completed_games);
                        const ConfidenceInterval b_ci =
                            Wilson95ConfidenceInterval(overall_wins[1], completed_games);
                        std::cerr << "result id=" << (game_num + 1) << " completed=" << completed_games << "/"
                                  << num_games << " pair=" << (pair_num + 1) << " leg=" << (leg + 1)
                                  << " first=" << AgentLabel(seat_to_agent[0]) << " winner=" << WinnerLabel(winner)
                                  << " final_score=" << FormatScoreVector(result.final_scores)
                                  << " a_win_rate="
                                  << FormatRate(static_cast<double>(overall_wins[0]) / completed_games)
                                  << " a_ci95=" << FormatConfidenceInterval(a_ci)
                                  << " b_win_rate="
                                  << FormatRate(static_cast<double>(overall_wins[1]) / completed_games)
                                  << " b_ci95=" << FormatConfidenceInterval(b_ci)
                                  << " draws=" << overall_draws << std::endl;
                    }
                }
            }
        }));
    }

    for (auto &future : futures) {
        future.get();
    }

    if (!quiet) {
        std::cerr << "Number of games played: " << completed_games << std::endl;
        std::cerr << "Number of distinct games played: " << histories.size() << std::endl;
        std::cerr << "Players: A=" << p1_spec.type << ", B=" << p2_spec.type << std::endl;
        std::cerr << "Overall wins: " << absl::StrJoin(overall_wins, ", ") << std::endl;
        std::cerr << "Overall losses: " << overall_wins[1] << ", " << overall_wins[0] << std::endl;
        std::cerr << "Overall draws: " << overall_draws << std::endl;
    }

    return 0;
}
