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
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(std::string, game, "carcassonne", "The name of the game to load.");
ABSL_FLAG(std::string, az_path, "", "Path to AZ experiment.");
ABSL_FLAG(std::string, az_graph_def, "vpnet.pb",
          "AZ graph definition file name.");
ABSL_FLAG(int, az_checkpoint, -1, "Checkpoint of AZ model.");
ABSL_FLAG(std::string, device, "/cpu:0",
          "Torch device, e.g. /cpu:0, cpu, cuda:0.");
ABSL_FLAG(uint_fast32_t, seed, 0, "Seed for chance sampling.");
ABSL_FLAG(int, top_k, 20, "How many top policy actions to print.");
ABSL_FLAG(bool, sample_chance, true,
          "Whether to sample chance nodes before inference.");
ABSL_FLAG(int, random_decision_steps, 0,
          "How many random player decision actions to apply before inference. "
          "Chance actions are sampled but do not count toward this total.");
ABSL_FLAG(int, max_simulations, 300, "How many MCTS simulations to run.");
ABSL_FLAG(double, uct_c, 2, "PUCT exploration constant.");
ABSL_FLAG(double, mcts_policy_temperature, 1,
          "Temperature applied to MCTS visit counts before normalization.");
ABSL_FLAG(double, policy_alpha, 0,
          "Root Dirichlet noise alpha for MCTS. Default 0 disables noise.");
ABSL_FLAG(double, policy_epsilon, 0,
          "Root Dirichlet noise mix for MCTS. Default 0 disables noise.");
ABSL_FLAG(bool, mcts_solve, false, "Whether to enable MCTS-Solver.");
ABSL_FLAG(int64_t, max_memory_mb, 1000, "MCTS memory limit in MB.");

namespace {

uint_fast32_t Seed() {
  uint_fast32_t seed = absl::GetFlag(FLAGS_seed);
  return seed != 0 ? seed : absl::ToUnixMicros(absl::Now());
}

open_spiel::Action GetAction(const open_spiel::State& state,
                             const std::string& action_str) {
  for (open_spiel::Action action : state.LegalActions()) {
    if (action_str == state.ActionToString(state.CurrentPlayer(), action)) {
      return action;
    }
  }
  return open_spiel::kInvalidAction;
}

double PolicyEntropy(const open_spiel::ActionsAndProbs& policy) {
  double entropy = 0;
  for (const auto& [action, probability] : policy) {
    if (probability > 0) {
      entropy -= probability * std::log(probability);
    }
  }
  return entropy;
}

open_spiel::ActionsAndProbs SortedPolicy(open_spiel::ActionsAndProbs policy) {
  std::sort(policy.begin(), policy.end(),
            [](const auto& left, const auto& right) {
              return left.second > right.second;
            });
  return policy;
}

double PolicySum(const open_spiel::ActionsAndProbs& policy) {
  double sum = 0;
  for (const auto& [action, probability] : policy) {
    sum += probability;
  }
  return sum;
}

void PrintPolicy(const std::string& label,
                 const open_spiel::ActionsAndProbs& policy,
                 const open_spiel::State& state,
                 open_spiel::Player current_player, int top_k) {
  open_spiel::ActionsAndProbs sorted_policy = SortedPolicy(policy);
  const double entropy = PolicyEntropy(policy);
  const double normalized_entropy =
      policy.size() > 1 ? entropy / std::log(policy.size()) : 0;
  top_k = std::max(0, std::min(top_k, static_cast<int>(sorted_policy.size())));

  std::cout << label << " legal sum: " << PolicySum(policy) << "\n";
  std::cout << label << " entropy: " << entropy << "\n";
  std::cout << label << " normalized entropy: " << normalized_entropy << "\n";
  std::cout << "Top " << label << " actions:\n";
  for (int i = 0; i < top_k; ++i) {
    const auto& [action, probability] = sorted_policy[i];
    std::cout << i + 1 << ". action=" << action
              << " prob=" << probability << " "
              << state.ActionToString(current_player, action) << "\n";
  }
}

class DirectVPNetEvaluator : public open_spiel::algorithms::Evaluator {
 public:
  explicit DirectVPNetEvaluator(
      open_spiel::algorithms::torch_az::VPNetModel* model)
      : model_(model) {}

  std::vector<double> Evaluate(const open_spiel::State& state) override {
    double player0_value = Inference(state).value;
    return {player0_value, -player0_value};
  }

  open_spiel::ActionsAndProbs Prior(
      const open_spiel::State& state) override {
    if (state.IsChanceNode()) {
      return state.ChanceOutcomes();
    }
    return Inference(state).policy;
  }

 private:
  open_spiel::algorithms::torch_az::VPNetModel::InferenceOutputs Inference(
      const open_spiel::State& state) {
    open_spiel::algorithms::torch_az::VPNetModel::InferenceInputs input{
        state.LegalActions(), state.ObservationTensor()};
    return model_->Inference(std::vector<decltype(input)>{input})[0];
  }

  open_spiel::algorithms::torch_az::VPNetModel* model_;
};

void SampleChanceUntilDecision(open_spiel::State* state, std::mt19937* rng,
                               std::vector<std::string>* history) {
  while (state->IsChanceNode()) {
    open_spiel::ActionsAndProbs outcomes = state->ChanceOutcomes();
    open_spiel::Action action = open_spiel::SampleAction(outcomes, *rng).first;
    history->push_back(state->ActionToString(state->CurrentPlayer(), action));
    state->ApplyAction(action);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<char*> positional_args = absl::ParseCommandLine(argc, argv);
  std::mt19937 rng(Seed());
  if (absl::GetFlag(FLAGS_max_simulations) <= 0) {
    open_spiel::SpielFatalError("--max_simulations must be > 0.");
  }
  if (absl::GetFlag(FLAGS_mcts_policy_temperature) < 0) {
    open_spiel::SpielFatalError("--mcts_policy_temperature must be >= 0.");
  }

  const std::string game_name = absl::GetFlag(FLAGS_game);
  const std::string az_path = absl::GetFlag(FLAGS_az_path);
  if (az_path.empty()) {
    open_spiel::SpielFatalError("AlphaZero path must be specified.");
  }

  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(game_name);
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  std::vector<std::string> history;
  for (int i = 1; i < positional_args.size(); ++i) {
    if (state->IsTerminal()) {
      open_spiel::SpielFatalError("Cannot apply actions after terminal state.");
    }
    const std::string action_str = positional_args[i];
    open_spiel::Action action = GetAction(*state, action_str);
    if (action == open_spiel::kInvalidAction) {
      open_spiel::SpielFatalError("Invalid action: " + action_str);
    }
    history.push_back(action_str);
    state->ApplyAction(action);
  }

  const int random_decision_steps =
      absl::GetFlag(FLAGS_random_decision_steps);
  if (random_decision_steps < 0) {
    open_spiel::SpielFatalError("--random_decision_steps must be >= 0.");
  }

  for (int step = 0; step < random_decision_steps && !state->IsTerminal();
       ++step) {
    if (state->IsChanceNode()) {
      if (!absl::GetFlag(FLAGS_sample_chance)) {
        open_spiel::SpielFatalError(
            "Random stepping reached a chance node. Use --sample_chance=true.");
      }
      SampleChanceUntilDecision(state.get(), &rng, &history);
    }
    if (state->IsTerminal()) {
      break;
    }

    const open_spiel::Player player = state->CurrentPlayer();
    std::vector<open_spiel::Action> legal_actions = state->LegalActions();
    open_spiel::Action action =
        legal_actions[std::uniform_int_distribution<int>(
            0, static_cast<int>(legal_actions.size()) - 1)(rng)];
    history.push_back(state->ActionToString(player, action));
    state->ApplyAction(action);
  }

  if (absl::GetFlag(FLAGS_sample_chance)) {
    SampleChanceUntilDecision(state.get(), &rng, &history);
  }

  if (state->IsTerminal()) {
    open_spiel::SpielFatalError("State is terminal; no inference to run.");
  }
  if (state->IsChanceNode()) {
    open_spiel::SpielFatalError(
        "State is a chance node. Use --sample_chance=true or pass actions "
        "that reach a player decision state.");
  }

  const open_spiel::Player current_player = state->CurrentPlayer();
  std::vector<open_spiel::Action> legal_actions = state->LegalActions();
  std::vector<float> observation = state->ObservationTensor();

  open_spiel::algorithms::torch_az::VPNetModel model(
      *game, az_path, absl::GetFlag(FLAGS_az_graph_def),
      absl::GetFlag(FLAGS_device));
  model.LoadCheckpoint(absl::GetFlag(FLAGS_az_checkpoint));

  std::vector<open_spiel::algorithms::torch_az::VPNetModel::InferenceInputs>
      inputs;
  inputs.push_back({legal_actions, observation});
  std::vector<open_spiel::algorithms::torch_az::VPNetModel::InferenceOutputs>
      outputs = model.Inference(inputs);
  const auto& output = outputs[0];

  const int top_k =
      std::max(0, std::min(absl::GetFlag(FLAGS_top_k),
                           static_cast<int>(legal_actions.size())));

  auto evaluator = std::make_shared<DirectVPNetEvaluator>(&model);
  open_spiel::algorithms::MCTSBot bot(
      *game, evaluator, absl::GetFlag(FLAGS_uct_c),
      absl::GetFlag(FLAGS_max_simulations),
      absl::GetFlag(FLAGS_max_memory_mb), absl::GetFlag(FLAGS_mcts_solve),
      Seed(), /*verbose=*/false,
      open_spiel::algorithms::ChildSelectionPolicy::PUCT,
      absl::GetFlag(FLAGS_policy_alpha),
      absl::GetFlag(FLAGS_policy_epsilon),
      /*dont_return_chance_node=*/true);
  std::unique_ptr<open_spiel::algorithms::SearchNode> root =
      bot.MCTSearch(*state);
  open_spiel::ActionsAndProbs mcts_policy;
  mcts_policy.reserve(root->children.size());
  const double mcts_temperature =
      absl::GetFlag(FLAGS_mcts_policy_temperature);
  if (mcts_temperature == 0) {
    mcts_policy.emplace_back(root->BestChild().action, 1.0);
  } else {
    for (const open_spiel::algorithms::SearchNode& child : root->children) {
      mcts_policy.emplace_back(
          child.action, std::pow(child.explore_count, 1.0 / mcts_temperature));
    }
    open_spiel::NormalizePolicy(&mcts_policy);
  }
  const double root_value =
      root->explore_count > 0 ? root->total_reward / root->explore_count : 0;

  std::cout << "Game: " << game_name << "\n";
  std::cout << "Model path: " << az_path << "\n";
  std::cout << "Graph def: " << absl::GetFlag(FLAGS_az_graph_def) << "\n";
  std::cout << "Checkpoint: " << absl::GetFlag(FLAGS_az_checkpoint) << "\n";
  std::cout << "Device: " << model.Device() << "\n";
  std::cout << "Random decision steps: " << random_decision_steps << "\n";
  std::cout << "History: " << absl::StrJoin(history, ", ") << "\n";
  std::cout << "Current player: " << current_player << "\n";
  std::cout << "State:\n" << state->ToString() << "\n";
  std::cout << "Observation shape: "
            << absl::StrJoin(game->ObservationTensorShape(), "x") << "\n";
  std::cout << "Observation tensor size: " << observation.size() << "\n";
  std::cout << "Legal actions: " << legal_actions.size() << "\n";
  std::cout << "Value/player0: " << output.value << "\n";
  PrintPolicy("Network policy", output.policy, *state, current_player, top_k);
  std::cout << "MCTS simulations: " << absl::GetFlag(FLAGS_max_simulations)
            << "\n";
  std::cout << "MCTS root visits: " << root->explore_count << "\n";
  std::cout << "MCTS root value/current-player: " << root_value << "\n";
  PrintPolicy("MCTS policy", mcts_policy, *state, current_player, top_k);

  return 0;
}
