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

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/algorithms/alpha_zero_torch/model.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(std::string, game, "carcassonne", "The name of the game to load.");
ABSL_FLAG(std::string, az_path, "", "Path to teacher AZ experiment.");
ABSL_FLAG(std::string, az_graph_def, "vpnet.pb",
          "AZ graph definition file name.");
ABSL_FLAG(int, az_checkpoint, -1, "Teacher checkpoint.");
ABSL_FLAG(std::string, nn_model, "resnet",
          "Student model type when not initializing from a checkpoint.");
ABSL_FLAG(int, nn_width, 128,
          "Student model width when not initializing from a checkpoint.");
ABSL_FLAG(int, nn_depth, 10,
          "Student model depth when not initializing from a checkpoint.");
ABSL_FLAG(std::string, device, "/cpu:0",
          "Torch device, e.g. /cpu:0, cpu, cuda:0.");
ABSL_FLAG(uint_fast32_t, seed, 1, "Random seed.");
ABSL_FLAG(int, samples, 256, "Number of fixed training states.");
ABSL_FLAG(int, holdout_samples, 0,
          "Number of additional teacher states for holdout metrics.");
ABSL_FLAG(int, random_decision_steps, 30,
          "Random player decision actions to apply before each sampled state.");
ABSL_FLAG(int, max_simulations, 320, "MCTS simulations for policy targets.");
ABSL_FLAG(std::string, teacher, "az",
          "Fixed-data teacher: az or pure_mcts. pure_mcts uses UCT with a "
          "RandomRolloutEvaluator and does not use the AZ network for targets.");
ABSL_FLAG(int, rollout_count, 10,
          "Random rollouts per pure_mcts leaf evaluation.");
ABSL_FLAG(double, uct_c, 2, "PUCT exploration constant.");
ABSL_FLAG(double, policy_alpha, 1, "Root Dirichlet noise alpha for MCTS.");
ABSL_FLAG(double, policy_epsilon, 0.25, "Root Dirichlet noise mix for MCTS.");
ABSL_FLAG(double, mcts_policy_temperature, 1,
          "Temperature applied to MCTS visit counts before normalization.");
ABSL_FLAG(bool, init_from_checkpoint, true,
          "Initialize the overfit student from the teacher checkpoint.");
ABSL_FLAG(double, learning_rate, 0.001, "Policy-only overfit learning rate.");
ABSL_FLAG(double, weight_decay, 0, "Overfit student weight decay.");
ABSL_FLAG(int, batch_size, 64, "Policy-only overfit batch size.");
ABSL_FLAG(int, train_steps, 500, "Policy-only overfit gradient steps.");
ABSL_FLAG(int, report_every, 50, "Print metrics every N training steps.");
ABSL_FLAG(bool, use_vpnet_learn, true,
          "Use VPNetModel::Learn on fixed TrainInputs instead of a local "
          "policy-only training loop.");
ABSL_FLAG(std::string, student_path, "/tmp/az_policy_overfit_student",
          "Temporary path for the VPNetModel::Learn student graph config.");
ABSL_FLAG(bool, save_final_checkpoint, false,
          "Save checkpoint-0 before training and checkpoint--1 after training.");
ABSL_FLAG(bool, save_best_holdout_checkpoint, false,
          "Save checkpoint--3 whenever holdout KL reaches a new best.");
ABSL_FLAG(bool, save_checkpoint_every_report, false,
          "Save checkpoint-<step> at every report interval.");
ABSL_FLAG(bool, hardcoded_single_sample, false,
          "Bypass MCTS/replay and train one real observation against a "
          "manually hardcoded one-hot policy target.");
ABSL_FLAG(int, hardcoded_target_action, -1,
          "Action id to use for --hardcoded_single_sample. If negative, uses "
          "the first legal action in the sampled state.");
ABSL_FLAG(bool, bn_calibration, false,
          "Load a checkpoint, update only BatchNorm running statistics with "
          "random real observations, and save a calibrated checkpoint.");
ABSL_FLAG(std::string, bn_calibrated_path, "/tmp/az_bn_calibrated",
          "Output path for --bn_calibration calibrated checkpoint.");
ABSL_FLAG(bool, diagnose_self_play_targets, false,
          "Compare AZ self-play MCTS targets against pure rollout MCTS targets.");
ABSL_FLAG(bool, diagnose_root_value_returns, false,
          "Run MCTS self-play games and compare each root value against the "
          "final returns.");
ABSL_FLAG(bool, diagnose_mcts_stability, false,
          "For fixed states, rerun root MCTS with multiple simulation counts "
          "and seeds to measure policy target stability.");
ABSL_FLAG(std::string, diagnose_sim_counts, "160,320,640,1280",
          "Comma-separated simulation counts for --diagnose_mcts_stability.");
ABSL_FLAG(int, diagnose_repeats, 5,
          "Independent MCTS repeats per state and simulation count for "
          "--diagnose_mcts_stability.");
ABSL_FLAG(int, diagnose_games, 32,
          "Number of full games for --diagnose_root_value_returns.");
ABSL_FLAG(int, diagnose_cases, 5,
          "How many individual diagnosis cases to print.");
ABSL_FLAG(int, diagnose_top_k, 8,
          "How many actions to print for each diagnosis policy.");

namespace {

using open_spiel::Action;
using open_spiel::ActionsAndProbs;
using open_spiel::State;
using open_spiel::algorithms::SearchNode;
using open_spiel::algorithms::torch_az::Model;
using open_spiel::algorithms::torch_az::ModelConfig;
using open_spiel::algorithms::torch_az::VPNetModel;

struct Sample {
  std::vector<Action> legal_actions;
  std::vector<float> observation;
  ActionsAndProbs target_policy;
  double target_value = 0;
  int current_player = 0;
};

struct ValueStats {
  int count = 0;
  double target_sum = 0;
  double prediction_sum = 0;
  double target_square_sum = 0;
  double prediction_square_sum = 0;
  double product_sum = 0;
  double squared_error_sum = 0;
  double sign_correct = 0;

  void Add(double prediction, double target) {
    ++count;
    target_sum += target;
    prediction_sum += prediction;
    target_square_sum += target * target;
    prediction_square_sum += prediction * prediction;
    product_sum += prediction * target;
    const double error = prediction - target;
    squared_error_sum += error * error;
    sign_correct += (prediction >= 0) == (target >= 0) ? 1 : 0;
  }

  double TargetMean() const { return count > 0 ? target_sum / count : 0; }
  double PredictionMean() const {
    return count > 0 ? prediction_sum / count : 0;
  }
  double MSE() const { return count > 0 ? squared_error_sum / count : 0; }
  double SignAccuracy() const { return count > 0 ? sign_correct / count : 0; }
  double Correlation() const {
    if (count <= 1) return std::numeric_limits<double>::quiet_NaN();
    const double n = count;
    const double numerator = n * product_sum - prediction_sum * target_sum;
    const double prediction_var =
        n * prediction_square_sum - prediction_sum * prediction_sum;
    const double target_var = n * target_square_sum - target_sum * target_sum;
    const double denominator =
        std::sqrt(std::max(0.0, prediction_var) * std::max(0.0, target_var));
    return denominator > 0 ? numerator / denominator
                           : std::numeric_limits<double>::quiet_NaN();
  }
};

struct Metrics {
  double target_entropy = 0;
  double policy_entropy = 0;
  double cross_entropy = 0;
  double kl = 0;
  double value_mse = 0;
  double top1_agreement = 0;
  ValueStats value_all;
  std::array<ValueStats, 2> value_by_player;
};

void AddValueMetric(Metrics* metrics, const Sample& sample,
                    double prediction) {
  metrics->value_all.Add(prediction, sample.target_value);
  if (sample.current_player == 0 || sample.current_player == 1) {
    metrics->value_by_player[sample.current_player].Add(prediction,
                                                        sample.target_value);
  }
}

struct SplitValueStats {
  ValueStats all;
  std::array<ValueStats, 2> by_player;

  void Add(int current_player, double prediction, double target) {
    all.Add(prediction, target);
    if (current_player == 0 || current_player == 1) {
      by_player[current_player].Add(prediction, target);
    }
  }
};

struct RootValueRecord {
  int current_player = 0;
  double root_value_current = 0;
  double root_value_p0 = 0;
};

void PrintRootValueStats(const std::string& label,
                         const SplitValueStats& stats);

struct PredictionStats {
  double policy_entropy = 0;
  double normalized_policy_entropy = 0;
  double max_probability = 0;
  double value_abs = 0;
};

struct SearchTarget {
  ActionsAndProbs policy;
  double player0_value = 0;
};

struct DiagnosticCase {
  int index = 0;
  int current_player = 0;
  std::string state_string;
  std::vector<std::pair<Action, std::string>> action_strings;
  ActionsAndProbs network_prior;
  ActionsAndProbs az_policy;
  ActionsAndProbs pure_policy;
  double network_value = 0;
  double az_value = 0;
  double pure_value = 0;
};

std::string TorchDeviceName(const std::string& device) {
  if (!device.empty() && device[0] == '/') return device.substr(1);
  return device;
}

ModelConfig LoadModelConfig(const std::string& path,
                            const std::string& filename) {
  std::ifstream file(absl::StrCat(path, "/", filename));
  ModelConfig config;
  file >> config;
  return config;
}

ModelConfig NewModelConfig(const open_spiel::Game& game) {
  return ModelConfig{/*observation_tensor_shape=*/game.ObservationTensorShape(),
                     /*number_of_actions=*/game.NumDistinctActions(),
                     /*nn_depth=*/absl::GetFlag(FLAGS_nn_depth),
                     /*nn_width=*/absl::GetFlag(FLAGS_nn_width),
                     /*learning_rate=*/absl::GetFlag(FLAGS_learning_rate),
                     /*weight_decay=*/absl::GetFlag(FLAGS_weight_decay),
                     /*nn_model=*/absl::GetFlag(FLAGS_nn_model)};
}

ModelConfig BaseModelConfig(const open_spiel::Game& game,
                            const std::string& path,
                            const std::string& filename) {
  ModelConfig config;
  const std::string config_path = absl::StrCat(path, "/", filename);
  if (!path.empty() && std::filesystem::exists(config_path)) {
    config = LoadModelConfig(path, filename);
  } else {
    config = NewModelConfig(game);
  }
  config.learning_rate = absl::GetFlag(FLAGS_learning_rate);
  config.weight_decay = absl::GetFlag(FLAGS_weight_decay);
  return config;
}

double Entropy(const ActionsAndProbs& policy) {
  double entropy = 0;
  for (const auto& [action, prob] : policy) {
    if (prob > 0) entropy -= prob * std::log(prob);
  }
  return entropy;
}

double AverageTargetEntropy(const std::vector<Sample>& samples) {
  double entropy = 0;
  for (const Sample& sample : samples) {
    entropy += Entropy(sample.target_policy);
  }
  return samples.empty() ? 0 : entropy / samples.size();
}

double NormalizedEntropy(const ActionsAndProbs& policy, int legal_action_count) {
  return legal_action_count > 1 ? Entropy(policy) / std::log(legal_action_count)
                                : 0;
}

double PolicyProb(const ActionsAndProbs& policy, Action action) {
  for (const auto& [candidate, prob] : policy) {
    if (candidate == action) return prob;
  }
  return 0;
}

double MaxProbability(const ActionsAndProbs& policy) {
  double max_prob = 0;
  for (const auto& [action, prob] : policy) {
    max_prob = std::max(max_prob, prob);
  }
  return max_prob;
}

double CrossEntropy(const ActionsAndProbs& target,
                    const ActionsAndProbs& prediction) {
  double cross_entropy = 0;
  for (const auto& [action, target_prob] : target) {
    if (target_prob > 0) {
      cross_entropy -= target_prob *
                       std::log(std::max(1e-12, PolicyProb(prediction, action)));
    }
  }
  return cross_entropy;
}

double KL(const ActionsAndProbs& target, const ActionsAndProbs& prediction) {
  return CrossEntropy(target, prediction) - Entropy(target);
}

double L1Distance(const ActionsAndProbs& left, const ActionsAndProbs& right,
                  const std::vector<Action>& legal_actions) {
  double distance = 0;
  for (Action action : legal_actions) {
    distance += std::abs(PolicyProb(left, action) - PolicyProb(right, action));
  }
  return distance;
}

Action TopAction(const ActionsAndProbs& policy) {
  return std::max_element(policy.begin(), policy.end(),
                          [](const auto& a, const auto& b) {
                            return a.second < b.second;
                          })
      ->first;
}

ActionsAndProbs SortedPolicy(ActionsAndProbs policy) {
  std::sort(policy.begin(), policy.end(),
            [](const auto& left, const auto& right) {
              return left.second > right.second;
            });
  return policy;
}

void SampleChanceUntilDecision(State* state, std::mt19937* rng) {
  while (state->IsChanceNode()) {
    Action action = open_spiel::SampleAction(state->ChanceOutcomes(), *rng).first;
    state->ApplyAction(action);
  }
}

std::unique_ptr<State> RandomState(const open_spiel::Game& game,
                                   int random_decision_steps,
                                   std::mt19937* rng) {
  std::unique_ptr<State> state = game.NewInitialState();
  for (int step = 0; step < random_decision_steps && !state->IsTerminal();
       ++step) {
    SampleChanceUntilDecision(state.get(), rng);
    if (state->IsTerminal()) break;
    std::vector<Action> legal_actions = state->LegalActions();
    Action action = legal_actions[std::uniform_int_distribution<int>(
        0, static_cast<int>(legal_actions.size()) - 1)(*rng)];
    state->ApplyAction(action);
  }
  SampleChanceUntilDecision(state.get(), rng);
  return state;
}

std::vector<Sample> CollectRandomObservationSamples(
    const open_spiel::Game& game, int num_samples, int random_decision_steps,
    std::mt19937* rng) {
  std::vector<Sample> samples;
  samples.reserve(num_samples);
  while (samples.size() < num_samples) {
    std::unique_ptr<State> state = RandomState(game, random_decision_steps, rng);
    if (state->IsTerminal() || state->IsChanceNode() ||
        state->LegalActions().empty()) {
      continue;
    }
    samples.push_back({state->LegalActions(), state->ObservationTensor(),
                       /*target_policy=*/{}, /*target_value=*/0,
                       state->CurrentPlayer()});
  }
  return samples;
}

class DirectVPNetEvaluator : public open_spiel::algorithms::Evaluator {
 public:
  explicit DirectVPNetEvaluator(VPNetModel* model) : model_(model) {}

  std::vector<double> Evaluate(const State& state) override {
    double player0_value = Inference(state).value;
    return {player0_value, -player0_value};
  }

  ActionsAndProbs Prior(const State& state) override {
    if (state.IsChanceNode()) return state.ChanceOutcomes();
    return Inference(state).policy;
  }

 private:
  VPNetModel::InferenceOutputs Inference(const State& state) {
    VPNetModel::InferenceInputs input{state.LegalActions(),
                                      state.ObservationTensor()};
    return model_->Inference(std::vector<VPNetModel::InferenceInputs>{input})[0];
  }

  VPNetModel* model_;
};

SearchTarget MCTSTarget(const open_spiel::Game& game, const State& state,
                        std::shared_ptr<open_spiel::algorithms::Evaluator>
                            evaluator,
                        int seed,
                        int max_simulations,
                        open_spiel::algorithms::ChildSelectionPolicy
                            child_selection_policy,
                        double dirichlet_alpha,
                        double dirichlet_epsilon) {
  open_spiel::algorithms::MCTSBot bot(
      game, std::move(evaluator), absl::GetFlag(FLAGS_uct_c),
      max_simulations,
      /*max_memory_mb=*/1000,
      /*solve=*/false, seed,
      /*verbose=*/false, child_selection_policy, dirichlet_alpha,
      dirichlet_epsilon,
      /*dont_return_chance_node=*/true);
  std::unique_ptr<SearchNode> root = bot.MCTSearch(state);
  ActionsAndProbs policy;
  policy.reserve(root->children.size());
  double temperature = absl::GetFlag(FLAGS_mcts_policy_temperature);
  if (temperature == 0) {
    policy.emplace_back(root->BestChild().action, 1.0);
  } else {
    for (const SearchNode& child : root->children) {
      policy.emplace_back(child.action,
                          std::pow(child.explore_count, 1.0 / temperature));
    }
    open_spiel::NormalizePolicy(&policy);
  }

  double root_value =
      root->explore_count > 0 ? root->total_reward / root->explore_count : 0;
  double player0_value = state.CurrentPlayer() == 0 ? root_value : -root_value;
  return {policy, player0_value};
}

SearchTarget MCTSTarget(const open_spiel::Game& game, const State& state,
                        std::shared_ptr<open_spiel::algorithms::Evaluator>
                            evaluator,
                        int seed,
                        open_spiel::algorithms::ChildSelectionPolicy
                            child_selection_policy,
                        double dirichlet_alpha,
                        double dirichlet_epsilon) {
  return MCTSTarget(game, state, std::move(evaluator), seed,
                    absl::GetFlag(FLAGS_max_simulations),
                    child_selection_policy, dirichlet_alpha,
                    dirichlet_epsilon);
}

ActionsAndProbs MCTSPolicy(const open_spiel::Game& game, const State& state,
                           std::shared_ptr<open_spiel::algorithms::Evaluator>
                               evaluator,
                           int seed) {
  return MCTSTarget(
             game, state, std::move(evaluator), seed,
             open_spiel::algorithms::ChildSelectionPolicy::PUCT,
             absl::GetFlag(FLAGS_policy_alpha),
	             absl::GetFlag(FLAGS_policy_epsilon))
	      .policy;
}

std::vector<int> ParseSimCounts(const std::string& counts_text) {
  std::vector<int> counts;
  for (absl::string_view part :
       absl::StrSplit(counts_text, ',', absl::SkipWhitespace())) {
    int value = 0;
    if (!absl::SimpleAtoi(part, &value) || value <= 0) {
      open_spiel::SpielFatalError(
          absl::StrCat("Invalid --diagnose_sim_counts entry: ", part));
    }
    counts.push_back(value);
  }
  if (counts.empty()) {
    open_spiel::SpielFatalError("--diagnose_sim_counts must not be empty.");
  }
  std::sort(counts.begin(), counts.end());
  counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
  return counts;
}

struct PolicyComparisonStats {
  int count = 0;
  double top1 = 0;
  double kl = 0;
  double l1 = 0;

  void Add(const ActionsAndProbs& left, const ActionsAndProbs& right,
           const std::vector<Action>& legal_actions) {
    ++count;
    top1 += TopAction(left) == TopAction(right) ? 1 : 0;
    kl += KL(left, right);
    l1 += L1Distance(left, right, legal_actions);
  }

  void Print(const std::string& label) const {
    const double denom = count > 0 ? count : 1;
    std::cout << label
              << " comparisons=" << count
              << " top1=" << top1 / denom
              << " kl=" << kl / denom
              << " l1=" << l1 / denom << "\n";
  }
};

void PrintPolicyTable(const std::string& label, const ActionsAndProbs& policy,
                      const DiagnosticCase& sample, int top_k);

void RunMCTSStabilityDiagnosis(
    const open_spiel::Game& game,
    std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator,
    open_spiel::algorithms::ChildSelectionPolicy child_selection_policy,
    double dirichlet_alpha, double dirichlet_epsilon, std::mt19937* rng) {
  if (absl::GetFlag(FLAGS_mcts_policy_temperature) < 0) {
    open_spiel::SpielFatalError("--mcts_policy_temperature must be >= 0.");
  }
  const int repeats = absl::GetFlag(FLAGS_diagnose_repeats);
  if (repeats <= 0) {
    open_spiel::SpielFatalError("--diagnose_repeats must be positive.");
  }

  std::vector<int> sim_counts =
      ParseSimCounts(absl::GetFlag(FLAGS_diagnose_sim_counts));

  std::vector<std::unique_ptr<State>> states;
  states.reserve(absl::GetFlag(FLAGS_samples));
  while (states.size() < absl::GetFlag(FLAGS_samples)) {
    std::unique_ptr<State> state = RandomState(
        game, absl::GetFlag(FLAGS_random_decision_steps), rng);
    if (state->IsTerminal() || state->IsChanceNode() ||
        state->LegalActions().empty()) {
      continue;
    }
    states.push_back(std::move(state));
  }

  std::vector<std::vector<std::vector<ActionsAndProbs>>> policies(
      sim_counts.size(),
      std::vector<std::vector<ActionsAndProbs>>(
          states.size(), std::vector<ActionsAndProbs>(repeats)));

  int base_seed = absl::GetFlag(FLAGS_seed);
  if (base_seed == 0) base_seed = 1;
  for (int sim_index = 0; sim_index < sim_counts.size(); ++sim_index) {
    for (int state_index = 0; state_index < states.size(); ++state_index) {
      for (int repeat = 0; repeat < repeats; ++repeat) {
        const int search_seed = base_seed + 1000000 * sim_index +
                                1000 * state_index + repeat;
        policies[sim_index][state_index][repeat] =
            MCTSTarget(game, *states[state_index], evaluator, search_seed,
                       sim_counts[sim_index], child_selection_policy,
                       dirichlet_alpha, dirichlet_epsilon)
                .policy;
      }
    }
  }

  std::cout << "mode=diagnose_mcts_stability"
            << " samples=" << states.size()
            << " repeats=" << repeats
            << " sim_counts=" << absl::GetFlag(FLAGS_diagnose_sim_counts)
            << " teacher=" << absl::GetFlag(FLAGS_teacher)
            << " rollout_count=" << absl::GetFlag(FLAGS_rollout_count)
            << " uct_c=" << absl::GetFlag(FLAGS_uct_c)
            << " policy_alpha=" << dirichlet_alpha
            << " policy_epsilon=" << dirichlet_epsilon
            << " random_decision_steps="
            << absl::GetFlag(FLAGS_random_decision_steps) << "\n";

  for (int sim_index = 0; sim_index < sim_counts.size(); ++sim_index) {
    double entropy = 0;
    double norm_entropy = 0;
    double top_prob = 0;
    int policy_count = 0;
    PolicyComparisonStats within;
    for (int state_index = 0; state_index < states.size(); ++state_index) {
      const std::vector<Action> legal_actions = states[state_index]->LegalActions();
      for (int repeat = 0; repeat < repeats; ++repeat) {
        const ActionsAndProbs& policy =
            policies[sim_index][state_index][repeat];
        entropy += Entropy(policy);
        norm_entropy += NormalizedEntropy(policy, legal_actions.size());
        top_prob += MaxProbability(policy);
        ++policy_count;
      }
      for (int left = 0; left < repeats; ++left) {
        for (int right = left + 1; right < repeats; ++right) {
          within.Add(policies[sim_index][state_index][left],
                     policies[sim_index][state_index][right], legal_actions);
        }
      }
    }
    const double denom = policy_count > 0 ? policy_count : 1;
    std::cout << "sim=" << sim_counts[sim_index]
              << " entropy=" << entropy / denom
              << " normalized_entropy=" << norm_entropy / denom
              << " top_prob=" << top_prob / denom << "\n";
    within.Print(absl::StrCat("within_seed sim=", sim_counts[sim_index]));
  }

  const int max_index = sim_counts.size() - 1;
  for (int sim_index = 0; sim_index + 1 < sim_counts.size(); ++sim_index) {
    PolicyComparisonStats vs_next;
    PolicyComparisonStats vs_max;
    for (int state_index = 0; state_index < states.size(); ++state_index) {
      const std::vector<Action> legal_actions = states[state_index]->LegalActions();
      for (int left_repeat = 0; left_repeat < repeats; ++left_repeat) {
        for (int right_repeat = 0; right_repeat < repeats; ++right_repeat) {
          vs_next.Add(policies[sim_index][state_index][left_repeat],
                      policies[sim_index + 1][state_index][right_repeat],
                      legal_actions);
          vs_max.Add(policies[sim_index][state_index][left_repeat],
                     policies[max_index][state_index][right_repeat],
                     legal_actions);
        }
      }
    }
    vs_next.Print(absl::StrCat("vs_next sim=", sim_counts[sim_index],
                               "->", sim_counts[sim_index + 1]));
    vs_max.Print(absl::StrCat("vs_max sim=", sim_counts[sim_index],
                              "->", sim_counts[max_index]));
  }

  const int case_count =
      std::max(0, std::min(absl::GetFlag(FLAGS_diagnose_cases),
                           static_cast<int>(states.size())));
  const int top_k = absl::GetFlag(FLAGS_diagnose_top_k);
  for (int case_index = 0; case_index < case_count; ++case_index) {
    std::cout << "\nstability_case=" << case_index
              << " current_player=" << states[case_index]->CurrentPlayer()
              << " legal_actions=" << states[case_index]->LegalActions().size()
              << "\nstate:\n" << states[case_index]->ToString() << "\n";
    DiagnosticCase printable;
    printable.current_player = states[case_index]->CurrentPlayer();
    for (Action action : states[case_index]->LegalActions()) {
      printable.action_strings.push_back(
          {action, states[case_index]->ActionToString(
                       states[case_index]->CurrentPlayer(), action)});
    }
    for (int sim_index = 0; sim_index < sim_counts.size(); ++sim_index) {
      PrintPolicyTable(absl::StrCat("sim=", sim_counts[sim_index],
                                    " repeat=0"),
                       policies[sim_index][case_index][0], printable, top_k);
    }
  }
}

std::string ActionString(const DiagnosticCase& sample, Action action) {
  for (const auto& [candidate, action_string] : sample.action_strings) {
    if (candidate == action) return action_string;
  }
  return absl::StrCat("<action ", action, ">");
}

void PrintPolicyTable(const std::string& label, const ActionsAndProbs& policy,
                      const DiagnosticCase& sample, int top_k) {
  ActionsAndProbs sorted = SortedPolicy(policy);
  top_k = std::max(0, std::min(top_k, static_cast<int>(sorted.size())));
  std::cout << label << " entropy=" << Entropy(policy)
            << " normalized_entropy="
            << NormalizedEntropy(policy, sample.action_strings.size())
            << " top_prob=" << MaxProbability(policy) << "\n";
  for (int i = 0; i < top_k; ++i) {
    const auto& [action, prob] = sorted[i];
    std::cout << "  " << i + 1 << ". action=" << action
              << " prob=" << prob << " " << ActionString(sample, action)
              << "\n";
  }
}

void RunSelfPlayTargetDiagnosis(const open_spiel::Game& game,
                                VPNetModel* teacher, std::mt19937* rng) {
  if (absl::GetFlag(FLAGS_mcts_policy_temperature) < 0) {
    open_spiel::SpielFatalError("--mcts_policy_temperature must be >= 0.");
  }

  const int sample_count = absl::GetFlag(FLAGS_samples);
  int az_search_seed = absl::GetFlag(FLAGS_seed);
  if (az_search_seed == 0) az_search_seed = 1;
  int pure_search_seed = az_search_seed + 1000000;

  auto az_evaluator = std::make_shared<DirectVPNetEvaluator>(teacher);
  auto pure_evaluator =
      std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(
          absl::GetFlag(FLAGS_rollout_count), az_search_seed + 2000000);

  std::vector<DiagnosticCase> cases;
  cases.reserve(sample_count);
  while (cases.size() < sample_count) {
    std::unique_ptr<State> state = game.NewInitialState();
    while (!state->IsTerminal() && cases.size() < sample_count) {
      SampleChanceUntilDecision(state.get(), rng);
      if (state->IsTerminal() || state->LegalActions().empty()) break;

      const int current_player = state->CurrentPlayer();
      const std::vector<Action> legal_actions = state->LegalActions();
      const std::vector<float> observation = state->ObservationTensor();
      VPNetModel::InferenceOutputs prior =
          teacher->Inference({VPNetModel::InferenceInputs{legal_actions,
                                                          observation}})[0];

      SearchTarget az_target = MCTSTarget(
          game, *state, az_evaluator, az_search_seed++,
          open_spiel::algorithms::ChildSelectionPolicy::PUCT,
          absl::GetFlag(FLAGS_policy_alpha),
          absl::GetFlag(FLAGS_policy_epsilon));
      SearchTarget pure_target = MCTSTarget(
          game, *state, pure_evaluator, pure_search_seed++,
          open_spiel::algorithms::ChildSelectionPolicy::UCT,
          /*dirichlet_alpha=*/0, /*dirichlet_epsilon=*/0);

      DiagnosticCase sample;
      sample.index = cases.size();
      sample.current_player = current_player;
      sample.state_string = state->ToString();
      sample.network_prior = prior.policy;
      sample.az_policy = az_target.policy;
      sample.pure_policy = pure_target.policy;
      sample.network_value = prior.value;
      sample.az_value = az_target.player0_value;
      sample.pure_value = pure_target.player0_value;
      sample.action_strings.reserve(legal_actions.size());
      for (Action action : legal_actions) {
        sample.action_strings.push_back(
            {action, state->ActionToString(current_player, action)});
      }
      cases.push_back(std::move(sample));

      Action action = open_spiel::SampleAction(az_target.policy, *rng).first;
      state->ApplyAction(action);
    }
  }

  double az_entropy = 0;
  double az_norm_entropy = 0;
  double az_top_prob = 0;
  double pure_entropy = 0;
  double pure_norm_entropy = 0;
  double pure_top_prob = 0;
  double prior_entropy = 0;
  double prior_norm_entropy = 0;
  double prior_top_prob = 0;
  double az_pure_top1 = 0;
  double az_pure_ce = 0;
  double az_pure_kl = 0;
  double az_pure_l1 = 0;
  double prior_az_top1 = 0;
  double prior_az_ce = 0;
  double prior_az_kl = 0;
  double prior_az_l1 = 0;
  double value_abs_diff = 0;
  double az_value_sum = 0;
  double pure_value_sum = 0;
  double az_value_sq_sum = 0;
  double pure_value_sq_sum = 0;
  double value_product_sum = 0;

  for (const DiagnosticCase& sample : cases) {
    const std::vector<Action> legal_actions = [&]() {
      std::vector<Action> actions;
      actions.reserve(sample.action_strings.size());
      for (const auto& [action, action_string] : sample.action_strings) {
        actions.push_back(action);
      }
      return actions;
    }();

    az_entropy += Entropy(sample.az_policy);
    az_norm_entropy +=
        NormalizedEntropy(sample.az_policy, sample.action_strings.size());
    az_top_prob += MaxProbability(sample.az_policy);
    pure_entropy += Entropy(sample.pure_policy);
    pure_norm_entropy +=
        NormalizedEntropy(sample.pure_policy, sample.action_strings.size());
    pure_top_prob += MaxProbability(sample.pure_policy);
    prior_entropy += Entropy(sample.network_prior);
    prior_norm_entropy +=
        NormalizedEntropy(sample.network_prior, sample.action_strings.size());
    prior_top_prob += MaxProbability(sample.network_prior);

    az_pure_top1 += TopAction(sample.az_policy) == TopAction(sample.pure_policy)
                        ? 1
                        : 0;
    az_pure_ce += CrossEntropy(sample.az_policy, sample.pure_policy);
    az_pure_kl += KL(sample.az_policy, sample.pure_policy);
    az_pure_l1 +=
        L1Distance(sample.az_policy, sample.pure_policy, legal_actions);

    prior_az_top1 += TopAction(sample.network_prior) == TopAction(sample.az_policy)
                         ? 1
                         : 0;
    prior_az_ce += CrossEntropy(sample.az_policy, sample.network_prior);
    prior_az_kl += KL(sample.az_policy, sample.network_prior);
    prior_az_l1 +=
        L1Distance(sample.network_prior, sample.az_policy, legal_actions);

    const double value_diff = sample.az_value - sample.pure_value;
    value_abs_diff += std::abs(value_diff);
    az_value_sum += sample.az_value;
    pure_value_sum += sample.pure_value;
    az_value_sq_sum += sample.az_value * sample.az_value;
    pure_value_sq_sum += sample.pure_value * sample.pure_value;
    value_product_sum += sample.az_value * sample.pure_value;
  }

  const double denom = cases.size();
  const double value_corr_num =
      denom * value_product_sum - az_value_sum * pure_value_sum;
  const double value_corr_den =
      std::sqrt(std::max(0.0, denom * az_value_sq_sum -
                                  az_value_sum * az_value_sum) *
                std::max(0.0, denom * pure_value_sq_sum -
                                  pure_value_sum * pure_value_sum));
  const double value_corr =
      value_corr_den > 0 ? value_corr_num / value_corr_den
                         : std::numeric_limits<double>::quiet_NaN();

  std::cout << "mode=diagnose_self_play_targets"
            << " samples=" << cases.size()
            << " max_simulations=" << absl::GetFlag(FLAGS_max_simulations)
            << " rollout_count=" << absl::GetFlag(FLAGS_rollout_count)
            << " policy_alpha=" << absl::GetFlag(FLAGS_policy_alpha)
            << " policy_epsilon=" << absl::GetFlag(FLAGS_policy_epsilon)
            << " mcts_policy_temperature="
            << absl::GetFlag(FLAGS_mcts_policy_temperature) << "\n";
  std::cout << "az_target entropy=" << az_entropy / denom
            << " normalized_entropy=" << az_norm_entropy / denom
            << " top_prob=" << az_top_prob / denom << "\n";
  std::cout << "pure_mcts_target entropy=" << pure_entropy / denom
            << " normalized_entropy=" << pure_norm_entropy / denom
            << " top_prob=" << pure_top_prob / denom << "\n";
  std::cout << "network_prior entropy=" << prior_entropy / denom
            << " normalized_entropy=" << prior_norm_entropy / denom
            << " top_prob=" << prior_top_prob / denom << "\n";
  std::cout << "az_vs_pure top1=" << az_pure_top1 / denom
            << " ce=" << az_pure_ce / denom
            << " kl=" << az_pure_kl / denom
            << " l1=" << az_pure_l1 / denom << "\n";
  std::cout << "prior_vs_az top1=" << prior_az_top1 / denom
            << " ce=" << prior_az_ce / denom
            << " kl=" << prior_az_kl / denom
            << " l1=" << prior_az_l1 / denom << "\n";
  std::cout << "value az_mean=" << az_value_sum / denom
            << " pure_mean=" << pure_value_sum / denom
            << " abs_diff=" << value_abs_diff / denom
            << " corr=" << value_corr << "\n";

  const int case_count =
      std::max(0, std::min(absl::GetFlag(FLAGS_diagnose_cases),
                           static_cast<int>(cases.size())));
  const int top_k = absl::GetFlag(FLAGS_diagnose_top_k);
  for (int i = 0; i < case_count; ++i) {
    const DiagnosticCase& sample = cases[i];
    std::cout << "\ncase=" << sample.index
              << " current_player=" << sample.current_player
              << " legal_actions=" << sample.action_strings.size()
              << " az_value_p0=" << sample.az_value
              << " pure_value_p0=" << sample.pure_value
              << " network_value_p0=" << sample.network_value
              << " az_top_matches_pure="
              << (TopAction(sample.az_policy) == TopAction(sample.pure_policy)
                      ? "true"
                      : "false")
              << "\n";
    std::cout << "state:\n" << sample.state_string << "\n";
    PrintPolicyTable("network_prior", sample.network_prior, sample, top_k);
    PrintPolicyTable("az_target", sample.az_policy, sample, top_k);
    PrintPolicyTable("pure_mcts_target", sample.pure_policy, sample, top_k);
  }
}

void RunRootValueReturnsDiagnosis(
    const open_spiel::Game& game,
    std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator,
    open_spiel::algorithms::ChildSelectionPolicy child_selection_policy,
    double dirichlet_alpha, double dirichlet_epsilon, std::mt19937* rng) {
  if (absl::GetFlag(FLAGS_mcts_policy_temperature) < 0) {
    open_spiel::SpielFatalError("--mcts_policy_temperature must be >= 0.");
  }

  const int game_count = absl::GetFlag(FLAGS_diagnose_games);
  int search_seed = absl::GetFlag(FLAGS_seed);
  if (search_seed == 0) search_seed = 1;

  SplitValueStats root_p0_vs_return0;
  SplitValueStats root_current_vs_return_current;
  SplitValueStats root_current_vs_return0;
  SplitValueStats root_p0_vs_return_current;
  int total_states = 0;
  int p0_wins = 0;
  int p1_wins = 0;
  int draws = 0;

  for (int game_index = 0; game_index < game_count; ++game_index) {
    std::unique_ptr<State> state = game.NewInitialState();
    std::vector<RootValueRecord> records;

    while (!state->IsTerminal()) {
      SampleChanceUntilDecision(state.get(), rng);
      if (state->IsTerminal()) break;
      if (state->LegalActions().empty()) break;

      const int current_player = state->CurrentPlayer();
      SearchTarget target = MCTSTarget(
          game, *state, evaluator, search_seed++, child_selection_policy,
          dirichlet_alpha, dirichlet_epsilon);
      const double root_value_p0 = target.player0_value;
      const double root_value_current =
          current_player == 0 ? root_value_p0 : -root_value_p0;
      records.push_back({current_player, root_value_current, root_value_p0});

      Action action = open_spiel::SampleAction(target.policy, *rng).first;
      state->ApplyAction(action);
    }

    if (!state->IsTerminal()) {
      open_spiel::SpielFatalError("Root-value diagnosis ended before terminal.");
    }

    const std::vector<double> returns = state->Returns();
    if (returns[0] > 0) {
      ++p0_wins;
    } else if (returns[0] < 0) {
      ++p1_wins;
    } else {
      ++draws;
    }

    for (const RootValueRecord& record : records) {
      root_p0_vs_return0.Add(record.current_player, record.root_value_p0,
                             returns[0]);
      root_current_vs_return_current.Add(
          record.current_player, record.root_value_current,
          returns[record.current_player]);
      root_current_vs_return0.Add(record.current_player,
                                  record.root_value_current, returns[0]);
      root_p0_vs_return_current.Add(record.current_player,
                                    record.root_value_p0,
                                    returns[record.current_player]);
    }
    total_states += records.size();
  }

  std::cout << "mode=diagnose_root_value_returns"
            << " games=" << game_count
            << " states=" << total_states
            << " avg_states_per_game="
            << (game_count > 0 ? static_cast<double>(total_states) / game_count
                               : 0)
            << " p0_wins=" << p0_wins
            << " p1_wins=" << p1_wins
            << " draws=" << draws
            << " teacher=" << absl::GetFlag(FLAGS_teacher)
            << " max_simulations=" << absl::GetFlag(FLAGS_max_simulations)
            << " rollout_count=" << absl::GetFlag(FLAGS_rollout_count)
            << " mcts_policy_temperature="
            << absl::GetFlag(FLAGS_mcts_policy_temperature) << "\n";
  PrintRootValueStats("root_p0_vs_return0", root_p0_vs_return0);
  PrintRootValueStats("root_current_vs_return_current",
                      root_current_vs_return_current);
  PrintRootValueStats("root_current_vs_return0_wrong_view",
                      root_current_vs_return0);
  PrintRootValueStats("root_p0_vs_return_current_wrong_view",
                      root_p0_vs_return_current);
}

std::vector<Sample> CollectPureMCTSSelfPlaySamples(
    const open_spiel::Game& game, int num_samples,
    std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator,
    std::mt19937* rng) {
  std::vector<Sample> samples;
  samples.reserve(num_samples);
  int search_seed = absl::GetFlag(FLAGS_seed);
  if (search_seed == 0) search_seed = 1;

  while (samples.size() < num_samples) {
    std::unique_ptr<State> state = game.NewInitialState();
    while (!state->IsTerminal() && samples.size() < num_samples) {
      SampleChanceUntilDecision(state.get(), rng);
      if (state->IsTerminal()) break;
      if (state->LegalActions().empty()) break;

      SearchTarget target = MCTSTarget(
          game, *state, evaluator, search_seed++,
          open_spiel::algorithms::ChildSelectionPolicy::UCT,
          /*dirichlet_alpha=*/0, /*dirichlet_epsilon=*/0);
      samples.push_back({state->LegalActions(), state->ObservationTensor(),
                         target.policy, target.player0_value,
                         state->CurrentPlayer()});

      Action action = open_spiel::SampleAction(target.policy, *rng).first;
      state->ApplyAction(action);
    }
  }
  return samples;
}

torch::Tensor ObservationTensor(const std::vector<Sample>& samples,
                                const std::vector<int>& indices,
                                int flat_input_size,
                                const torch::Device& device) {
  std::vector<float> raw(indices.size() * flat_input_size);
  for (int i = 0; i < indices.size(); ++i) {
    std::copy(samples[indices[i]].observation.begin(),
              samples[indices[i]].observation.end(),
              raw.begin() + i * flat_input_size);
  }
  return torch::from_blob(raw.data(),
                          {static_cast<int64_t>(indices.size()),
                           flat_input_size})
      .clone()
      .to(device);
}

torch::Tensor MaskTensor(const std::vector<Sample>& samples,
                         const std::vector<int>& indices, int num_actions,
                         const torch::Device& device) {
  std::vector<uint8_t> raw(indices.size() * num_actions, 0);
  for (int i = 0; i < indices.size(); ++i) {
    for (Action action : samples[indices[i]].legal_actions) {
      raw[i * num_actions + action] = 1;
    }
  }
  return torch::from_blob(raw.data(),
                          {static_cast<int64_t>(indices.size()), num_actions},
                          torch::TensorOptions().dtype(torch::kByte))
      .clone()
      .to(device);
}

torch::Tensor TargetTensor(const std::vector<Sample>& samples,
                           const std::vector<int>& indices, int num_actions,
                           const torch::Device& device) {
  std::vector<float> raw(indices.size() * num_actions, 0);
  for (int i = 0; i < indices.size(); ++i) {
    for (const auto& [action, prob] : samples[indices[i]].target_policy) {
      raw[i * num_actions + action] = prob;
    }
  }
  return torch::from_blob(raw.data(),
                          {static_cast<int64_t>(indices.size()), num_actions})
      .clone()
      .to(device);
}

Metrics Evaluate(Model& model, const std::vector<Sample>& samples,
                 int flat_input_size, int num_actions,
                 const torch::Device& device, bool train_mode) {
  if (train_mode) {
    model->train();
  } else {
    model->eval();
  }
  torch::NoGradGuard no_grad;
  Metrics metrics;
  std::vector<int> one_index(1);
  for (int i = 0; i < samples.size(); ++i) {
    one_index[0] = i;
    torch::Tensor obs =
        ObservationTensor(samples, one_index, flat_input_size, device);
    torch::Tensor mask = MaskTensor(samples, one_index, num_actions, device);
    torch::Tensor target = TargetTensor(samples, one_index, num_actions, device);
    std::vector<torch::Tensor> outputs = model->forward(obs, mask);
    torch::Tensor value = outputs[0].to(torch::kCPU);
    torch::Tensor policy = outputs[1].to(torch::kCPU);
    torch::Tensor target_cpu = target.to(torch::kCPU);
    auto value_acc = value.accessor<float, 2>();
    auto policy_acc = policy.accessor<float, 2>();
    auto target_acc = target_cpu.accessor<float, 2>();

    ActionsAndProbs predicted;
    predicted.reserve(samples[i].legal_actions.size());
    for (Action action : samples[i].legal_actions) {
      predicted.emplace_back(action, policy_acc[0][action]);
    }

    double ce = 0;
    for (Action action = 0; action < num_actions; ++action) {
      double target_prob = target_acc[0][action];
      if (target_prob > 0) {
        ce -= target_prob *
              std::log(std::max(1e-12, static_cast<double>(policy_acc[0][action])));
      }
    }
    double target_entropy = Entropy(samples[i].target_policy);
    metrics.cross_entropy += ce;
    metrics.kl += ce - target_entropy;
    metrics.target_entropy += target_entropy;
    metrics.policy_entropy += Entropy(predicted);
    const double predicted_value = value_acc[0][0];
    const double value_error = predicted_value - samples[i].target_value;
    metrics.value_mse += value_error * value_error;
    AddValueMetric(&metrics, samples[i], predicted_value);
    metrics.top1_agreement +=
        TopAction(predicted) == TopAction(samples[i].target_policy) ? 1 : 0;
  }
  double denom = samples.size();
  metrics.cross_entropy /= denom;
  metrics.kl /= denom;
  metrics.target_entropy /= denom;
  metrics.policy_entropy /= denom;
  metrics.value_mse /= denom;
  metrics.top1_agreement /= denom;
  return metrics;
}

PredictionStats EvaluatePredictionStats(Model& model,
                                        const std::vector<Sample>& samples,
                                        int flat_input_size, int num_actions,
                                        const torch::Device& device) {
  model->eval();
  torch::NoGradGuard no_grad;
  PredictionStats stats;
  std::vector<int> one_index(1);
  for (int i = 0; i < samples.size(); ++i) {
    one_index[0] = i;
    torch::Tensor obs =
        ObservationTensor(samples, one_index, flat_input_size, device);
    torch::Tensor mask = MaskTensor(samples, one_index, num_actions, device);
    std::vector<torch::Tensor> outputs = model->forward(obs, mask);
    torch::Tensor value = outputs[0].to(torch::kCPU);
    torch::Tensor policy = outputs[1].to(torch::kCPU);
    auto value_acc = value.accessor<float, 2>();
    auto policy_acc = policy.accessor<float, 2>();

    double entropy = 0;
    double max_prob = 0;
    for (Action action : samples[i].legal_actions) {
      double prob = policy_acc[0][action];
      if (prob > 0) entropy -= prob * std::log(prob);
      max_prob = std::max(max_prob, prob);
    }
    stats.policy_entropy += entropy;
    stats.normalized_policy_entropy +=
        samples[i].legal_actions.size() > 1
            ? entropy / std::log(samples[i].legal_actions.size())
            : 0;
    stats.max_probability += max_prob;
    stats.value_abs += std::abs(static_cast<double>(value_acc[0][0]));
  }
  double denom = samples.size();
  stats.policy_entropy /= denom;
  stats.normalized_policy_entropy /= denom;
  stats.max_probability /= denom;
  stats.value_abs /= denom;
  return stats;
}

void PrintPredictionStats(const std::string& label,
                          const PredictionStats& stats) {
  std::cout << label << " policy_entropy=" << stats.policy_entropy
            << " normalized_policy_entropy="
            << stats.normalized_policy_entropy
            << " max_probability=" << stats.max_probability
            << " value_abs=" << stats.value_abs << "\n";
}

void RunBatchNormCalibration(Model& model, const std::vector<Sample>& samples,
                             int flat_input_size, int num_actions,
                             const torch::Device& device,
                             int batch_size, int steps,
                             std::mt19937* rng) {
  model->train();
  torch::NoGradGuard no_grad;
  std::uniform_int_distribution<int> sample_dist(0, samples.size() - 1);
  for (int step = 0; step < steps; ++step) {
    std::vector<int> indices;
    indices.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      indices.push_back(sample_dist(*rng));
    }
    torch::Tensor obs = ObservationTensor(samples, indices, flat_input_size,
                                          device);
    torch::Tensor mask = MaskTensor(samples, indices, num_actions, device);
    model->forward(obs, mask);
  }
}

Metrics EvaluateVPNet(VPNetModel* model, const std::vector<Sample>& samples) {
  std::vector<VPNetModel::InferenceInputs> inputs;
  inputs.reserve(samples.size());
  for (const Sample& sample : samples) {
    inputs.push_back({sample.legal_actions, sample.observation});
  }

  std::vector<VPNetModel::InferenceOutputs> outputs = model->Inference(inputs);
  Metrics metrics;
  for (int i = 0; i < samples.size(); ++i) {
    const Sample& sample = samples[i];
    const VPNetModel::InferenceOutputs& output = outputs[i];

    double ce = 0;
    for (const auto& [target_action, target_prob] : sample.target_policy) {
      double predicted_prob = 0;
      for (const auto& [predicted_action, prob] : output.policy) {
        if (predicted_action == target_action) {
          predicted_prob = prob;
          break;
        }
      }
      if (target_prob > 0) {
        ce -= target_prob * std::log(std::max(1e-12, predicted_prob));
      }
    }

    double target_entropy = Entropy(sample.target_policy);
    double value_error = output.value - sample.target_value;
    metrics.cross_entropy += ce;
    metrics.kl += ce - target_entropy;
    metrics.target_entropy += target_entropy;
    metrics.policy_entropy += Entropy(output.policy);
    metrics.value_mse += value_error * value_error;
    AddValueMetric(&metrics, sample, output.value);
    metrics.top1_agreement +=
        TopAction(output.policy) == TopAction(sample.target_policy) ? 1 : 0;
  }

  double denom = samples.size();
  metrics.cross_entropy /= denom;
  metrics.kl /= denom;
  metrics.target_entropy /= denom;
  metrics.policy_entropy /= denom;
  metrics.value_mse /= denom;
  metrics.top1_agreement /= denom;
  return metrics;
}

void PrintMetrics(int step, const Metrics& metrics) {
  std::cout << "step=" << step << " ce=" << metrics.cross_entropy
            << " kl=" << metrics.kl
            << " target_entropy=" << metrics.target_entropy
            << " policy_entropy=" << metrics.policy_entropy
            << " value_mse=" << metrics.value_mse
            << " top1=" << metrics.top1_agreement << "\n";
}

void PrintValueStats(const std::string& mode, int step,
                     const std::string& player_label,
                     const ValueStats& stats) {
  if (stats.count == 0) return;
  std::cout << "value_split mode=" << mode
            << " step=" << step
            << " player=" << player_label
            << " n=" << stats.count
            << " target_mean=" << stats.TargetMean()
            << " pred_mean=" << stats.PredictionMean()
            << " mse=" << stats.MSE()
            << " sign_acc=" << stats.SignAccuracy()
            << " corr=" << stats.Correlation() << "\n";
}

void PrintRootValueStats(const std::string& label,
                         const SplitValueStats& stats) {
  PrintValueStats(label, /*step=*/0, "all", stats.all);
  PrintValueStats(label, /*step=*/0, "0", stats.by_player[0]);
  PrintValueStats(label, /*step=*/0, "1", stats.by_player[1]);
}

void PrintModeMetrics(const std::string& mode, int step,
                      const Metrics& metrics) {
  std::cout << "mode=" << mode << " ";
  PrintMetrics(step, metrics);
  PrintValueStats(mode, step, "all", metrics.value_all);
  PrintValueStats(mode, step, "0", metrics.value_by_player[0]);
  PrintValueStats(mode, step, "1", metrics.value_by_player[1]);
}

void PrintSavedModelTrainEvalMetrics(const std::string& checkpoint_base,
                                     const ModelConfig& config,
                                     const std::string& device_name,
                                     const torch::Device& device,
                                     const open_spiel::Game& game,
                                     const std::vector<Sample>& samples,
                                     int step) {
  Model train_model(config, device_name);
  train_model->to(device);
  torch::load(train_model, absl::StrCat(checkpoint_base, ".pt"), device);
  PrintModeMetrics(
      "train", step,
      Evaluate(train_model, samples, game.ObservationTensorSize(),
               game.NumDistinctActions(), device, /*train_mode=*/true));

  Model eval_model(config, device_name);
  eval_model->to(device);
  torch::load(eval_model, absl::StrCat(checkpoint_base, ".pt"), device);
  PrintModeMetrics(
      "eval", step,
      Evaluate(eval_model, samples, game.ObservationTensorSize(),
               game.NumDistinctActions(), device, /*train_mode=*/false));
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string az_path = absl::GetFlag(FLAGS_az_path);
  const std::string teacher_mode = absl::GetFlag(FLAGS_teacher);
  const bool needs_input_checkpoint =
      absl::GetFlag(FLAGS_diagnose_self_play_targets) ||
      absl::GetFlag(FLAGS_bn_calibration) || teacher_mode == "az" ||
      absl::GetFlag(FLAGS_init_from_checkpoint);
  if (az_path.empty() && needs_input_checkpoint) {
    open_spiel::SpielFatalError(
        "--az_path must be specified when loading a teacher/checkpoint.");
  }
  if (absl::GetFlag(FLAGS_samples) <= 0 ||
      absl::GetFlag(FLAGS_batch_size) <= 0 ||
      absl::GetFlag(FLAGS_train_steps) < 0 ||
      absl::GetFlag(FLAGS_holdout_samples) < 0 ||
      absl::GetFlag(FLAGS_diagnose_games) <= 0 ||
      absl::GetFlag(FLAGS_diagnose_repeats) <= 0) {
    open_spiel::SpielFatalError(
        "samples, holdout_samples, batch_size, train_steps, diagnose_games, "
        "or diagnose_repeats invalid.");
  }

  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(absl::GetFlag(FLAGS_game));
  const std::string device_name = TorchDeviceName(absl::GetFlag(FLAGS_device));
  torch::Device device(device_name);
  std::mt19937 rng(absl::GetFlag(FLAGS_seed));

  ModelConfig config =
      BaseModelConfig(*game, az_path, absl::GetFlag(FLAGS_az_graph_def));

  if (absl::GetFlag(FLAGS_diagnose_self_play_targets)) {
    VPNetModel teacher(*game, az_path, absl::GetFlag(FLAGS_az_graph_def),
                       absl::GetFlag(FLAGS_device));
    teacher.LoadCheckpoint(absl::GetFlag(FLAGS_az_checkpoint));
    RunSelfPlayTargetDiagnosis(*game, &teacher, &rng);
    return 0;
  }

  if (absl::GetFlag(FLAGS_bn_calibration)) {
    std::vector<Sample> calibration_samples = CollectRandomObservationSamples(
        *game, absl::GetFlag(FLAGS_samples),
        absl::GetFlag(FLAGS_random_decision_steps), &rng);

    Model model(config, device_name);
    model->to(device);
    torch::load(model,
                absl::StrCat(az_path, "/checkpoint-",
                             absl::GetFlag(FLAGS_az_checkpoint), ".pt"),
                device);

    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(config.learning_rate));
    const std::string original_optimizer_path =
        absl::StrCat(az_path, "/checkpoint-",
                     absl::GetFlag(FLAGS_az_checkpoint), "-optimizer.pt");
    if (std::filesystem::exists(original_optimizer_path)) {
      torch::load(optimizer, original_optimizer_path, device);
    }

    const int flat_input_size = game->ObservationTensorSize();
    const int num_actions = game->NumDistinctActions();
    std::cout << "mode=bn_calibration"
              << " samples=" << calibration_samples.size()
              << " batch_size=" << absl::GetFlag(FLAGS_batch_size)
              << " steps=" << absl::GetFlag(FLAGS_train_steps)
              << " random_decision_steps="
              << absl::GetFlag(FLAGS_random_decision_steps)
              << " input_checkpoint=" << az_path << "/checkpoint-"
              << absl::GetFlag(FLAGS_az_checkpoint) << "\n";
    PrintPredictionStats(
        "before",
        EvaluatePredictionStats(model, calibration_samples, flat_input_size,
                                num_actions, device));

    RunBatchNormCalibration(model, calibration_samples, flat_input_size,
                            num_actions, device,
                            absl::GetFlag(FLAGS_batch_size),
                            absl::GetFlag(FLAGS_train_steps), &rng);

    PrintPredictionStats(
        "after",
        EvaluatePredictionStats(model, calibration_samples, flat_input_size,
                                num_actions, device));

    const std::string output_path = absl::GetFlag(FLAGS_bn_calibrated_path);
    std::filesystem::create_directories(output_path);
    open_spiel::algorithms::torch_az::CreateGraphDef(
        *game, config.learning_rate, config.weight_decay, output_path,
        absl::GetFlag(FLAGS_az_graph_def), config.nn_model, config.nn_width,
        config.nn_depth);
    const std::string checkpoint_base =
        absl::StrCat(output_path, "/checkpoint-",
                     absl::GetFlag(FLAGS_az_checkpoint));
    torch::save(model, absl::StrCat(checkpoint_base, ".pt"));
    torch::save(optimizer, absl::StrCat(checkpoint_base, "-optimizer.pt"));
    std::cout << "saved_calibrated_checkpoint=" << checkpoint_base << "\n";
    return 0;
  }

  if (absl::GetFlag(FLAGS_hardcoded_single_sample)) {
    std::unique_ptr<State> state;
    for (int attempt = 0; attempt < 1000; ++attempt) {
      state = RandomState(*game, absl::GetFlag(FLAGS_random_decision_steps),
                          &rng);
      if (!state->IsTerminal() && !state->IsChanceNode() &&
          !state->LegalActions().empty()) {
        break;
      }
    }
    if (state == nullptr || state->IsTerminal() || state->IsChanceNode() ||
        state->LegalActions().empty()) {
      open_spiel::SpielFatalError("Could not sample a decision state.");
    }

    std::vector<Action> legal_actions = state->LegalActions();
    Action target_action = absl::GetFlag(FLAGS_hardcoded_target_action);
    if (target_action < 0) target_action = legal_actions.front();
    if (std::find(legal_actions.begin(), legal_actions.end(), target_action) ==
        legal_actions.end()) {
      open_spiel::SpielFatalError(absl::StrCat(
          "--hardcoded_target_action=", target_action, " is not legal."));
    }

    Sample sample{legal_actions,
                  state->ObservationTensor(),
                  ActionsAndProbs{{target_action, 1.0}},
                  /*target_value=*/0.0,
                  state->CurrentPlayer()};
    std::vector<Sample> samples{sample};

    const std::string student_path = absl::GetFlag(FLAGS_student_path);
    std::filesystem::create_directories(student_path);
    open_spiel::algorithms::torch_az::CreateGraphDef(
        *game, config.learning_rate, config.weight_decay, student_path,
        absl::GetFlag(FLAGS_az_graph_def), config.nn_model, config.nn_width,
        config.nn_depth);
    VPNetModel student(*game, student_path, absl::GetFlag(FLAGS_az_graph_def),
                       absl::GetFlag(FLAGS_device));
    if (absl::GetFlag(FLAGS_init_from_checkpoint)) {
      student.LoadCheckpoint(absl::StrCat(
          az_path, "/checkpoint-", absl::GetFlag(FLAGS_az_checkpoint)));
    }

    VPNetModel::TrainInputs fixed_input{
        sample.legal_actions, sample.observation, sample.target_policy,
        sample.target_value};
    std::vector<VPNetModel::TrainInputs> batch(
        absl::GetFlag(FLAGS_batch_size), fixed_input);

    std::cout << "mode=hardcoded_single_sample"
              << " current_player=" << state->CurrentPlayer()
              << " legal_actions=" << legal_actions.size()
              << " target_action=" << target_action
              << " target_action_string=\""
              << state->ActionToString(state->CurrentPlayer(), target_action)
              << "\""
              << " target_value=0"
              << " init_from_checkpoint="
              << (absl::GetFlag(FLAGS_init_from_checkpoint) ? "true"
                                                             : "false")
              << " trainer=VPNetModel::Learn"
              << " lr=" << absl::GetFlag(FLAGS_learning_rate)
              << " weight_decay=" << absl::GetFlag(FLAGS_weight_decay)
              << " batch_size=" << absl::GetFlag(FLAGS_batch_size) << "\n";

    constexpr int kInspectCheckpointStep = -999999;
    std::string inspect_checkpoint =
        student.SaveCheckpoint(kInspectCheckpointStep);
    PrintSavedModelTrainEvalMetrics(inspect_checkpoint, config, device_name,
                                    device, *game, samples, 0);
    VPNetModel::LossInfo last_loss;
    for (int step = 1; step <= absl::GetFlag(FLAGS_train_steps); ++step) {
      last_loss = student.Learn(batch);
      if (step % absl::GetFlag(FLAGS_report_every) == 0 ||
          step == absl::GetFlag(FLAGS_train_steps)) {
        std::cout << "learn_step=" << step
                  << " policy_loss=" << last_loss.Policy()
                  << " value_loss=" << last_loss.Value()
                  << " l2=" << last_loss.L2()
                  << " total_loss=" << last_loss.Total() << "\n";
        inspect_checkpoint = student.SaveCheckpoint(kInspectCheckpointStep);
        PrintSavedModelTrainEvalMetrics(inspect_checkpoint, config, device_name,
                                        device, *game, samples, step);
      }
    }
    return 0;
  }

  std::vector<Sample> samples;
  samples.reserve(absl::GetFlag(FLAGS_samples));
  std::vector<Sample> holdout_samples;
  holdout_samples.reserve(absl::GetFlag(FLAGS_holdout_samples));

  std::unique_ptr<VPNetModel> az_teacher;
  std::shared_ptr<open_spiel::algorithms::Evaluator> teacher_evaluator;
  if (teacher_mode == "az") {
    az_teacher = std::make_unique<VPNetModel>(
        *game, az_path, absl::GetFlag(FLAGS_az_graph_def),
        absl::GetFlag(FLAGS_device));
    az_teacher->LoadCheckpoint(absl::GetFlag(FLAGS_az_checkpoint));
    teacher_evaluator = std::make_shared<DirectVPNetEvaluator>(az_teacher.get());
  } else if (teacher_mode == "pure_mcts") {
    teacher_evaluator =
        std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(
            absl::GetFlag(FLAGS_rollout_count), absl::GetFlag(FLAGS_seed));
  } else {
    open_spiel::SpielFatalError(
        absl::StrCat("Unknown --teacher=", teacher_mode,
                     ". Expected az or pure_mcts."));
  }

  if (absl::GetFlag(FLAGS_diagnose_root_value_returns)) {
    const bool use_az_teacher = teacher_mode == "az";
    RunRootValueReturnsDiagnosis(
        *game, teacher_evaluator,
        use_az_teacher ? open_spiel::algorithms::ChildSelectionPolicy::PUCT
                       : open_spiel::algorithms::ChildSelectionPolicy::UCT,
        use_az_teacher ? absl::GetFlag(FLAGS_policy_alpha) : 0,
        use_az_teacher ? absl::GetFlag(FLAGS_policy_epsilon) : 0, &rng);
    return 0;
  }

  if (absl::GetFlag(FLAGS_diagnose_mcts_stability)) {
    const bool use_az_teacher = teacher_mode == "az";
    RunMCTSStabilityDiagnosis(
        *game, teacher_evaluator,
        use_az_teacher ? open_spiel::algorithms::ChildSelectionPolicy::PUCT
                       : open_spiel::algorithms::ChildSelectionPolicy::UCT,
        use_az_teacher ? absl::GetFlag(FLAGS_policy_alpha) : 0,
        use_az_teacher ? absl::GetFlag(FLAGS_policy_epsilon) : 0, &rng);
    return 0;
  }

  auto collect_teacher_samples = [&](int sample_count,
                                     const std::string& split_name) {
    std::vector<Sample> collected;
    if (sample_count == 0) return collected;

    if (teacher_mode == "az") {
      std::cout << "Collecting " << sample_count
                << " fixed AZ-guided MCTS " << split_name
                << " policy targets...\n";
      collected.reserve(sample_count);
      for (int i = 0; i < sample_count; ++i) {
        std::unique_ptr<State> state = RandomState(
            *game, absl::GetFlag(FLAGS_random_decision_steps), &rng);
        if (state->IsTerminal() || state->IsChanceNode()) {
          --i;
          continue;
        }
        VPNetModel::InferenceInputs inference_input{
            state->LegalActions(), state->ObservationTensor()};
        double target_value =
            az_teacher->Inference({inference_input})[0].value;
        ActionsAndProbs policy = MCTSPolicy(
            *game, *state, teacher_evaluator,
            absl::GetFlag(FLAGS_seed) + static_cast<int>(collected.size()));
        collected.push_back(Sample{state->LegalActions(),
                                   state->ObservationTensor(), policy,
                                   target_value, state->CurrentPlayer()});
      }
      return collected;
    }

    std::cout << "Collecting " << sample_count
              << " fixed pure-MCTS self-play " << split_name
              << " policy targets...\n";
    return CollectPureMCTSSelfPlaySamples(*game, sample_count,
                                          teacher_evaluator, &rng);
  };

  samples = collect_teacher_samples(absl::GetFlag(FLAGS_samples), "train");
  holdout_samples = collect_teacher_samples(
      absl::GetFlag(FLAGS_holdout_samples), "holdout");
  const double target_entropy = AverageTargetEntropy(samples);
  const double holdout_target_entropy = AverageTargetEntropy(holdout_samples);

  if (absl::GetFlag(FLAGS_use_vpnet_learn)) {
    const std::string student_path = absl::GetFlag(FLAGS_student_path);
    std::filesystem::create_directories(student_path);
    open_spiel::algorithms::torch_az::CreateGraphDef(
        *game, absl::GetFlag(FLAGS_learning_rate),
        absl::GetFlag(FLAGS_weight_decay), student_path,
        absl::GetFlag(FLAGS_az_graph_def), config.nn_model, config.nn_width,
        config.nn_depth);
    VPNetModel student(*game, student_path, absl::GetFlag(FLAGS_az_graph_def),
                       absl::GetFlag(FLAGS_device));
    if (absl::GetFlag(FLAGS_init_from_checkpoint)) {
      student.LoadCheckpoint(absl::StrCat(
          az_path, "/checkpoint-", absl::GetFlag(FLAGS_az_checkpoint)));
    }

    std::vector<VPNetModel::TrainInputs> fixed_data;
    fixed_data.reserve(samples.size());
    for (const Sample& sample : samples) {
      fixed_data.push_back({sample.legal_actions, sample.observation,
                            sample.target_policy, sample.target_value});
    }

    std::cout << "samples=" << samples.size()
              << " holdout_samples=" << holdout_samples.size()
              << " teacher=" << teacher_mode
              << " avg_target_entropy=" << target_entropy
              << " avg_holdout_target_entropy=" << holdout_target_entropy
              << " trainer=VPNetModel::Learn"
              << " init_from_checkpoint="
              << (absl::GetFlag(FLAGS_init_from_checkpoint) ? "true"
                                                             : "false")
              << " nn_model=" << config.nn_model
              << " nn_width=" << config.nn_width
              << " nn_depth=" << config.nn_depth
              << " lr=" << absl::GetFlag(FLAGS_learning_rate)
              << " weight_decay=" << absl::GetFlag(FLAGS_weight_decay)
              << " batch_size=" << absl::GetFlag(FLAGS_batch_size) << "\n";

    if (absl::GetFlag(FLAGS_save_final_checkpoint)) {
      std::string initial_checkpoint = student.SaveCheckpoint(0);
      std::cout << "saved_initial_checkpoint=" << initial_checkpoint << "\n";
    }

    constexpr int kBestHoldoutCheckpointStep =
        VPNetModel::kInvalidCheckpointStep - 1;
    PrintModeMetrics("train", 0, EvaluateVPNet(&student, samples));
    double best_holdout_kl = std::numeric_limits<double>::infinity();
    int best_holdout_step = -1;
    if (!holdout_samples.empty()) {
      Metrics holdout_metrics = EvaluateVPNet(&student, holdout_samples);
      PrintModeMetrics("holdout", 0, holdout_metrics);
      best_holdout_kl = holdout_metrics.kl;
      best_holdout_step = 0;
      if (absl::GetFlag(FLAGS_save_best_holdout_checkpoint)) {
        std::string best_checkpoint =
            student.SaveCheckpoint(kBestHoldoutCheckpointStep);
        std::cout << "saved_best_holdout_checkpoint=" << best_checkpoint
                  << " source_step=0"
                  << " holdout_kl=" << best_holdout_kl << "\n";
      }
    }
    std::uniform_int_distribution<int> sample_dist(0, fixed_data.size() - 1);
    for (int step = 1; step <= absl::GetFlag(FLAGS_train_steps); ++step) {
      std::vector<VPNetModel::TrainInputs> batch;
      batch.reserve(absl::GetFlag(FLAGS_batch_size));
      for (int i = 0; i < absl::GetFlag(FLAGS_batch_size); ++i) {
        batch.push_back(fixed_data[sample_dist(rng)]);
      }
      student.Learn(batch);

      if (step % absl::GetFlag(FLAGS_report_every) == 0 ||
          step == absl::GetFlag(FLAGS_train_steps)) {
        PrintModeMetrics("train", step, EvaluateVPNet(&student, samples));
        if (absl::GetFlag(FLAGS_save_checkpoint_every_report)) {
          std::string report_checkpoint = student.SaveCheckpoint(step);
          std::cout << "saved_report_checkpoint=" << report_checkpoint
                    << "\n";
        }
        if (!holdout_samples.empty()) {
          Metrics holdout_metrics = EvaluateVPNet(&student, holdout_samples);
          PrintModeMetrics("holdout", step, holdout_metrics);
          if (absl::GetFlag(FLAGS_save_best_holdout_checkpoint) &&
              holdout_metrics.kl < best_holdout_kl) {
            best_holdout_kl = holdout_metrics.kl;
            best_holdout_step = step;
            std::string best_checkpoint =
                student.SaveCheckpoint(kBestHoldoutCheckpointStep);
            std::cout << "saved_best_holdout_checkpoint=" << best_checkpoint
                      << " source_step=" << best_holdout_step
                      << " holdout_kl=" << best_holdout_kl << "\n";
          }
        }
      }
    }
    if (absl::GetFlag(FLAGS_save_final_checkpoint)) {
      std::string final_checkpoint =
          student.SaveCheckpoint(VPNetModel::kMostRecentCheckpointStep);
      std::cout << "saved_final_checkpoint=" << final_checkpoint << "\n";
    }
    return 0;
  }

  Model student(config, device_name);
  student->to(device);
  if (absl::GetFlag(FLAGS_init_from_checkpoint)) {
    torch::load(student, absl::StrCat(az_path, "/checkpoint-",
                                      absl::GetFlag(FLAGS_az_checkpoint),
                                      ".pt"),
                device);
  }
  torch::optim::Adam optimizer(
      student->parameters(),
      torch::optim::AdamOptions(absl::GetFlag(FLAGS_learning_rate)));

  const int flat_input_size = game->ObservationTensorSize();
  const int num_actions = game->NumDistinctActions();
  std::cout << "samples=" << samples.size()
            << " holdout_samples=" << holdout_samples.size()
            << " avg_target_entropy=" << target_entropy
            << " avg_holdout_target_entropy=" << holdout_target_entropy
            << " init_from_checkpoint="
            << (absl::GetFlag(FLAGS_init_from_checkpoint) ? "true" : "false")
            << " lr=" << absl::GetFlag(FLAGS_learning_rate)
            << " batch_size=" << absl::GetFlag(FLAGS_batch_size) << "\n";

  PrintModeMetrics("train", 0,
                   Evaluate(student, samples, flat_input_size, num_actions,
                            device, /*train_mode=*/false));
  if (!holdout_samples.empty()) {
    PrintModeMetrics("holdout", 0,
                     Evaluate(student, holdout_samples, flat_input_size,
                              num_actions, device, /*train_mode=*/false));
  }

  std::uniform_int_distribution<int> sample_dist(0, samples.size() - 1);
  for (int step = 1; step <= absl::GetFlag(FLAGS_train_steps); ++step) {
    std::vector<int> indices;
    indices.reserve(absl::GetFlag(FLAGS_batch_size));
    for (int i = 0; i < absl::GetFlag(FLAGS_batch_size); ++i) {
      indices.push_back(sample_dist(rng));
    }

    torch::Tensor obs = ObservationTensor(samples, indices, flat_input_size, device);
    torch::Tensor mask = MaskTensor(samples, indices, num_actions, device);
    torch::Tensor target = TargetTensor(samples, indices, num_actions, device);

    student->train();
    optimizer.zero_grad();
    torch::Tensor policy = student->forward(obs, mask)[1];
    torch::Tensor policy_loss =
        torch::mean(torch::sum(-target * torch::log(torch::clamp(policy, 1e-12, 1)),
                               /*dim=*/-1));
    policy_loss.backward();
    optimizer.step();

    if (step % absl::GetFlag(FLAGS_report_every) == 0 ||
        step == absl::GetFlag(FLAGS_train_steps)) {
      PrintModeMetrics("train", step,
                       Evaluate(student, samples, flat_input_size, num_actions,
                                device, /*train_mode=*/false));
      if (!holdout_samples.empty()) {
        PrintModeMetrics("holdout", step,
                         Evaluate(student, holdout_samples, flat_input_size,
                                  num_actions, device,
                                  /*train_mode=*/false));
      }
    }
  }

  return 0;
}
