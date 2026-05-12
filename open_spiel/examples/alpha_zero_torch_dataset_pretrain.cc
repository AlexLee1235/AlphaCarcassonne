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

#include <nop/serializer.h>
#include <nop/structure.h>
#include <nop/utility/stream_reader.h>
#include <nop/utility/stream_writer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "c10/util/Exception.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/alpha_zero_torch/device_manager.h"
#include "open_spiel/algorithms/alpha_zero_torch/model.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(std::string, mode, "inspect", "generate, train, or inspect.");
ABSL_FLAG(std::string, game, "carcassonne", "The game to load.");
ABSL_FLAG(std::string, dataset, "", "Train dataset path.");
ABSL_FLAG(std::string, holdout_dataset, "", "Holdout dataset path.");
ABSL_FLAG(int, samples, 256, "Number of train samples to generate.");
ABSL_FLAG(int, holdout_samples, 0, "Number of holdout samples to generate.");
ABSL_FLAG(std::string, teacher, "pure_mcts",
          "Teacher used for generation: pure_mcts or az.");
ABSL_FLAG(int, max_simulations, 320, "MCTS simulations per target.");
ABSL_FLAG(int, rollout_count, 10, "Random rollouts per pure-MCTS leaf.");
ABSL_FLAG(double, uct_c, 2, "UCT exploration constant.");
ABSL_FLAG(double, mcts_policy_temperature, 1,
          "Temperature applied to MCTS visit counts.");
ABSL_FLAG(double, policy_alpha, 1, "AZ teacher Dirichlet noise alpha.");
ABSL_FLAG(double, policy_epsilon, 0.25, "AZ teacher Dirichlet noise epsilon.");
ABSL_FLAG(double, temperature, 1,
          "Temperature used to sample AZ self-play actions before drop.");
ABSL_FLAG(int, temperature_drop, 10,
          "Drop AZ self-play action temperature to 0 after this many history "
          "entries, matching alpha_zero.cc.");
ABSL_FLAG(int, num_workers, 1, "Parallel CPU workers for generation.");
ABSL_FLAG(uint_fast32_t, seed, 1, "Random seed.");

ABSL_FLAG(std::string, az_path, "", "Checkpoint source path.");
ABSL_FLAG(std::string, az_graph_def, "vpnet.pb", "Graph definition file.");
ABSL_FLAG(int, az_checkpoint, -1, "Checkpoint step to load.");
ABSL_FLAG(bool, init_from_checkpoint, false,
          "Initialize student from --az_path/--az_checkpoint.");
ABSL_FLAG(std::string, student_path, "/tmp/az_dataset_pretrain_student",
          "Output path for student graph and checkpoints.");
ABSL_FLAG(std::string, nn_model, "resnet", "Student model type.");
ABSL_FLAG(int, nn_width, 128, "Student model width.");
ABSL_FLAG(int, nn_depth, 10, "Student model depth.");
ABSL_FLAG(double, learning_rate, 0.001, "Student learning rate.");
ABSL_FLAG(double, weight_decay, 0.0001, "Student weight decay.");
ABSL_FLAG(std::string, device, "/cpu:0", "Torch device, e.g. /cuda:0.");
ABSL_FLAG(int, batch_size, 64, "Training batch size.");
ABSL_FLAG(int, train_steps, 500, "Gradient steps.");
ABSL_FLAG(int, report_every, 50, "Print metrics every N steps.");
ABSL_FLAG(bool, save_final_checkpoint, false,
          "Save checkpoint-0 before training and checkpoint--1 after training.");
ABSL_FLAG(bool, save_best_holdout_checkpoint, false,
          "Save checkpoint--3 whenever holdout KL reaches a new best.");
ABSL_FLAG(bool, save_checkpoint_every_report, false,
          "Save checkpoint-<step> at each report.");
ABSL_FLAG(bool, quiet_torch_warnings, true,
          "Suppress LibTorch warnings from this diagnostic binary.");

namespace {

using open_spiel::Action;
using open_spiel::ActionsAndProbs;
using open_spiel::State;
using open_spiel::algorithms::SearchNode;
using open_spiel::algorithms::torch_az::DeviceManager;
using open_spiel::algorithms::torch_az::ModelConfig;
using open_spiel::algorithms::torch_az::VPNetEvaluator;
using open_spiel::algorithms::torch_az::VPNetModel;

constexpr int kFormatVersion = 1;
constexpr int kBestHoldoutCheckpointStep =
    VPNetModel::kInvalidCheckpointStep - 1;

class QuietTorchWarningHandler : public c10::WarningHandler {
 public:
  void process(const c10::Warning&) override {}
};

struct DatasetSample {
  std::vector<Action> legal_actions;
  std::vector<float> observation;
  ActionsAndProbs target_policy;
  double target_value = 0;
  int current_player = 0;

  NOP_STRUCTURE(DatasetSample, legal_actions, observation, target_policy,
                target_value, current_player);
};

struct DatasetHeader {
  int format_version = kFormatVersion;
  std::string game_string;
  std::vector<int> observation_tensor_shape;
  int num_distinct_actions = 0;
  std::string teacher = "pure_mcts";
  int max_simulations = 0;
  int rollout_count = 0;
  double mcts_policy_temperature = 1;

  NOP_STRUCTURE(DatasetHeader, format_version, game_string,
                observation_tensor_shape, num_distinct_actions, teacher,
                max_simulations, rollout_count, mcts_policy_temperature);
};

struct DatasetFile {
  DatasetHeader header;
  std::vector<DatasetSample> samples;

  NOP_STRUCTURE(DatasetFile, header, samples);
};

struct SearchTarget {
  ActionsAndProbs policy;
  ActionsAndProbs action_policy;
  double player0_value = 0;
  Action best_action = open_spiel::kInvalidAction;
};

struct TargetStats {
  int count = 0;
  double sum = 0;
  double square_sum = 0;
  double min = std::numeric_limits<double>::infinity();
  double max = -std::numeric_limits<double>::infinity();
  int positive = 0;
  int negative = 0;
  int zero = 0;

  void Add(double value) {
    ++count;
    sum += value;
    square_sum += value * value;
    min = std::min(min, value);
    max = std::max(max, value);
    if (value > 0) {
      ++positive;
    } else if (value < 0) {
      ++negative;
    } else {
      ++zero;
    }
  }

  double Mean() const { return count > 0 ? sum / count : 0; }
  double StdDev() const {
    if (count <= 0) return 0;
    const double mean = Mean();
    return std::sqrt(std::max(0.0, square_sum / count - mean * mean));
  }
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
        std::sqrt(std::max(0.0, prediction_var) *
                  std::max(0.0, target_var));
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

std::string TorchDeviceName(const std::string& device) {
  if (!device.empty() && device[0] == '/') return device.substr(1);
  return device;
}

double Entropy(const ActionsAndProbs& policy) {
  double entropy = 0;
  for (const auto& [action, prob] : policy) {
    if (prob > 0) entropy -= prob * std::log(prob);
  }
  return entropy;
}

double AverageTargetEntropy(const std::vector<DatasetSample>& samples) {
  double entropy = 0;
  for (const DatasetSample& sample : samples) {
    entropy += Entropy(sample.target_policy);
  }
  return samples.empty() ? 0 : entropy / samples.size();
}

std::string ShapeString(const std::vector<int>& shape) {
  std::string result = "[";
  for (int i = 0; i < shape.size(); ++i) {
    if (i > 0) absl::StrAppend(&result, ",");
    absl::StrAppend(&result, shape[i]);
  }
  absl::StrAppend(&result, "]");
  return result;
}

double PolicyProb(const ActionsAndProbs& policy, Action action) {
  for (const auto& [candidate, prob] : policy) {
    if (candidate == action) return prob;
  }
  return 0;
}

double CrossEntropy(const ActionsAndProbs& target,
                    const ActionsAndProbs& prediction) {
  double cross_entropy = 0;
  for (const auto& [action, target_prob] : target) {
    if (target_prob > 0) {
      cross_entropy -=
          target_prob * std::log(std::max(1e-12, PolicyProb(prediction, action)));
    }
  }
  return cross_entropy;
}

Action TopAction(const ActionsAndProbs& policy) {
  SPIEL_CHECK_FALSE(policy.empty());
  return std::max_element(policy.begin(), policy.end(),
                          [](const auto& a, const auto& b) {
                            return a.second < b.second;
                          })
      ->first;
}

void AddValueMetric(Metrics* metrics, const DatasetSample& sample,
                    double prediction) {
  metrics->value_all.Add(prediction, sample.target_value);
  if (sample.current_player == 0 || sample.current_player == 1) {
    metrics->value_by_player[sample.current_player].Add(prediction,
                                                        sample.target_value);
  }
}

void SampleChanceUntilDecision(State* state, std::mt19937* rng) {
  while (state->IsChanceNode()) {
    Action action = open_spiel::SampleAction(state->ChanceOutcomes(), *rng).first;
    state->ApplyAction(action);
  }
}

ActionsAndProbs VisitCountPolicy(const SearchNode& root, double temperature) {
  ActionsAndProbs policy;
  policy.reserve(root.children.size());
  if (root.children.empty()) return policy;
  if (temperature == 0) {
    policy.emplace_back(root.BestChild().action, 1.0);
    return policy;
  }

  double weight_sum = 0;
  for (const SearchNode& child : root.children) {
    const double weight = std::pow(child.explore_count, 1.0 / temperature);
    policy.emplace_back(child.action, weight);
    weight_sum += weight;
  }
  if (weight_sum <= 0) {
    const double uniform_prob = 1.0 / root.children.size();
    for (auto& [action, prob] : policy) {
      prob = uniform_prob;
    }
  } else {
    open_spiel::NormalizePolicy(&policy);
  }
  return policy;
}

SearchTarget MCTSTarget(
    const open_spiel::Game& game, const State& state,
    const std::shared_ptr<open_spiel::algorithms::Evaluator>& evaluator,
    int seed,
    open_spiel::algorithms::ChildSelectionPolicy child_selection_policy,
    double dirichlet_alpha, double dirichlet_epsilon) {
  open_spiel::algorithms::MCTSBot bot(
      game, evaluator, absl::GetFlag(FLAGS_uct_c),
      absl::GetFlag(FLAGS_max_simulations),
      /*max_memory_mb=*/1000,
      /*solve=*/false, seed,
      /*verbose=*/false, child_selection_policy,
      dirichlet_alpha, dirichlet_epsilon,
      /*dont_return_chance_node=*/true);
  std::unique_ptr<SearchNode> root = bot.MCTSearch(state);

  const double root_value =
      root->explore_count > 0 ? root->total_reward / root->explore_count : 0;
  const double player0_value =
      state.CurrentPlayer() == 0 ? root_value : -root_value;
  return {VisitCountPolicy(*root, absl::GetFlag(FLAGS_mcts_policy_temperature)),
          VisitCountPolicy(*root, absl::GetFlag(FLAGS_temperature)),
          player0_value, root->BestChild().action};
}

SearchTarget MCTSTarget(
    const open_spiel::Game& game, const State& state,
    const std::shared_ptr<open_spiel::algorithms::Evaluator>& evaluator,
    int seed) {
  return MCTSTarget(game, state, evaluator, seed,
                    open_spiel::algorithms::ChildSelectionPolicy::UCT,
                    /*dirichlet_alpha=*/0, /*dirichlet_epsilon=*/0);
}

std::vector<DatasetSample> CollectPureMCTSSelfPlaySamples(
    const open_spiel::Game& game, int num_samples, int worker_id,
    uint_fast32_t seed) {
  std::vector<DatasetSample> samples;
  samples.reserve(num_samples);
  std::mt19937 rng(seed + 1000003 * worker_id);
  int search_seed = static_cast<int>(seed + 2000003 * worker_id + 1);
  auto evaluator = std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(
      absl::GetFlag(FLAGS_rollout_count),
      static_cast<int>(seed + 3000017 * worker_id + 1));

  while (samples.size() < num_samples) {
    std::unique_ptr<State> state = game.NewInitialState();
    while (!state->IsTerminal() && samples.size() < num_samples) {
      SampleChanceUntilDecision(state.get(), &rng);
      if (state->IsTerminal()) break;
      if (state->LegalActions().empty()) break;

      SearchTarget target = MCTSTarget(game, *state, evaluator, search_seed++);
      samples.push_back({state->LegalActions(), state->ObservationTensor(),
                         target.policy, target.player0_value,
                         state->CurrentPlayer()});

      Action action = open_spiel::SampleAction(target.policy, rng).first;
      state->ApplyAction(action);
    }
  }
  return samples;
}

std::vector<DatasetSample> CollectAZMCTSSelfPlaySamples(
    const open_spiel::Game& game, int num_samples,
    const std::shared_ptr<open_spiel::algorithms::Evaluator>& evaluator,
    int worker_id, uint_fast32_t seed) {
  std::vector<DatasetSample> samples;
  samples.reserve(num_samples);
  std::mt19937 rng(seed + 1000003 * worker_id);
  int search_seed = static_cast<int>(seed + 2000003 * worker_id + 1);

  while (samples.size() < num_samples) {
    std::unique_ptr<State> state = game.NewInitialState();
    std::vector<DatasetSample> pending_game_samples;
    const int remaining_samples =
        num_samples - static_cast<int>(samples.size());
    pending_game_samples.reserve(
        std::min<int>(game.MaxGameLength(), remaining_samples));
    int history_size = 0;

    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        Action action =
            open_spiel::SampleAction(state->ChanceOutcomes(), rng).first;
        state->ApplyAction(action);
        ++history_size;
        continue;
      }
      if (state->LegalActions().empty()) break;

      SearchTarget target = MCTSTarget(
          game, *state, evaluator, search_seed++,
          open_spiel::algorithms::ChildSelectionPolicy::PUCT,
          absl::GetFlag(FLAGS_policy_alpha),
          absl::GetFlag(FLAGS_policy_epsilon));

      if (samples.size() + pending_game_samples.size() < num_samples) {
        pending_game_samples.push_back(
            {state->LegalActions(), state->ObservationTensor(), target.policy,
             /*target_value=*/0, state->CurrentPlayer()});
      }

      Action action = target.best_action;
      if (history_size < absl::GetFlag(FLAGS_temperature_drop)) {
        SPIEL_CHECK_FALSE(target.action_policy.empty());
        action = open_spiel::SampleAction(target.action_policy, rng).first;
      }
      state->ApplyAction(action);
      ++history_size;
    }

    if (!state->IsTerminal()) {
      continue;
    }
    const std::vector<double> returns = state->Returns();
    SPIEL_CHECK_FALSE(returns.empty());
    const double player0_return = returns[0];
    for (DatasetSample& sample : pending_game_samples) {
      sample.target_value = player0_return;
      samples.push_back(std::move(sample));
    }
  }
  SPIEL_CHECK_EQ(samples.size(), num_samples);
  return samples;
}

DatasetHeader MakeHeader(const open_spiel::Game& game,
                         const std::string& game_string) {
  const std::string teacher = absl::GetFlag(FLAGS_teacher);
  return DatasetHeader{/*format_version=*/kFormatVersion,
                       /*game_string=*/game_string,
                       /*observation_tensor_shape=*/game.ObservationTensorShape(),
                       /*num_distinct_actions=*/game.NumDistinctActions(),
                       /*teacher=*/teacher,
                       /*max_simulations=*/absl::GetFlag(FLAGS_max_simulations),
                       /*rollout_count=*/
                       teacher == "az" ? 0 : absl::GetFlag(FLAGS_rollout_count),
                       /*mcts_policy_temperature=*/
                       absl::GetFlag(FLAGS_mcts_policy_temperature)};
}

void EnsureParentDirectory(const std::string& path) {
  std::filesystem::path fs_path(path);
  if (!fs_path.parent_path().empty()) {
    std::filesystem::create_directories(fs_path.parent_path());
  }
}

void SaveDataset(const DatasetFile& dataset, const std::string& path) {
  EnsureParentDirectory(path);
  nop::Serializer<nop::StreamWriter<std::ofstream>> serializer{path};
  auto status = serializer.Write(dataset);
  if (!status) {
    open_spiel::SpielFatalError(absl::StrCat(
        "Failed to write dataset: ", path,
        " error=", status.GetErrorMessage()));
  }
}

DatasetFile LoadDataset(const std::string& path) {
  std::error_code error;
  const bool exists = std::filesystem::exists(path, error);
  if (error || !exists) {
    open_spiel::SpielFatalError(absl::StrCat(
        "Dataset file does not exist: ", path));
  }
  const uintmax_t bytes = std::filesystem::file_size(path, error);
  if (error || bytes == 0) {
    open_spiel::SpielFatalError(absl::StrCat(
        "Dataset file is empty or unreadable: ", path));
  }
  DatasetFile dataset;
  nop::Deserializer<nop::StreamReader<std::ifstream>> deserializer{path};
  auto status = deserializer.Read(&dataset);
  if (!status) {
    open_spiel::SpielFatalError(absl::StrCat(
        "Failed to read dataset: ", path,
        " error=", status.GetErrorMessage(),
        ". Regenerate this file with --mode=generate."));
  }
  return dataset;
}

void ValidateDataset(const DatasetFile& dataset, const open_spiel::Game& game,
                     const std::string& expected_game_string) {
  if (dataset.header.format_version != kFormatVersion) {
    open_spiel::SpielFatalError(absl::StrCat(
        "Unsupported dataset format_version=", dataset.header.format_version,
        " expected=", kFormatVersion));
  }
  if (dataset.header.game_string != expected_game_string) {
    open_spiel::SpielFatalError(absl::StrCat(
        "Dataset game mismatch. dataset_game='", dataset.header.game_string,
        "' requested_game='", expected_game_string,
        "'. Use --game='", dataset.header.game_string,
        "' or regenerate the dataset."));
  }
  if (dataset.header.observation_tensor_shape != game.ObservationTensorShape()) {
    open_spiel::SpielFatalError(absl::StrCat(
        "Dataset observation shape mismatch. dataset_shape=",
        ShapeString(dataset.header.observation_tensor_shape),
        " requested_shape=", ShapeString(game.ObservationTensorShape()),
        ". Regenerate the dataset after observation-plane changes."));
  }
  if (dataset.header.num_distinct_actions != game.NumDistinctActions()) {
    open_spiel::SpielFatalError(absl::StrCat(
        "Dataset action count mismatch. dataset_actions=",
        dataset.header.num_distinct_actions,
        " requested_actions=", game.NumDistinctActions(),
        ". Regenerate the dataset."));
  }
  for (const DatasetSample& sample : dataset.samples) {
    SPIEL_CHECK_EQ(sample.observation.size(), game.ObservationTensorSize());
    SPIEL_CHECK_TRUE(sample.target_value >= -1.000001);
    SPIEL_CHECK_TRUE(sample.target_value <= 1.000001);
  }
}

DatasetFile GenerateDataset(const std::string& game_string, int sample_count,
                            uint_fast32_t seed) {
  const std::string teacher = absl::GetFlag(FLAGS_teacher);
  if (teacher != "pure_mcts" && teacher != "az") {
    open_spiel::SpielFatalError(
        absl::StrCat("Unknown --teacher=", teacher,
                     ". Expected pure_mcts or az."));
  }
  SPIEL_CHECK_GE(sample_count, 0);
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame(game_string);
  DatasetFile dataset;
  dataset.header = MakeHeader(*game, game_string);
  if (sample_count == 0) return dataset;

  const int worker_count = std::max(1, absl::GetFlag(FLAGS_num_workers));
  std::vector<std::future<std::vector<DatasetSample>>> futures;
  futures.reserve(worker_count);
  std::shared_ptr<DeviceManager> az_device_manager;
  std::shared_ptr<open_spiel::algorithms::Evaluator> az_evaluator;

  if (teacher == "az") {
    const std::string az_path = absl::GetFlag(FLAGS_az_path);
    if (az_path.empty()) {
      open_spiel::SpielFatalError("--az_path is required when --teacher=az.");
    }
    const int inference_batch_size = std::max(1, std::min(64, worker_count));
    const int inference_threads = inference_batch_size > 1 ? 1 : 0;
    az_device_manager = std::make_shared<DeviceManager>();
    VPNetModel teacher_model(*game, az_path, absl::GetFlag(FLAGS_az_graph_def),
                             absl::GetFlag(FLAGS_device));
    teacher_model.LoadCheckpoint(absl::GetFlag(FLAGS_az_checkpoint));
    az_device_manager->AddDevice(std::move(teacher_model));
    az_evaluator = std::make_shared<VPNetEvaluator>(
        az_device_manager.get(), inference_batch_size, inference_threads,
        /*cache_size=*/0);

    for (int worker = 0; worker < worker_count; ++worker) {
      const int worker_samples =
          sample_count / worker_count + (worker < sample_count % worker_count);
      futures.push_back(std::async(std::launch::async, [=]() {
        std::shared_ptr<const open_spiel::Game> worker_game =
            open_spiel::LoadGame(game_string);
        return CollectAZMCTSSelfPlaySamples(*worker_game, worker_samples,
                                            az_evaluator, worker, seed);
      }));
    }
  } else {
    for (int worker = 0; worker < worker_count; ++worker) {
      const int worker_samples =
          sample_count / worker_count + (worker < sample_count % worker_count);
      futures.push_back(std::async(std::launch::async, [=]() {
        std::shared_ptr<const open_spiel::Game> worker_game =
            open_spiel::LoadGame(game_string);
        return CollectPureMCTSSelfPlaySamples(*worker_game, worker_samples,
                                              worker, seed);
      }));
    }
  }

  dataset.samples.reserve(sample_count);
  for (auto& future : futures) {
    std::vector<DatasetSample> worker_samples = future.get();
    dataset.samples.insert(dataset.samples.end(),
                           std::make_move_iterator(worker_samples.begin()),
                           std::make_move_iterator(worker_samples.end()));
  }
  SPIEL_CHECK_EQ(dataset.samples.size(), sample_count);
  return dataset;
}

void PrintTargetStats(const std::string& label,
                      const std::vector<DatasetSample>& samples) {
  TargetStats stats;
  std::array<TargetStats, 2> by_player;
  for (const DatasetSample& sample : samples) {
    stats.Add(sample.target_value);
    if (sample.current_player == 0 || sample.current_player == 1) {
      by_player[sample.current_player].Add(sample.target_value);
    }
  }
  std::cout << "[dataset " << label << "]\n"
            << "  samples: " << stats.count << "\n"
            << "  target_entropy: " << AverageTargetEntropy(samples) << "\n"
            << "  value: mean=" << stats.Mean()
            << " std=" << stats.StdDev()
            << " min=" << (stats.count ? stats.min : 0)
            << " max=" << (stats.count ? stats.max : 0) << "\n"
            << "  value_counts: pos=" << stats.positive
            << " neg=" << stats.negative
            << " zero=" << stats.zero << "\n";
  for (int player = 0; player < 2; ++player) {
    const TargetStats& p = by_player[player];
    std::cout << "  player" << player
              << ": samples=" << p.count
              << " mean=" << p.Mean()
              << " std=" << p.StdDev()
              << " pos=" << p.positive
              << " neg=" << p.negative
              << " zero=" << p.zero << "\n";
  }
}

Metrics EvaluateVPNet(VPNetModel* model,
                      const std::vector<DatasetSample>& samples) {
  constexpr int kEvalBatchSize = 1024;
  Metrics metrics;
  if (samples.empty()) return metrics;

  for (int start = 0; start < samples.size(); start += kEvalBatchSize) {
    const int end = std::min<int>(samples.size(), start + kEvalBatchSize);
    std::vector<VPNetModel::InferenceInputs> inputs;
    inputs.reserve(end - start);
    for (int i = start; i < end; ++i) {
      inputs.push_back({samples[i].legal_actions, samples[i].observation});
    }

    std::vector<VPNetModel::InferenceOutputs> outputs = model->Inference(inputs);
    for (int local = 0; local < outputs.size(); ++local) {
      const DatasetSample& sample = samples[start + local];
      const VPNetModel::InferenceOutputs& output = outputs[local];
      const double ce = CrossEntropy(sample.target_policy, output.policy);
      const double target_entropy = Entropy(sample.target_policy);
      const double value_error = output.value - sample.target_value;
      metrics.cross_entropy += ce;
      metrics.kl += ce - target_entropy;
      metrics.target_entropy += target_entropy;
      metrics.policy_entropy += Entropy(output.policy);
      metrics.value_mse += value_error * value_error;
      metrics.top1_agreement +=
          TopAction(output.policy) == TopAction(sample.target_policy) ? 1 : 0;
      AddValueMetric(&metrics, sample, output.value);
    }
  }

  const double denom = samples.size();
  metrics.cross_entropy /= denom;
  metrics.kl /= denom;
  metrics.target_entropy /= denom;
  metrics.policy_entropy /= denom;
  metrics.value_mse /= denom;
  metrics.top1_agreement /= denom;
  return metrics;
}

void PrintValueStats(const std::string& player_label,
                     const ValueStats& stats) {
  if (stats.count == 0) return;
  std::cout << "  value " << player_label
            << ": n=" << stats.count
            << " mse=" << stats.MSE()
            << " sign=" << stats.SignAccuracy()
            << " corr=" << stats.Correlation()
            << " target_mean=" << stats.TargetMean()
            << " pred_mean=" << stats.PredictionMean() << "\n";
}

void PrintModeMetrics(const std::string& mode, int step,
                      const Metrics& metrics) {
  std::cout << "[" << mode << " step=" << step << "]\n"
            << "  policy: ce=" << metrics.cross_entropy
            << " kl=" << metrics.kl
            << " target_H=" << metrics.target_entropy
            << " pred_H=" << metrics.policy_entropy
            << " top1=" << metrics.top1_agreement << "\n"
            << "  value_mse: " << metrics.value_mse << "\n";
  PrintValueStats("all", metrics.value_all);
  PrintValueStats("p0_turn", metrics.value_by_player[0]);
  PrintValueStats("p1_turn", metrics.value_by_player[1]);
}

ModelConfig LoadModelConfig(const std::string& path,
                            const std::string& graph_def) {
  std::ifstream file(std::filesystem::path(path) / graph_def);
  SPIEL_CHECK_TRUE(file.good());
  ModelConfig config;
  file >> config;
  return config;
}

void SaveModelConfig(const std::string& path, const std::string& graph_def,
                     const ModelConfig& config) {
  std::ofstream file(std::filesystem::path(path) / graph_def);
  SPIEL_CHECK_TRUE(file.good());
  file << config;
}

void SaveCheckpointModelConfig(const std::string& source_path,
                               const std::string& target_path,
                               const std::string& graph_def) {
  ModelConfig config = LoadModelConfig(source_path, graph_def);
  config.learning_rate = absl::GetFlag(FLAGS_learning_rate);
  config.weight_decay = absl::GetFlag(FLAGS_weight_decay);
  SaveModelConfig(target_path, graph_def, config);
}

std::vector<VPNetModel::TrainInputs> ToTrainInputs(
    const std::vector<DatasetSample>& samples) {
  std::vector<VPNetModel::TrainInputs> train_inputs;
  train_inputs.reserve(samples.size());
  for (const DatasetSample& sample : samples) {
    train_inputs.push_back({sample.legal_actions, sample.observation,
                            sample.target_policy, sample.target_value});
  }
  return train_inputs;
}

int RunGenerate() {
  const std::string game_string = absl::GetFlag(FLAGS_game);
  const std::string dataset_path = absl::GetFlag(FLAGS_dataset);
  SPIEL_CHECK_FALSE(dataset_path.empty());

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame(game_string);
  const std::string teacher = absl::GetFlag(FLAGS_teacher);
  std::cout << "[generate]\n"
            << "  game: " << game_string << "\n"
            << "  obs_shape: " << ShapeString(game->ObservationTensorShape())
            << "\n"
            << "  actions: " << game->NumDistinctActions() << "\n"
            << "  teacher: " << teacher
            << " sim=" << absl::GetFlag(FLAGS_max_simulations)
            << " rollouts="
            << (teacher == "az" ? 0 : absl::GetFlag(FLAGS_rollout_count))
            << " temp=" << absl::GetFlag(FLAGS_mcts_policy_temperature)
            << "\n"
            << "  workers: " << absl::GetFlag(FLAGS_num_workers)
            << "\n"
            << "  train_out: " << dataset_path << "\n";
  if (teacher == "az") {
    std::cout << "  az_path: " << absl::GetFlag(FLAGS_az_path)
              << " checkpoint=" << absl::GetFlag(FLAGS_az_checkpoint)
              << " device=" << absl::GetFlag(FLAGS_device) << "\n"
              << "  puct: alpha=" << absl::GetFlag(FLAGS_policy_alpha)
              << " epsilon=" << absl::GetFlag(FLAGS_policy_epsilon)
              << " action_temp=" << absl::GetFlag(FLAGS_temperature)
              << " temp_drop=" << absl::GetFlag(FLAGS_temperature_drop)
              << "\n";
  }
  if (absl::GetFlag(FLAGS_holdout_samples) > 0) {
    std::cout << "  holdout_out: " << absl::GetFlag(FLAGS_holdout_dataset)
              << "\n";
  }

  std::cout << "[generate train] samples=" << absl::GetFlag(FLAGS_samples)
            << "\n";
  DatasetFile train =
      GenerateDataset(game_string, absl::GetFlag(FLAGS_samples),
                      absl::GetFlag(FLAGS_seed));
  SaveDataset(train, dataset_path);
  std::cout << "[saved train] " << dataset_path << "\n";
  PrintTargetStats("train", train.samples);

  const int holdout_count = absl::GetFlag(FLAGS_holdout_samples);
  const std::string holdout_path = absl::GetFlag(FLAGS_holdout_dataset);
  if (holdout_count > 0) {
    SPIEL_CHECK_FALSE(holdout_path.empty());
    std::cout << "[generate holdout] samples=" << holdout_count << "\n";
    DatasetFile holdout =
        GenerateDataset(game_string, holdout_count,
                        absl::GetFlag(FLAGS_seed) + 50000019);
    SaveDataset(holdout, holdout_path);
    std::cout << "[saved holdout] " << holdout_path << "\n";
    PrintTargetStats("holdout", holdout.samples);
  }
  return 0;
}

int RunInspect() {
  const std::string game_string = absl::GetFlag(FLAGS_game);
  const std::string dataset_path = absl::GetFlag(FLAGS_dataset);
  SPIEL_CHECK_FALSE(dataset_path.empty());
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame(game_string);
  DatasetFile dataset = LoadDataset(dataset_path);
  ValidateDataset(dataset, *game, game_string);

  std::cout << "[inspect]\n"
            << "  dataset: " << dataset_path << "\n"
            << "  format: " << dataset.header.format_version << "\n"
            << "  game: " << dataset.header.game_string << "\n"
            << "  obs_shape: "
            << ShapeString(dataset.header.observation_tensor_shape) << "\n"
            << "  actions: " << dataset.header.num_distinct_actions << "\n"
            << "  teacher: " << dataset.header.teacher
            << " sim=" << dataset.header.max_simulations
            << " rollouts=" << dataset.header.rollout_count
            << " temp=" << dataset.header.mcts_policy_temperature << "\n";
  PrintTargetStats("inspect", dataset.samples);
  return 0;
}

int RunTrain() {
  const std::string game_string = absl::GetFlag(FLAGS_game);
  const std::string dataset_path = absl::GetFlag(FLAGS_dataset);
  SPIEL_CHECK_FALSE(dataset_path.empty());
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame(game_string);
  DatasetFile train = LoadDataset(dataset_path);
  ValidateDataset(train, *game, game_string);

  DatasetFile holdout;
  const std::string holdout_path = absl::GetFlag(FLAGS_holdout_dataset);
  if (!holdout_path.empty()) {
    holdout = LoadDataset(holdout_path);
    ValidateDataset(holdout, *game, game_string);
  }

  const std::string student_path = absl::GetFlag(FLAGS_student_path);
  std::filesystem::create_directories(student_path);
  if (absl::GetFlag(FLAGS_init_from_checkpoint)) {
    const std::string az_path = absl::GetFlag(FLAGS_az_path);
    SPIEL_CHECK_FALSE(az_path.empty());
    SaveCheckpointModelConfig(az_path, student_path,
                              absl::GetFlag(FLAGS_az_graph_def));
  } else {
    open_spiel::algorithms::torch_az::CreateGraphDef(
        *game, absl::GetFlag(FLAGS_learning_rate),
        absl::GetFlag(FLAGS_weight_decay), student_path,
        absl::GetFlag(FLAGS_az_graph_def), absl::GetFlag(FLAGS_nn_model),
        absl::GetFlag(FLAGS_nn_width), absl::GetFlag(FLAGS_nn_depth));
  }
  const ModelConfig student_config =
      LoadModelConfig(student_path, absl::GetFlag(FLAGS_az_graph_def));

  VPNetModel student(*game, student_path, absl::GetFlag(FLAGS_az_graph_def),
                     absl::GetFlag(FLAGS_device));
  if (absl::GetFlag(FLAGS_init_from_checkpoint)) {
    student.LoadCheckpoint(absl::StrCat(absl::GetFlag(FLAGS_az_path),
                                        "/checkpoint-",
                                        absl::GetFlag(FLAGS_az_checkpoint)));
  }

  std::vector<VPNetModel::TrainInputs> fixed_data = ToTrainInputs(train.samples);
  SPIEL_CHECK_FALSE(fixed_data.empty());

  std::cout << "[train]\n"
            << "  game: " << game_string << "\n"
            << "  train_dataset: " << dataset_path << "\n"
            << "  train_samples: " << train.samples.size() << "\n"
            << "  holdout_dataset: " << holdout_path << "\n"
            << "  holdout_samples: " << holdout.samples.size() << "\n"
            << "  student_path: " << student_path << "\n"
            << "  init_from_checkpoint: "
            << (absl::GetFlag(FLAGS_init_from_checkpoint) ? "true" : "false")
            << "\n"
            << "  model: " << student_config.nn_model
            << " width=" << student_config.nn_width
            << " depth=" << student_config.nn_depth << "\n"
            << "  optimizer: lr=" << student_config.learning_rate
            << " weight_decay=" << student_config.weight_decay
            << " batch_size=" << absl::GetFlag(FLAGS_batch_size)
            << " steps=" << absl::GetFlag(FLAGS_train_steps) << "\n"
            << "  device: " << absl::GetFlag(FLAGS_device) << "\n";

  if (absl::GetFlag(FLAGS_save_final_checkpoint)) {
    std::string initial_checkpoint = student.SaveCheckpoint(0);
    std::cout << "[saved initial] " << initial_checkpoint << "\n";
  }

  PrintModeMetrics("train", 0, EvaluateVPNet(&student, train.samples));
  double best_holdout_kl = std::numeric_limits<double>::infinity();
  if (!holdout.samples.empty()) {
    Metrics holdout_metrics = EvaluateVPNet(&student, holdout.samples);
    PrintModeMetrics("holdout", 0, holdout_metrics);
    best_holdout_kl = holdout_metrics.kl;
    if (absl::GetFlag(FLAGS_save_best_holdout_checkpoint)) {
      std::string best_checkpoint =
          student.SaveCheckpoint(kBestHoldoutCheckpointStep);
      std::cout << "[saved best] " << best_checkpoint
                << " source_step=0"
                << " holdout_kl=" << best_holdout_kl << "\n";
    }
  }

  std::mt19937 rng(absl::GetFlag(FLAGS_seed) + 900001);
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
      PrintModeMetrics("train", step, EvaluateVPNet(&student, train.samples));
      if (absl::GetFlag(FLAGS_save_checkpoint_every_report)) {
        std::string report_checkpoint = student.SaveCheckpoint(step);
        std::cout << "[saved report] " << report_checkpoint << "\n";
      }
      if (!holdout.samples.empty()) {
        Metrics holdout_metrics = EvaluateVPNet(&student, holdout.samples);
        PrintModeMetrics("holdout", step, holdout_metrics);
        if (absl::GetFlag(FLAGS_save_best_holdout_checkpoint) &&
            holdout_metrics.kl < best_holdout_kl) {
          best_holdout_kl = holdout_metrics.kl;
          std::string best_checkpoint =
              student.SaveCheckpoint(kBestHoldoutCheckpointStep);
          std::cout << "[saved best] " << best_checkpoint
                    << " source_step=" << step
                    << " holdout_kl=" << best_holdout_kl << "\n";
        }
      }
    }
  }

  if (absl::GetFlag(FLAGS_save_final_checkpoint)) {
    std::string final_checkpoint =
        student.SaveCheckpoint(VPNetModel::kMostRecentCheckpointStep);
    std::cout << "[saved final] " << final_checkpoint << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  static QuietTorchWarningHandler quiet_torch_warning_handler;
  if (absl::GetFlag(FLAGS_quiet_torch_warnings)) {
    c10::WarningUtils::set_warning_handler(&quiet_torch_warning_handler);
  }
  const std::string mode = absl::GetFlag(FLAGS_mode);
  if (mode == "generate") return RunGenerate();
  if (mode == "train") return RunTrain();
  if (mode == "inspect") return RunInspect();
  open_spiel::SpielFatalError(
      absl::StrCat("Unknown --mode=", mode, ". Expected generate, train, or inspect."));
}
