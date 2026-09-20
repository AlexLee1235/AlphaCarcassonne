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

#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"

#include <torch/torch.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <fstream>  // For ifstream/ofstream.
#include <map>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/algorithms/alpha_zero_torch/model.h"
#include "open_spiel/games/carcassonne/carcassonne.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {
namespace {

std::map<std::string, std::vector<int64_t>> TensorShapes(
    const torch::nn::Module& module) {
  std::map<std::string, std::vector<int64_t>> shapes;
  for (const auto& parameter : module.named_parameters()) {
    shapes[parameter.key()] = parameter.value().sizes().vec();
  }
  for (const auto& buffer : module.named_buffers()) {
    shapes[buffer.key()] = buffer.value().sizes().vec();
  }
  return shapes;
}

// torch::load resizes every tensor to whatever the file holds instead of
// rejecting a mismatch, so a checkpoint from another architecture (e.g. the
// old 1-filter flatten value head) would only fail later, with an opaque shape
// error inside forward(). Check the shapes right after loading instead.
void LoadModelChecked(Model& model, const std::string& file,
                      const torch::Device& device) {
  const std::map<std::string, std::vector<int64_t>> expected =
      TensorShapes(*model);
  torch::load(model, file, device);
  for (const auto& [name, shape] : TensorShapes(*model)) {
    auto it = expected.find(name);
    if (it == expected.end() || it->second != shape) {
      SpielFatalError(absl::StrCat(
          "Checkpoint ", file, " does not match the model architecture: ", name,
          " is [", absl::StrJoin(shape, ","), "] in the checkpoint but [",
          it == expected.end() ? std::string("missing")
                               : absl::StrJoin(it->second, ","),
          "] in the model. Checkpoints saved before the global-pooling value "
          "head cannot be loaded; train from scratch."));
    }
  }
}

}  // namespace

// Saves a struct that holds initialization data for the model to a file.
//
// The TensorFlow version creates a TensorFlow graph definition when
// CreateGraphDef is called. To avoid having to change this, allow calls to
// CreateGraphDef, however now it simply saves a struct to a file which can
// then be loaded and used to initialize a model.
bool SaveModelConfig(const std::string& path, const std::string& filename,
                     const ModelConfig& net_config) {
  std::ofstream file;
  file.open(absl::StrCat(path, "/", filename));

  if (!file) {
    return false;
  } else {
    file << net_config;
  }
  file.close();

  return true;
}

// Loads a struct that holds initialization data for the model from a file.
//
// The TensorFlow version creates a TensorFlow graph definition when
// CreateGraphDef is called. To avoid having to change this, allow calls to
// CreateGraphDef, however now it simply saves a struct to a file which can
// then be loaded and used to initialize a model.
ModelConfig LoadModelConfig(const std::string& path,
                            const std::string& filename) {
  std::ifstream file;
  file.open(absl::StrCat(path, "/", filename));
  ModelConfig net_config;

  file >> net_config;
  file.close();

  return net_config;
}

// Modifies a given device string to one that can be accepted by the
// Torch library.
//
// The Torch library accepts 'cpu', 'cpu:0', 'cuda:0', 'cuda:1',
// 'cuda:2', 'cuda:3'..., but complains when there's a slash in front
// of the device name.
//
// Currently, this function only disregards a slash if it exists at the
// beginning of the device string, more functionality can be added if
// needed.
std::string TorchDeviceName(const std::string& device) {
  if (device[0] == '/') {
    return device.substr(1);
  }
  return device;
}

int LastPlacedObservationPlane(const Game& game) {
  if (game.GetType().short_name == "carcassonne") {
    return carcassonne::kLastPlacedPlane;
  }
  return -1;
}

int GlobalObservationFeatures(const Game& game) {
  if (game.GetType().short_name == "carcassonne") {
    static_assert(carcassonne::kGlobalFeaturePlane ==
                  carcassonne::kObservationPlanes - 1);
    return carcassonne::kGlobalFeatures;
  }
  return 0;
}

bool CreateGraphDef(const Game& game, double learning_rate, double weight_decay,
                    const std::string& path, const std::string& filename,
                    std::string nn_model, int nn_width, int nn_depth,
                    bool verbose) {
  ModelConfig net_config = {
      /*observation_tensor_shape=*/game.ObservationTensorShape(),
      /*number_of_actions=*/game.NumDistinctActions(),
      /*nn_depth=*/nn_depth,
      /*nn_width=*/nn_width,
      /*learning_rate=*/learning_rate,
      /*weight_decay=*/weight_decay,
      /*nn_model=*/nn_model,
      /*last_placed_plane=*/LastPlacedObservationPlane(game),
      /*global_features=*/GlobalObservationFeatures(game)};

  return SaveModelConfig(path, filename, net_config);
}

VPNetModel::VPNetModel(const Game& game, const std::string& path,
                       const std::string& file_name, const std::string& device)
    : device_(device),
      path_(path),
      flat_input_size_(game.ObservationTensorSize()),
      num_actions_(game.NumDistinctActions()),
      model_config_(LoadModelConfig(path, file_name)),
      model_(model_config_, TorchDeviceName(device)),
      model_optimizer_(
          model_->parameters(),
          torch::optim::AdamOptions(  // NOLINT(misc-include-cleaner)
              model_config_.learning_rate)),
      torch_device_(TorchDeviceName(device)) {
  // Some assumptions that we can remove eventually. The value net returns
  // a single value in terms of player 0 and the game is assumed to be zero-sum,
  // so player 1 can just be -value.
  SPIEL_CHECK_EQ(game.NumPlayers(), 2);
  SPIEL_CHECK_EQ(game.GetType().utility, GameType::Utility::kZeroSum);

  // Put this model on the specified device. Inference is the common case, so
  // the model stays in eval mode; Learn switches to train mode and back.
  model_->to(torch_device_);
  model_->eval();
}

std::string VPNetModel::SaveCheckpoint(int step) {
  std::string full_path = absl::StrCat(path_, "/checkpoint-", step);

  torch::save(model_, absl::StrCat(full_path, ".pt"));
  torch::save(model_optimizer_, absl::StrCat(full_path, "-optimizer.pt"));

  return full_path;
}

void VPNetModel::LoadCheckpoint(int step) {
  // Load checkpoint from the path given at its initialization.
  LoadCheckpoint(absl::StrCat(path_, "/checkpoint-", step));
}

void VPNetModel::LoadCheckpoint(const std::string& path) {
  LoadModelChecked(model_, absl::StrCat(path, ".pt"), torch_device_);
  torch::load(model_optimizer_, absl::StrCat(path, "-optimizer.pt"),
              torch_device_);
  model_->eval();
}

void VPNetModel::LoadCheckpointWeightsOnly(int step) {
  LoadCheckpointWeightsOnly(absl::StrCat(path_, "/checkpoint-", step));
}

void VPNetModel::LoadCheckpointWeightsOnly(const std::string& path) {
  LoadModelChecked(model_, absl::StrCat(path, ".pt"), torch_device_);
  model_->eval();
}

VPNetModel::InferenceStaging::InferenceStaging(int max_batch_size,
                                               int flat_input_size,
                                               int num_actions, bool pinned)
    : max_batch_size_(max_batch_size),
      flat_input_size_(flat_input_size),
      num_actions_(num_actions),
      observations_(torch::empty(
          {max_batch_size, flat_input_size},
          torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(pinned))),
      legal_mask_(torch::empty(
          {max_batch_size, num_actions},
          torch::TensorOptions().dtype(torch::kBool).pinned_memory(pinned))) {}

void VPNetModel::InferenceStaging::Pack(
    const std::vector<InferenceInputs>& inputs) {
  batch_size_ = inputs.size();
  SPIEL_CHECK_GT(batch_size_, 0);
  SPIEL_CHECK_LE(batch_size_, max_batch_size_);

  // Write into the tensors through raw pointers: assigning to a torch::Tensor
  // element by element is far slower than this.
  float* observations = observations_.data_ptr<float>();
  bool* legal_mask = legal_mask_.data_ptr<bool>();
  std::fill(legal_mask, legal_mask + batch_size_ * num_actions_, false);
  for (int batch = 0; batch < batch_size_; ++batch) {
    SPIEL_CHECK_EQ(inputs[batch].observations.size(), flat_input_size_);
    std::copy(inputs[batch].observations.begin(),
              inputs[batch].observations.end(),
              observations + batch * flat_input_size_);
    for (Action action : inputs[batch].legal_actions) {
      legal_mask[batch * num_actions_ + action] = true;
    }
  }
}

std::vector<VPNetModel::InferenceOutputs> VPNetModel::InferenceStaging::Unpack(
    const std::vector<InferenceInputs>& inputs) const {
  SPIEL_CHECK_EQ(static_cast<int>(inputs.size()), batch_size_);
  // Accessors rather than .item<>(): the outputs are already on the host, and
  // .item<>() on a device tensor would synchronize on every element.
  auto value = value_.accessor<float, 2>();
  auto policy = policy_.accessor<float, 2>();

  std::vector<InferenceOutputs> outputs;
  outputs.reserve(batch_size_);
  for (int batch = 0; batch < batch_size_; ++batch) {
    ActionsAndProbs state_policy;
    state_policy.reserve(inputs[batch].legal_actions.size());
    for (Action action : inputs[batch].legal_actions) {
      state_policy.push_back({action, policy[batch][action]});
    }
    outputs.push_back(
        {static_cast<double>(value[batch][0]), std::move(state_policy)});
  }
  return outputs;
}

// One model runs one batch at a time: DeviceManager::DeviceLoan holds that
// model's own mutex while the caller has it. Two models are independent even
// on the same device, so several of them can share a GPU and run in parallel.
void VPNetModel::RunInference(InferenceStaging* staging) {
  const int batch_size = staging->batch_size_;
  SPIEL_CHECK_GT(batch_size, 0);
  SPIEL_CHECK_LE(batch_size, staging->max_batch_size_);

  // NoGradGuard prevents LibTorch from building the autograd computation
  // graph, saving memory and compute since we never backprop through
  // inference.
  torch::NoGradGuard no_grad;

  // The host buffers are pinned, so these copies are asynchronous; the copy
  // back below waits for them, which is also what makes it safe to pack the
  // next batch into the same buffers afterwards.
  torch::Tensor observations =
      staging->observations_.narrow(0, 0, batch_size)
          .to(torch_device_, /*non_blocking=*/true);
  torch::Tensor legal_mask =
      staging->legal_mask_.narrow(0, 0, batch_size)
          .to(torch_device_, /*non_blocking=*/true);

  std::vector<torch::Tensor> torch_outputs = model_(observations, legal_mask);

  // Move the outputs back in one transfer each.
  staging->value_ = torch_outputs[0].to(torch::kCPU).contiguous();
  staging->policy_ = torch_outputs[1].to(torch::kCPU).contiguous();
}

std::vector<VPNetModel::InferenceOutputs> VPNetModel::Inference(
    const std::vector<InferenceInputs>& inputs) {
  // For callers without their own staging. A caller running batch after batch
  // should keep an InferenceStaging instead, and pack and unpack outside the
  // model's lock.
  InferenceStaging staging(inputs.size(), flat_input_size_, num_actions_,
                           /*pinned=*/false);
  staging.Pack(inputs);
  RunInference(&staging);
  return staging.Unpack(inputs);
}

VPNetModel::LossInfo VPNetModel::Learn(const std::vector<TrainInputs>& inputs) {
  return Learn(inputs, /*policy_loss_weight=*/1.0, /*value_loss_weight=*/1.0,
               /*l2_loss_weight=*/1.0);
}

VPNetModel::LossInfo VPNetModel::Learn(const std::vector<TrainInputs>& inputs,
                                       double policy_loss_weight,
                                       double value_loss_weight,
                                       double l2_loss_weight) {
  int training_batch_size = inputs.size();

  std::vector<float> raw_train_inputs(training_batch_size * flat_input_size_);
  std::vector<uint8_t> raw_legal_mask(training_batch_size * num_actions_, 0);
  std::vector<float> raw_policy_targets(training_batch_size * num_actions_, 0);
  std::vector<float> raw_value_targets(training_batch_size);

  for (int batch = 0; batch < training_batch_size; ++batch) {
    std::copy(inputs[batch].observations.begin(),
              inputs[batch].observations.end(),
              raw_train_inputs.begin() + (batch * flat_input_size_));
    for (Action action : inputs[batch].legal_actions) {
      raw_legal_mask[num_actions_ * batch + action] = 1;
    }
    for (const auto &[action, probability] : inputs[batch].policy) {
      raw_policy_targets[num_actions_ * batch + action] = probability;
    }
    raw_value_targets[batch] = inputs[batch].value;
  }

  // Torch tensors by default use a dense, row-aligned memory layout.
  //   - Their default data type is a 32-bit float
  //   - Use the byte data type for boolean
  // Clone first to take ownership from the raw blob, then move to device.
  torch::Tensor torch_train_inputs =
      torch::from_blob(raw_train_inputs.data(),
                       {training_batch_size, flat_input_size_})
          .clone()
          .to(torch_device_);
  torch::Tensor torch_train_legal_mask =
      torch::from_blob(raw_legal_mask.data(),
                       {training_batch_size, num_actions_},
                       torch::TensorOptions().dtype(torch::kBool))
          .clone()
          .to(torch_device_);
  torch::Tensor torch_policy_targets =
      torch::from_blob(raw_policy_targets.data(),
                       {training_batch_size, num_actions_})
          .clone()
          .to(torch_device_);
  torch::Tensor torch_value_targets =
      torch::from_blob(raw_value_targets.data(), {training_batch_size, 1})
          .clone()
          .to(torch_device_);

  // Run a training step and get the losses.
  model_->train();
  model_->zero_grad();

  std::vector<torch::Tensor> torch_outputs =
      model_->losses(torch_train_inputs, torch_train_legal_mask,
                     torch_policy_targets, torch_value_targets);

  torch::Tensor total_loss = policy_loss_weight * torch_outputs[0] +
                             value_loss_weight * torch_outputs[1] +
                             l2_loss_weight * torch_outputs[2];

  total_loss.backward();

  model_optimizer_.step();

  model_->eval();

  return LossInfo(torch_outputs[0].item<float>(),
                  torch_outputs[1].item<float>(),
                  torch_outputs[2].item<float>(),
                  torch_outputs[3].item<float>(),
                  torch_outputs[4].item<float>(),
                  torch_outputs[5].item<float>());
}

}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel
