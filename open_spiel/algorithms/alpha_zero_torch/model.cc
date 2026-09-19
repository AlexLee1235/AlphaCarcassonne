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

#include "open_spiel/algorithms/alpha_zero_torch/model.h"

#include <torch/torch.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {

std::istream& operator>>(std::istream& stream, ModelConfig& config) {
  int channels;
  int height;
  int width;

  stream >> channels >> height >> width >> config.number_of_actions >>
      config.nn_depth >> config.nn_width >> config.learning_rate >>
      config.weight_decay >> config.nn_model;

  // Written by newer versions only; a file without it predates the conv policy
  // head and describes a game with no last-placed plane.
  if (!(stream >> config.last_placed_plane)) {
    config.last_placed_plane = -1;
    stream.clear();
  }
  // Likewise for files that predate the global feature vector.
  if (!(stream >> config.global_features)) {
    config.global_features = 0;
    stream.clear();
  }

  config.observation_tensor_shape = {channels, height, width};

  return stream;
}

std::ostream& operator<<(std::ostream& stream, const ModelConfig& config) {
  int shape_dim = config.observation_tensor_shape.size();
  int height = shape_dim > 1 ? config.observation_tensor_shape[1] : 1;
  int width = shape_dim > 2 ? config.observation_tensor_shape[2] : 1;

  stream << config.observation_tensor_shape[0] << " " << height << " " << width
         << " " << config.number_of_actions << " " << config.nn_depth << " "
         << config.nn_width << " " << config.learning_rate << " "
         << config.weight_decay << " " << config.nn_model << " "
         << config.last_placed_plane << " " << config.global_features;
  return stream;
}

ResInputBlockImpl::ResInputBlockImpl(const ResInputBlockConfig& config)
    : conv_(torch::nn::Conv2dOptions(
                /*input_channels=*/config.input_channels,
                /*output_channels=*/config.filters,
                /*kernel_size=*/config.kernel_size)
                .stride(1)
                .padding(config.padding)
                .dilation(1)
                .groups(1)
                .bias(true)
                .padding_mode(torch::kZeros)),
      batch_norm_(torch::nn::BatchNorm2dOptions(
                      /*num_features=*/config.filters)
                      .eps(0.001)      // Make it the same as TF.
                      .momentum(0.01)  // Torch momentum = 1 - TF momentum.
                      .affine(true)
                      .track_running_stats(true)) {
  channels_ = config.input_channels;
  height_ = config.input_height;
  width_ = config.input_width;

  register_module("input_conv", conv_);
  register_module("input_batch_norm", batch_norm_);
}

torch::Tensor ResInputBlockImpl::forward(torch::Tensor x,
                                         torch::Tensor global_bias) {
  torch::Tensor output = conv_(x.reshape({-1, channels_, height_, width_}));
  if (global_bias.defined()) {
    output = output + global_bias.unsqueeze(/*dim=*/2).unsqueeze(/*dim=*/3);
  }
  output = torch::relu(batch_norm_(output));

  return output;
}

ResTorsoBlockImpl::ResTorsoBlockImpl(const ResTorsoBlockConfig& config,
                                     int layer)
    : conv1_(torch::nn::Conv2dOptions(
                 /*input_channels=*/config.input_channels,
                 /*output_channels=*/config.filters,
                 /*kernel_size=*/config.kernel_size)
                 .stride(1)
                 .padding(config.padding)
                 .dilation(1)
                 .groups(1)
                 .bias(true)
                 .padding_mode(torch::kZeros)),
      conv2_(torch::nn::Conv2dOptions(
                 /*input_channels=*/config.filters,
                 /*output_channels=*/config.filters,
                 /*kernel_size=*/config.kernel_size)
                 .stride(1)
                 .padding(config.padding)
                 .dilation(1)
                 .groups(1)
                 .bias(true)
                 .padding_mode(torch::kZeros)),
      batch_norm1_(torch::nn::BatchNorm2dOptions(
                       /*num_features=*/config.filters)
                       .eps(0.001)      // Make it the same as TF.
                       .momentum(0.01)  // Torch momentum = 1 - TF momentum.
                       .affine(true)
                       .track_running_stats(true)),
      batch_norm2_(torch::nn::BatchNorm2dOptions(
                       /*num_features=*/config.filters)
                       .eps(0.001)      // Make it the same as TF.
                       .momentum(0.01)  // Torch momentum = 1 - TF momentum.
                       .affine(true)
                       .track_running_stats(true)) {
  register_module("res_" + std::to_string(layer) + "_conv_1", conv1_);
  register_module("res_" + std::to_string(layer) + "_conv_2", conv2_);
  register_module("res_" + std::to_string(layer) + "_batch_norm_1",
                  batch_norm1_);
  register_module("res_" + std::to_string(layer) + "_batch_norm_2",
                  batch_norm2_);
}

torch::Tensor ResTorsoBlockImpl::forward(torch::Tensor x,
                                         torch::Tensor global_bias) {
  torch::Tensor residual = x;

  torch::Tensor output = conv1_(x);
  if (global_bias.defined()) {
    output = output + global_bias.unsqueeze(/*dim=*/2).unsqueeze(/*dim=*/3);
  }
  output = torch::relu(batch_norm1_(output));
  output = batch_norm2_(conv2_(output));
  output += residual;
  output = torch::relu(output);

  return output;
}

ResOutputBlockImpl::ResOutputBlockImpl(const ResOutputBlockConfig& config)
    : value_conv_(torch::nn::Conv2dOptions(
                      /*input_channels=*/config.input_channels,
                      /*output_channels=*/config.value_filters,
                      /*kernel_size=*/config.kernel_size)
                      .stride(1)
                      .padding(config.padding)
                      .dilation(1)
                      .groups(1)
                      .bias(true)
                      .padding_mode(torch::kZeros)),
      value_batch_norm_(
          torch::nn::BatchNorm2dOptions(
              /*num_features=*/config.value_filters)
              .eps(0.001)      // Make it the same as TF.
              .momentum(0.01)  // Torch momentum = 1 - TF momentum.
              .affine(true)
              .track_running_stats(true)),
      value_linear1_(torch::nn::LinearOptions(
                         /*in_features=*/config.value_linear_in_features,
                         /*out_features=*/config.value_linear_out_features)
                         .bias(true)),
      value_linear2_(torch::nn::LinearOptions(
                         /*in_features=*/config.value_linear_out_features,
                         /*out_features=*/1)
                         .bias(true)),
      policy_conv_(torch::nn::Conv2dOptions(
                       /*input_channels=*/config.input_channels,
                       /*output_channels=*/config.policy_filters,
                       /*kernel_size=*/config.kernel_size)
                       .stride(1)
                       .padding(config.padding)
                       .dilation(1)
                       .groups(1)
                       .bias(true)
                       .padding_mode(torch::kZeros)),
      policy_batch_norm_(
          torch::nn::BatchNorm2dOptions(
              /*num_features=*/config.policy_filters)
              .eps(0.001)      // Make it the same as TF.
              .momentum(0.01)  // Torch momentum = 1 - TF momentum.
              .affine(true)
              .track_running_stats(true)),
      policy_observation_size_(config.policy_observation_size),
      policy_conv_planes_(config.policy_conv_planes),
      policy_extra_actions_(config.policy_extra_actions),
      global_features_(config.global_features) {
  register_module("value_conv", value_conv_);
  register_module("value_batch_norm", value_batch_norm_);
  register_module("value_linear_1", value_linear1_);
  register_module("value_linear_2", value_linear2_);
  register_module("policy_conv", policy_conv_);
  register_module("policy_batch_norm", policy_batch_norm_);

  if (policy_conv_planes_ > 0) {
    auto one_by_one = [](int in_channels, int out_channels) {
      return torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels,
                                                        out_channels,
                                                        /*kernel_size=*/1)
                                   .stride(1)
                                   .padding(0)
                                   .dilation(1)
                                   .groups(1)
                                   .bias(true)
                                   .padding_mode(torch::kZeros));
    };
    // Whether a move is good depends on board-wide counts (tiles left, meeples
    // left, the score gap) that the trunk cannot carry to every cell, so a
    // pooled branch adds them back as a per-channel bias.
    policy_gpool_conv_ =
        one_by_one(config.input_channels, config.policy_filters);
    policy_gpool_linear_ = torch::nn::Linear(
        torch::nn::LinearOptions(2 * config.policy_filters + global_features_,
                                 config.policy_filters)
            .bias(true));
    register_module("policy_gpool_conv", policy_gpool_conv_);
    register_module("policy_gpool_linear", policy_gpool_linear_);
    // One 1x1 filter per per-cell action, shared by every cell.
    policy_placement_conv_ =
        one_by_one(config.policy_filters, policy_conv_planes_);
    register_module("policy_placement_conv", policy_placement_conv_);
    if (policy_extra_actions_ > 0) {
      // The remaining actions belong to the cell just played, so they are read
      // from that cell instead of from the whole board.
      policy_cell_conv_ =
          one_by_one(config.policy_filters, policy_extra_actions_);
      register_module("policy_cell_conv", policy_cell_conv_);
    }
  } else {
    policy_linear_ = torch::nn::Linear(
        torch::nn::LinearOptions(
            /*in_features=*/config.policy_linear_in_features,
            /*out_features=*/config.policy_linear_out_features)
            .bias(true));
    register_module("policy_linear", policy_linear_);
  }
}

std::vector<torch::Tensor> ResOutputBlockImpl::forward(
    torch::Tensor x, torch::Tensor mask, torch::Tensor last_placed_plane,
    torch::Tensor global_features) {
  if (global_features_ > 0 && !global_features.defined()) {
    throw std::runtime_error("This output block needs the global features.");
  }
  // [B, value_filters, H, W] -> [B, value_filters, H*W]
  torch::Tensor value_output =
      torch::relu(value_batch_norm_(value_conv_(x))).flatten(2);
  // Global mean ++ max pooling -> [B, 2 * value_filters]. The mean is a
  // translation-invariant board sum; the max keeps "is there one big feature".
  value_output =
      torch::cat({value_output.mean(/*dim=*/2), value_output.amax(/*dim=*/2)},
                 /*dim=*/1);
  if (global_features_ > 0) {
    value_output = torch::cat({value_output, global_features}, /*dim=*/1);
  }
  value_output = torch::relu(value_linear1_(value_output));
  value_output = torch::tanh(value_linear2_(value_output));

  torch::Tensor policy_hidden = policy_conv_(x);
  torch::Tensor policy_logits;
  if (policy_conv_planes_ > 0) {
    torch::Tensor pooled = policy_gpool_conv_(x).flatten(2);
    pooled = torch::cat({pooled.mean(/*dim=*/2), pooled.amax(/*dim=*/2)},
                        /*dim=*/1);
    if (global_features_ > 0) {
      pooled = torch::cat({pooled, global_features}, /*dim=*/1);
    }
    // [B, policy_filters] -> the same bias on every cell.
    policy_hidden = policy_hidden + policy_gpool_linear_(pooled)
                                        .unsqueeze(/*dim=*/2)
                                        .unsqueeze(/*dim=*/3);
    policy_hidden = torch::relu(policy_batch_norm_(policy_hidden));
    // [B, planes, H, W] -> [B, H * W * planes]. The game indexes an action as
    // (cell * planes + plane), so the planes have to be the fastest axis.
    policy_logits = policy_placement_conv_(policy_hidden)
                        .permute({0, 2, 3, 1})
                        .contiguous()
                        .view({policy_hidden.size(0), -1});
    if (policy_extra_actions_ > 0) {
      if (!last_placed_plane.defined()) {
        throw std::runtime_error(
            "The conv policy head needs the last-placed plane.");
      }
      // The plane is a one-hot of the cell just played, so multiplying by it
      // and summing reads that cell's logits without an index lookup.
      torch::Tensor cell_logits =
          (policy_cell_conv_(policy_hidden) * last_placed_plane)
              .sum(/*dim=*/{2, 3});
      policy_logits = torch::cat({policy_logits, cell_logits}, /*dim=*/1);
    }
  } else {
    policy_hidden = torch::relu(policy_batch_norm_(policy_hidden));
    policy_logits =
        policy_linear_(policy_hidden.view({-1, policy_observation_size_}));
  }
  policy_logits = torch::where(mask, policy_logits,
                               -(1 << 16) * torch::ones_like(policy_logits));

  return {value_output, policy_logits};
}

MLPBlockImpl::MLPBlockImpl(const int in_features, const int out_features)
    : linear_(torch::nn::LinearOptions(
                         /*in_features=*/in_features,
                         /*out_features=*/out_features)
                         .bias(true)) {
  register_module("linear", linear_);
}

torch::Tensor MLPBlockImpl::forward(torch::Tensor x) {
  return torch::relu(linear_(x));
}

MLPOutputBlockImpl::MLPOutputBlockImpl(const int nn_width,
                                       const int policy_linear_out_features)
    : value_linear1_(torch::nn::LinearOptions(
                         /*in_features=*/nn_width,
                         /*out_features=*/nn_width)
                         .bias(true)),
      value_linear2_(torch::nn::LinearOptions(
                         /*in_features=*/nn_width,
                         /*out_features=*/1)
                         .bias(true)),
      policy_linear1_(torch::nn::LinearOptions(
                          /*input_channels=*/nn_width,
                          /*output_channels=*/nn_width)
                          .bias(true)),
      policy_linear2_(torch::nn::LinearOptions(
                          /*in_features=*/nn_width,
                          /*out_features=*/policy_linear_out_features)
                          .bias(true)) {
  register_module("value_linear_1", value_linear1_);
  register_module("value_linear_2", value_linear2_);
  register_module("policy_linear_1", policy_linear1_);
  register_module("policy_linear_2", policy_linear2_);
}

std::vector<torch::Tensor> MLPOutputBlockImpl::forward(torch::Tensor x,
                                                       torch::Tensor mask) {
  torch::Tensor value_output = torch::relu(value_linear1_(x));
  value_output = torch::tanh(value_linear2_(value_output));

  torch::Tensor policy_logits = torch::relu(policy_linear1_(x));
  policy_logits = policy_linear2_(policy_logits);
  policy_logits = torch::where(mask, policy_logits,
                               -(1 << 16) * torch::ones_like(policy_logits));

  return {value_output, policy_logits};
}

ModelImpl::ModelImpl(const ModelConfig& config, const std::string& device)
    : device_(device),
      num_torso_blocks_(config.nn_depth),
      weight_decay_(config.weight_decay) {
  // Save config.nn_model to class
  nn_model_ = config.nn_model;
  last_placed_plane_ = config.last_placed_plane;
  global_features_ = config.nn_model == "resnet" ? config.global_features : 0;

  int input_size = 1;
  for (const auto& num : config.observation_tensor_shape) {
    if (num > 0) {
      input_size *= num;
    }
  }
  // Decide if resnet or MLP
  if (config.nn_model == "resnet") {
    int obs_dims = config.observation_tensor_shape.size();
    int channels = config.observation_tensor_shape[0];
    int height = obs_dims > 1 ? config.observation_tensor_shape[1] : 1;
    int width = obs_dims > 2 ? config.observation_tensor_shape[2] : 1;
    input_channels_ = channels;
    input_height_ = height;
    input_width_ = width;
    // With a global feature vector the last plane holds it; only the planes
    // before it are a picture of the board.
    const int board_channels = global_features_ > 0 ? channels - 1 : channels;
    if (global_features_ > 0 &&
        (channels < 2 || global_features_ > height * width ||
         last_placed_plane_ >= board_channels)) {
      throw std::runtime_error(
          "The global features do not fit the observation's last plane.");
    }

    ResInputBlockConfig input_config = {/*input_channels=*/board_channels,
                                        /*input_height=*/height,
                                        /*input_width=*/width,
                                        /*filters=*/config.nn_width,
                                        /*kernel_size=*/3,
                                        /*padding=*/1};

    ResTorsoBlockConfig residual_config = {/*input_channels=*/config.nn_width,
                                           /*filters=*/config.nn_width,
                                           /*kernel_size=*/3,
                                           /*padding=*/1};

    // The value head pools over the board, so its size does not depend on
    // height/width. With nn_width=32 it has 18,017 parameters, none of them
    // position-dependent (the old 1-filter flatten head had 7,200 of 7,300).
    constexpr int kValueFilters = 32;
    constexpr int kValueHidden = 256;

    // A conv policy head needs the actions to factor into a few choices per
    // board cell plus a handful that belong to no cell. Carcassonne's do: four
    // tile rotations per cell (900) and then six meeple moves. Games whose
    // action count is nothing like 4 * height * width (tic_tac_toe,
    // connect_four, othello) keep the dense head.
    constexpr int kPolicyRotations = 4;
    constexpr int kMaxExtraActions = 16;
    constexpr int kPolicyConvFilters = 32;
    const int placement_actions = kPolicyRotations * height * width;
    const int extra_actions = config.number_of_actions - placement_actions;
    // The per-cell actions alone are not enough: the actions tied to the cell
    // just played are read from the plane that marks it.
    const bool conv_policy =
        extra_actions >= 0 && extra_actions <= kMaxExtraActions &&
        (extra_actions == 0 || config.last_placed_plane >= 0);

    ResOutputBlockConfig output_config = {
        /*input_channels=*/config.nn_width,
        /*value_filters=*/kValueFilters,
        /*policy_filters=*/conv_policy ? kPolicyConvFilters : 2,
        /*kernel_size=*/1,
        /*padding=*/0,
        // mean ++ max ++ global features
        /*value_linear_in_features=*/2 * kValueFilters + global_features_,
        /*value_linear_out_features=*/kValueHidden,
        /*policy_linear_in_features=*/2 * width * height,
        /*policy_linear_out_features=*/config.number_of_actions,
        /*policy_observation_size=*/2 * width * height,
        /*policy_conv_planes=*/conv_policy ? kPolicyRotations : 0,
        /*policy_extra_actions=*/conv_policy ? extra_actions : 0,
        /*global_features=*/global_features_};

    layers_->push_back(ResInputBlock(input_config));
    for (int i = 0; i < num_torso_blocks_; i++) {
      layers_->push_back(ResTorsoBlock(residual_config, i));
    }
    layers_->push_back(ResOutputBlock(output_config));

    register_module("layers", layers_);

    if (global_features_ > 0) {
      auto bias = [&](const std::string& name) {
        return register_module(
            name, torch::nn::Linear(torch::nn::LinearOptions(
                                        global_features_, config.nn_width)
                                        .bias(true)));
      };
      // The same vector again every second block, so blocks deep in the trunk
      // do not depend on it having survived the ones before.
      constexpr int kGlobalBiasEvery = 2;
      global_input_bias_ = bias("global_bias_input");
      global_block_biases_.assign(num_torso_blocks_,
                                  torch::nn::Linear(nullptr));
      for (int i = kGlobalBiasEvery - 1; i < num_torso_blocks_;
           i += kGlobalBiasEvery) {
        global_block_biases_[i] = bias("global_bias_res_" + std::to_string(i));
      }
    }

  } else if (config.nn_model == "mlp") {
    layers_->push_back(MLPBlock(input_size, config.nn_width));
    for (int i = 0; i < num_torso_blocks_; i++) {
      layers_->push_back(MLPBlock(config.nn_width, config.nn_width));
    }
    layers_->push_back(
        MLPOutputBlock(config.nn_width, config.number_of_actions));

    register_module("layers", layers_);
  } else {
    throw std::runtime_error("Unknown nn_model: " + config.nn_model);
  }
}

std::vector<torch::Tensor> ModelImpl::forward(torch::Tensor x,
                                              torch::Tensor mask) {
  std::vector<torch::Tensor> output = this->forward_(x, mask);
  return {output[0], torch::softmax(output[1], 1)};
}

std::vector<torch::Tensor> ModelImpl::losses(torch::Tensor inputs,
                                             torch::Tensor masks,
                                             torch::Tensor policy_targets,
                                             torch::Tensor value_targets) {
  std::vector<torch::Tensor> output = this->forward_(inputs, masks);

  torch::Tensor value_predictions = output[0];
  torch::Tensor policy_predictions = output[1];
  torch::Tensor log_policy_predictions = torch::log_softmax(policy_predictions, 1);

  // Policy loss (cross-entropy).
  torch::Tensor policy_loss = torch::sum(
      -policy_targets * log_policy_predictions, -1);
  policy_loss = torch::mean(policy_loss);

  torch::Tensor target_entropy = torch::sum(
      -policy_targets * torch::log(torch::clamp(policy_targets, 1e-12, 1)), -1);
  target_entropy = torch::mean(target_entropy);

  torch::Tensor policy_probs = torch::softmax(policy_predictions, 1);
  torch::Tensor pred_entropy =
      torch::sum(-policy_probs * log_policy_predictions, -1);
  pred_entropy = torch::mean(pred_entropy);

  torch::Tensor policy_kl = policy_loss - target_entropy;

  // Value loss (mean-squared error).
  torch::nn::MSELoss mse_loss;
  torch::Tensor value_loss = mse_loss(value_predictions, value_targets);

  // L2 regularization loss (weights only).
  torch::Tensor l2_regularization_loss = torch::full(
      {1, 1}, 0, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  for (auto& named_parameter : this->named_parameters()) {
    // named_parameter is essentially a key-value pair:
    //   {key, value} == {std::string name, torch::Tensor parameter}
    std::string parameter_name = named_parameter.key();

    // Do not include bias' in the loss.
    if (absl::StrContains(parameter_name, "bias")) {
      continue;
    }

    // Copy TensorFlow's l2_loss function.
    // https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    l2_regularization_loss +=
        weight_decay_ * torch::sum(torch::square(named_parameter.value())) / 2;
  }

  return {policy_loss, value_loss, l2_regularization_loss, target_entropy,
          pred_entropy, policy_kl};
}

std::vector<torch::Tensor> ModelImpl::forward_(torch::Tensor x,
                                               torch::Tensor mask) {
  std::vector<torch::Tensor> output;
  if (this->nn_model_ == "resnet") {
    torch::Tensor board =
        x.view({-1, input_channels_, input_height_, input_width_});
    // [B, global_features], cut from the last plane, which is not convolved.
    torch::Tensor global;
    if (global_features_ > 0) {
      global = board.select(/*dim=*/1, input_channels_ - 1)
                   .flatten(/*start_dim=*/1)
                   .slice(/*dim=*/1, 0, global_features_);
      board = board.slice(/*dim=*/1, 0, input_channels_ - 1);
    }
    // The policy head reads the actions that belong to the cell just played
    // from this plane of the observation, so keep it before the trunk runs.
    torch::Tensor last_placed_plane;
    if (last_placed_plane_ >= 0) {
      last_placed_plane =
          board.slice(/*dim=*/1, last_placed_plane_, last_placed_plane_ + 1);
    }
    for (int i = 0; i < num_torso_blocks_ + 2; i++) {
      if (i == 0) {
        torch::Tensor bias;
        if (global.defined()) {
          bias = global_input_bias_(global);
        }
        x = layers_[i]->as<ResInputBlock>()->forward(board, bias);
      } else if (i >= num_torso_blocks_ + 1) {
        output = layers_[i]->as<ResOutputBlock>()->forward(
            x, mask, last_placed_plane, global);
      } else {
        torch::Tensor bias;
        if (global.defined() && !global_block_biases_[i - 1].is_empty()) {
          bias = global_block_biases_[i - 1](global);
        }
        x = layers_[i]->as<ResTorsoBlock>()->forward(x, bias);
      }
    }
  } else if (this->nn_model_ == "mlp") {
    for (int i = 0; i < num_torso_blocks_ + 1; i++) {
      x = layers_[i]->as<MLPBlock>()->forward(x);
    }
    output = layers_[num_torso_blocks_ + 1]->as<MLPOutputBlockImpl>()
        ->forward(x, mask);
  } else {
    throw std::runtime_error("Unknown nn_model: " + this->nn_model_);
  }
  return output;
}

}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel
