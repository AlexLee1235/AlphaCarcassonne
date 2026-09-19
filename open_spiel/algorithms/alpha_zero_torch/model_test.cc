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
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {
namespace {

void TestModelCreation() {
  std::cout << "\n~-~-~-~- TestModelCreation -~-~-~-~" << std::endl;

  // This fork only builds a few games (clobber is not one of them).
  std::shared_ptr<const Game> game = LoadGame("othello");

  ModelConfig net_config = {
      /*observation_tensor_shape=*/game->ObservationTensorShape(),
      /*number_of_actions=*/game->NumDistinctActions(),
      /*nn_depth=*/8,
      /*nn_width=*/128,
      /*learning_rate=*/0.001,
      /*weight_decay=*/0.001};
  Model net(net_config, "cpu:0");

  std::cout << "Good! The network looks like:\n" << net << std::endl;
}

void TestModelInference() {
  std::cout << "\n~-~-~-~- TestModelInference -~-~-~-~" << std::endl;

  // Othello's observation is 3 x 8 x 8, same as the upstream clobber(8x8).
  const int channels = 3;
  const int rows = 8;
  const int columns = 8;

  std::shared_ptr<const Game> game = LoadGame("othello");
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  ModelConfig net_config = {
      /*observation_tensor_shape=*/game->ObservationTensorShape(),
      /*number_of_actions=*/game->NumDistinctActions(),
      /*nn_depth=*/rows + 1,
      /*nn_width=*/128,
      /*learning_rate=*/0.001,
      /*weight_decay=*/0.001};
  Model net(net_config, "cpu:0");

  std::vector<float> observation_vector = state->ObservationTensor();
  torch::Tensor observation_tensor = torch::from_blob(
      observation_vector.data(), {1, channels * rows * columns});
  torch::Tensor mask = torch::full({1, game->NumDistinctActions()}, false,
                                   torch::TensorOptions().dtype(torch::kByte));

  for (Action action : state->LegalActions()) {
    mask[0][action] = true;
  }

  std::cout << "Input:\n"
            << observation_tensor.view({channels, rows, columns}) << std::endl;
  std::cout << "Mask:\n" << mask << std::endl;

  std::vector<torch::Tensor> output = net(observation_tensor, mask);

  std::cout << "Output:\n" << output << std::endl;

  // Check value and policy.
  SPIEL_CHECK_EQ((int)output.size(), 2);
  SPIEL_CHECK_EQ(output[0].numel(), 1);
  SPIEL_CHECK_EQ(output[1].numel(), game->NumDistinctActions());

  // Check mask's influence on the policy.
  for (int i = 0; i < game->NumDistinctActions(); i++) {
    if (mask[0][i].item<bool>()) {
      SPIEL_CHECK_GT(output[1][0][i].item<float>(), 0.0);
    } else {
      SPIEL_CHECK_EQ(output[1][0][i].item<float>(), 0.0);
    }
  }

  std::cout << "Value:\n" << output[0] << std::endl;
  std::cout << "Policy:\n" << output[1] << std::endl;
}

// The value head pools over the board instead of flattening it, so it has no
// position-dependent parameters: its parameters must not depend on the board
// size, and its value must not change when the board cells are permuted.
void TestValueHeadGlobalPooling() {
  std::cout << "\n~-~-~-~- TestValueHeadGlobalPooling -~-~-~-~" << std::endl;

  const int nn_width = 32;
  std::vector<std::map<std::string, std::vector<int64_t>>> head_shapes;
  // Boards of different sizes: 3x3 and 6x7.
  for (const char* game_name : {"tic_tac_toe", "connect_four"}) {
    std::shared_ptr<const Game> game = LoadGame(game_name);
    ModelConfig net_config = {
        /*observation_tensor_shape=*/game->ObservationTensorShape(),
        /*number_of_actions=*/game->NumDistinctActions(),
        /*nn_depth=*/2,
        /*nn_width=*/nn_width,
        /*learning_rate=*/0.001,
        /*weight_decay=*/0.001};
    Model net(net_config, "cpu:0");

    std::map<std::string, std::vector<int64_t>> shapes;
    int64_t value_parameters = 0;
    for (const auto& parameter : net->named_parameters()) {
      if (absl::StrContains(parameter.key(), ".value_")) {
        shapes[parameter.key()] = parameter.value().sizes().vec();
        value_parameters += parameter.value().numel();
      }
    }
    // conv 32*32+32, BN 2*32, FC 64*256+256, FC 256+1.
    SPIEL_CHECK_EQ(value_parameters, 18017);
    head_shapes.push_back(shapes);
  }
  SPIEL_CHECK_TRUE(head_shapes[0] == head_shapes[1]);

  const int batch = 4;
  const int height = 6;
  const int width = 7;
  const int num_actions = 10;
  ResOutputBlockConfig config = {
      /*input_channels=*/nn_width,
      /*value_filters=*/32,
      /*policy_filters=*/2,
      /*kernel_size=*/1,
      /*padding=*/0,
      /*value_linear_in_features=*/2 * 32,
      /*value_linear_out_features=*/256,
      /*policy_linear_in_features=*/2 * height * width,
      /*policy_linear_out_features=*/num_actions,
      /*policy_observation_size=*/2 * height * width,
      /*policy_conv_planes=*/0,
      /*policy_extra_actions=*/0,
      /*global_features=*/0};
  ResOutputBlock head(config);
  head->eval();
  torch::NoGradGuard no_grad;

  torch::manual_seed(0);
  torch::Tensor x = torch::randn({batch, nn_width, height, width});
  torch::Tensor permuted = x.flatten(2)
                               .index_select(2, torch::randperm(height * width))
                               .view({batch, nn_width, height, width});
  torch::Tensor mask = torch::ones(
      {batch, num_actions}, torch::TensorOptions().dtype(torch::kBool));

  torch::Tensor value = head->forward(x, mask)[0];
  torch::Tensor permuted_value = head->forward(permuted, mask)[0];
  std::cout << "Value:\n" << value << "\nPermuted value:\n" << permuted_value
            << std::endl;
  SPIEL_CHECK_TRUE(value.sizes().vec() == std::vector<int64_t>({batch, 1}));
  SPIEL_CHECK_TRUE(torch::allclose(value, permuted_value, /*rtol=*/1e-5,
                                   /*atol=*/1e-6));
  // Not vacuous: different boards still give different values.
  SPIEL_CHECK_GT((value.max() - value.min()).item<float>(), 0.0);
}

// The conv policy head scores each board cell with shared 1x1 filters, so a
// logit has to land on the action the game means: the game reads action
// (cell * planes + plane), which only holds if the planes are the fastest axis
// when the conv output is flattened. The meeple actions are read from the cell
// the last-placed plane marks, and a pooled branch biases every cell with
// board-wide information.
void TestConvPolicyHead() {
  std::cout << "\n~-~-~-~- TestConvPolicyHead -~-~-~-~" << std::endl;

  const int batch = 2;
  const int nn_width = 32;
  const int height = 6;
  const int width = 7;
  const int planes = 4;
  const int cell_actions = 6;
  const int placement_actions = planes * height * width;
  const int num_actions = placement_actions + cell_actions;

  ResOutputBlockConfig config = {
      /*input_channels=*/nn_width,
      /*value_filters=*/32,
      /*policy_filters=*/32,
      /*kernel_size=*/1,
      /*padding=*/0,
      /*value_linear_in_features=*/2 * 32,
      /*value_linear_out_features=*/256,
      /*policy_linear_in_features=*/2 * height * width,
      /*policy_linear_out_features=*/num_actions,
      /*policy_observation_size=*/2 * height * width,
      /*policy_conv_planes=*/planes,
      /*policy_extra_actions=*/cell_actions,
      /*global_features=*/0};
  ResOutputBlock head(config);
  head->eval();
  torch::NoGradGuard no_grad;

  torch::manual_seed(0);
  torch::Tensor x = torch::randn({batch, nn_width, height, width});
  torch::Tensor mask = torch::ones(
      {batch, num_actions}, torch::TensorOptions().dtype(torch::kBool));
  // The cell the game just played on, as a one-hot plane.
  const int played_y = 2;
  const int played_x = 5;
  torch::Tensor last_placed = torch::zeros({batch, 1, height, width});
  last_placed.index_put_({torch::indexing::Slice(), 0, played_y, played_x}, 1.0);

  torch::Tensor logits = head->forward(x, mask, last_placed)[1];
  SPIEL_CHECK_TRUE(logits.sizes().vec() ==
                   std::vector<int64_t>({batch, num_actions}));

  // The pooled branch biases every cell from the whole board, so a cell's
  // logits move when any cell changes. That is the point of it, but it hides
  // which logits belong to which cell, so check it separately: with the pooled
  // branch silenced every layer left is 1x1.
  torch::Tensor far = x.clone();
  far.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 0, 0},
                 x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                          0, 0}) + 1.0);
  const float pooled_effect =
      (head->forward(far, mask, last_placed)[1] - logits)
          .abs()
          .index({torch::indexing::Slice(), planes * (3 * width + 3)})
          .max()
          .item<float>();
  SPIEL_CHECK_GT(pooled_effect, 1e-5);

  for (auto& parameter : head->named_parameters()) {
    if (absl::StrContains(parameter.key(), "policy_gpool_linear")) {
      parameter.value().zero_();
    }
  }
  logits = head->forward(x, mask, last_placed)[1];

  // Changing one cell now moves exactly that cell's per-cell logits, and the
  // meeple logits only when it is the cell the last-placed plane marks.
  for (const std::pair<int, int>& cell : {std::make_pair(3, 4),
                                          std::make_pair(played_y, played_x)}) {
    torch::Tensor changed = x.clone();
    changed.index_put_({torch::indexing::Slice(), torch::indexing::Slice(),
                        cell.first, cell.second},
                       x.index({torch::indexing::Slice(),
                                torch::indexing::Slice(), cell.first,
                                cell.second}) + 1.0);
    torch::Tensor moved =
        (head->forward(changed, mask, last_placed)[1] - logits).abs().amax(0);
    const int index = cell.first * width + cell.second;
    for (int action = 0; action < placement_actions; ++action) {
      const float delta = moved[action].item<float>();
      if (action / planes == index) {
        SPIEL_CHECK_GT(delta, 1e-5);
      } else {
        SPIEL_CHECK_LT(delta, 1e-5);
      }
    }
    const float meeple_moved =
        moved.slice(0, placement_actions, num_actions).max().item<float>();
    if (index == played_y * width + played_x) {
      SPIEL_CHECK_GT(meeple_moved, 1e-5);
    } else {
      SPIEL_CHECK_LT(meeple_moved, 1e-5);
    }
  }
  std::cout << "Per-cell logits follow the cell, meeple logits follow the "
               "last-placed plane." << std::endl;

  // A real Carcassonne model: 1770 actions = 4 * 21 * 21 cells + 6 meeple
  // moves, and a policy head that no longer holds most of the network's
  // parameters.
  std::shared_ptr<const Game> game = LoadGame("carcassonne");
  ModelConfig net_config = {
      /*observation_tensor_shape=*/game->ObservationTensorShape(),
      /*number_of_actions=*/game->NumDistinctActions(),
      /*nn_depth=*/2,
      /*nn_width=*/nn_width,
      /*learning_rate=*/0.001,
      /*weight_decay=*/0.001,
      /*nn_model=*/"resnet",
      /*last_placed_plane=*/LastPlacedObservationPlane(*game),
      /*global_features=*/GlobalObservationFeatures(*game)};
  SPIEL_CHECK_GE(net_config.last_placed_plane, 0);
  Model net(net_config, "cpu:0");
  int64_t policy_parameters = 0;
  int64_t largest_tensor = 0;
  for (const auto& parameter : net->named_parameters()) {
    if (absl::StrContains(parameter.key(), ".policy_")) {
      policy_parameters += parameter.value().numel();
    }
    largest_tensor = std::max(largest_tensor, parameter.value().numel());
  }
  // conv 32*32+32, gpool conv 32*32+32, gpool FC (64+70)*32+32, BN 2*32,
  // placement conv 4*32+4, meeple conv 6*32+6. The 70 are the global
  // features joined to the pooled branch.
  SPIEL_CHECK_EQ(policy_parameters, 6826);
  // A dense head would be Linear(2*21*21 -> 1770), 1,562,910 on its own.
  SPIEL_CHECK_LT(largest_tensor, 50000);

  // A game whose actions do not factor per cell keeps the dense head.
  std::shared_ptr<const Game> othello = LoadGame("othello");
  ModelConfig othello_config = {
      /*observation_tensor_shape=*/othello->ObservationTensorShape(),
      /*number_of_actions=*/othello->NumDistinctActions(),
      /*nn_depth=*/2,
      /*nn_width=*/nn_width,
      /*learning_rate=*/0.001,
      /*weight_decay=*/0.001};
  Model othello_net(othello_config, "cpu:0");
  bool has_dense_policy = false;
  for (const auto& parameter : othello_net->named_parameters()) {
    has_dense_policy |= absl::StrContains(parameter.key(), "policy_linear");
  }
  SPIEL_CHECK_TRUE(has_dense_policy);
}

// Carcassonne packs a vector of board-wide features into the first cells of
// its last observation plane. The model has to keep that plane out of the
// convolutions, read only the vector's own cells, and pass the vector to the
// trunk and to both heads.
void TestGlobalFeatures() {
  std::cout << "\n~-~-~-~- TestGlobalFeatures -~-~-~-~" << std::endl;
  torch::NoGradGuard no_grad;
  torch::manual_seed(0);

  std::shared_ptr<const Game> game = LoadGame("carcassonne");
  const std::vector<int> shape = game->ObservationTensorShape();
  const int channels = shape[0];
  const int cells = shape[1] * shape[2];
  const int global_features = GlobalObservationFeatures(*game);
  SPIEL_CHECK_GT(global_features, 0);
  SPIEL_CHECK_LE(global_features, cells);
  const int nn_width = 32;
  ModelConfig config = {
      /*observation_tensor_shape=*/shape,
      /*number_of_actions=*/game->NumDistinctActions(),
      /*nn_depth=*/4,
      /*nn_width=*/nn_width,
      /*learning_rate=*/0.001,
      /*weight_decay=*/0.001,
      /*nn_model=*/"resnet",
      /*last_placed_plane=*/LastPlacedObservationPlane(*game),
      /*global_features=*/global_features};

  // The config survives the vpnet.pb round trip.
  std::stringstream stream;
  stream << config;
  ModelConfig read_back;
  stream >> read_back;
  SPIEL_CHECK_EQ(read_back.global_features, global_features);
  SPIEL_CHECK_EQ(read_back.last_placed_plane, config.last_placed_plane);

  Model net(config, "cpu:0");
  net->eval();
  int bias_layers = 0;
  int64_t bias_parameters = 0;
  for (const auto& parameter : net->named_parameters()) {
    if (absl::StrContains(parameter.key(), "input_conv.weight")) {
      // The packed plane is not convolved.
      SPIEL_CHECK_EQ(parameter.value().size(1), channels - 1);
    }
    if (absl::StrContains(parameter.key(), "global_bias_")) {
      bias_parameters += parameter.value().numel();
      bias_layers += absl::StrContains(parameter.key(), "weight");
    }
  }
  // After the input conv, then in residual blocks 1 and 3 of 0-3.
  SPIEL_CHECK_EQ(bias_layers, 3);
  SPIEL_CHECK_EQ(bias_parameters, 3 * (global_features * nn_width + nn_width));

  // Real observations: one tile decision and one meeple decision.
  std::vector<float> observations;
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) state->ApplyAction(state->LegalActions()[0]);
  for (int i = 0; i < 2; ++i) {
    const std::vector<float> obs = state->ObservationTensor();
    observations.insert(observations.end(), obs.begin(), obs.end());
    state->ApplyAction(state->LegalActions()[0]);
  }
  const int batch = 2;
  torch::Tensor obs = torch::from_blob(observations.data(),
                                       {batch, channels * cells})
                          .clone();
  torch::Tensor mask =
      torch::ones({batch, game->NumDistinctActions()},
                  torch::TensorOptions().dtype(torch::kBool));
  const std::vector<torch::Tensor> base = net->forward(obs, mask);
  auto moved = [&](const torch::Tensor& changed, int output) {
    return (net->forward(changed, mask)[output] - base[output])
        .abs()
        .max()
        .item<float>();
  };

  // The cells of the packed plane past the vector are not read.
  torch::Tensor padding = obs.clone();
  padding.view({batch, channels, cells})
      .select(/*dim=*/1, channels - 1)
      .slice(/*dim=*/1, global_features, cells) += 1.0f;
  SPIEL_CHECK_EQ(moved(padding, 0), 0.0f);
  SPIEL_CHECK_EQ(moved(padding, 1), 0.0f);

  // Every entry of the vector reaches the value and the policy.
  for (int i = 0; i < global_features; ++i) {
    torch::Tensor changed = obs.clone();
    changed.view({batch, channels, cells})
        .select(/*dim=*/1, channels - 1)
        .select(/*dim=*/1, i) += 1.0f;
    SPIEL_CHECK_GT(moved(changed, 0), 0.0f);
    SPIEL_CHECK_GT(moved(changed, 1), 0.0f);
  }

  // Each head reads the vector directly, not only through the trunk.
  const int height = 6;
  const int width = 7;
  const int planes = 4;
  const int num_actions = planes * height * width + 6;
  ResOutputBlockConfig head_config = {
      /*input_channels=*/nn_width,
      /*value_filters=*/32,
      /*policy_filters=*/32,
      /*kernel_size=*/1,
      /*padding=*/0,
      /*value_linear_in_features=*/2 * 32 + global_features,
      /*value_linear_out_features=*/256,
      /*policy_linear_in_features=*/2 * height * width,
      /*policy_linear_out_features=*/num_actions,
      /*policy_observation_size=*/2 * height * width,
      /*policy_conv_planes=*/planes,
      /*policy_extra_actions=*/6,
      /*global_features=*/global_features};
  ResOutputBlock head(head_config);
  head->eval();
  torch::Tensor x = torch::randn({batch, nn_width, height, width});
  torch::Tensor last_placed = torch::zeros({batch, 1, height, width});
  last_placed.index_put_({torch::indexing::Slice(), 0, 2, 3}, 1.0);
  torch::Tensor head_mask = torch::ones(
      {batch, num_actions}, torch::TensorOptions().dtype(torch::kBool));
  torch::Tensor global = torch::randn({batch, global_features});
  const std::vector<torch::Tensor> head_base =
      head->forward(x, head_mask, last_placed, global);
  const std::vector<torch::Tensor> head_changed =
      head->forward(x, head_mask, last_placed, global + 1.0f);
  SPIEL_CHECK_GT((head_changed[0] - head_base[0]).abs().max().item<float>(),
                 1e-6);
  SPIEL_CHECK_GT((head_changed[1] - head_base[1]).abs().max().item<float>(),
                 1e-6);
  bool threw = false;
  try {
    head->forward(x, head_mask, last_placed);
  } catch (const std::exception&) {
    threw = true;
  }
  SPIEL_CHECK_TRUE(threw);
  std::cout << "The packed plane stays out of the convolutions; every global "
               "feature reaches both heads." << std::endl;
}

void TestCUDAAVailability() {
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available!" << std::endl;
  } else {
    std::cout << "CUDA is not available." << std::endl;
  }
}

}  // namespace
}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::algorithms::torch_az::TestModelCreation();
  open_spiel::algorithms::torch_az::TestModelInference();
  open_spiel::algorithms::torch_az::TestValueHeadGlobalPooling();
  open_spiel::algorithms::torch_az::TestConvPolicyHead();
  open_spiel::algorithms::torch_az::TestGlobalFeatures();
  open_spiel::algorithms::torch_az::TestCUDAAVailability();
}
