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

#ifndef OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_MODEL_H_
#define OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_MODEL_H_

#include <torch/torch.h>

#include <iostream>
#include <string>
#include <vector>

namespace open_spiel {
namespace algorithms {
namespace torch_az {

struct ResInputBlockConfig {
  int input_channels;
  int input_height;
  int input_width;
  int filters;
  int kernel_size;
  int padding;
};

struct ResTorsoBlockConfig {
  int input_channels;
  int filters;
  int kernel_size;
  int padding;
  int layer;
};

struct ResOutputBlockConfig {
  int input_channels;
  int value_filters;
  int policy_filters;
  int kernel_size;
  int padding;
  int value_linear_in_features;
  int value_linear_out_features;
  int policy_linear_in_features;
  int policy_linear_out_features;
  int policy_observation_size;
  // Logits per board cell, e.g. one per tile rotation. 0 keeps the dense head
  // that maps flattened board features to every action with one linear layer.
  int policy_conv_planes;
  // Actions that belong to no cell (Carcassonne: the meeple moves). They come
  // after the per-cell actions, as the game's action encoding has them.
  int policy_extra_actions;
  // Length of the board-wide feature vector joined to the pooled features of
  // both heads (see ModelConfig::global_features); 0 for none.
  int global_features;
};

// Information for the model. This should be enough for any type of model
// (residual, convultional, or MLP). It needs to be saved/loaded to/from
// a file so the input and output stream operators are overload.
struct ModelConfig {
  std::vector<int> observation_tensor_shape;
  int number_of_actions;
  int nn_depth;
  int nn_width;
  double learning_rate;
  double weight_decay;
  std::string nn_model = "resnet";
  // Observation plane holding a one-hot of the cell just played, or -1 when the
  // game has none. The conv policy head reads the actions that belong to that
  // cell (Carcassonne's meeple moves) from it.
  int last_placed_plane = -1;
  // When > 0, the last observation plane is not a picture of the board: its
  // first global_features values are a vector of board-wide quantities
  // (scores, the deck, the phase). The model takes it out of the convolutions
  // and adds it as a per-channel bias to the trunk and to both heads.
  int global_features = 0;
};
std::istream& operator>>(std::istream& stream, ModelConfig& config);
std::ostream& operator<<(std::ostream& stream, const ModelConfig& config);

// A block of the residual model's network that handles the input. It consists
// of one convolutional layer (CONV) and one batch normalization (BN) layer, and
// the output is passed through a rectified linear unit function (RELU).
//
// Illustration:
//   [Input Tensor] --> CONV --> BN --> RELU
//
// There is only one input block per model. global_bias, if given, is
// [B, filters] and is added to every cell after the CONV.
class ResInputBlockImpl : public torch::nn::Module {
 public:
  ResInputBlockImpl(const ResInputBlockConfig& config);
  torch::Tensor forward(torch::Tensor x, torch::Tensor global_bias = {});

 private:
  int channels_;
  int height_;
  int width_;
  torch::nn::Conv2d conv_;
  torch::nn::BatchNorm2d batch_norm_;
};
TORCH_MODULE(ResInputBlock);

// A block of the residual model's network that makes up the 'torso'. It
// consists of two convolutional layers (CONV) and two batchnormalization layers
// (BN). The activation function is rectified linear unit (RELU). The input to
// the layer is added to the output before the final activation function.
//
// Illustration:
//   [Input Tensor] --> CONV --> BN --> RELU --> CONV --> BN --> + --> RELU
//          \___________________________________________________/
//
// Unlike the input and output blocks, one can specify how many of these torso
// blocks they want in their model. global_bias, if given, is [B, filters] and
// is added to every cell after the first CONV.
class ResTorsoBlockImpl : public torch::nn::Module {
 public:
  ResTorsoBlockImpl(const ResTorsoBlockConfig& config, int layer);
  torch::Tensor forward(torch::Tensor x, torch::Tensor global_bias = {});

 private:
  torch::nn::Conv2d conv1_;
  torch::nn::Conv2d conv2_;
  torch::nn::BatchNorm2d batch_norm1_;
  torch::nn::BatchNorm2d batch_norm2_;
};
TORCH_MODULE(ResTorsoBlock);

// A block of the residual model's network that creates the output. It consists
// of a value and policy head. The value head takes the input through one
// convoluational layer (CONV), one batch normalization layers (BN), global
// average and global max pooling over the board (POOL), and two linear layers
// (LIN). The output activation function is tanh (TANH), the rectified linear
// activation function (RELU) is within. The policy head
// consists of one convolutional layer, batch normalization layer, and linear
// layer. There is no softmax activation function in this layer. The softmax
// on the output is applied in the forward function of the residual model.
// This design was chosen because the loss function of the residual model
// requires the policy logits, not the policy distribution. By providing the
// policy logits as output, the residual model can either apply the softmax
// activation function, or calculate the loss using Torch's log softmax
// function.
//
// The value head pools instead of flattening: a flatten + LIN readout gives
// every board cell its own weights, which only get gradient when that cell is
// occupied, and a single ReLU'd filter cannot carry a signed per-cell score.
// With several filters and mean/max pooling the readout has no
// position-dependent parameters, the mean is a hard-coded board-wide sum, and
// the head no longer depends on the board size.
//
// The policy head has two forms. When the actions factor into a few choices
// per board cell (policy_conv_planes), 1x1 CONVs score every cell directly:
// the weights are shared by all cells, so the head neither has to learn which
// action index belongs to which cell nor needs one weight per (cell, action).
// A pooled branch adds a board-wide bias to every cell, because whether a move
// is good depends on global counts the trunk cannot carry that far. The actions
// tied to the cell just played (Carcassonne's meeple moves) are read from that
// cell with the last-placed plane as a one-hot. Games whose actions do not
// factor that way keep the dense readout.
//
// A board-wide feature vector, when the game has one, is joined to the pooled
// features of the value head and of the policy head's pooled branch.
//
// Illustration:
//                    --> CONV --> BN --> RELU --> POOL(mean ++ max) --> LIN
//                                                 --> RELU --> LIN --> TANH
//   [Input Tensor] --
//                    --> CONV --+--> BN --> RELU --> CONV --> per-cell actions
//                    |          |                 \-> CONV * last_placed_plane
//                    |          |                     --> the cell's actions
//                    \-> CONV --> POOL --> LIN (bias added above)
//                                       (or, dense: --> LIN)  (no SOFTMAX here)
//
// There is only one output block per model.
class ResOutputBlockImpl : public torch::nn::Module {
 public:
  ResOutputBlockImpl(const ResOutputBlockConfig& config);
  // last_placed_plane is [B, 1, H, W], required by the conv policy head.
  // global_features is [B, config.global_features], required when that is > 0.
  std::vector<torch::Tensor> forward(torch::Tensor x, torch::Tensor mask,
                                     torch::Tensor last_placed_plane = {},
                                     torch::Tensor global_features = {});

 private:
  torch::nn::Conv2d value_conv_;
  torch::nn::BatchNorm2d value_batch_norm_;
  torch::nn::Linear value_linear1_;
  torch::nn::Linear value_linear2_;
  torch::nn::Conv2d policy_conv_;
  torch::nn::BatchNorm2d policy_batch_norm_;
  // Only one of these two policy readouts is built and registered.
  torch::nn::Linear policy_linear_{nullptr};
  torch::nn::Conv2d policy_gpool_conv_{nullptr};
  torch::nn::Linear policy_gpool_linear_{nullptr};
  torch::nn::Conv2d policy_placement_conv_{nullptr};
  torch::nn::Conv2d policy_cell_conv_{nullptr};
  int policy_observation_size_;
  int policy_conv_planes_;
  int policy_extra_actions_;
  int global_features_;
};
TORCH_MODULE(ResOutputBlock);

// A dense block with ReLU activation.
class MLPBlockImpl : public torch::nn::Module {
 public:
  MLPBlockImpl(const int in_features, const int out_features);
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::nn::Linear linear_;
};
TORCH_MODULE(MLPBlock);

class MLPOutputBlockImpl : public torch::nn::Module {
 public:
  MLPOutputBlockImpl(const int nn_width, const int policy_linear_out_features);
  std::vector<torch::Tensor> forward(torch::Tensor x, torch::Tensor mask);

 private:
  torch::nn::Linear value_linear1_;
  torch::nn::Linear value_linear2_;
  torch::nn::Linear policy_linear1_;
  torch::nn::Linear policy_linear2_;
};
TORCH_MODULE(MLPOutputBlock);

// The model class that interacts with the VPNet. The ResInputBlock,
// ResTorsoBlock, and ResOutputBlock are not to be used by the VPNet directly.
class ModelImpl : public torch::nn::Module {
 public:
  ModelImpl(const ModelConfig& config, const std::string& device);
  std::vector<torch::Tensor> forward(torch::Tensor x, torch::Tensor mask);
  std::vector<torch::Tensor> losses(torch::Tensor inputs, torch::Tensor masks,
                                    torch::Tensor policy_targets,
                                    torch::Tensor value_targets);

 private:
  std::vector<torch::Tensor> forward_(torch::Tensor x, torch::Tensor mask);
  torch::nn::ModuleList layers_;
  torch::Device device_;
  int num_torso_blocks_;
  double weight_decay_;
  std::string nn_model_;
  // Shape of one observation, and the plane the conv policy head reads the
  // cell's own actions from (-1 when the game has no such plane).
  int input_channels_ = 0;
  int input_height_ = 0;
  int input_width_ = 0;
  int last_placed_plane_ = -1;
  // The board-wide feature vector (ModelConfig::global_features) enters the
  // trunk as a per-channel bias: once after the input convolution and again in
  // every second residual block. global_block_biases_ has one entry per
  // residual block, empty for blocks without one.
  int global_features_ = 0;
  torch::nn::Linear global_input_bias_{nullptr};
  std::vector<torch::nn::Linear> global_block_biases_;
};
TORCH_MODULE(Model);

}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_MODEL_H_
