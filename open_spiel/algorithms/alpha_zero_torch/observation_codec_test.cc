#include "open_spiel/algorithms/alpha_zero_torch/observation_codec.h"

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/games/carcassonne/carcassonne.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/serializable_circular_buffer.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {
namespace {

constexpr int kPlaneSize = BOARD_SIZE * BOARD_SIZE;

// Calls `f` with the observation of every decision state of `games` random
// games, from each player's side.
void ForEachObservation(
    const Game& game, int games, std::mt19937* rng,
    const std::function<void(const std::vector<float>&)>& f) {
  for (int g = 0; g < games; ++g) {
    std::unique_ptr<State> state = game.NewInitialState();
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        state->ApplyAction(SampleAction(state->ChanceOutcomes(), *rng).first);
        continue;
      }
      for (Player player = 0; player < game.NumPlayers(); ++player) {
        f(state->ObservationTensor(player));
      }
      const std::vector<Action> legal = state->LegalActions();
      state->ApplyAction(legal[std::uniform_int_distribution<int>(
          0, legal.size() - 1)(*rng)]);
    }
  }
}

// Decoding gives back the observation bit for bit.
void CheckRoundTrip(const ObservationCodec& codec,
                    const std::vector<float>& observation) {
  const std::vector<float> decoded = codec.Decode(codec.Encode(observation));
  SPIEL_CHECK_EQ(decoded.size(), observation.size());
  SPIEL_CHECK_EQ(std::memcmp(decoded.data(), observation.data(),
                             observation.size() * sizeof(float)),
                 0);
}

void CarcassonneRoundTripTest() {
  std::shared_ptr<const Game> game = LoadGame("carcassonne");
  const ObservationCodec codec(*game);
  SPIEL_CHECK_EQ(codec.NumNumerators(),
                 carcassonne::kSpatialPlanes * kPlaneSize);
  SPIEL_CHECK_EQ(codec.NumFloats(), carcassonne::kGlobalFeatures);
  std::mt19937 rng(20261005);
  int observations = 0;
  ForEachObservation(*game, 5, &rng,
                     [&](const std::vector<float>& observation) {
                       CheckRoundTrip(codec, observation);
                       ++observations;
                     });
  SPIEL_CHECK_GT(observations, 0);
}

// TryEncode refuses what it cannot store, and keeps the extremes it can.
void CarcassonneMisfitTest() {
  std::shared_ptr<const Game> game = LoadGame("carcassonne");
  const ObservationCodec codec(*game);
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) state->ApplyAction(state->LegalActions()[0]);
  const std::vector<float> valid = state->ObservationTensor(0);
  CompactObservation compact;
  SPIEL_CHECK_TRUE(codec.TryEncode(valid, &compact));

  struct Plant {
    int plane;
    int cell;
    float value;
    bool fits;
  };
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<Plant> plants = {
      {carcassonne::kOccupiedPlane, 0, 0.5f, false},  // Not n / 1.
      {carcassonne::kFeatureMyMeeplesPlane, 0, 200 / 7.0f, false},  // n > 127.
      {carcassonne::kFeatureMyMeeplesPlane, 0,
       std::nextafter(1 / 7.0f, 1.0f), false},
      {carcassonne::kFeatureOpensPlane, 0, nan, false},
      {carcassonne::kFeatureScorePlane, 0, inf, false},
      {carcassonne::kFeatureScorePlane, 0, -inf, false},
      {carcassonne::kFeatureScorePlane, 0, -1e30f, false},
      {carcassonne::kMonasteryOwnerPlane, 0, 128.0f, false},
      // Past the global vector, outside every segment.
      {carcassonne::kGlobalFeaturePlane, carcassonne::kGlobalFeatures, 1.0f,
       false},
      {carcassonne::kFeatureSignedScorePlane, 0, -1.0f, true},
      {carcassonne::kMonasteryOwnerPlane, 0, -128.0f, true},
      {carcassonne::kMonasteryOwnerPlane, 0, 127.0f, true},
      // The global vector is kept as floats.
      {carcassonne::kGlobalFeaturePlane, 0, 0.123f, true},
  };
  for (const Plant& plant : plants) {
    std::vector<float> observation = valid;
    observation[plant.plane * kPlaneSize + plant.cell] = plant.value;
    SPIEL_CHECK_EQ(codec.TryEncode(observation, &compact), plant.fits);
    if (plant.fits) CheckRoundTrip(codec, observation);
  }
}

// A game whose planes the codec knows nothing about is kept as floats.
void OtherGameTest() {
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  const ObservationCodec codec(*game);
  SPIEL_CHECK_EQ(codec.NumNumerators(), 0);
  SPIEL_CHECK_EQ(codec.NumFloats(), game->ObservationTensorSize());
  std::mt19937 rng(1);
  ForEachObservation(*game, 3, &rng,
                     [&](const std::vector<float>& observation) {
                       CheckRoundTrip(codec, observation);
                     });
}

// What the replay buffer saves comes back unchanged.
void SerializationTest() {
  std::shared_ptr<const Game> game = LoadGame("carcassonne");
  const ObservationCodec codec(*game);
  std::mt19937 rng(7);
  std::vector<CompactObservation> compact;
  ForEachObservation(*game, 1, &rng,
                     [&](const std::vector<float>& observation) {
                       if (compact.size() < 16) {
                         compact.push_back(codec.Encode(observation));
                       }
                     });
  SerializableCircularBuffer<CompactObservation> saved(compact.size());
  for (const CompactObservation& observation : compact) saved.Add(observation);
  const std::string path =
      file::GetTmpDir() + "/observation_codec_test_buffer.data";
  saved.SaveBuffer(path);
  SerializableCircularBuffer<CompactObservation> loaded(compact.size());
  loaded.LoadBuffer(path);
  file::Remove(path);
  SPIEL_CHECK_TRUE(loaded.Data() == compact);
}

}  // namespace
}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel

int main() {
  open_spiel::algorithms::torch_az::CarcassonneRoundTripTest();
  open_spiel::algorithms::torch_az::CarcassonneMisfitTest();
  open_spiel::algorithms::torch_az::OtherGameTest();
  open_spiel::algorithms::torch_az::SerializationTest();
}
