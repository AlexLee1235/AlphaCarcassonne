#ifndef OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_OBSERVATION_CODEC_H_
#define OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_OBSERVATION_CODEC_H_

#include <cstdint>
#include <limits>  // Before libnop, which uses std::numeric_limits without it.
#include <string>
#include <vector>

#include <nop/structure.h>

#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {

// An observation as trajectories and the replay buffer keep it. A plane whose
// values are all n / d, for an int8 n and a denominator d fixed per plane, is
// kept as the numerators n; anything else as floats. For Carcassonne that is
// a quarter of the float size, and libnop writes an integral vector as one
// block where it writes a float vector one tagged value at a time.
struct CompactObservation {
  std::vector<int8_t> numerators;
  std::vector<float> floats;

  bool operator==(const CompactObservation& other) const {
    return numerators == other.numerators && floats == other.floats;
  }

  NOP_STRUCTURE(CompactObservation, numerators, floats);
};

// Turns a game's observations into CompactObservations and back. Decoding
// divides each numerator by its denominator, the operation the observation
// was built with, so it gives back the same floats (a -0.0 would come back
// as +0.0; observations do not hold one). Games that do not say what their
// planes hold (see the constructor) are kept as floats.
class ObservationCodec {
 public:
  explicit ObservationCodec(const Game& game);

  // False if some value is not n / d for an int8 n in its plane, or a value
  // that no plane keeps is not 0.
  bool TryEncode(absl::Span<const float> observation,
                 CompactObservation* compact) const;
  // TryEncode, or SpielFatalError naming the first value that does not fit.
  CompactObservation Encode(absl::Span<const float> observation) const;

  void Decode(const CompactObservation& compact,
              absl::Span<float> observation) const;
  std::vector<float> Decode(const CompactObservation& compact) const;

  int NumNumerators() const { return num_numerators_; }
  int NumFloats() const { return num_floats_; }
  // The stored values of one observation, in bytes.
  int CompactBytes() const {
    return num_numerators_ * sizeof(int8_t) + num_floats_ * sizeof(float);
  }

 private:
  // A run of observation values stored the same way: as numerators over
  // `denominator`, or as floats when it is 0.
  struct Segment {
    int begin;
    int size;
    float denominator;
  };

  std::string DescribeMisfit(absl::Span<const float> observation) const;

  std::vector<Segment> segments_;  // In order; values outside them are 0.
  int flat_size_;
  int plane_size_;
  int num_numerators_ = 0;
  int num_floats_ = 0;
};

}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_OBSERVATION_CODEC_H_
