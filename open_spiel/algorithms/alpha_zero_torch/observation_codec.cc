#include "open_spiel/algorithms/alpha_zero_torch/observation_codec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/carcassonne/carcassonne.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {

ObservationCodec::ObservationCodec(const Game& game)
    : flat_size_(game.ObservationTensorSize()),
      plane_size_(flat_size_ / game.ObservationTensorShape()[0]) {
  if (game.GetType().short_name == "carcassonne") {
    SPIEL_CHECK_EQ(flat_size_, carcassonne::kObservationTensorSize);
    for (int plane = 0; plane < carcassonne::kSpatialPlanes; ++plane) {
      const float denominator =
          carcassonne::ObservationPlaneDenominator(plane);
      SPIEL_CHECK_GT(denominator, 0.0f);
      segments_.push_back({plane * plane_size_, plane_size_, denominator});
    }
    // Only the start of the global plane holds the global vector, which is
    // not made of fractions; the rest of that plane is 0.
    SPIEL_CHECK_EQ(carcassonne::ObservationPlaneDenominator(
                       carcassonne::kGlobalFeaturePlane),
                   0.0f);
    segments_.push_back({carcassonne::kGlobalFeaturePlane * plane_size_,
                         carcassonne::kGlobalFeatures, 0.0f});
  } else {
    segments_.push_back({0, flat_size_, 0.0f});
  }
  for (const Segment& segment : segments_) {
    if (segment.denominator == 0.0f) {
      num_floats_ += segment.size;
    } else {
      num_numerators_ += segment.size;
    }
  }
}

bool ObservationCodec::TryEncode(absl::Span<const float> observation,
                                 CompactObservation* compact) const {
  SPIEL_CHECK_EQ(static_cast<int>(observation.size()), flat_size_);
  compact->numerators.resize(num_numerators_);
  compact->floats.resize(num_floats_);
  int8_t* numerators = compact->numerators.data();
  float* floats = compact->floats.data();
  bool fits = true;
  int pos = 0;
  for (const Segment& segment : segments_) {
    for (; pos < segment.begin; ++pos) fits &= observation[pos] == 0.0f;
    const float* in = observation.data() + segment.begin;
    // Locals: numerators is a char type, which may alias anything, so
    // reading the segment each iteration would hide the trip count from the
    // vectorizer.
    const int size = segment.size;
    const float denominator = segment.denominator;
    if (denominator == 0.0f) {
      std::copy(in, in + size, floats);
      floats += size;
    } else {
      // Checked for every value and accumulated rather than branched on, so
      // the loop stays vectorizable; DescribeMisfit finds the culprit.
      uint32_t misfit = 0;
      for (int i = 0; i < size; ++i) {
        // Rounds by adding 1.5 * 2^23, which leaves the nearest integer in
        // the low mantissa bits. Unlike a float-to-int conversion this is
        // defined for every input (out of range or NaN, it gives a numerator
        // the checks reject), and the loop vectorizes; clamping first and
        // converting does not, under GCC's default -ftrapping-math.
        const float shifted = in[i] * denominator + 12582912.0f;
        uint32_t bits;
        std::memcpy(&bits, &shifted, sizeof(bits));
        const uint32_t n = bits - 0x4B400000u;
        misfit |= (n + 128u) & ~255u;  // Not an int8.
        misfit |= static_cast<float>(static_cast<int32_t>(n)) / denominator !=
                  in[i];
        numerators[i] = static_cast<int8_t>(n);
      }
      fits &= misfit == 0;
      numerators += size;
    }
    pos = segment.begin + size;
  }
  for (; pos < flat_size_; ++pos) fits &= observation[pos] == 0.0f;
  return fits;
}

CompactObservation ObservationCodec::Encode(
    absl::Span<const float> observation) const {
  CompactObservation compact;
  if (!TryEncode(observation, &compact)) {
    SpielFatalError(DescribeMisfit(observation));
  }
  return compact;
}

// Why TryEncode refused `observation`: the first value that does not fit,
// found one value at a time.
std::string ObservationCodec::DescribeMisfit(
    absl::Span<const float> observation) const {
  auto describe = [this](int index, float value, const std::string& why) {
    return absl::StrFormat(
        "ObservationCodec: observation value %.9g at index %d (plane %d, "
        "cell %d) %s. The codec and the game's observation planes disagree.",
        value, index, index / plane_size_, index % plane_size_, why);
  };
  int pos = 0;
  for (const Segment& segment : segments_) {
    for (; pos < segment.begin; ++pos) {
      if (observation[pos] != 0.0f) {
        return describe(pos, observation[pos], "is not 0, and no plane keeps it");
      }
    }
    const float denominator = segment.denominator;
    if (denominator != 0.0f) {
      for (int i = segment.begin; i < segment.begin + segment.size; ++i) {
        const float value = observation[i];
        const float scaled = value * denominator;
        if (!(scaled >= -128.0f && scaled <= 127.0f) ||
            static_cast<float>(std::lround(scaled)) / denominator != value) {
          return describe(i, value,
                          absl::StrFormat("is not n / %g for an int8 n",
                                          denominator));
        }
      }
    }
    pos = segment.begin + segment.size;
  }
  for (; pos < flat_size_; ++pos) {
    if (observation[pos] != 0.0f) {
      return describe(pos, observation[pos], "is not 0, and no plane keeps it");
    }
  }
  return "ObservationCodec: refused an observation in which every value fits.";
}

void ObservationCodec::Decode(const CompactObservation& compact,
                              absl::Span<float> observation) const {
  SPIEL_CHECK_EQ(static_cast<int>(observation.size()), flat_size_);
  SPIEL_CHECK_EQ(static_cast<int>(compact.numerators.size()), num_numerators_);
  SPIEL_CHECK_EQ(static_cast<int>(compact.floats.size()), num_floats_);
  const int8_t* numerators = compact.numerators.data();
  const float* floats = compact.floats.data();
  float* begin = observation.data();
  int pos = 0;
  for (const Segment& segment : segments_) {
    std::fill(begin + pos, begin + segment.begin, 0.0f);
    float* out = begin + segment.begin;
    const int size = segment.size;
    const float denominator = segment.denominator;
    if (denominator == 0.0f) {
      std::copy(floats, floats + size, out);
      floats += size;
    } else {
      // Division, not a multiplication by 1 / d: it is the operation the
      // observation was built with, so the result is the same float.
      for (int i = 0; i < size; ++i) {
        out[i] = static_cast<float>(numerators[i]) / denominator;
      }
      numerators += size;
    }
    pos = segment.begin + size;
  }
  std::fill(begin + pos, begin + flat_size_, 0.0f);
}

std::vector<float> ObservationCodec::Decode(
    const CompactObservation& compact) const {
  std::vector<float> observation(flat_size_);
  Decode(compact, absl::MakeSpan(observation));
  return observation;
}

}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel
