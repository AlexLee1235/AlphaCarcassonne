// Measures what it would cost to keep Carcassonne observations in the replay
// buffer as int8 instead of float. Every spatial plane holds n / d for a small
// integer n and a denominator d fixed per plane (1 for the 0/1 planes); only
// the first kGlobalFeatures cells of the global plane are real floats. So a
// plane can be stored as its numerators and decoded with the same float
// division the observation was built with, which gives back the same bits.
//
// A standalone experiment: nothing in the training loop uses this codec. It
// first checks on random games that the encoding is lossless, then times
// encoding, decoding, a learner-sized batch and, with --serialize_path, the
// libnop save/load that replay_buffer.data goes through.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>  // Before libnop, which uses std::numeric_limits unincluded.
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <nop/structure.h>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/carcassonne/carcassonne.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/serializable_circular_buffer.h"

ABSL_FLAG(int, games, 20, "Random games to collect observations from.");
ABSL_FLAG(int, seed, 1, "Random seed.");
ABSL_FLAG(int, repeats, 5, "Timed runs per measurement, after one warm-up.");
ABSL_FLAG(int, batch_size, 2048, "Observations per simulated training batch.");
ABSL_FLAG(int, batches_per_step, 128,
          "Batches per learner step, to scale the batch timings.");
ABSL_FLAG(int, buffer_size, 262144,
          "Replay buffer size, to scale the memory and file sizes.");
ABSL_FLAG(std::string, serialize_path, "",
          "If set, save and load the observations with libnop at this path "
          "prefix (<path>.float.data, <path>.compact.data; removed after).");

namespace open_spiel {
namespace carcassonne {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int kPlaneSize = BOARD_SIZE * BOARD_SIZE;

// Keeps the optimizer from dropping work whose result is never read.
double g_sink = 0;

double Seconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

// Planes sharing a denominator, in plane order. The denominators mirror the
// normalization constants in carcassonne.cc, which are private there; the
// validation pass is what says they still match.
struct PlaneGroup {
  const char* name;
  int first;
  int count;
  float denominator;
};

const std::vector<PlaneGroup>& PlaneGroups() {
  static const std::vector<PlaneGroup> groups = {
      {"occupied, terrain, shield, monastery, side links", kOccupiedPlane,
       kFrontierPlane - kOccupiedPlane, 1.0f},
      {"frontier, legal placements, last placed", kFrontierPlane,
       kLastPlacedPlane + 1 - kFrontierPlane, 1.0f},
      {"feature opens", kFeatureOpensPlane, 4, 6.0f},
      {"feature score", kFeatureScorePlane, 4, 12.0f},
      {"feature meeples (mine, theirs)", kFeatureMyMeeplesPlane, 8, 7.0f},
      {"feature signed score", kFeatureSignedScorePlane, 4, 12.0f},
      {"monastery coverage", kMonasteryCoveragePlane, 1, 9.0f},
      {"monastery owner", kMonasteryOwnerPlane, 1, 1.0f},
      {"field farmers (mine, theirs)", kFieldMyFarmersPlane,
       2 * HALF_EDGE_COUNT, 7.0f},
      {"field score", kFieldScorePlane, HALF_EDGE_COUNT, 30.0f},
      {"field size", kFieldSizePlane, HALF_EDGE_COUNT, 30.0f},
      {"field open cities", kFieldOpenCitiesPlane, HALF_EDGE_COUNT, 10.0f},
      {"inner field farmers (mine, theirs)", kInnerFieldMyFarmersPlane, 2,
       7.0f},
      {"inner field score", kInnerFieldScorePlane, 1, 30.0f},
      {"inner field size", kInnerFieldSizePlane, 1, 30.0f},
      {"inner field open cities", kInnerFieldOpenCitiesPlane, 1, 10.0f},
  };
  return groups;
}

// The groups must cover the spatial planes exactly once, in order; a plane
// added to the observation without a group here stops the program.
void CheckPlaneGroups() {
  int next = 0;
  for (const PlaneGroup& group : PlaneGroups()) {
    if (group.first != next) {
      SpielFatalError(absl::StrFormat(
          "Plane group '%s' starts at plane %d, expected %d.", group.name,
          group.first, next));
    }
    next += group.count;
  }
  SPIEL_CHECK_EQ(next, kSpatialPlanes);
  SPIEL_CHECK_EQ(kGlobalFeaturePlane, kSpatialPlanes);
  SPIEL_CHECK_EQ(kObservationPlanes, kSpatialPlanes + 1);
}

// The int8 numerator of `value` over `denominator`, when value is exactly
// that numerator / denominator.
bool Numerator(float value, float denominator, int* numerator) {
  const float scaled = value * denominator;
  if (!(scaled >= -128.0f && scaled <= 127.0f)) return false;
  *numerator = static_cast<int>(std::lround(scaled));
  return static_cast<float>(*numerator) / denominator == value;
}

struct CompactObservation {
  std::vector<int8_t> numerators;
  std::vector<float> floats;

  bool operator==(const CompactObservation& other) const {
    return numerators == other.numerators && floats == other.floats;
  }

  NOP_STRUCTURE(CompactObservation, numerators, floats);
};

class Codec {
 public:
  Codec() {
    std::map<float, int> table_of;
    for (const PlaneGroup& group : PlaneGroups()) {
      if (!table_of.count(group.denominator)) {
        table_of[group.denominator] = tables_.size();
        std::array<float, 256> table;
        for (int n = -128; n < 128; ++n) {
          table[static_cast<uint8_t>(n)] =
              static_cast<float>(n) / group.denominator;
        }
        tables_.push_back(table);
      }
      for (int plane = group.first; plane < group.first + group.count;
           ++plane) {
        segments_.push_back({plane * kPlaneSize, kPlaneSize, group.denominator,
                             table_of[group.denominator]});
        num_numerators_ += kPlaneSize;
      }
    }
    // The rest of the global plane is outside every segment: always 0.
    segments_.push_back(
        {kGlobalFeaturePlane * kPlaneSize, kGlobalFeatures, 0.0f, -1});
    num_floats_ = kGlobalFeatures;
  }

  int NumNumerators() const { return num_numerators_; }
  int NumFloats() const { return num_floats_; }

  // Writes `observation` into `compact`. Checked, it also returns whether
  // every value came back exactly (and every value outside the segments is
  // 0); the check is accumulated rather than branched on, so the loop stays
  // vectorizable. The validation pass reports where a value went wrong.
  template <bool kChecked>
  bool Encode(const float* observation, CompactObservation* compact) const {
    compact->numerators.resize(num_numerators_);
    compact->floats.resize(num_floats_);
    int8_t* numerators = compact->numerators.data();
    float* floats = compact->floats.data();
    bool ok = true;
    int pos = 0;
    for (const Segment& segment : segments_) {
      if (kChecked) {
        for (; pos < segment.begin; ++pos) ok &= observation[pos] == 0.0f;
      }
      const float* in = observation + segment.begin;
      if (segment.denominator == 0.0f) {
        std::copy(in, in + segment.size, floats);
        floats += segment.size;
      } else {
        // Locals: numerators is a char type, which may alias anything, so
        // reading segment.size each iteration would hide the trip count.
        const float denominator = segment.denominator;
        const int size = segment.size;
        uint32_t bad = 0;
        for (int i = 0; i < size; ++i) {
          // Rounds by adding 1.5 * 2^23, which leaves the nearest integer in
          // the low mantissa bits. Unlike a float-to-int conversion this is
          // defined for every input (out of range or NaN, it gives a
          // numerator the checks reject), and the loop vectorizes; clamping
          // first and converting does not, under GCC's default
          // -ftrapping-math.
          const float shifted = in[i] * denominator + 12582912.0f;
          uint32_t bits;
          std::memcpy(&bits, &shifted, sizeof(bits));
          const uint32_t n = bits - 0x4B400000u;
          if (kChecked) {
            bad |= (n + 128u) & ~255u;  // Not an int8.
            bad |= static_cast<float>(static_cast<int32_t>(n)) / denominator !=
                   in[i];
          }
          numerators[i] = static_cast<int8_t>(n);
        }
        ok &= bad == 0;
        numerators += segment.size;
      }
      pos = segment.begin + segment.size;
    }
    if (kChecked) {
      for (; pos < kObservationTensorSize; ++pos) {
        ok &= observation[pos] == 0.0f;
      }
    }
    return ok;
  }

  // Writes the observation back into `observation`, by division (the same
  // operation the observation was built with) or by a table of the 256
  // possible quotients per denominator.
  template <bool kTable>
  void Decode(const CompactObservation& compact, float* observation) const {
    SPIEL_CHECK_EQ(static_cast<int>(compact.numerators.size()),
                   num_numerators_);
    SPIEL_CHECK_EQ(static_cast<int>(compact.floats.size()), num_floats_);
    const int8_t* numerators = compact.numerators.data();
    const float* floats = compact.floats.data();
    int pos = 0;
    for (const Segment& segment : segments_) {
      std::fill(observation + pos, observation + segment.begin, 0.0f);
      float* out = observation + segment.begin;
      if (segment.denominator == 0.0f) {
        std::copy(floats, floats + segment.size, out);
        floats += segment.size;
      } else if (kTable) {
        const float* table = tables_[segment.table].data();
        for (int i = 0; i < segment.size; ++i) {
          out[i] = table[static_cast<uint8_t>(numerators[i])];
        }
        numerators += segment.size;
      } else {
        const float denominator = segment.denominator;
        for (int i = 0; i < segment.size; ++i) {
          out[i] = static_cast<float>(numerators[i]) / denominator;
        }
        numerators += segment.size;
      }
      pos = segment.begin + segment.size;
    }
    std::fill(observation + pos, observation + kObservationTensorSize, 0.0f);
  }

 private:
  struct Segment {
    int begin;
    int size;
    float denominator;  // 0: kept as float.
    int table;
  };

  std::vector<Segment> segments_;
  std::vector<std::array<float, 256>> tables_;
  int num_numerators_ = 0;
  int num_floats_ = 0;
};

struct Corpus {
  std::vector<std::unique_ptr<State>> states;
  std::vector<Player> players;
  std::vector<std::vector<float>> observations;
};

// Every decision state of random games, observed by the player to move, as
// PlayGame records them.
Corpus Collect(int games, std::mt19937* rng) {
  std::shared_ptr<const Game> game = LoadGame("carcassonne");
  Corpus corpus;
  for (int g = 0; g < games; ++g) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        state->ApplyAction(SampleAction(state->ChanceOutcomes(), *rng).first);
        continue;
      }
      const Player player = state->CurrentPlayer();
      corpus.states.push_back(state->Clone());
      corpus.players.push_back(player);
      corpus.observations.push_back(state->ObservationTensor(player));
      const std::vector<Action> legal = state->LegalActions();
      state->ApplyAction(legal[std::uniform_int_distribution<int>(
          0, legal.size() - 1)(*rng)]);
    }
  }
  return corpus;
}

// Checks every value of every observation against its plane's denominator,
// then the codec's round trip. Prints what it finds; false if anything is off.
bool Validate(const Corpus& corpus, const Codec& codec) {
  struct GroupStats {
    int min_n = 0;
    int max_n = 0;
    int64_t nonzero = 0;
    int64_t values = 0;
    int64_t failures = 0;
  };
  const std::vector<PlaneGroup>& groups = PlaneGroups();
  std::vector<GroupStats> stats(groups.size());
  int64_t global_tail_nonzero = 0;
  int printed_failures = 0;
  for (const std::vector<float>& observation : corpus.observations) {
    for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
      const PlaneGroup& group = groups[g];
      GroupStats& s = stats[g];
      for (int plane = group.first; plane < group.first + group.count;
           ++plane) {
        for (int cell = 0; cell < kPlaneSize; ++cell) {
          const float value = observation[plane * kPlaneSize + cell];
          int n = 0;
          ++s.values;
          if (!Numerator(value, group.denominator, &n)) {
            ++s.failures;
            if (printed_failures++ < 10) {
              absl::PrintF(
                  "  NOT n/%g: plane %d (%s) x=%d y=%d value=%.9g "
                  "value*d=%.9g\n",
                  group.denominator, plane, group.name, cell % BOARD_SIZE,
                  cell / BOARD_SIZE, value, value * group.denominator);
            }
            continue;
          }
          s.min_n = std::min(s.min_n, n);
          s.max_n = std::max(s.max_n, n);
          s.nonzero += n != 0;
        }
      }
    }
    for (int cell = kGlobalFeatures; cell < kPlaneSize; ++cell) {
      global_tail_nonzero +=
          observation[kGlobalFeaturePlane * kPlaneSize + cell] != 0.0f;
    }
  }

  absl::PrintF("\n=== validation: %d observations ===\n",
               corpus.observations.size());
  absl::PrintF("%-50s %6s %5s %5s %5s %9s %9s\n", "planes", "range", "d",
               "min n", "max n", "nonzero%", "failures");
  bool ok = true;
  int64_t nonzero = 0;
  int64_t values = 0;
  for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
    const PlaneGroup& group = groups[g];
    const GroupStats& s = stats[g];
    absl::PrintF("%-50s %2d-%-3d %5g %5d %5d %8.2f%% %9d\n", group.name,
                 group.first, group.first + group.count - 1,
                 group.denominator, s.min_n, s.max_n,
                 100.0 * s.nonzero / s.values, s.failures);
    ok &= s.failures == 0;
    nonzero += s.nonzero;
    values += s.values;
  }
  absl::PrintF("%-50s %6d %5s   kept as float (%d values)\n",
               "global vector", kGlobalFeaturePlane, "-", kGlobalFeatures);
  absl::PrintF("%-50s %6d %5s   nonzero after cell %d: %d\n",
               "global plane tail", kGlobalFeaturePlane, "-", kGlobalFeatures,
               global_tail_nonzero);
  absl::PrintF("spatial values nonzero overall: %.2f%%\n",
               100.0 * nonzero / values);
  ok &= global_tail_nonzero == 0;

  // The codec itself: checked encode must accept every observation, and
  // both decoders must give back equal values. Equal is not quite identical
  // bits: -0.0 comes back as +0.0, so count those apart.
  int64_t rejected = 0;
  int64_t value_mismatches[2] = {0, 0};
  int64_t negative_zeros[2] = {0, 0};
  std::vector<float> decoded(kObservationTensorSize);
  CompactObservation compact;
  for (const std::vector<float>& observation : corpus.observations) {
    rejected += !codec.Encode<true>(observation.data(), &compact);
    for (int table = 0; table < 2; ++table) {
      if (table) {
        codec.Decode<true>(compact, decoded.data());
      } else {
        codec.Decode<false>(compact, decoded.data());
      }
      for (int i = 0; i < kObservationTensorSize; ++i) {
        if (decoded[i] != observation[i]) {
          ++value_mismatches[table];
        } else if (std::memcmp(&decoded[i], &observation[i], sizeof(float))) {
          ++negative_zeros[table];
        }
      }
    }
  }
  absl::PrintF(
      "codec: %d numerators + %d floats; checked encode rejected %d; "
      "decode mismatches: division %d, table %d; -0.0 -> +0.0: %d / %d\n",
      codec.NumNumerators(), codec.NumFloats(), rejected, value_mismatches[0],
      value_mismatches[1], negative_zeros[0], negative_zeros[1]);
  ok &= rejected == 0 && value_mismatches[0] == 0 && value_mismatches[1] == 0;

  // And it must refuse what it cannot store: one bad value planted in an
  // otherwise valid observation each time.
  struct Plant {
    const char* what;
    int plane;
    int cell;
    float value;
    bool accept;
  };
  const float kNaN = std::numeric_limits<float>::quiet_NaN();
  const float kInf = std::numeric_limits<float>::infinity();
  const std::vector<Plant> plants = {
      {"0.5 in a 0/1 plane", kOccupiedPlane, 0, 0.5f, false},
      {"200/7 (not an int8)", kFeatureMyMeeplesPlane, 0, 200 / 7.0f, false},
      {"1/7 one ulp up", kFeatureMyMeeplesPlane, 0,
       std::nextafter(1 / 7.0f, 1.0f), false},
      {"NaN", kFeatureOpensPlane, 0, kNaN, false},
      {"+inf", kFeatureScorePlane, 0, kInf, false},
      {"-1e30", kFeatureScorePlane, 0, -1e30f, false},
      {"nonzero in the global plane tail", kGlobalFeaturePlane,
       kGlobalFeatures, 1.0f, false},
      {"-12/12 (valid, negative)", kFeatureSignedScorePlane, 0, -1.0f, true},
      {"-128/1 (valid, int8 minimum)", kMonasteryOwnerPlane, 0, -128.0f,
       true},
  };
  int planted_ok = 0;
  for (const Plant& plant : plants) {
    std::vector<float> observation = corpus.observations.front();
    observation[plant.plane * kPlaneSize + plant.cell] = plant.value;
    const bool accepted = codec.Encode<true>(observation.data(), &compact);
    if (accepted == plant.accept) {
      ++planted_ok;
    } else {
      absl::PrintF("  planted %s: %s, expected %s\n", plant.what,
                   accepted ? "accepted" : "rejected",
                   plant.accept ? "accepted" : "rejected");
    }
  }
  absl::PrintF("planted values handled as expected: %d / %d\n", planted_ok,
               plants.size());
  ok &= planted_ok == static_cast<int>(plants.size());
  return ok;
}

struct Timing {
  double median;
  double min;
};

Timing Summarize(std::vector<double> seconds) {
  std::sort(seconds.begin(), seconds.end());
  return {seconds[seconds.size() / 2], seconds.front()};
}

// One warm-up, then `repeats` timed runs of `run`.
template <typename F>
Timing Time(int repeats, F&& run) {
  run();
  std::vector<double> seconds;
  for (int r = 0; r < repeats; ++r) {
    const Clock::time_point start = Clock::now();
    run();
    seconds.push_back(Seconds(start, Clock::now()));
  }
  return Summarize(seconds);
}

void PrintPerObservation(const char* name, Timing timing, int count) {
  const double us = 1e6 * timing.median / count;
  const double float_gb_per_s =
      kObservationTensorSize * sizeof(float) * count / timing.median / 1e9;
  absl::PrintF("%-36s %9.2f %9.2f %9.2f\n", name, us,
               1e6 * timing.min / count, float_gb_per_s);
}

void TimePerObservation(const Corpus& corpus,
                        const std::vector<CompactObservation>& compact,
                        const Codec& codec, int repeats) {
  const int count = corpus.observations.size();
  std::vector<float> buffer(kObservationTensorSize);
  CompactObservation scratch;
  auto sink_buffer = [&buffer]() {
    for (int i = 0; i < kObservationTensorSize; i += 997) g_sink += buffer[i];
  };

  absl::PrintF("\n=== per observation: %d observations, %d repeats ===\n",
               count, repeats);
  absl::PrintF("%-36s %9s %9s %9s\n", "", "us median", "us min",
               "GB/s float");
  PrintPerObservation(
      "ObservationTensor() (reference)", Time(repeats, [&]() {
        for (int i = 0; i < count; ++i) {
          corpus.states[i]->ObservationTensor(corpus.players[i],
                                              absl::MakeSpan(buffer));
        }
        sink_buffer();
      }),
      count);
  PrintPerObservation(
      "copy float observation (memcpy)", Time(repeats, [&]() {
        for (int i = 0; i < count; ++i) {
          std::memcpy(buffer.data(), corpus.observations[i].data(),
                      kObservationTensorSize * sizeof(float));
        }
        sink_buffer();
      }),
      count);
  // Encoding into a fresh CompactObservation each time, as an actor would.
  PrintPerObservation(
      "encode, checked (new object)", Time(repeats, [&]() {
        int ok = 0;
        for (int i = 0; i < count; ++i) {
          CompactObservation out;
          ok += codec.Encode<true>(corpus.observations[i].data(), &out);
          g_sink += out.numerators[i % out.numerators.size()];
        }
        g_sink += ok;
      }),
      count);
  PrintPerObservation(
      "encode, unchecked (new object)", Time(repeats, [&]() {
        for (int i = 0; i < count; ++i) {
          CompactObservation out;
          codec.Encode<false>(corpus.observations[i].data(), &out);
          g_sink += out.numerators[i % out.numerators.size()];
        }
      }),
      count);
  PrintPerObservation(
      "encode, checked (reused object)", Time(repeats, [&]() {
        int ok = 0;
        for (int i = 0; i < count; ++i) {
          ok += codec.Encode<true>(corpus.observations[i].data(), &scratch);
        }
        g_sink += ok;
      }),
      count);
  PrintPerObservation(
      "decode, division", Time(repeats, [&]() {
        for (int i = 0; i < count; ++i) {
          codec.Decode<false>(compact[i], buffer.data());
        }
        sink_buffer();
      }),
      count);
  PrintPerObservation(
      "decode, lookup table", Time(repeats, [&]() {
        for (int i = 0; i < count; ++i) {
          codec.Decode<true>(compact[i], buffer.data());
        }
        sink_buffer();
      }),
      count);
}

// What a learner batch costs: draw samples out of the buffer (a copy, as
// CircularBuffer::Sample makes), lay them out contiguously for the device (as
// PackTrainBatch does), free the samples. The contiguous buffer is allocated
// once, since both paths need the same one.
void TimeBatches(const Corpus& corpus,
                 const std::vector<CompactObservation>& compact,
                 const Codec& codec, int batch_size, int batches_per_step,
                 int repeats, std::mt19937* rng) {
  const int count = corpus.observations.size();
  std::vector<float> batch(static_cast<size_t>(batch_size) *
                           kObservationTensorSize);
  enum Phase { kSample, kLayOut, kFree, kNumPhases };
  enum Path { kFloat, kDivision, kTable, kNumPaths };
  std::vector<double> seconds[kNumPaths][kNumPhases];
  std::uniform_int_distribution<int> pick(0, count - 1);

  for (int r = -1; r < repeats; ++r) {  // r == -1 is the warm-up.
    std::vector<int> indices(batch_size);
    for (int& index : indices) index = pick(*rng);

    for (int path = 0; path < kNumPaths; ++path) {
      Clock::time_point t0 = Clock::now();
      Clock::time_point t1;
      Clock::time_point t2;
      if (path == kFloat) {
        std::vector<std::vector<float>> sampled;
        sampled.reserve(batch_size);
        for (int index : indices) {
          sampled.push_back(corpus.observations[index]);
        }
        t1 = Clock::now();
        for (int b = 0; b < batch_size; ++b) {
          std::copy(sampled[b].begin(), sampled[b].end(),
                    batch.begin() + static_cast<size_t>(b) *
                                        kObservationTensorSize);
        }
        t2 = Clock::now();
      } else {
        std::vector<CompactObservation> sampled;
        sampled.reserve(batch_size);
        for (int index : indices) sampled.push_back(compact[index]);
        t1 = Clock::now();
        for (int b = 0; b < batch_size; ++b) {
          float* out =
              batch.data() + static_cast<size_t>(b) * kObservationTensorSize;
          if (path == kTable) {
            codec.Decode<true>(sampled[b], out);
          } else {
            codec.Decode<false>(sampled[b], out);
          }
        }
        t2 = Clock::now();
      }
      // The sampled copies are released as their block ends, between t2
      // and t3.
      const Clock::time_point t3 = Clock::now();
      g_sink += batch[static_cast<size_t>(r + 1) % batch.size()];
      if (r >= 0) {
        seconds[path][kSample].push_back(Seconds(t0, t1));
        seconds[path][kLayOut].push_back(Seconds(t1, t2));
        seconds[path][kFree].push_back(Seconds(t2, t3));
      }
    }
  }

  absl::PrintF(
      "\n=== per batch of %d (random picks from %d observations), "
      "%d repeats ===\n",
      batch_size, count, repeats);
  absl::PrintF("%-30s %10s %10s %10s %10s %14s\n", "", "sample ms",
               "lay out ms", "free ms", "total ms", "s per step x" +
               std::to_string(batches_per_step));
  const char* names[kNumPaths] = {"float (now)", "int8, decode by division",
                                  "int8, decode by table"};
  for (int path = 0; path < kNumPaths; ++path) {
    double total = 0;
    double phase_ms[kNumPhases];
    for (int phase = 0; phase < kNumPhases; ++phase) {
      phase_ms[phase] = 1e3 * Summarize(seconds[path][phase]).median;
      total += phase_ms[phase];
    }
    absl::PrintF("%-30s %10.1f %10.1f %10.1f %10.1f %14.2f\n", names[path],
                 phase_ms[kSample], phase_ms[kLayOut], phase_ms[kFree], total,
                 total * batches_per_step / 1e3);
  }
}

void PrintMemory(const Codec& codec, int buffer_size) {
  const double float_bytes =
      sizeof(std::vector<float>) + kObservationTensorSize * sizeof(float);
  const double compact_bytes = sizeof(CompactObservation) +
                               codec.NumNumerators() * sizeof(int8_t) +
                               codec.NumFloats() * sizeof(float);
  absl::PrintF("\n=== memory per observation (allocator overhead not counted) "
               "===\n");
  absl::PrintF("float:   %8.0f bytes   x %d = %6.2f GB\n", float_bytes,
               buffer_size, float_bytes * buffer_size / 1e9);
  absl::PrintF("compact: %8.0f bytes   x %d = %6.2f GB   (%.2fx smaller)\n",
               compact_bytes, buffer_size, compact_bytes * buffer_size / 1e9,
               float_bytes / compact_bytes);
}

// Saves `items` through SerializableCircularBuffer, the class that writes
// replay_buffer.data, then loads them back and compares.
template <typename T>
void TimeSerialization(const char* name, const std::vector<T>& items,
                       const std::string& file, int buffer_size) {
  double write_s;
  {
    SerializableCircularBuffer<T> out(items.size());
    for (const T& item : items) out.Add(item);
    const Clock::time_point start = Clock::now();
    out.SaveBuffer(file);
    write_s = Seconds(start, Clock::now());
  }
  const double bytes = std::filesystem::file_size(file);
  SerializableCircularBuffer<T> in(items.size());
  const Clock::time_point start = Clock::now();
  in.LoadBuffer(file);
  const double read_s = Seconds(start, Clock::now());
  const bool same = in.Data() == items;
  std::filesystem::remove(file);

  const double scale = static_cast<double>(buffer_size) / items.size();
  absl::PrintF(
      "%-8s %9.1f MB %8.0f B/obs  write %6.2f s (%6.0f MB/s)  read %6.2f s "
      "(%6.0f MB/s)  same after load: %s\n",
      name, bytes / 1e6, bytes / items.size(), write_s, bytes / write_s / 1e6,
      read_s, bytes / read_s / 1e6, same ? "yes" : "NO");
  absl::PrintF("%-8s   x %d: %6.2f GB, write ~%.0f s, read ~%.0f s\n", "",
               buffer_size, bytes * scale / 1e9, write_s * scale,
               read_s * scale);
}

int Main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const int games = absl::GetFlag(FLAGS_games);
  const int repeats = absl::GetFlag(FLAGS_repeats);
  const int batch_size = absl::GetFlag(FLAGS_batch_size);
  const int batches_per_step = absl::GetFlag(FLAGS_batches_per_step);
  const int buffer_size = absl::GetFlag(FLAGS_buffer_size);
  const std::string serialize_path = absl::GetFlag(FLAGS_serialize_path);
  SPIEL_CHECK_GT(games, 0);
  SPIEL_CHECK_GT(repeats, 0);
  SPIEL_CHECK_GT(batch_size, 0);

  CheckPlaneGroups();
  const Codec codec;
  std::mt19937 rng(absl::GetFlag(FLAGS_seed));

  const Clock::time_point start = Clock::now();
  Corpus corpus = Collect(games, &rng);
  absl::PrintF("collected %d observations from %d games in %.1f s "
               "(observation: %d planes x %d x %d = %d floats)\n",
               corpus.observations.size(), games,
               Seconds(start, Clock::now()), kObservationPlanes, BOARD_SIZE,
               BOARD_SIZE, kObservationTensorSize);

  if (!Validate(corpus, codec)) {
    absl::PrintF("\nvalidation FAILED: the encoding is not lossless as "
                 "specified; timings skipped.\n");
    return 1;
  }

  std::vector<CompactObservation> compact(corpus.observations.size());
  for (int i = 0; i < static_cast<int>(compact.size()); ++i) {
    codec.Encode<true>(corpus.observations[i].data(), &compact[i]);
  }

  TimePerObservation(corpus, compact, codec, repeats);
  TimeBatches(corpus, compact, codec, batch_size, batches_per_step, repeats,
              &rng);
  PrintMemory(codec, buffer_size);

  if (!serialize_path.empty()) {
    absl::PrintF("\n=== libnop save / load, %d observations (written to the "
                 "page cache; not fsynced) ===\n",
                 corpus.observations.size());
    TimeSerialization("float", corpus.observations,
                      serialize_path + ".float.data", buffer_size);
    TimeSerialization("compact", compact, serialize_path + ".compact.data",
                      buffer_size);
  }

  absl::PrintF("\nchecksum (ignore): %g\n", g_sink);
  return 0;
}

}  // namespace
}  // namespace carcassonne
}  // namespace open_spiel

int main(int argc, char** argv) {
  return open_spiel::carcassonne::Main(argc, argv);
}
