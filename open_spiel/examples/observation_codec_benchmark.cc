// Measures what keeping Carcassonne observations as int8 costs and saves: the
// ObservationCodec that trajectories and the replay buffer use (see
// algorithms/alpha_zero_torch/observation_codec.h). Every spatial plane holds
// n / d for an int8 n and a denominator d fixed per plane, and decoding
// divides by d again, which gives back the same bits.
//
// It first checks on random games that the encoding is lossless, then times
// encoding, decoding, a learner-sized batch and, with --serialize_path, the
// libnop save/load that replay_buffer.data goes through.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/algorithms/alpha_zero_torch/observation_codec.h"
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

using algorithms::torch_az::CompactObservation;
using algorithms::torch_az::ObservationCodec;
using Clock = std::chrono::steady_clock;

constexpr int kPlaneSize = VIEW_SIZE * VIEW_SIZE;

// Keeps the optimizer from dropping work whose result is never read.
double g_sink = 0;

double Seconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

// Planes reported together, in plane order; each group shares one
// denominator (checked).
struct PlaneGroup {
  const char* name;
  int first;
  int count;
};

const std::vector<PlaneGroup>& PlaneGroups() {
  static const std::vector<PlaneGroup> groups = {
      {"occupied, terrain, shield, monastery, side links", kOccupiedPlane,
       kFrontierPlane - kOccupiedPlane},
      {"frontier, legal placements, last placed", kFrontierPlane,
       kLastPlacedPlane + 1 - kFrontierPlane},
      {"feature opens", kFeatureOpensPlane, 4},
      {"feature score", kFeatureScorePlane, 4},
      {"feature meeples (mine, theirs)", kFeatureMyMeeplesPlane, 8},
      {"feature signed score", kFeatureSignedScorePlane, 4},
      {"monastery coverage", kMonasteryCoveragePlane, 1},
      {"monastery owner", kMonasteryOwnerPlane, 1},
      {"field farmers (mine, theirs)", kFieldMyFarmersPlane,
       2 * HALF_EDGE_COUNT},
      {"field score", kFieldScorePlane, HALF_EDGE_COUNT},
      {"field size", kFieldSizePlane, HALF_EDGE_COUNT},
      {"field open cities", kFieldOpenCitiesPlane, HALF_EDGE_COUNT},
      {"inner field farmers (mine, theirs)", kInnerFieldMyFarmersPlane, 2},
      {"inner field score", kInnerFieldScorePlane, 1},
      {"inner field size", kInnerFieldSizePlane, 1},
      {"inner field open cities", kInnerFieldOpenCitiesPlane, 1},
      {"feature big meeple (mine, theirs)", kFeatureMyBigMeeplePlane, 8},
      {"monastery big meeple", kMonasteryBigMeeplePlane, 1},
      {"feature inn or cathedral", kFeatureInnCathedralPlane, 4},
      {"feature builder (mine, theirs)", kFeatureMyBuilderPlane, 8},
      {"field pig (mine, theirs)", kFieldMyPigPlane, 2 * HALF_EDGE_COUNT},
      // One group per kind: each has its own denominator.
      {"feature wine", kFeatureGoodsPlane, 4},
      {"feature wheat", kFeatureGoodsPlane + 4, 4},
      {"feature cloth", kFeatureGoodsPlane + 8, 4},
      {"piece spots (mine, theirs)", kMyPiecePlane, 2 * kPieceSpotPlanes},
      {"builder and pig tiles, dragon", kMyBuilderTilePlane,
       kSpatialPlanes - kMyBuilderTilePlane},
  };
  return groups;
}

// The groups must cover the spatial planes exactly once, in order, each with
// one denominator; a plane added to the observation without a group here
// stops the program.
void CheckPlaneGroups() {
  int next = 0;
  for (const PlaneGroup& group : PlaneGroups()) {
    if (group.first != next) {
      SpielFatalError(absl::StrFormat(
          "Plane group '%s' starts at plane %d, expected %d.", group.name,
          group.first, next));
    }
    for (int plane = group.first; plane < group.first + group.count;
         ++plane) {
      SPIEL_CHECK_EQ(ObservationPlaneDenominator(plane),
                     ObservationPlaneDenominator(group.first));
    }
    next += group.count;
  }
  SPIEL_CHECK_EQ(next, kSpatialPlanes);
}

struct Corpus {
  std::vector<std::unique_ptr<State>> states;
  std::vector<Player> players;
  std::vector<std::vector<float>> observations;
  std::vector<SideGroups> groups;  // To rotate the observations.
};

// Every decision state of random games, observed by the player to move, as
// PlayGame records them.
Corpus Collect(const Game& game, int games, std::mt19937* rng) {
  Corpus corpus;
  for (int g = 0; g < games; ++g) {
    std::unique_ptr<State> state = game.NewInitialState();
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        state->ApplyAction(SampleAction(state->ChanceOutcomes(), *rng).first);
        continue;
      }
      const Player player = state->CurrentPlayer();
      corpus.states.push_back(state->Clone());
      corpus.players.push_back(player);
      corpus.observations.push_back(state->ObservationTensor(player));
      corpus.groups.push_back(
          GetSideGroups(static_cast<const CarcassonneState&>(*state)));
      const std::vector<Action> legal = state->LegalActions();
      state->ApplyAction(legal[std::uniform_int_distribution<int>(
          0, legal.size() - 1)(*rng)]);
    }
  }
  return corpus;
}

// Checks every value of every observation against its plane's denominator,
// then the codec's round trip. Prints what it finds; false if anything is off.
bool Validate(const Corpus& corpus, const ObservationCodec& codec) {
  struct GroupStats {
    long min_n = 0;
    long max_n = 0;
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
      const float denominator = ObservationPlaneDenominator(group.first);
      GroupStats& s = stats[g];
      for (int plane = group.first; plane < group.first + group.count;
           ++plane) {
        for (int cell = 0; cell < kPlaneSize; ++cell) {
          const float value = observation[plane * kPlaneSize + cell];
          const float scaled = value * denominator;
          ++s.values;
          const long n = std::lround(scaled);
          if (!(scaled >= -128.0f && scaled <= 127.0f) ||
              static_cast<float>(n) / denominator != value) {
            ++s.failures;
            if (printed_failures++ < 10) {
              absl::PrintF(
                  "  NOT n/%g: plane %d (%s) x=%d y=%d value=%.9g\n",
                  denominator, plane, group.name, cell % VIEW_SIZE,
                  cell / VIEW_SIZE, value);
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
                 ObservationPlaneDenominator(group.first), s.min_n, s.max_n,
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

  // The codec itself: it must accept every observation and give back the
  // same bits.
  int64_t rejected = 0;
  int64_t mismatches = 0;
  std::vector<float> decoded(kObservationTensorSize);
  CompactObservation compact;
  for (const std::vector<float>& observation : corpus.observations) {
    if (!codec.TryEncode(observation, &compact)) {
      ++rejected;
      continue;
    }
    codec.Decode(compact, absl::MakeSpan(decoded));
    mismatches += std::memcmp(decoded.data(), observation.data(),
                              decoded.size() * sizeof(float)) != 0;
  }
  absl::PrintF(
      "codec: %d numerators + %d floats; rejected %d observations; %d "
      "decoded differently\n",
      codec.NumNumerators(), codec.NumFloats(), rejected, mismatches);
  ok &= rejected == 0 && mismatches == 0;
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
                        const ObservationCodec& codec, int repeats) {
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
  // A fresh CompactObservation each time, as PlayGame makes them.
  PrintPerObservation(
      "Encode (new object)", Time(repeats, [&]() {
        for (int i = 0; i < count; ++i) {
          const CompactObservation out = codec.Encode(corpus.observations[i]);
          g_sink += out.numerators[i % out.numerators.size()];
        }
      }),
      count);
  PrintPerObservation(
      "TryEncode (reused object)", Time(repeats, [&]() {
        int fits = 0;
        for (int i = 0; i < count; ++i) {
          fits += codec.TryEncode(corpus.observations[i], &scratch);
        }
        g_sink += fits;
      }),
      count);
  PrintPerObservation(
      "Decode (into a buffer)", Time(repeats, [&]() {
        for (int i = 0; i < count; ++i) {
          codec.Decode(compact[i], absl::MakeSpan(buffer));
        }
        sink_buffer();
      }),
      count);
}

// What a learner batch costs with augment_rotations, up to packing it for
// the device (the same either way): draw samples out of the buffer (a copy,
// as CircularBuffer::Sample makes), put each in a random orientation as a
// float observation, free what is left. Before, the float copy was rotated
// into a new vector; now, as ToTrainInputs in alpha_zero.cc does, a rotated
// sample is decoded into one scratch observation and rotated from there.
void TimeBatches(const Corpus& corpus,
                 const std::vector<CompactObservation>& compact,
                 const ObservationCodec& codec, int batch_size,
                 int batches_per_step, int repeats, std::mt19937* rng) {
  const int count = corpus.observations.size();
  enum Phase { kSample, kToFloats, kFree, kNumPhases };
  enum Path { kFloat, kCompact, kNumPaths };
  std::vector<double> seconds[kNumPaths][kNumPhases];
  std::uniform_int_distribution<int> pick(0, count - 1);
  std::uniform_int_distribution<int> rotation(0, kNumBoardRotations - 1);

  for (int r = -1; r < repeats; ++r) {  // r == -1 is the warm-up.
    std::vector<int> indices(batch_size);
    std::vector<int> rotations(batch_size);
    for (int b = 0; b < batch_size; ++b) {
      indices[b] = pick(*rng);
      rotations[b] = rotation(*rng);
    }

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
          if (rotations[b] == 0) continue;
          std::vector<float> rotated(kObservationTensorSize);
          RotateObservation(sampled[b], rotations[b],
                            corpus.groups[indices[b]],
                            absl::MakeSpan(rotated));
          sampled[b].swap(rotated);
        }
        g_sink += sampled[(r + 1) % batch_size][0];
        t2 = Clock::now();
      } else {
        std::vector<CompactObservation> sampled;
        sampled.reserve(batch_size);
        for (int index : indices) sampled.push_back(compact[index]);
        t1 = Clock::now();
        std::vector<float> scratch(kObservationTensorSize);
        std::vector<std::vector<float>> decoded;
        decoded.reserve(batch_size);
        for (int b = 0; b < batch_size; ++b) {
          if (rotations[b] == 0) {
            decoded.push_back(codec.Decode(sampled[b]));
            continue;
          }
          codec.Decode(sampled[b], absl::MakeSpan(scratch));
          decoded.emplace_back(kObservationTensorSize);
          RotateObservation(scratch, rotations[b], corpus.groups[indices[b]],
                            absl::MakeSpan(decoded.back()));
        }
        g_sink += decoded[(r + 1) % batch_size][0];
        t2 = Clock::now();
      }
      // The block's vectors are released as it ends, between t2 and t3.
      const Clock::time_point t3 = Clock::now();
      if (r >= 0) {
        seconds[path][kSample].push_back(Seconds(t0, t1));
        seconds[path][kToFloats].push_back(Seconds(t1, t2));
        seconds[path][kFree].push_back(Seconds(t2, t3));
      }
    }
  }

  absl::PrintF(
      "\n=== per batch of %d (random picks from %d observations), "
      "%d repeats ===\n",
      batch_size, count, repeats);
  absl::PrintF("%-24s %10s %10s %10s %10s %14s\n", "", "sample ms",
               "rot+dec ms", "free ms", "total ms",
               "s per step x" + std::to_string(batches_per_step));
  const char* names[kNumPaths] = {"float (before)", "int8 (ObservationCodec)"};
  for (int path = 0; path < kNumPaths; ++path) {
    double total = 0;
    double phase_ms[kNumPhases];
    for (int phase = 0; phase < kNumPhases; ++phase) {
      phase_ms[phase] = 1e3 * Summarize(seconds[path][phase]).median;
      total += phase_ms[phase];
    }
    absl::PrintF("%-24s %10.1f %10.1f %10.1f %10.1f %14.2f\n", names[path],
                 phase_ms[kSample], phase_ms[kToFloats], phase_ms[kFree],
                 total, total * batches_per_step / 1e3);
  }
}

void PrintMemory(const ObservationCodec& codec, int buffer_size) {
  const double float_bytes =
      sizeof(std::vector<float>) + kObservationTensorSize * sizeof(float);
  const double compact_bytes =
      sizeof(CompactObservation) + codec.CompactBytes();
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
  std::shared_ptr<const Game> game = LoadGame("carcassonne");
  const ObservationCodec codec(*game);
  std::mt19937 rng(absl::GetFlag(FLAGS_seed));

  const Clock::time_point start = Clock::now();
  Corpus corpus = Collect(*game, games, &rng);
  absl::PrintF("collected %d observations from %d games in %.1f s "
               "(observation: %d planes x %d x %d = %d floats)\n",
               corpus.observations.size(), games,
               Seconds(start, Clock::now()), kObservationPlanes, VIEW_SIZE,
               VIEW_SIZE, kObservationTensorSize);

  if (!Validate(corpus, codec)) {
    absl::PrintF("\nvalidation FAILED: the encoding is not lossless; "
                 "timings skipped.\n");
    return 1;
  }

  std::vector<CompactObservation> compact;
  compact.reserve(corpus.observations.size());
  for (const std::vector<float>& observation : corpus.observations) {
    compact.push_back(codec.Encode(observation));
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
