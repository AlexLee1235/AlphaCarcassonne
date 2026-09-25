#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>

#include "open_spiel/algorithms/alpha_zero_torch/device_manager.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/games/carcassonne/carcassonne.h"
#include "open_spiel/games/carcassonne/game/game.hpp"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

namespace {

using json = nlohmann::json;

constexpr double kDefaultUctC = 2.0;
constexpr int kDefaultRolloutCount = 10;
constexpr int kDefaultMaxMemoryMb = 1000;
constexpr bool kDefaultSolve = true;

std::string ShapeString(const std::vector<int> &shape) {
    std::string result = "[";
    for (int i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            result += ",";
        }
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

int ShapeSize(const std::vector<int> &shape) {
    int size = 1;
    for (const int dim : shape) {
        if (dim > 0) {
            size *= dim;
        }
    }
    return size;
}

std::string EnvString(const char *name, const std::string &default_value) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return value;
}

int EnvInt(const char *name, int default_value) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return std::stoi(value);
}

int PositiveEnvInt(const char *name, int default_value) {
    return std::max(1, EnvInt(name, default_value));
}

int FallbackSimulations() {
    const char *az_value = std::getenv("CARCASSONNE_AZ_SIMULATIONS");
    if (az_value != nullptr && az_value[0] != '\0') {
        return std::max(1, std::stoi(az_value));
    }
    return PositiveEnvInt("CARCASSONNE_MCTS_SIMULATIONS", 200);
}

std::tuple<int, int, int> DecodeTileAction(open_spiel::Action action) {
    if (action < 0 || action >= open_spiel::carcassonne::kTileActionCount) {
        throw std::runtime_error("Bot did not return a tile placement action.");
    }
    const int rot = action % 4;
    action /= 4;
    const int x = action % BOARD_SIZE;
    const int y = action / BOARD_SIZE;
    return {x, y, rot};
}

int DecodeMeepleAction(open_spiel::Action action) {
    if (action < open_spiel::carcassonne::kMeepleActionOffset ||
        action >= open_spiel::carcassonne::kNumDistinctPlayerActions) {
        throw std::runtime_error("Bot did not return a meeple action.");
    }
    return action - open_spiel::carcassonne::kMeepleActionOffset - 1;
}

open_spiel::algorithms::torch_az::ModelConfig LoadModelConfig(const std::string &path,
                                                              const std::string &filename) {
    const std::string full_path = path + "/" + filename;
    std::ifstream file(full_path);
    if (!file) {
        throw std::runtime_error("AlphaZero model config not found: " + full_path);
    }

    open_spiel::algorithms::torch_az::ModelConfig config;
    file >> config;
    if (!file) {
        throw std::runtime_error("Could not parse AlphaZero model config: " + full_path);
    }
    return config;
}

// Which side the network's value is for. Runs trained with
// --value_is_current_player=true (0916 onwards) predict the value for the
// player to move; older runs predict it for player 0. Reading it wrong makes
// the search maximise the opponent's value whenever the bot is not player 0.
// Order: CARCASSONNE_AZ_VALUE_IS_CURRENT_PLAYER, then <az_path>/config.json
// (written by the training run), then false.
bool ResolveValueIsCurrentPlayer(const std::string &az_path) {
    const std::string forced = EnvString("CARCASSONNE_AZ_VALUE_IS_CURRENT_PLAYER", "");
    if (forced == "true" || forced == "1") {
        return true;
    }
    if (forced == "false" || forced == "0") {
        return false;
    }
    if (!forced.empty()) {
        throw std::runtime_error("CARCASSONNE_AZ_VALUE_IS_CURRENT_PLAYER must be true or false, got " + forced);
    }

    std::ifstream file(az_path + "/config.json");
    if (!file) {
        return false;
    }
    const json config = json::parse(file, /*cb=*/nullptr, /*allow_exceptions=*/false);
    if (!config.is_object()) {
        return false;
    }
    const auto it = config.find("value_is_current_player");
    return it != config.end() && it->is_boolean() && it->get<bool>();
}

void ValidateModelConfig(const open_spiel::Game &game,
                         const open_spiel::algorithms::torch_az::ModelConfig &config) {
    const std::vector<int> game_shape = game.ObservationTensorShape();
    if (config.observation_tensor_shape != game_shape) {
        throw std::runtime_error("AlphaZero model observation shape " +
                                 ShapeString(config.observation_tensor_shape) +
                                 " does not match current game shape " + ShapeString(game_shape) +
                                 ". Use a checkpoint trained after the latest observation-plane changes.");
    }

    const int config_size = ShapeSize(config.observation_tensor_shape);
    if (config_size != game.ObservationTensorSize()) {
        throw std::runtime_error("AlphaZero model observation size " + std::to_string(config_size) +
                                 " does not match current game size " +
                                 std::to_string(game.ObservationTensorSize()) + ".");
    }

    if (config.number_of_actions != game.NumDistinctActions()) {
        throw std::runtime_error("AlphaZero model action count " + std::to_string(config.number_of_actions) +
                                 " does not match current game action count " +
                                 std::to_string(game.NumDistinctActions()) + ".");
    }
}

class CarcassonneBotCli {
  public:
    CarcassonneBotCli()
        : game_(std::make_shared<open_spiel::carcassonne::CarcassonneGame>(open_spiel::GameParameters{})) {}

    json Handle(const json &request) {
        const std::string command = request.value("cmd", "");
        if (command == "reset") {
            mirror_ = Carcassonne();
            return Ok();
        }
        if (command == "info") {
            return Info();
        }
        if (command == "apply_draw") {
            mirror_.drawTile(request.at("type").get<int>());
            return Ok();
        }
        if (command == "apply_tile") {
            mirror_.placeTile(request.at("x").get<int>(), request.at("y").get<int>(), request.at("rot").get<int>());
            return Ok();
        }
        if (command == "apply_meeple") {
            mirror_.placeMeeple(request.at("pos").get<int>());
            return Ok();
        }
        if (command == "choose") {
            return Choose(request);
        }
        throw std::runtime_error("Unknown command: " + command);
    }

  private:
    json Ok() const { return json{{"ok", true}}; }

    json Info() const {
        const std::vector<int> shape = game_->ObservationTensorShape();
        json info{{"ok", true},
                  {"observation_shape", shape},
                  {"observation_tensor_size", game_->ObservationTensorSize()},
                  {"num_distinct_actions", game_->NumDistinctActions()}};
        const std::string az_path = EnvString("CARCASSONNE_AZ_PATH", "");
        if (!az_path.empty()) {
            info["value_is_current_player"] = ResolveValueIsCurrentPlayer(az_path);
        }
        return info;
    }

    open_spiel::carcassonne::CarcassonneState MakeState() const {
        return open_spiel::carcassonne::CarcassonneState(game_, mirror_);
    }

    open_spiel::Action ChooseRandom(const open_spiel::State &state, int seed) const {
        auto bot = open_spiel::MakeUniformRandomBot(state.CurrentPlayer(), seed);
        return bot->Step(state);
    }

    open_spiel::Action ChooseMcts(const open_spiel::State &state, int simulations, int seed) const {
        auto evaluator =
            std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(kDefaultRolloutCount, seed);
        open_spiel::algorithms::MCTSBot bot(*game_, evaluator, kDefaultUctC, simulations, kDefaultMaxMemoryMb,
                                            kDefaultSolve, seed, /*verbose=*/false);
        return bot.Step(state);
    }

    void EnsureAlphaZero() {
        if (az_evaluator_ != nullptr) {
            return;
        }

        const std::string az_path = EnvString("CARCASSONNE_AZ_PATH", "");
        if (az_path.empty()) {
            throw std::runtime_error("CARCASSONNE_AZ_PATH must be set for AlphaZero.");
        }
        const std::string graph_def = EnvString("CARCASSONNE_AZ_GRAPH_DEF", "vpnet.pb");
        ValidateModelConfig(*game_, LoadModelConfig(az_path, graph_def));

        az_device_manager_ = std::make_unique<open_spiel::algorithms::torch_az::DeviceManager>();
        az_device_manager_->AddDevice(
            open_spiel::algorithms::torch_az::VPNetModel(*game_, az_path, graph_def, "/cpu:0"));
        az_device_manager_->Get(0, 0)->LoadCheckpointWeightsOnly(EnvInt("CARCASSONNE_AZ_CHECKPOINT", -1));
        az_evaluator_ = std::make_shared<open_spiel::algorithms::torch_az::VPNetEvaluator>(
            az_device_manager_.get(), PositiveEnvInt("CARCASSONNE_AZ_BATCH_SIZE", 1),
            PositiveEnvInt("CARCASSONNE_AZ_THREADS", 1), PositiveEnvInt("CARCASSONNE_AZ_CACHE_SIZE", 16384),
            PositiveEnvInt("CARCASSONNE_AZ_CACHE_SHARDS", 1), /*batch_wait_ms=*/1,
            ResolveValueIsCurrentPlayer(az_path));
    }

    open_spiel::Action ChooseAlphaZero(const open_spiel::State &state, int simulations, int seed) {
        EnsureAlphaZero();
        open_spiel::algorithms::MCTSBot bot(*game_, az_evaluator_, kDefaultUctC, simulations, kDefaultMaxMemoryMb,
                                            kDefaultSolve, seed, /*verbose=*/false,
                                            open_spiel::algorithms::ChildSelectionPolicy::PUCT, 0, 0,
                                            /*dont_return_chance_node=*/true);
        return bot.Step(state);
    }

    json Choose(const json &request) {
        auto state = MakeState();
        if (state.IsTerminal()) {
            throw std::runtime_error("Cannot choose an action from a terminal state.");
        }
        if (state.IsChanceNode()) {
            throw std::runtime_error("Cannot choose an action from a chance state.");
        }

        const std::string bot = request.value("bot", "");
        const int seed = request.value("seed", EnvInt("CARCASSONNE_BOT_SEED", 1));
        open_spiel::Action action = open_spiel::kInvalidAction;
        if (bot == "random") {
            action = ChooseRandom(state, seed);
        } else if (bot == "mcts") {
            action = ChooseMcts(state, request.value("simulations", PositiveEnvInt("CARCASSONNE_MCTS_SIMULATIONS", 200)),
                                seed);
        } else if (bot == "alphazero" || bot == "az") {
            action = ChooseAlphaZero(state, request.value("simulations", FallbackSimulations()), seed);
        } else {
            throw std::runtime_error("Unknown bot: " + bot);
        }

        if (mirror_.current_phase == PHASE_TILE) {
            auto [x, y, rot] = DecodeTileAction(action);
            return json{{"ok", true}, {"kind", "tile"}, {"x", x}, {"y", y}, {"rot", rot}};
        }
        if (mirror_.current_phase == PHASE_MEEPLE) {
            return json{{"ok", true}, {"kind", "meeple"}, {"pos", DecodeMeepleAction(action)}};
        }
        throw std::runtime_error("Bot returned an action for an unsupported phase.");
    }

    std::shared_ptr<const open_spiel::Game> game_;
    Carcassonne mirror_;
    std::unique_ptr<open_spiel::algorithms::torch_az::DeviceManager> az_device_manager_;
    std::shared_ptr<open_spiel::algorithms::torch_az::VPNetEvaluator> az_evaluator_;
};

json Error(const std::exception &exc) { return json{{"ok", false}, {"error", exc.what()}}; }

} // namespace

int main() {
    CarcassonneBotCli cli;
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            if (line.empty()) {
                continue;
            }
            std::cout << cli.Handle(json::parse(line)).dump() << std::endl;
        } catch (const std::exception &exc) {
            std::cout << Error(exc).dump() << std::endl;
        }
    }
    return 0;
}
