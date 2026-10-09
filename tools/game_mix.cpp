// game_mix —— 產生混合規則訓練的遊戲清單,給 alpha_zero_torch_example 的 --game_mix。
//
// 列出所有規則組合:旅館與大教堂、商人與建築師、公主與龍各 off / tiles / on,河流 off / on,共 54 種。
// 每種用隨機對局量平均決策數(放磚、放 meeple、選格、選位置、龍步都算),再定權重:
//   states(預設)  ∝ 1 / 平均決策數:每種在 replay buffer 佔一樣多的狀態,短的組合局數較多。
//   games          都一樣:每種抽到的局數一樣,長的組合佔的狀態較多。
// 權重正規化成平均 1。自我對弈的局長跟隨機對局不同(cutoff 會提早結束),
// 實際的分配看 learner.jsonl 的 "mix" 欄,要調就改 run 目錄 config.json 的權重再續訓。
//
// 用法: ./build/game_mix [每種局數=100] [states|games] > mix.json
//   stdout 是 JSON 清單([{"game", "weight", "decisions"}, ...]),stderr 印每種佔的局數與狀態比例。
#include "common.hpp"

#include <cstring>

int main(int argc, char **argv) {
    const int games_per_rules = argc > 1 ? atoi(argv[1]) : 100;
    const std::string mode = argc > 2 ? argv[2] : "states";
    if (games_per_rules <= 0 || (mode != "states" && mode != "games")) {
        fprintf(stderr, "用法: %s [每種局數=100] [states|games] > mix.json\n", argv[0]);
        return 2;
    }

    const char *const modes[] = {"off", "tiles", "on"};
    std::vector<diag::GameRules> all_rules;
    for (int ic = 0; ic < 3; ++ic) {
        for (int tb = 0; tb < 3; ++tb) {
            for (int river = 0; river < 2; ++river) {
                for (int pd = 0; pd < 3; ++pd) {
                    const std::string text = std::string("inns_cathedrals=") + modes[ic] + ",traders_builders=" +
                                             modes[tb] + ",river=" + (river ? "on" : "off") +
                                             ",princess_dragon=" + modes[pd];
                    diag::GameRules rules;
                    std::string error;
                    if (!diag::ParseGameString(text, &rules, &error)) {
                        fprintf(stderr, "%s\n", error.c_str());
                        return 1;
                    }
                    all_rules.push_back(rules);
                }
            }
        }
    }

    std::mt19937 rng(20261009);
    const int count = static_cast<int>(all_rules.size());
    std::vector<double> decisions(count);
    for (int i = 0; i < count; ++i) {
        long long total = 0;
        for (int g = 0; g < games_per_rules; ++g) {
            Carcassonne game = diag::NewGame(all_rules[i]);
            while (game.current_phase != PHASE_TERMINAL) {
                if (game.current_phase == PHASE_CHANCE) {
                    if (!diag::SampleDraw(game, rng)) break;
                    continue;
                }
                if (!diag::RandomStep(game, rng)) break;
                ++total;
            }
        }
        decisions[i] = static_cast<double>(total) / games_per_rules;
    }

    std::vector<double> weights(count);
    double weight_sum = 0;
    for (int i = 0; i < count; ++i) {
        weights[i] = mode == "states" ? 1.0 / decisions[i] : 1.0;
        weight_sum += weights[i];
    }
    double state_sum = 0;
    for (int i = 0; i < count; ++i) {
        weights[i] *= count / weight_sum;
        state_sum += weights[i] * decisions[i];
    }

    fprintf(stderr, "%d 種規則組合,每種 %d 局隨機對局,權重照 %s\n", count, games_per_rules, mode.c_str());
    fprintf(stderr, "  %-8s %-6s %-6s  %s\n", "決策數", "局數", "狀態", "規則");
    printf("[\n");
    for (int i = 0; i < count; ++i) {
        printf("  {\"game\": \"%s\", \"weight\": %.4f, \"decisions\": %.1f}%s\n",
               diag::OpenSpielGameString(all_rules[i]).c_str(), weights[i], decisions[i], i + 1 < count ? "," : "");
        fprintf(stderr, "  %8.1f %5.2f%% %5.2f%%  %s\n", decisions[i], 100 * weights[i] / count,
                100 * weights[i] * decisions[i] / state_sum, diag::GameName(all_rules[i]).c_str());
    }
    printf("]\n");
    return 0;
}
