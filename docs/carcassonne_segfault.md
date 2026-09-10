# Carcassonne 訓練 Segmentation fault 檢查（2026-09-10）

已確認一個不需要 CUDA 就能重現的崩潰缺陷：`alpha_zero_torch` 的
`PlayGame()` 漏掉「抽牌直接進入終局」的情況。這個函式由 actor 與
evaluator 共用，因此任何一個工作執行緒觸發，都可能中斷整個訓練程序。
目前沒有 vast.ai 那次崩潰的 backtrace，不能斷言它是唯一原因。

## 觸發流程與修正

1. 牌堆剩最後一張牌，且棋盤沒有合法放置位置。
2. `Carcassonne::drawTile()` 消耗該牌，呼叫 `resolveNoMoreDraws()`，
   結算分數並切到 `PHASE_TERMINAL`。這是引擎正常的終局行為。
3. 舊版 `PlayGame()` 只在玩家動作後檢查終局；抽牌分支執行完便進入下一輪。
4. 終局的 `CurrentPlayer()` 是 `kTerminalPlayerId == -4`。
   舊版接著使用 `(*bots)[player]->MCTSearch(*state)`，即 `bots[-4]`，
   越界取得 bot 指標並呼叫成員函式。

修正是在每輪迴圈開頭先處理終局並保存 `Returns()`，另在 bot 存取前
檢查玩家索引。一般玩家動作後的終局優先順序、cutoff、觀測格式、
模型形狀及 checkpoint 格式均維持原有行為。

`carcassonne_test_utils.h` 保存一組全部經過合法動作檢查的固定棋局：
已放置 70 回合，剩全城堡牌（chance action `2`）。抽取後立即終局，
分數為 48:39，回報為 `[1, -1]`。測試不依賴隨機碰到這個罕見局面。

## 驗證

在本機 Ubuntu / WSL、GCC 13 上：

- 完整 `alpha_zero_torch_example` 訓練執行檔重新編譯與連結成功。
- 依專案 CMake 設定建置的 `torch_play_test` 與 `carcassonne_test`，
  正式 CTest 執行結果為 2/2 通過。
- 從原始版本取出的 `PlayGame()`，使用上述棋局與
  AddressSanitizer / UndefinedBehaviorSanitizer，確實產生 SEGV。
  堆疊為 `PlayGame -> MCTSBot::MCTSearch`，UBSan 回報無效的 `MCTSBot` 物件。
- 修正版通過同一測試，並通過初始即終局、玩家動作終局、cutoff 測試。
- 卡卡頌既有測試與新增的最後一張無法放置牌、clone、終局觀測測試，
  在上述 sanitizers 下通過。
- 引擎額外跑完 8 個執行緒、10,000 局完整遊戲，共 2,129,366 個動作、
  133,691 次 clone 比對。檢查牌堆計數、米寶數量、米寶觀測寫入範圍、
  棋盤與複本一致性，未發現 ASan / UBSan 錯誤。

Sanitizer 檢查涵蓋引擎、OpenSpiel 的卡卡頌介面、MCTS 及測試中的
`PlayGame()`；沒有對整套 LibTorch / CUDA 做 sanitizer instrumentation。
這些結果不等同於在 RTX 5090 上完成數小時訓練，也不是 ThreadSanitizer
對所有資料競爭的證明。

## 引擎與並行路徑檢查

- 合法棋局的 feature 索引落在 288 個 edge slots 內；棋盤座標由
  frontier 的邊界檢查產生，合法 tile moves 輸出陣列有 900 個元素。
- `MAX_FRONTIER_CELLS = 146` 符合 72 張相鄰方塊的 frontier 上限；
  修道院追蹤容量 6 與牌組內的 6 張修道院牌相符。
- 引擎 clone 以值複製各模組；牌型表為唯讀。每個 actor 使用自己的
  state 與 MCTS bot。`DisjointSet::find() const` 會做路徑壓縮，
  因此不能據此宣稱同一個 state 可安全供多執行緒同時讀取；目前的
  訓練呼叫路徑沒有找到這種共用方式。
- 模型由 `DeviceManager::DeviceLoan` 的 model mutex 保護；推論 cache
  的 Get / Set / Clear 也有鎖。未找到足以重現另一個崩潰的證據。

`--max_memory_mb` 是每個 MCTS 的樹節點預算，不是整個訓練程序的 RAM
上限。2048 actors、replay buffer、trajectory queue、推論 cache 會另外
占用記憶體。這組 replay buffer 的 float 觀測資料本身，填滿時即約
17.6 GiB（262,144 × 80 × 15 × 15 × 4 bytes）。目前 trajectory queue 容量使用
`replay_buffer_size / replay_buffer_reuse` 個「完整棋局」，這組參數為
87,381 局。這是獨立的記憶體壓力風險，不能用它取代已重現的越界原因。

## Ubuntu 更新後驗證

在 repository 根目錄依既有設定重新產生 build，再編譯相關目標：

```bash
source open_spiel/scripts/global_variables.sh
source venv/bin/activate
(
  cd build
  cmake -DPython3_EXECUTABLE=python \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_PREFIX_PATH="${LIBCXXWRAP_JULIA_DIR}" \
    -DBUILD_TYPE=Testing \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ../open_spiel
)
cmake --build build --target alpha_zero_torch_example carcassonne_test torch_play_test -j2
ctest --test-dir build --output-on-failure -R '^(carcassonne_test|torch_play_test)$'
```

接著使用原訓練指令驗證。如果仍有 Segmentation fault，可把原指令完整
接在 `gdb --args` 後面執行；保留所有原本的參數。在 gdb 裡輸入：

```text
set pagination off
set logging file carcassonne-gdb.txt
set logging enabled on
run
thread apply all bt
```

最後一行在程式因 SIGSEGV 停住後執行，用來取得真正出錯的 CPU / 引擎 /
LibTorch 呼叫位置。若需行號與區域變數，請依相同的專案設定另建帶
`-g -fno-omit-frame-pointer` 的版本。診斷用輸出及本機壓力測試產物保存在
`build/carcassonne_crash_check/`。
