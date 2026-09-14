# tools/ — AlphaCarcassonne 診斷工具

量測「observation / value head / trunk 到底有沒有在做我們以為它在做的事」。
完整分析寫在專案文件 **`feature-representation-redesign.md`**；這裡的每支程式
對應該文件的一個小節，原始碼開頭都註明了參考結果。

這些工具**不需要** OpenSpiel 本體、CMake 或 libtorch：
C++ 的部分只依賴 `open_spiel/games/carcassonne/game/` 底下的純引擎，
Python 的部分只需要 numpy。

---

## Quickstart

```bash
cd tools
make            # 編譯 6 支 C++ 診斷程式到 build/
make run        # 依序跑一遍，輸出存到 results/（含 occupancy.csv）

# 權重分析（需要一個 checkpoint）
cd results
python3 ../readckpt.py /path/to/checkpoint-193.pt ck.npz
python3 ../check_value_head.py ck.npz          # 讀同目錄的 occupancy.csv
python3 ../erf.py            ck.npz
```

`make run` 全跑一遍約數分鐘（`erf.py` 另外約 1 分鐘）。
每支程式都吃一個「局數」參數，想快一點就 `./build/diag_value_gap 300`。

---

## 工具一覽

| 程式 | 量什麼 | 文件小節 | checkpoint-193 / 隨機對局的參考結果 |
|---|---|---|---|
| `diag_value_gap` | 每個決策點的 `banked` vs `static`（= banked + 終局待結分）分差，預測最終勝者的準確率 | §2.1 | **最後一手 0.717 vs 0.992**；E\|pending\| 穩定在 6 分，而平均最終分差只有 8.62 |
| `diag_local_decomp` | 「完美 per-cell CNN + 全域求和」在現有 80 plane 下能否重建 pending | §2.2 | **exact 99.17%**、MAE 0.008 → 資訊是夠的，問題在架構與監督 |
| `diag_board` / `diag_board25` | bounding box、每手合法落點、跨度分佈、終局待結分組成 | §3.6 | 15×15 有 **97.7%** 的對局碰到邊界；真實跨度 **31%** 超過 15 → 建議 21 |
| `diag_opens` | (A) 有盾磚型的城市元件數 (B) `opens` vs 最終封口率 | §3.2 / §3.3 | (A) 全部只有 1 個 → 盾不用廣播 (B) opens=1 封口率是 opens=2 的 **7.6 倍** → opens 必須加 |
| `diag_occupancy` | 每格「有磚」的機率熱圖，並輸出 `occupancy.csv` | §2.4-c | 中心 100% / 角 0.5% → **185×** 失衡 |
| `readckpt.py` | 純 numpy 讀 libtorch checkpoint → `.npz` | — | 589,217 個 float，對得上 2.43MB |
| `check_value_head.py` | value head 三項健康檢查 | §2.4-c | `β/\|γ\|=0.07`（47% 被 ReLU 截斷）、`cos(eff,1)=0.048`、占用率 ≤5% 的格子權重還在初始值 |
| `erf.py` | trunk 有效感受野、端到端敏感度、有號探針 | §2.4-b | ERF 在 r=7 只剩 **0.29**、r=10 剩 0.047；有號探針只有 **89/225** 格符號正確 |

---

## 每支程式在回答什麼問題

**`diag_value_gap`** — 觀測裡唯一的分數訊號是 `kScoreDiffPlane`（已入袋分差）。
它離「足以判斷勝負」還差多少？答案：最後一手只有 71.7% 準，加上終局待結分才 99.2%。
那 28 個百分點就是 value head 該補、但補不起來的缺口。

**`diag_local_decomp`** — 那缺口是「資訊不在輸入裡」還是「網路沒學會」？
模擬一個完美的 per-cell CNN，只用網路看得到的東西（邊地形、shield、
磚內 link、以及 `LogModule::getMeepleMap` 已廣播的元件 meeple 數），
每格輸出 `sign × (1 + shield)` 再求和。99.17% 完全吻合 →
**資訊是夠的**，問題在架構（§2.4-b/c）與監督（§2.4-a）。

**`diag_board` / `diag_board25`** — `BOARD_SIZE=15` 在起始磚 (7,7) 各方向只有 7 格。
把引擎複製一份改成 25 再跑同樣的對局，就能看出被裁掉多少。
順帶：邊界會製造**永遠填不掉的假開口**，污染 `opens` 的語意。

**`diag_opens`** — 決定「元件盾牌數」和「opens」該不該廣播。
盾：所有有盾磚型都只有一個城市元件，歸屬無歧義，本地 plane 就夠 → **不用加**。
opens：無法由局部推導（每格只看得到自己那個開口，元件總數要沿元件求和），
而且是封口機率的強單調預測子 → **必須加**。

**`diag_occupancy`** — 現在的 value head 是 `flatten(225) → Linear(225,32)`，
每個位置有自己獨立的權重，而那個權重只有在該格有磚時才拿得到梯度。
中心 100%、角落 0.5%，差 185 倍。配 `check_value_head.py` 看實際後果。

**`check_value_head.py`** — 三個問題：ReLU 有沒有在砍格子（`β/|γ|`）、
head 有沒有在做空間求和（`cos(eff, 1)`）、每個位置的權重有沒有被訓練到
（`W1` per-position std vs 占用率）。三個答案都是「沒有」。

**`erf.py`** — 理論感受野半徑 16 覆蓋整個 15×15，那 trunk 為什麼不自己把
全域量算好廣播出去？因為**卷積沒有求和運算子，只有局部加權平均**：
把 k 個 3×3 核卷在一起是 σ≈√(2k/3) 的離散高斯，不是平坦核
（Luo et al. 2016）。這支把 17 個核用 FFT 卷在一起直接量，並加上 head 的
位置權重算端到端敏感度，最後用一個語意明確的方向（「我多一個 meeple」）
做有號探針。

---

## 備註

- **`#define private public`**：`common.hpp` 用這招直接存取 `Carcassonne` 的
  `features` / `monasteries` / `logs`。診斷工具需要看到引擎內部，但不該為了診斷
  而放寬正式程式碼的封裝。（嚴格說是 UB，實務上對這個用途沒問題。）
- **線性化的 caveat**：`erf.py` 把 ReLU 當成恆等、BN 折進 conv、residual 當成
  `I + W₂∘W₁`。這是 Jacobian 的「全開」線性化，不是任何真實狀態下的 Jacobian。
  ReLU 只會削弱傳遞，所以 ERF 那張表是**上界**。
- **隨機對局的 caveat**：所有 C++ 診斷都用均勻隨機下法。結構性的結論
  （資訊夠不夠、跨度會不會超出、opens 的單調性）不受影響，但**絕對數值**
  （例如封口率）跟真實 self-play 分佈不同，會被低估。
- **改成 global pooling 之後**（`model.cc` 的 value head 已改成 32 filters +
  mean ⊕ max pooling）：兩支工具都會依 checkpoint 的形狀自動判斷架構。
  - `check_value_head.py`：舊架構的 `cos(eff,1)` 與「每格權重 vs 占用率」失去意義
    （位置權重被硬編碼了），改成印 `value_conv` 各通道的 BN γ（大部分塌到 0 表示
    通道用不完，可以減）、每個通道在 mean / max 兩個分支被讀出的程度，以及線性化後
    的正負號（有號求和需要「加分」與「扣分」兩種通道同時存在）。
  - `erf.py`：(2)(3) 改用 mean pooling 分支線性化（每格權重 1/(H·W)），max 分支
    不可線性化、沒有計入；棋盤邊長取自 `occupancy.csv`，沒有就用 15。
    有號探針 (3) 的正號格數應該從 89/225 大幅上升（CLAUDE.md §8 第 4 項）。
  - head 所在的 `layers.N.` 由 residual block 數推出，不再寫死 `layers.9.`
    （`nn_depth` ≠ 8 的 checkpoint 也能讀）。
- **舊 checkpoint 不能載入新架構**：`VPNetModel::LoadCheckpoint` 會檢查形狀並直接報錯，
  從頭訓練（`init_checkpoint` 留空）。但工具仍可讀舊 checkpoint 做對照。
- **`make clean`** 會清掉 `build/` 和 `results/`（兩者都已在 `.gitignore`）。
