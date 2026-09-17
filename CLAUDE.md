# AlphaCarcassonne 診斷 + Feature / 訓練重新設計

> **依據**
> - 原始碼：`open_spiel/games/carcassonne/{carcassonne.h,carcassonne.cc,game/*}`、
>   `open_spiel/algorithms/alpha_zero_torch/{model.cc,model.h,alpha_zero.cc,vpevaluator.cc,vpnet.cc}`、
>   `open_spiel/algorithms/mcts.cc`
> - 訓練紀錄：`0904/learner (7).jsonl`（step 193）、`0904/config.json`；
>   **`0916/learner.jsonl`（step 62）、`0916/config.json`** —— 做完 §7 #1 之後的對照組
> - **權重實測**：`0904/checkpoint-193.pt`、**`0916/checkpoint-62.pt`**（用 `tools/readckpt.py` 讀出，純 numpy）
> - **模擬實測**：`tools/` 底下的 6 支 C++ 診斷程式（見附錄）
>
> **設定**：2 人、基本版、**關閉農夫**。
>
> **修訂紀錄**
> | 版本 | 內容 |
> |---|---|
> | v1 | 主張「CNN 在結構上算不出終局待結分」。**錯誤**，v2 已整節重寫 |
> | v2 | 修正 §2；刪除「元件大小 / 盾牌數 / 封口旗標需要廣播」三個錯誤建議 |
> | v3 | §2.4-c 從論證升級為權重實測；加入參數預算與占用率熱圖 |
> | v4 | 加入 trunk 有效感受野的實測（§2.4-b）；新增 §2.4-d 損失函數；修正 §3.8 對和局的誇大；把 WDL 從落地順序第 1 步拆出（§7）；所有量測程式收進 `tools/`（見附錄） |
> | v5 | 新增 §3.10（meeple 動作空間：去重是對的、索引語義才是弱點）；**修正 §5 的 meeple head** —— 原本的 `pooled → MLP` 設計不好，改成空間讀取（新增 §5.4）；§6.5 補上旋轉增強必須置換 meeple 邊動作 |
> | **v7（本版）** | 新增 **§5.5 policy head 改法整理版**（含順序對齊、gpool 偏置、與旋轉增強的關係、**#5 必須排在 #4 之後**）與 **§9.4 程式碼**；§2.4-e 補上 `0916` 實測的 `policy_linear` 列範數衰減（邊角剩 15%、起始磚格歸零），把 Adam+耦合 L2 的機制講成「固定步長收縮」。**第一次實機驗證**：`0916` 跑了 §7 #1（value head 32 通道 + global pooling）+ 旋轉增強，**eval 由 −0.93 翻正到 +0.14**（新增 §1.5）。更正 v1–v6 的兩處記述：`temperature_drop` 在 0904 就已經是 10（不是 30），以及 0904 的真正病灶是**過擬合**（train MSE 0.32 vs self-play raw accuracy 0.50–0.60），不只是欠擬合。§7 #1 標記完成、§8 測試 5 標記通過 |
| v6 | **移除分差的 bucket one-hot**，改多尺度 clip（§4.3、§9.2）；§2.4-c 補上 **Adam + 耦合 L2** 的機制，解釋那 93 格為何低於初始值；**更正 §6.6** —— optimizer 是 Adam（1e-4 屬正常值），且 gradient update 是 24,704 次 vs AGZ 700k（差 28× 不是 3600×）；新增 §6.5b（`reuse` 不要 ×4）與 §6.7 的 held-out loss；§6.5 標記為已實作並驗證；§7 重排 |

---

## 0. TL;DR

| # | 問題 | 證據 | 嚴重度 |
|---|---|---|---|
| 1 | **value 的「和」只存在於最後一步**，trunk 傳不動、policy head 看不到。實測 trunk 有效感受野在 r=7 已衰減到 **0.29**、r=10 剩 **0.047** | §2.4-b | 最高 |
| 2 | **value head 的形狀跟「有號空間求和」不合**：1 通道 + ReLU 砍掉 **47%** 的格子；有效讀出與均勻求和的 cosine 只有 **0.048**；93 個週邊格子的權重**還停在初始值** | §2.4-c | 最高 |
| 3 | **沒有對「數值」的直接監督**：唯一訊號是對 ±1 做 MSE，中間隔著硬閾值 | §2.4-a | 高 |
| 4 | **`opens` 完全不在觀測裡** —— 唯一無法由局部推導的全域量，而封口 ×2 是最大的單一分數槓桿 | §3.3 | 高 |
| 5 | **15×15 棋盤 31% 的對局會被截斷**，還製造永遠填不掉的假開口 | §3.6 | 中高 |
| 6 | **68% 的參數卡在 policy head 的一層全連接**，而且實測**正在歸零**：`policy_linear` 的 per-(x,y) 列範數，邊角只剩初始值的 **15%**，起始磚那格是**乾淨的 0.000** | §2.4-e、§5.5 | **最高（未解）** |
| 7 | **tanh 飽和在錯得最離譜時把梯度掐掉**（stage 3 的 max 處衰減 1.1 萬倍） | §2.4-d | 中 |
| 8 | 62% 的輸入張量是空間常數廣播平面 | §3.5 | 中 |
| 9 | `temperature_drop` 單位錯誤、`policy_alpha=1.0` 過大、800 sims 幾乎沒有深度 | §6 | 中 |

**最重要的單一指標（已更新）**：

| 版本 | 訓練量 | eval vs 800-sim rollout MCTS |
|---|---|---|
| `0904`（原始架構） | step 193 / 16.1M states | **−0.656**（193 步從沒翻正過，最好 −0.656） |
| `0916`（做完 #1 + 旋轉增強） | step 62 / **4.07M** states | **+0.14**（step 37 附近翻正，之後 20 步穩定在 +0.12～+0.18） |

**用 1/4 的資料量，從 −0.93 翻到 +0.14。** 上表 1–9 項裡，只有第 2 項（value head 形狀）被修掉。
其餘 8 項仍然成立，見 §1.5 的殘餘頭空間量測。

---

## 1. 現況指標

`learner (7).jsonl` step 193（462 局、113,344 條 trajectory、16.1M states）：

| stage（0=開局, 6=最後一個決策） | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| value_accuracy | 0.548 | 0.500 | **0.491** | 0.587 | 0.593 | **0.751** | 0.970 |
| \|value\| 平均 | 0.195 | 0.468 | 0.493 | 0.601 | 0.630 | 0.525 | 0.802 |

其他：`policy_kl = 0.436`、`policy_pred_entropy 1.504 > policy_target_entropy 1.063`（policy 也 underfit）、
`value loss = 0.256`、outcomes 237 / 209 / 16（和局 3.5%）、game_length 141.9。

**「最後一手 24% 判反」= stage 5 的 0.751。** stage 用 `(states.size()-1)*stage/6` 取樣，
141.9 個決策狀態下 stage 5 ≈ 距終局約 12 手；stage 6 才是真正最後一手（0.970）。

兩件事比 24% 更嚴重：

- **stage 1–4 都在 0.49–0.59（擲硬幣），但 |value| 已經 0.47–0.63。** 自信滿滿且零資訊。
- `value_accuracies` 統計的是 **MCTS root value**，不是網路原始輸出。stage 6 的 0.970 主要是搜尋走到終局換來的。
  **建議另外記錄 raw NN value 的 sign accuracy**，否則分不出「網路不會」還是「搜尋不夠」。
  → **已實作**（`0916` 的 `raw_value_accuracy` / `raw_value_prediction`），見 §1.5。

---

## 1.5 第一次實機驗證：`0904` → `0916`

### 1.5.1 到底改了什麼

實際比對 `config.json` 與 `checkpoint-62.pt` 的張量形狀（`tools/readckpt.py`）：

| 項目 | 0904 | 0916 |
|---|---|---|
| **value head** | `conv(32→1,k=1)` → BN(1) → `view(225)` → `Linear(225→32)` → `Linear(32→1)` | **`conv(32→1)` → `conv(32→32,k=1)`；flatten(2) → `mean ‖ amax`(dim=2) → `Linear(64→256)` → `Linear(256→1)`** |
| value head 參數 | 7,303（1.2%） | 18,081（3.0%） |
| `augment_rotations` | 無 | **true** |
| `replay_buffer_reuse` | 3 | 4 |
| `evaluators` | 1 | 4 |
| `inference_batch_size` / `cache` | 64 / 262,144 | 128 / 2,621,440 |
| 其餘 | `nn_width 32`, `nn_depth 8`, `lr 1e-4`, `weight_decay 1e-4`, `policy_alpha 1.0`, `max_simulations 800`, `temperature_drop 10`, `train_batch_size 2048` | **完全相同** |

**沒有動到的**：input conv 仍是 `(32, 80, 3, 3)` → 觀測仍是 80 planes、`BOARD_SIZE` 仍是 15；
policy head 仍是 `Linear(450→906)`（408,606 參數 = **68.1%**）；optimizer 仍是 `Adam` + `losses()` 裡的顯式 L2。

也就是說 **§7 的 #2、#3、#4、#5、#8、#9、#10、#11 一個都還沒做**。翻正只靠 #1 + 旋轉增強。

> 修正 v1–v6 的記述：`temperature_drop` 在 0904 就已經是 10，**不是 30**。§6.1 對這個值的批評仍然成立
> （10 個 history entry ≈ 3.3 手 ≈ 全局的 2.3%），只是它不是這一輪的變因。

### 1.5.2 對照表

同訓練量對照（0904 step 46 = 4.02M states vs 0916 step 62 = 4.07M states）：

| stage（0=開局, 6=最後一手） | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 0904 `value_accuracy` | 0.555 | 0.502 | 0.570 | 0.586 | 0.627 | 0.734 | 0.964 |
| **0916 `value_accuracy`** | 0.511 | **0.576** | **0.615** | **0.671** | **0.760** | **0.859** | 0.976 |
| 差 | −4.4 | **+7.4** | **+4.5** | **+8.5** | **+13.3** | **+12.5** | +1.2 |
| 0916 `raw_value_accuracy`（網路原始輸出） | 0.504 | 0.552 | 0.561 | 0.591 | 0.706 | 0.848 | **0.918** |

`0904` 跑到 **step 193（16.1M states，4 倍資料）**，value_accuracy 是
`[0.548, 0.500, 0.491, 0.587, 0.593, 0.751, 0.970]` —— **跟 step 46 沒有差別**。
這就是 §2.4-c 說的：1 通道瓶頸不是「學得慢」，是**學不動**。

### 1.5.3 value loss 上升 0.316 → 0.612 是好事

| | 0904 @ 4.0M | 0916 @ 4.1M |
|---|---|---|
| train `value` loss（buffer 上的 MSE） | **0.316** | **0.612** |
| self-play `raw_value_accuracy` stage 1–4 | ~0.50–0.59（當時沒記，用 MCTS root 值近似） | 0.552 / 0.561 / 0.591 / 0.706 |
| `policy_kl` | 0.450 | **0.345** |
| `policy_target_entropy` | 1.095 | 1.275 |

target 是 `Returns() = ±1`（`model.cc` 裡是純 `MSELoss`）。
**MSE 0.316 對 ±1 的 target，意味著網路在 buffer 上的輸出 |v| ≈ 0.8 且 sign 正確率 ≈ 95%。**
但同一時刻 self-play 的新鮮局面只有 50–60% 正確。

> **0904 的病不只是欠擬合，是嚴重過擬合**：train MSE 0.32 / 有效泛化 55%。
> 這也補上了 §6.5 那個問題的答案 ——「為什麼加了旋轉增強不是直接快四倍」：
> 它的收益不在 sample efficiency 的線性倍數，而在**把 buffer 的有效多樣性 ×4、壓掉記憶**。
> 0916 的 train loss 升高、泛化大幅改善，是教科書式的正則化曲線。

`policy_target_entropy` 1.095 → 1.275 說明 MCTS 的訪問分佈變寬（value 不再自信地亂剪枝），
`policy_kl` 0.450 → 0.345 說明 policy 反而更好擬合了。

### 1.5.4 eval 曲線（不是雜訊）

```
0916:  step  4  −1.00      step 32  −0.18      step 48  +0.14
       step 12 −0.909      step 36  −0.02      step 52  +0.12
       step 20 −0.622      step 40  +0.06      step 56  +0.14
       step 24 −0.413      step 44  +0.18      step 60  +0.12
       step 28 −0.340                          step 62  +0.14  (n=116)

0904:  step 16 −1.00   step 48 −0.931   step 80 −0.60   step 112 −1.00
       step 64 −1.00   step 96 −0.714   step 128 −0.882  step 160 −0.75
       step 176 −0.70  step 192 −0.708
```

單點 +0.14 在 50 局視窗下約 1 個標準誤（SE ≈ √(0.96/50) ≈ 0.14），但 **25 步單調上升 + 20 步穩定在正值**不是雜訊。
`evaluators 1 → 4` 也讓每點的樣本數從 n=5～30 變成 n=116，0904 的 eval 數字本來就不可信（但 193 步全是負的，結論不變）。

### 1.5.5 殘餘頭空間（§7 #2 的預測仍然成立）

最後一手（stage 6）：

| | 準確率 |
|---|---|
| `sign(banked)` 單獨（= 現在觀測裡唯一的分數訊號） | 0.717（§2.1 實測） |
| **0916 網路原始輸出** | **0.918** |
| 0916 MCTS root（搜尋補上 +5.8 分） | 0.976 |
| `sign(static_diff)`（banked + pending） | **0.992**（§2.1 實測） |

**網路自己走到 0.918，但 §7 #2 那條「把 pending 直接餵進去」的 0.992 還在原地。**
中局的落差更大：stage 4 raw 只有 0.706，而 §2.1 的 `static` 在 60–70% 進度已經是 0.852。
`raw_value_accuracy` 這一欄現在就是 #2 / #3 的儀表板 —— 做完之後 stage 4–6 應該分別往 0.85 / 0.93 / 0.99 走。

---

## 2. 診斷

### 2.1 缺口有多大

3000 局隨機對局，在每個 tile 決策點記錄兩個量，看它們預測最終勝者的準確率：

- `banked` = `player_scores[me] − player_scores[opp]` —— **這正是 `kScoreDiffPlane` 餵給網路的東西**
- `static` = `banked` + 「現在就結束的話的終局補分差」

```
games=3000   draws=3.7%   mean|final score diff| = 8.62

bucket        n   acc(banked)  acc(static)  E|pending|  E|banked|  corr(static,final)
 0- 10%   23955     0.537        0.573        1.79       0.50        0.225
20- 30%   21000     0.605        0.666        5.45       2.67        0.476
40- 50%   21000     0.663        0.750        6.30       4.07        0.691
60- 70%   21000     0.682        0.852        6.15       5.03        0.878
80- 90%   21000     0.712        0.933        6.34       5.66        0.965
90-100%   20999     0.719        0.972        6.48       5.91        0.989

LAST decision ply:   acc(banked) = 0.717      acc(static) = 0.992
```

**最後一手只看 banked 分差是 71.7%，加上 pending 是 99.2%。**
stage 5 的 0.751 幾乎就等於 banked baseline —— value head 整局大部分時間沒有提供
超過「分差那一格」以外的資訊。

pending 組成（雙方合計）：未完成城市 **19.8**、未完成道路 **15.7**、修道院 **7.7**。
而平均最終分差只有 8.62 —— **看不見的部分跟決勝幅度是同一個量級。**

### 2.2 資訊夠不夠？夠。

模擬「一個完美的 per-cell CNN + 全域求和」在**現有 80 個 plane** 下能算出什麼：
每格只用邊地形、`shield`、`monastery`、以及可從磚型還原的磚內 `link` 去重，
讀已廣播的元件 meeple 數決定正負號，每格輸出 `sign × (1 + shield)`，最後求和。
212,938 個狀態對照真實 pending：

```
exact match = 99.17%    sign match = 99.96%    MAE = 0.0083 分    max|err| = 1 分
```

（那 0.83% 的 ±1 誤差來自「同一元件離開一張磚又繞回來」的外部迴圈，磚內 `link` 無法辨識。極罕見。）

三個推論 —— **這三樣都不需要加**：

1. **元件 tile 數**：每格貢獻 1，求和自然得到。
2. **「封口了沒」旗標**：`settleCompletedFeatures` 在 feature 完成時清空 `meeple_count` 並收回 meeple，
   所以 **`meeple_count > 0` ⟹ 未完成**。pending 裡永遠不會出現 ×2。
3. **元件盾牌數**：每格本地的 `shield` 就夠（§3.2 已驗證所有有盾磚型都恰好單一城市元件）。

### 2.3 你們已經做過 feature engineering，而且做對了

`LogModule::getMeepleMap` 對每張已放磚的每一邊查 disjoint-set 的 root，
把**元件層級的** meeple 數寫進 plane 15–18（我方）與 20–23（對手）。
這是一個手工的 connected-component reduction，而且是最難的那個
（沿著細長元件傳播歸屬、又不能漏到空間相鄰但不連通的格子）。

§2.2 的 99.17% 完全依賴它。**所以爭論點不是「要不要做 feature engineering」，
而是「還要預先算哪些 reduction」。**

### 2.4 那為什麼還是學不起來

#### (a) 沒有對「數值」的直接監督

唯一的 value 訊號是對 ±1 做 MSE。從「pending 求和」到標籤中間隔著一個硬閾值
（`banked + pending ≷ 0`）再套 `tanh`。要學會一個必須精確到 ±1 分的算術電路，
梯度在遠離邊界時是 0（標籤不變），在邊界附近則被 chance node 的雜訊淹沒。
典型的「沒有計數監督卻要學會計數」。**KataGo 的 score head 解的就是這個。**

#### (b) 那個和只存在於最後一步 —— trunk 傳不動

理論上 8 blocks × 2 conv(3×3) 的感受野半徑是 16，覆蓋 15×15。但**卷積沒有「求和」運算子，
只有「局部加權平均」**：把 k 個 3×3 核卷在一起得到的是路徑數決定的離散高斯，σ ≈ √(2k/3)。
k=16 → σ ≈ 3.3。這是 Luo et al. 2016 *Understanding the Effective Receptive Field* 的主結果：
**ERF 只隨深度以 √L 成長，形狀是高斯不是平坦。**

**實測**（把 17 個核用 FFT 卷在一起；BN 折進 conv、residual 當 `I + W₂∘W₁`、ReLU 當恆等
—— ReLU 只會削弱傳遞，所以這是**上界**）：

| Chebyshev 距離 r | 0 | 1 | 2 | 3 | 4 | 5 | 6 | **7** | 8 | 9 | **10** | 12 | **14** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 相對中心敏感度 | 1.00 | 1.32 | 1.22 | 1.05 | 0.88 | 0.61 | 0.45 | **0.29** | 0.18 | 0.09 | **0.047** | 0.012 | **0.0023** |

若 trunk 在做均勻求和，這一列應該**一路都是 1.00**。
15×15 的板、起始磚在中心：角落到中心是 r=7（打 0.29 折），角落到角落是 r=14（0.0023，等於不存在）。

**端到端（trunk × head 的位置權重）**：head 那組看起來雜亂的位置權重**確實部分補償了** trunk 的衰減
（兩者合起來比任一個單獨看都平滑）。但只取真正的空間平面（0–24）：

```
中心(7,7) = 137,591    邊(7,1) = 67,081    角(1,1) = 28,731     max/min = 10.7×

占用率 >=50% 的 15 格：107,760   (1.00)
占用率 20-50% 的 52 格： 79,710   (0.74)
占用率  5-20% 的 65 格： 69,054   (0.64)
占用率 <= 5% 的 93 格： 48,647   (0.45)        corr(占用率, 敏感度) = +0.613
```

**週邊的一塊磚對 value 的影響力不到中心的一半。** 求和的正確答案是 1.00。

**有號探針**：取「我在這個元件多一個 meeple、對手少一個」的方向（plane 15 − plane 20）。
若網路真的在算 pending，這個擾動在每一格都應該讓 value 上升：

```
正號的格數 = 89 / 225        ← 60% 的格子符號是反的
mean = −9,902                ← 平均而言「我多一個 meeple」讓網路覺得局面變差
std / |mean| = 3.28
```

**網路連「我的 meeple 多是好事」這個方向都沒學到。**

> Caveat：以上三組都是線性化 Jacobian（ReLU 當恆等），不是任何真實狀態下的 Jacobian，
> 單獨看不算鐵證。但與行為證據一致：中盤 value_accuracy 0.49–0.59、輸給 rollout MCTS。

**為什麼 trunk 學不會傳（四個結構原因）**

1. **卷積是擴散不是求和**（上面）。要得到平坦核，每往外一層就得放大。
2. **BatchNorm 每層把訊號拉回單位變異數。** 「累積」與「正規化」直接對立；17 層 BN。
   實測 trunk 的 γ ≈ 0.95 —— 比 1 還小，**沒有在放大**。
3. **Residual 讓「留在原地」是預設。** `y = relu(x + F(x))`，identity 路徑增益 1；
   傳到 L 層外需要 L 個非平凡的 F 串接。局部是免費的，長程要主動學。
4. **32 通道的頻寬要跟 policy 共用。** 就算學會聚合，結果得存在通道裡並廣播到會被讀取的格子；
   但同一組 32 通道還要承載 policy head 需要的全部局部資訊。
   KataGo 的 **global pooling bias 層**就是為了不佔這個頻寬 —— 它在特定幾層以 per-channel bias
   注入 pooled 量，不需要它「存活」8 層卷積。

**對照：`mean over HW` 為什麼免費**

| | 3×3 conv 堆疊 | global average pooling |
|---|---|---|
| 有效核形狀 | 高斯，σ≈3.3 | **完全平坦，半徑無限** |
| 參數 | 每層 32×32×9 | **0** |
| 深度成本 | 16 層 | **0** |
| 中間 BN / ReLU 干擾 | 17 次 | 無 |
| 位置 (1,1) 的權重多久更新一次 | 只在該格有磚時（0.5%） | **每一步**（所有格共用 conv filter） |

#### (c) value head 的形狀跟「有號空間求和」不合

**現行架構**（`model.cc: ResOutputBlockImpl`，`nn_width=32`、80 planes @ 15×15；
`value_filters` 在 `model.cc:299` 硬編碼為 **1**）：

```
trunk 輸出 x                        [B, 32, 15, 15]
value_conv    Conv2d(32 → 1, k=1)   [B,  1, 15, 15]      33 params
value_bn      BatchNorm2d(1)        [B,  1, 15, 15]       2
relu                                [B,  1, 15, 15]   ← 全部 ≥ 0
view(-1, 225)                       [B, 225]
value_linear1 Linear(225 → 32), relu[B,  32]          7,232  (其中 7,200 個位置相依)
value_linear2 Linear(32 → 1), tanh  [B,   1]             33
                                                   ─────────
                                               合計   7,300  (全網路的 1.2%)
```

我們希望它算 `v_raw = Σ_cells s(c)`，`s(c)` 是那格的**有號**待結分。

**理論上 1 通道做得到嗎？做得到。** 若 `â_c = s(c) + C` 且 C 大到讓每格都 > 0，
ReLU 退化成恆等、求和完全正確。所以「1 通道不可能做有號求和」是**錯的**（v2 講太重）。
自然的 ReLU 寫法（A 學 `max(0,s)`、B 學 `max(0,−s)`、後面相減）需要 2 個通道，
且**分裂必須發生在 conv** —— `Linear(225→32)` 收到的已經是單一個被截斷過的場，
在那層做任何線性組合都只是同一個場的另一組位置權重：`Σw⁺·r − Σw⁻·r = Σ(w⁺−w⁻)·r`。
而 `s(c)` 的符號是**內容相依**（這格的元件現在誰佔多數），位置權重追不了。

**實際上有沒有走偏移那條路？讀 `checkpoint-193.pt`：沒有。**

**實測 1 — ReLU 正在砍掉將近一半的格子**

```
value_batch_norm:  γ = +1.4598   β = +0.1074   running_mean = +1.0251   running_std = 0.6839
BN 後的場:  mean = β = 0.107 ,  std = |γ| = 1.460
β / |γ| = 0.07        →   P(ReLU 前 < 0) ≈ 47.1%
```

要讓 ReLU 失效需要 `β/|γ| ≫ 1`，實際是 0.07。

**實測 2 — 有效讀出與空間求和幾乎正交**

`W2 @ W1` 攤成每格的有效權重（線性區近似）：

```
與全 1 向量的 cosine = +0.048      ← 若在做平移不變求和，應接近 1.0
sum = +2.035 ,  ||eff||₂ = 2.801 ,  std/|mean| = 20.6 ,  範圍 −0.496 ~ +0.539
```

**實測 3 — 週邊格子的權重還停在初始值**

每格「有磚」的機率（隨機對局 106,468 個狀態）：

```
   0   0   0   1   2   3   4   4   4   3   2   1   1   0   0
   0   1   1   2   5   7   9   9   8   7   4   3   1   1   0
   1   1   3   6   9  13  15  17  16  13   9   6   3   1   1
   1   3   6  10  14  22  26  28  26  21  15   9   5   3   1
   2   5   9  15  22  30  36  44  37  30  23  15   8   5   2
   3   7  12  20  29  40  48  59  51  41  30  21  12   6   3
   4   9  16  26  38  51  56  70  55  49  36  26  15   8   4
   4   9  17  28  42  57  69 100  71  56  42  27  17   9   4    ← 起始磚 (7,7)
   4   8  16  25  36  49  56  77  56  48  37  26  16   8   4
   3   6  12  19  30  41  49  61  48  42  32  23  12   6   3
   2   5   9  15  22  31  37  44  37  30  24  16   9   5   2
   1   2   5  10  15  21  25  30  26  22  16  10   5   3   1
   0   1   3   5   8  12  16  18  16  13   9   5   3   2   1
   0   1   1   3   4   6   9  10   9   6   4   3   1   1   0
   0   0   1   1   2   3   4   4   4   3   2   1   1   0   0

中心 100%   邊 9.1%   角 0.5%   →  185×
```

| 該格被佔用的機率 | 格數 | mean \|有效權重\| | `W1` 每格 std |
|---|---|---|---|
| ≥ 50% | 15 | 0.371 | **0.0699** |
| 20–50% | 52 | 0.211 | 0.0480 |
| 5–20% | 65 | 0.106 | 0.0376 |
| ≤ 5% | 93 | 0.087 | **0.0345** |

libtorch `Linear` 初始化 std ≈ **0.0385**。占用率 ≤5% 的 93 格是 0.0345 ——
比初始值還低（weight decay 壓的），**完全沒被訓練過**。

```
corr(占用率, W1 每格 std) = +0.607      ← 不含 W2，不受 ReLU 近似影響
corr(占用率, |有效權重|)   = +0.614
```

終局盤上約 72 格磚，大部分落在那 158 格「權重是未訓練噪音」的區域 ——
而那些正是後期才放下、pending 分數最集中的地方。

> 曾猜測「weight decay 打在 BN γ 上會讓 value head 退化」——**實測否定**：
> trunk 的 γ ≈ 0.95、value/policy head 的 γ ≈ 1.44。這條不成立。

**為什麼那 93 格會低於初始值：Adam + 耦合 L2**（v6 補上的機制）

```cpp
// vpnet.cc:178  optimizer 沒設 weight_decay
model_optimizer_(model_->parameters(), torch::optim::AdamOptions(lr))
// model.cc:393  L2 當成 loss 的一項
l2_regularization_loss += weight_decay_ * torch::sum(torch::square(w)) / 2;
// vpnet.cc      total_loss = policy + value + l2;  backward();  optimizer.step();
```

**這正是 AdamW 論文要修的那個組合。** L2 的梯度 `λw` 跟任務梯度混在一起進 Adam 的動量，
再被 `√v̂` 正規化。對一個任務梯度很小的參數，更新方向幾乎完全由 L2 項主導 ——
而 Adam 對近乎常數的梯度，步長 **≈ lr，與梯度大小無關**。

**於是：占用率越低 → 任務梯度越小 → `√v̂` 越小 → 衰減相對越狠。**
那 93 格被壓向 0 的速度快過它們學習的速度，所以停在 0.0345（低於初始的 0.0385）。
這把「占用率失衡」和「權重低於初始值」兩個觀測串成同一條因果鏈。

**修法**：改用 `torch::optim::AdamW`（`weight_decay` 設在 optimizer 上）並**移除 `losses()` 裡的
顯式 L2 項**，否則雙重衰減。AdamW 的衰減是 `-lr·λ·w`，不經過動量正規化。
`LossInfo` 的 `l2reg` 欄位可保留計算但不加進 `total_loss`，純當監控。
（換了 optimizer 型別之後舊的 `-optimizer.pt` 也載不了。）

**但優先度在架構之後**：做完 §7 #1（global pooling）與 #4（conv policy head）之後，
位置相依權重就不存在了，這個傷害自然消失。

#### (d) 損失函數：tanh 飽和在錯得最離譜時把梯度掐掉

MSE + tanh 對 pre-tanh logit 的梯度是 `2(v − z)·(1 − v²)`。`(1 − v²)` 在 |v|→1 時歸零。

用 step 193 的數據：

| stage | value_accuracy | \|v\| mean | \|v\| max | (1−v²) 在 max 處 |
|---|---|---|---|---|
| 1 | 0.500 | 0.468 | 0.98280 | 3.4e-02 |
| 2 | **0.491** | 0.493 | 0.99769 | 4.6e-03 |
| 3 | 0.587 | 0.601 | 0.99996 | **8.8e-05** |
| 5 | 0.751 | 0.525 | 0.99999 | **2.2e-05** |

stage 2 的準確率 49.1%、|v| 平均 0.49 —— **有一半的樣本是「自信地錯」**，
而最自信最錯的那些，梯度被衰減 200×–10000×。**網路愈是 confidently wrong 就愈學不到東西**，
這是個正回饋陷阱，而 stage 1–4 正好卡在這裡。

softmax + cross-entropy 沒有這一項（梯度 = `p − y`）：

| 預測往錯的方向 | MSE + tanh | CE + softmax |
|---|---|---|
| 略錯 (v=−0.2) | 2.30 | 0.60 |
| 錯 (v=−0.5) | 2.25 | 0.75 |
| 很錯 (v=−0.9) | 0.72 | 0.95 |
| 極錯 (v=−0.99) | 0.079 | 0.995 |
| (v=−0.999) | 0.008 | ~1.0 |

絕對尺度不可比（可用 loss weight 調），重點是**形狀**：MSE+tanh 在 v≈−0.33 達峰後往兩端衰減到 0；
CE 單調遞增到上限。

**但要注意這條的定位**（§7 的排序理由）：

- AlphaZero 原版就是 scalar tanh + MSE，在西洋棋（高水準和局率 ~50%）也能運作。
  這不是「設計錯了」，是「有更好的設計」。
- **不要為了 WDL 而做 WDL。** 真正解 (a) 的是 **score-margin 分佈頭**；
  WDL 可以直接從分數分佈積分導出（`P(>0)`, `P(=0)`, `P(<0)`），兩者互為一致性約束。
  既然改 WDL 一定得動 `losses()` 和 `vpnet.cc: Inference()`，就該一次做到位。

#### (e) 參數預算嚴重錯配

實際權重（`nn_width=32`, `nn_depth=8`；學習參數 588,104 + BN buffers 1,113 = 589,217 個 float，
對得上 2.43MB 的 checkpoint）：

| 部位 | 參數 | 佔比 |
|---|---|---|
| input block | 23,136 | 3.9% |
| 8 × residual block | 148,992 | 25.3% |
| **value head** | **7,300** | **1.2%** |
| **policy head**（`Linear(450→906)` 一層就 408,606） | **408,676** | **69.5%** |

**69.5% 的參數卡在 policy head 的一層全連接**，而它還得自己學會「index 906 對應哪個 (x,y,rot)」。
改成 conv policy head（§5.5）可以把這 40 萬參數還給 trunk。

另外：62% 的輸入張量是常數平面（§3.5）；193 個 training step（AGZ 是 700k）。

**（v7 新增）這不只是浪費，是主動壞掉 —— 在 `0916/checkpoint-62.pt` 上直接量得到。**

`policy_linear.weight` 是 `(906, 450)`。第 `a` 列只在動作 `a` 合法時才有梯度 ——
`torch::where(mask, logits, −65536)` 讓不合法動作的 logit 梯度**恰好是 0**。
把 900 個落子列按 `(x,y)` 攤開、對 4 個 rot 取平均的列範數：

```
Chebyshev 環 r     r=0     r=1     r=2     r=3     r=4     r=5     r=6     r=7
平均列範數        0.000   0.322   0.356   0.387   0.383   0.297   0.173   0.087
格數                 1       8      16      24      32      40      48      56
```

`Linear(450→906)` 的初始化是 `U(±1/√450)` → 期望列範數 **0.577**。

- **(7,7) 是起始磚，永遠不可能是合法落點 → 列範數是乾淨的 `0.000`。**
  這是完美的對照組：零任務梯度、只剩 L2，62 步之後歸零。
- 最常用的 r=3–4 也只剩 **0.387 / 0.383 = 初始值的 67%**。
- r=7（邊角，正是 §3.6 那 31% 會撞牆的對局用到的格子）剩 **0.087 = 15%**，形同死掉。
- 對照組：6 個 meeple 列（每個 meeple phase 幾乎都合法）是 **0.34–0.54**，
  其中 skip(900)=0.536、monastery(905)=0.526 最高 —— 出現頻率最高的兩個。

**這正是 §2.4-c 那個 Adam + 耦合 L2 機制的純粹版本，而且解釋了它為什麼這麼狠。**
一個任務梯度為零的參數，它唯一的梯度是 `wd·w`；Adam 會把梯度除以自己的 RMS，
所以更新量變成 `−lr·sign(w)` —— **固定步長的收縮，跟權重大小無關**。
62 個 learner step × 128 次更新 × lr 1e-4 = **總收縮預算 0.79**，而初始 |w| 平均只有 0.024。
不是指數衰減到很小，是**線性推到 0 然後釘在那裡**。

每個落子列拿到梯度的頻率：一個 tile phase 平均 ~31 個合法手 / 900 → 3.4%，
tile phase 約占 buffer 一半 → **每列約 1.7% 的 batch 有梯度**。meeple 列約 50%。差 **30 倍**。

conv head 沒有「每個位置一列」這種東西：4 個 rot 通道**每一個樣本、每一個合法格都貢獻梯度**，
共享權重讓有效梯度樣本數 ×900，衰減相對之下可以忽略。

### 2.5 對照：AlphaGo Zero 為什麼不需要這些

AGZ 的 value head **也是 1 個 filter**，圍棋上能動。三個原因：

1. **圍棋每一點被下到的頻率大致均勻**，361 個位置權重拿到的梯度差不多。
   §2.4-c 實測 3 那個 185× 的病在圍棋不存在 —— **同一個 head 在兩個遊戲裡條件數天差地遠**。
2. **圍棋的 value 剛好只需要一個純量場**（ownership），天然權重就是均勻的 1。
   終局分數 `Σ_cells ownership(c) − komi` 跟 CNN + global pooling 同構。
3. 256ch × 20–40 blocks + 4.9M 局。trunk 大到可以硬找出偏移解。

而 **KataGo 只有 1/50 的算力，還是把它換成 multi-channel + global pooling**，
並且加回 komi 當 global input、score head、ownership head。
我們要加的是**規則衍生**特徵（跟 komi、西洋棋的 50 步計數、合法手 mask 同類），不是啟發式知識。

### 2.6 「每一條路都是征子」—— 修正版

**靜態計分不是征子。** 它是求和，而歸屬已由 `getMeepleMap` 預先算好廣播。
真正征子性的只有兩件事，而且都**不在觀測裡**：

1. **一手合併兩個元件**：我在 A 城有 2 個 meeple、你在 B 城有 1 個，一張磚把 A、B 接起來 →
   15 分的歸屬瞬間確定，而 A、B 可能在盤面兩端。
2. **能不能封口**：12 格的開放城，封起來 24 分、封不起來 12 分。
   取決於開口數、開口形狀、牌堆剩什麼、對手能不能堵。**觀測裡沒有 `opens`。**

---

## 3. 觀測內容盤點

### 3.1 現有 80 個 plane

| plane | 內容 | 層級 |
|---|---|---|
| 0–11 | 4 邊 × 3 terrain one-hot | 每磚（本地） |
| 12 | `shield` | **每磚（本地，非元件總和）** |
| 13 | `monastery` | 每磚 |
| 14 | city connectivity flag（僅 type 14/15） | 每磚 |
| 15–18 | **我方 meeple 數，元件層級**，per side | **元件（已廣播）** |
| 19 | 我方已認領的修道院 | 每磚 |
| 20–23 | **對手 meeple 數，元件層級**，per side | **元件（已廣播）** |
| 24 | 對手已認領的修道院 | 每磚 |
| 25–39 | 手上這張磚（全盤廣播常數） | 全域 |
| 40 | 上一手落點 | 每磚 |
| 41–44 | 該格 rot=0..3 是否合法 | 每格 |
| 45–49 | 合法 meeple 位置（全盤廣播常數） | 全域 |
| 50–73 | 剩餘牌型比例（全盤廣播常數） | 全域 |
| 74–79 | 手上 meeple ×2、剩餘牌數、`tanh(分差/30)`、is_meeple_phase、is_player0 | 全域 |

**沒有的**：`opens`、元件 tile 數、元件盾牌數、元件 `getScore()`。

### 3.2 元件盾牌數：沒廣播，**不需要加**

所有有盾磚型在單張磚上都恰好只有 **1 個**城市元件：

```
type  3 (CCCC) ×1 → 1      type  9 (CGGC) ×2 → 1
type  5 (CCGC) ×1 → 1      type 11 (CRRC) ×2 → 1
type  7 (CCRC) ×2 → 1      type 13 (GCGC) ×2 → 1
```

盾的歸屬沒有歧義，`plane 12` 的本地 shield 就足夠，求和後自然得到元件盾數。
`Feature::getScore()` 用的 `(tile_mask & SHIELD_MASK).count()` 也是按磚計，與本地分解一致。

### 3.3 `opens`：沒廣播，**需要加**

**(1) 這是唯一無法由局部推導的全域量。** 每格能看到「我的北邊是城、北邊那格是空的」= 1 個開口，
但那是本地的；要得到「這個元件總共幾個開口」必須沿元件求和，而 global pooling
會把所有元件的開口混成一個沒用的總數。§2.2 的分解之所以成立，正是因為 pending **不需要** `opens`；
一旦要做前瞻就非它不可。

**(2) 封口 ×2 是最大的單一分數槓桿。**

**(3) 實測：`opens` 是封口機率的強單調預測子**（牌堆剩 30 張時取樣、帶 meeple 的未完成元件，4000 局隨機對局）：

```
opens    城市 n    最終封口%   E[分數增益] |  道路 n   最終封口%
  1       13030       7.6%        +1.18    |  16468      10.4%
  2        5854       1.0%        +1.03    |   9484       1.6%
  3        2549       0.1%        +1.25    |      0        --
  4         954       0.0%        +1.52    |      0        --
  5+        443       0.0%     +1.8~2.0    |      0        --
```

> 隨機對局的絕對封口率被嚴重低估（隨機下法不會刻意封城），但單調關係非常乾淨：
> `opens=1` 是 `opens=2` 的 7.6 倍、`opens≥3` 基本為零。有意識的對局裡差距只會更大。

**額外發現**：邊界會製造**假開口** —— 15×15 外緣，城市邊朝盤外時 `opens` 永遠消不掉，
但那個口實際上永遠填不了。97.7% 的對局會碰到邊界（§3.6），`opens` 的語意本身就被污染了。

### 3.4 `tanh(score_diff / 30)` 壓縮在最需要解析度的地方

分差 ±1 在輸入上只差 0.067，而勝負就是由 ±1 決定的。
應改成 **多尺度 `clip`**（`/3`、`/10`、`/30`），理由見 §4.3。

### 3.5 62% 的 plane 是空間常數

`kCurrentTile*`(15) + `kLegalMeeplePlane`(5) + `kRemainingTileTypePlane`(24) + `kGlobalFeaturePlanes`(6) = 50 / 80。
把 ~50 bit 膨脹成 11,250 個 float。**全域資訊應該走獨立向量輸入**（§4.3）。

### 3.6 `BOARD_SIZE = 15` 會裁掉真實對局

起始磚在 (7,7)，各方向只有 7 格。把 `BOARD_SIZE` 改成 25 重跑同樣 3000 局：

```
最大跨度 > 15 的對局： 31.1%      > 17: 4.9%      > 19: 0.4%      > 21: 0.0%
跨度分佈: 13:10.7%  14:27.6%  15:29.4%  16:17.8%  17:8.4%  18:3.4%  19:1.1%  20:0.4%
平均合法落點/手：15×15 → 30.8      25×25 → 32.6      碰到邊界的對局：97.7% → 1.3%
```

**建議 `BOARD_SIZE = 21`（覆蓋 99.9%）。** 順帶解掉 §3.3 的假開口問題。
（注意：做完 §7 第 1 項之後 value head 不再綁死棋盤大小，這一項會變成幾乎零成本。）

### 3.7 `HasCityConnectivityPlane(type==14||15)` 正確但脆弱

它剛好覆蓋兩個歧義情況（type 8/9 vs 14 的 CGGC、type 12/13 vs 15 的 GCGC）；
道路那邊沒有歧義。**目前是對的**，但改 deck 或加擴充就會靜默壞掉。
建議換成通用的 6 個 side-pair link plane（N-E, N-S, N-W, E-S, E-W, S-W）。

### 3.8 `Returns()` 只有 ±1 / 0，value head 是單一 tanh

和局佔 3.5%（16/462）。scalar 的限制是「必和」與「五五波」都輸出 0，
網路無法把那些樣本的 loss 壓到 0。

**但這兩種情況對 MCTS backup 是等價的**（期望回報都是 0），所以這條的實質效益有限。
改 WDL 的主要理由是 §2.4-d 的梯度形狀，不是和局表達力 —— 見 §7 的排序。

### 3.9 policy head 是 dense

`1×1 conv → 2 filters → flatten(450) → Linear(906)`：一層 408,606 參數（全網路 69.5%），
還得自己學會「index 906 對應哪個 (x,y,rot)」。**應該用 conv 直接輸出 4×H×W 的落子 logits**，
meeple 的 6 個動作另接小 head。而且 policy head 同樣看不到任何全域量（§2.4-b）。

→ **完整改法整理在 §5.5，程式碼在 §9.4；實測的權重衰減證據在 §2.4-e。**

### 3.10 meeple 動作空間：去重是對的，索引語義是弱點

```cpp
EncodeMeepleAction(pos) = 900 + pos + 1          // pos ∈ {-1, 0, 1, 2, 3, 4}
// -1 = 不放, 0..3 = 第 0..3 邊所屬的元件, 4 = 修道院
```

`FeatureModule::getLegalMeepleMoves` 對每個「沒人佔的元件」只給一個動作，
用該元件在這張磚上**最小的邊編號**代表。

**去重是必須的，而且做得比看起來更對。**

1. **meeple 放在元件上，不是邊上。** `FeatureModule::placeMeeple` 是
   `featureMap.getSetData(edgeIndex(id, pos)).meeple_count[player]++`，
   `getSetData` 內含 `find()` —— 指名哪一邊都落到同一個元件根。指北邊和指西邊是**完全相同的一手棋**。
2. **不去重的後果比「索引隨便選」嚴重得多**：同一個元件會變成 2–4 個重複動作 →
   MCTS 的 800 次訪問被拆散（那個元件被系統性低估）→ policy target `N(a)^(1/T)` 也被拆開 →
   合法手數虛增，`10/n` 的 Dirichlet α 也跟著算錯。
3. **用 `find()` 去重比用磚內 `link` 去重更正確。** 考慮 type 15（CGCG，磚內南北兩座城分開）：
   若那座城繞一圈從另一邊接回來，`find()` 會認出它們是同一個元件、只給一個動作。
   **這正是 §2.2 那個「只靠磚內 link」的局部分解會誤判的 0.83% 案例 —— 引擎在這裡是對的，近似才是錯的。**

**「最小邊編號」這個選擇本身也是對的。** 它是一個**內容無關的正規化**。
替代方案（例如「動作 901 永遠是最值錢的元件」）語義看似更清楚，但會讓動作意義隨盤面**不連續地換位** ——
對手多放一張磚讓城變大，901 和 902 的意思就互換，policy target 在相似局面之間完全不穩定，
MCTS 的轉置直覺也壞掉。而且那等於把一個價值啟發式塞進動作空間。

**容量驗證**：`ret` 最多裝 skip(1) + 元件(≤4) + 修道院(1) = 6，宣告是 `FixedVector<int,6>`（`push_back` 有 `assert`）。
實際填不滿 —— 修道院磚是 type 1（GGGG）與 type 2（GGRG），非草地邊 0 或 1 個 → 最多 3；
非修道院磚最多 5。安全，但只剩 1 格緩衝，**加擴充（帶修道院又帶城的磚）就會 assert**。值得加註解。

**真正的弱點：動作索引不帶語義，而且依賴旋轉。**

你在決定的是「放在那座 12 格、剩 1 個開口的城」還是「放在那條 2 格的路」，
但索引只說「北」。而且同一個元件對應到哪個索引，取決於磚的旋轉。
以 type 10（`CITY, ROAD, ROAD, CITY`, link `0,1,1,0`）為例：

```
rot 0:  edges = [C, R, R, C]
        i=0 城 → push 0     i=1 路 → push 1
        i=2 路（同根跳過）   i=3 城（同根跳過）        → 動作 {0, 1}

rot 1:  edges = [C, C, R, R]
        i=0 城 → push 0     i=1 城（同根跳過）
        i=2 路 → push 2     i=3 路（同根跳過）        → 動作 {0, 2}
```

同一張磚、同樣那兩個元件，「那條路」在 rot 0 是動作 1、在 rot 1 是動作 2。
網路得自己學會這個 join。**解法不是改排序規則，是讓 head 從空間上讀 —— 見 §5.4。**

---

## 4. Feature representation v2

原則：**空間的放張量、全域的放向量、無法由局部推導的 reduction 才預先算並廣播。**

### 4.1 board tensor：`48 × 21 × 21`

**A. 幾何 / 磚面（20 planes）**

| plane | 內容 |
|---|---|
| 0 | occupied |
| 1–12 | 4 邊 × 3 terrain one-hot |
| 13 | shield（本地即可，見 §3.2） |
| 14 | monastery |
| 15–20 | 6 × 磚內 side-pair 連通（取代 `HasCityConnectivityPlane`） |

**B. 動作 / 上下文（6 planes）**

| plane | 內容 |
|---|---|
| 21 | frontier |
| 22–25 | 手上這張磚在該格 rot=0..3 是否合法 |
| 26 | 上一手落點 |

**C. 元件層級 reduction（4 邊 × 5 = 20 planes）**

只保留**無法由局部推導**、或**必須讓 trunk / policy head 看到**的量：

| offset | 內容 | 正規化 | 為什麼 |
|---|---|---|---|
| +0 | **`opens`** | `min(opens,6)/6` | §3.3。唯一真正缺的全域量 |
| +1 | **`getScore()`**（現在結束的話值幾分） | `/12` clip | 讓 policy 知道量級，不必等 global pooling |
| +2 | **`2×getScore()`**（封口後值幾分） | `/24` clip | 與 +0 搭配才能評估「值不值得封／堵」 |
| +3 | 我方 meeple − 對手 meeple（已有，保留） | `/3` clip | 正負號 |
| +4 | 有號待結分 `sign × getScore()` | `/12` clip | 省掉網路做乘法 |

> **不要加**（v1 的錯誤建議，見 §2.2）：元件 tile 數、元件盾牌數、completed flag。

**D. 修道院（2 planes）**：`count3x3/9`、owner sign。

合計 **48 planes @ 21×21 = 21.2k float**（現在 80 @ 15×15 = 18k）。
砍掉 50 個常數平面、換進真正缺的元件量，成本幾乎持平。

> 實作提示：`FeatureModule::featureMap` 已經有全部欄位（`opens`、`getScore()`、`meeple_count[]`），
> 寫 observation 時 `getSetData(edgeIndex(id, side))` 取出來即可，**不需要新資料結構**。

### 4.2 移出張量的
`kRemainingTileTypePlane`(24)、`kCurrentTile*`(15)、`kLegalMeeplePlane`(5)、`kGlobalFeaturePlanes`(6)。

### 4.3 global vector（約 160 維，**不要廣播**）

| 維度 | 內容 |
|---|---|
| 2 | 我方 / 對手 banked 分數 `/40` |
| 1 | banked 分差 `clip(d/20)` |
| 2 | 我方 / 對手 pending 分數 `/20` |
| **3** | **`static_diff` = banked + pending，三個尺度 `clip(d/3)`、`clip(d/10)`、`clip(d/30)`** |
| 2 | 手上 meeple `/7` |
| 2 | 剩餘牌數 `/72`、已完成回合 `/36` |
| 24 | 剩餘牌型比例 |
| 24 | 手上磚型 one-hot |
| 8 | phase one-hot(2) + 6 個 meeple 動作合法遮罩 |
| 1 | 合法落點數 `/100` |

**為什麼還是要放 `static_diff`**（既然 §2.2 說資訊夠）：

1. 省掉一個必須精確到 ±1 分的算術電路，把 32 個通道的容量還給真正需要判斷的東西。
2. 讓 trunk 每一層和 policy head 都看得到（§2.4-b），而不是只有 value head 的最後一步。
3. 配合 score-margin 頭，等於對那個算術給出**直接監督**（§2.4-a）。
4. §2.4-c 的實測顯示現有 head 根本沒在做那個求和 —— 與其等它學會，不如直接給。

**為什麼是多尺度 clip，不是 bucket one-hot**（v6 修正，先前這裡寫的是 2×31 維 one-hot）

`clip(d/3)` 在 |d| ≤ 3 線性、之外飽和 —— 等於一個**保有解析度的軟正負號函數**，
而翻盤就發生在 0 附近。`/10` 和 `/30` 負責中段與大局的量級。3 維而不是 62 維。

當初主張 one-hot 的理由是「±1 分決定勝負，壓進一個 scalar 解析度不夠」。三個反駁：

1. **做了 score head（§7 #9）之後，閾值根本不在網路裡算。** 殘差讀出是
   `P(win) = Σ_{k > −static_diff} p(k)` —— 用精確的整數 `static_diff` 當積分下界，
   落在 bin 邊界上、零誤差。網路只負責預測 `Δ` 的分佈，而 `corr(static, Δ)` 只有
   −0.24 ~ +0.06（§2.1 的殘差分解），所以 `static_diff` 作為**輸入**只是弱的情境變數。
2. **BatchNorm 已經處理了尺度。** 空間常數平面經 conv 後對每個通道貢獻一個跨 batch 變動的常數，
   BN 把它正規化到單位變異數 —— 絕對尺度不是問題。剩下的只是「從純量做出銳利閾值」，
   那是幾個 ReLU 單元的事，學得起來。
3. **KataGo 把 komi 當純量餵**，而 komi 精確到 0.5 目就決定勝負。它**在輸出端**用分箱分佈
   （score belief），輸入端不用。

> **原則：輸入用純量，輸出用分箱。**
> 輸出分箱改變的是**損失函數**（稠密監督、序關係、避開 tanh 飽和 —— §2.4-a、§2.4-d），
> 輸入分箱只是一個**特徵變換**，網路自己學得出來。

**什麼時候值得回頭試 one-hot**：等 §7 #8（global vector 獨立輸入，31 維只佔 160 維的一小塊、成本可忽略）
**且** §7 #9（score head）還沒做（閾值還在網路裡算）—— 那個組合下值得當一次 ablation。
在那之前，31 個廣播平面會把張量從 84 推到 115 個 plane、其中 73% 是空間常數（正是 §3.5 在抱怨的事），
買的卻是 BN 大半已經給你的東西。

（誠實補充：one-hot 真正獨有的好處是 policy 對分差的最佳反應**分區間**而非單調 ——
小輸要搶、大輸要賭、小贏要穩、大贏也要穩。多尺度 clip 只能近似。但這個效益是「中等」，不是「必要」。）

**注入方式**（KataGo 做法）：`Linear(160 → C)` 後加到 input conv 輸出當 per-channel bias
（每 2 個 block 再做一次），**同時** concat 到 value / policy head 的 global pooling 之後。

---

## 5. 網路架構

```
trunk:   input conv(48→W) + global-bias  →  D × residual block (W ch, 3×3, BN, ReLU)
                                             ↑ 每 2 block 插一次 global pooling bias

policy:  1×1 conv(W→32) + global-pool bias → BN → ReLU        (見 §5.5)
         ├── 1×1 conv(32→4) → permute → 4·H·W 落子 logits     (取代 40 萬參數的 dense)
         └── 1×1 conv(32→6) ⊙ last_placed_plane → 6 meeple logits  (空間讀取，§5.4)
         concat → mask → softmax                               (合計 4,586 參數)

value:   1×1 conv(W→32) → [GlobalAvgPool ⊕ GlobalMaxPool](64) ⊕ global(160) → FC(256) → ReLU
         ├── score margin : 61-bin 分佈（−30..+30），HL-Gauss soft label
         ├── WDL          : 由分數分佈積分導出（P(>0), P(=0), P(<0)）
         └── (aux) 期望分差: scalar, Huber

aux:     ownership head — 1×1 conv(W→3)，每格 3 分類
         「終局時這格所屬 feature 的分數歸我／歸對手／無人」，loss 權重 0.15
```

### 5.1 value head 的三個改動各自對應一項實測

| 改動 | 解掉的問題 | 實測依據 |
|---|---|---|
| 1 → 32 通道 | 有號場可用兩通道自然表示（A=`max(0,s)`、B=`max(0,−s)`，pool 後相減），ReLU 不再破壞資訊；剩 30 個通道做別的聚合 | β/\|γ\| = 0.07，47% 被 ReLU 截斷 |
| flatten+Linear → global avg pooling | 求和硬編碼、平移不變；位置相依參數 7,200 → **0**；**梯度共享**（所有格共用 conv filter，占用率失衡自動消失）；head 不再綁死棋盤大小 | cos(eff, 1) = 0.048；93 格權重停在初始值；ERF r=7 只剩 0.29 |
| 加 max pooling | 補上「盤上有沒有某個特別大／快封口的元件」這種非加總資訊 | — |

### 5.2 tensor shape

```
trunk 輸出                                [B, W, 21, 21]
value_conv  Conv2d(W → 32, k=1) + BN + ReLU  [B, 32, 21, 21]
avg = mean over HW                        [B, 32]
max = max  over HW                        [B, 32]
cat(avg, max, global_vec)                 [B, 64 + G]
FC(64+G → 256) + ReLU                     [B, 256]
→ heads
```

參數（W=32，最小版 G=0）：conv 1,056 + BN 64 + FC1 16,640 + FC2 257 = **18,017**
（vs 現在 7,300，但**位置相依參數從 7,200 降到 0**）。

### 5.3 分三步做，第一步是自足的

| 步驟 | 改什麼 | 要動到哪些檔案 |
|---|---|---|
| **A（最小）** | 1→32 通道 + pooling，輸出仍是單一 tanh scalar | **只有 `model.h` / `model.cc`** |
| **B** | global vector concat 進 pooling 後面 | 加上輸入介面（§4.3） |
| **C** | score-margin 分佈頭（WDL 由它導出） | `model.cc: losses()`、`vpnet.cc: Inference()`、MCTS 的 value 轉換 |

**步驟 A 完全不用碰 `vpnet.cc`、`losses()`、`mcts.cc`** —— `Inference` 讀的還是 `value_acc[batch][0]`，
形狀不變。改約 10 行、不動任何介面，就把 §2.4-b/c 的問題消掉。

**注意**：形狀變了，舊 checkpoint 不能載入（`init_checkpoint` 留空，從頭訓練）。

### 5.4 meeple head 也要從空間讀，不要用 pooling + MLP

> 本版修正。先前寫的是 `pooled(W) ⊕ global → MLP → 6`，**那個設計不好**：
> global pooling 會把「剛放下那一格」整個洗掉，而那一格是唯一相關的地方。
> 就算補上 last-placed 那格的特徵，head 仍要從一個 W 維向量產生 6 個
> 「第 0/1/2/3 邊所屬元件」的 logits —— §3.10 那個 join 還是留在 head 裡做。

**改成跟落子頭一樣，用 1×1 conv 輸出，再從那一格取值：**

```cpp
torch::Tensor mmap = meeple_conv_(x);                          // 1×1 conv(W→6) → [B,6,H,W]
torch::Tensor meeple_logits = (mmap * last_plane).sum({2, 3}); // 用 kLastPlacedPlane 當 one-hot 取那格
```

`last_plane` 從觀測切出來即可（`kLastPlacedPlane`），可微，不用做索引。
參數 `W×6 + 6 = 198`（W=32）。

為什麼這好得多：

- **那一格的 trunk 特徵已經含有各邊所屬元件的資訊。** 加了 §4.1-C 的 per-side 元件平面之後，
  那一格的通道上就寫著「第 0 邊的元件值 12 分、剩 1 個開口」。head 只要做一次**本地讀取**，
  不用做全域 join。
- **「第 s 邊」從抽象索引變成 conv 的第 s 個輸出通道**，跟輸入的 per-side 平面天然對齊
  （plane 0–2 是北邊地形、元件平面 +0 也是北邊的 `opens`…）。這跟落子頭用 4 個通道表示 rotation
  是同一個道理，也讓它跟旋轉增強相容（§6.5）。

6 個輸出的順序要對上 `EncodeMeepleAction`：`[skip(-1), edge0, edge1, edge2, edge3, monastery]`。

---

### 5.5 policy head 改法（整理版，v7）

#### 5.5.1 現況與要解的四件事

```cpp
// ResOutputBlockImpl::forward，現況
policy_logits = relu(policy_batch_norm_(policy_conv_(x)));   // 1×1 conv(32→2)  [B,2,15,15]
policy_logits = policy_logits.view({-1, 450});
policy_logits = policy_linear_(policy_logits);               // Linear(450→906)  ← 408,606 參數
policy_logits = where(mask, policy_logits, -65536);
```

| # | 問題 | 依據 |
|---|---|---|
| P1 | **408,606 參數 = 全網路 68.1%**，全壓在一層 | §2.4-e |
| P2 | **每個位置一列、列梯度稀疏 + Adam 耦合 L2 → 系統性衰減**（實測 r=7 剩 15%、(7,7) 歸零） | §2.4-e（v7 新增） |
| P3 | **看不到任何全域量**（剩牌、剩 meeple、分差），只有那一格 32 維 trunk 特徵，而 trunk ERF r=7 只剩 0.29 | §2.4-b |
| P4 | **動作索引語義得自己學**：`(x,y,rot)` 的 (900) 與 meeple 邊索引的旋轉相依 join | §3.9, §3.10 |

#### 5.5.2 目標形狀

```
p  = conv1x1(W → 32)(trunk)                                  [B,32,H,W]
g  = conv1x1(W → 32)(trunk) → mean‖amax over HW → FC(64→32)  [B,32]      ← 全域池化偏置
p  = relu(BN(p + g[:, :, None, None]))                       [B,32,H,W]

tile   = conv1x1(32 → 4)(p)                                  [B,4,H,W]
       → permute(0,2,3,1).reshape(B, H*W*4)                  [B, 900]    ← 順序見 5.5.3
meeple = (conv1x1(32 → 6)(p) * last_plane).sum({2,3})        [B, 6]      ← §5.4

logits = cat({tile, meeple}, 1)                              [B, 906]
logits = where(mask, logits, -65536)
```

參數（W=32）：

| 部位 | 參數 |
|---|---|
| `policy_conv` 32→32 | 1,056 |
| `gpool_conv` 32→32 + `gpool_fc` 64→32 | 1,056 + 2,080 |
| BN(32) | 64 |
| `tile_conv` 32→4 | 132 |
| `meeple_conv` 32→6 | 198 |
| **合計** | **4,586**（vs 408,606，**89 倍**） |

省下的 40 萬參數可以直接換成 trunk：`nn_width 32 → 64` 的 8 層 residual 是
148,992 → 594,432，剛好吃掉這筆預算，而且錢花在會傳資訊的地方。

#### 5.5.3 順序對齊（唯一容易寫錯的地方）

```cpp
EncodeTileAction(x, y, rot) = ((y * BOARD_SIZE + x) * 4 + rot);   // rot 變化最快
```

conv 輸出 `[B, 4, H, W]` 直接 `view` 出來的順序是 `(rot * H + y) * W + x` —— **通道最慢、rot 最慢，錯的**。
必須先 permute：

```cpp
// [B,4,H,W] -> [B,H,W,4] -> [B, H*W*4]，索引 = (y*W + x)*4 + rot ✓
tile_logits = tile_logits.permute({0, 2, 3, 1}).contiguous().view({-1, kTileActionCount});
```

meeple 的 6 個通道順序必須是 `[skip(-1), edge0, edge1, edge2, edge3, monastery]`，
對上 `EncodeMeepleAction(pos) = 900 + pos + 1`。

**單元測試**：對每個 `(x,y,rot)` 建一個只有該動作合法的 mask，檢查
`argmax(logits) == EncodeTileAction(x,y,rot)`；再用一張手刻的 `[B,4,H,W]` 張量
（值設成 `(y*W+x)*4+rot`）跑 permute+view，斷言結果等於 `arange(900)`。

#### 5.5.4 為什麼 tile 和 meeple 可以直接 concat

`LegalActions()` 依 `current_phase` 分派：`PHASE_TILE` 只回落子動作、`PHASE_MEEPLE` 只回 meeple 動作。
**兩組動作永遠不會同時合法**，所以 softmax 永遠只在其中一組內部做，
兩個 head 之間的 logit 尺度不需要校準 —— 這是接兩個異質 head 時通常最麻煩的問題，在這裡自動消失。

`last_plane` 取 `kLastPlacedPlane`（`(last_x,last_y)` 的 one-hot）。在 `PHASE_MEEPLE` 它剛好是
剛放下那張磚。在 `PHASE_TILE` 它指向上一張磚，但那時 meeple logits 全被 mask 掉，無所謂。
值得加一個 debug 斷言：`PHASE_MEEPLE` 時 `last_plane.sum() == 1`。

#### 5.5.5 全域池化偏置為什麼是必要的，不是裝飾

這是 P3。「該不該把城封在這裡」取決於：牌庫還剩幾張能接的牌、對手還有幾個 meeple、
現在領先還是落後（領先就求穩、落後就要賭大城）。這些全是全域量。

現況下 policy head 只拿到 `(x,y)` 那一格的 32 維 trunk 向量。理論上 trunk 可以把全域量搬過去，
但 §2.4-b 實測 trunk ERF 在 r=7 已衰減到 **0.29**、r=10 剩 **0.047** —— 搬不動。
`gpool` 分支用 **3,136 個參數**給每個位置補上一個全域條件偏置，一步到位。
這跟 §5.1 給 value head 加 pooling 是同一個修法，也是 KataGo 的 policy head 做法。

> 若之後做 §4.3 的獨立 global vector，`g` 改成 `FC(64 + G → 32)` 即可，其餘不動。

#### 5.5.6 跟旋轉增強的關係（這條是 §6.5 的乘數）

§1.5.3 量到旋轉增強的收益遠小於 4 倍。dense head 是主因之一：

- **dense**：動作 `(x,y,0)` 與它旋轉後的像 `(x',y',1)` 是 `policy_linear` 裡**兩條毫不相干的 450 維權重列**。
  增強要教會的是 900 條列之間的一致性 —— 900 組獨立參數，每組還只有 1.7% 的 batch 有梯度（P2）。
- **conv**：整個落子頭只有 `32×4 = 128` 個權重，**所有位置共用**。
  旋轉增強要約束的參數少了 3,000 倍，而每個參數拿到的梯度樣本多了 900 倍。

**預測**：#4 做完之後，旋轉增強的邊際效益應該明顯上升，
而且 `policy_kl`（目前 0.345）應該再降一截。

#### 5.5.7 順序修正：#5（`BOARD_SIZE 21`）必須排在 #4 之後

§7 原本寫「#5 做完 #1 後幾乎零成本」—— 對 value head 成立（pooling 之後 head 不綁棋盤大小），
**對 policy head 完全相反**：

| | dense head | conv head |
|---|---|---|
| 15×15 | `Linear(450→906)` = 408,606 | 4,586 |
| **21×21** | `Linear(882→1770)` = **1,561,140（全網路 ~89%）** | **4,586（完全不變）** |

而且 21×21 新增的 864 條落子列，合法頻率比現在的邊角還低，會重演 P2 的歸零。
**先 #4 再 #5。**

#### 5.5.8 落地順序（#4 自己可以拆兩步）

| 步驟 | 改什麼 | 動到哪 | 是否自足 |
|---|---|---|---|
| **A** | tile 改 conv + permute、meeple 改空間讀取、concat | 只有 `model.h` / `model.cc` | **是**，介面不變（仍回傳 `[B,906]`） |
| **B** | 加 gpool 偏置分支 | 同上 | 是 |
| C | `nn_width 32 → 64`，把省下的參數還給 trunk | `config` | 是 |

A + B 合起來約 40 行，**不用碰 `vpnet.cc`、`losses()`、`mcts.cc`、`carcassonne.cc`**。
跟 §5.3 步驟 A 一樣，形狀變了所以舊 checkpoint 不能載入（`init_checkpoint` 留空）。

---

## 6. 訓練 / 搜尋設定

### 6.1 `temperature_drop` 的單位是錯的（bug 等級）

`alpha_zero.cc`：`if (history.size() >= temperature_drop)`。
`history` 每回合 push **3 筆**（chance draw + tile + meeple）。
設 30 → **第 10 回合（全 71 回合）就切成完全貪婪**。
→ 改成只數 decision ply，或設成 3 倍以上（≥120），或改線性退火 `T: 1.0 → 0.25`。

### 6.2 `policy_alpha = 1.0` 太大

Dirichlet(α=1) 在 n 維單體上是**均勻分佈**。慣例是 `α ≈ 10 / 平均合法手數`。
平均 30.8 個落點 → α ≈ 0.3；meeple 節點只有 2–5 個動作，α=1.0 幾乎把 root policy 洗掉。
→ per-node `α = 10 / |legal actions|`。

### 6.3 800 sims 在這個分支因子下幾乎沒有深度

平均 30.8 落點 × 最多 24 個 chance outcome，每回合還搜兩次（tile + meeple）。
有效前瞻約 **1 手**，而且自對弈極慢（`states_per_s = 34.7`，一局 ~113k 次推論）。

- **短期**：self-play 降到 200–300 sims，換 3–4 倍資料量。
- **中期（推薦）**：**afterstate value**。讓網路能直接評估 chance node（落子+meeple 之後、抽牌之前），
  MCTS 在 chance node 取網路值當 leaf value，而不是強制往下抽一張牌。
  把「對 24 種抽牌取期望」交給網路學而非取樣，方差大降、分支因子從 30×24 降到 30。
  （需要讓 `ObservationTensor` 在 `CurrentPlayer() == kChancePlayerId` 時也能寫出來，
  並在 `mcts.cc` 的 `while (... || (IsChanceNode && dont_return_chance_node_))` 開一個分支。）

### 6.4 value target 的方差

target 是最終 z，包含之後所有抽牌的運氣。建議混合搜尋值：
`target = λ·z + (1−λ)·Q_root`，λ ≈ 0.5–0.7。

### 6.5 資料增強（**已實作並驗證**）

> **狀態更新**：`augment_rotations` 已上線，而且做法正確 —— augmentation 在**採樣時**進行
> （`alpha_zero.cc:480`，"A fresh random orientation every time a state is sampled"），
> buffer 只存 1 份。不是插入時存 4 份（那會讓同樣大小的 buffer 只裝 1/4 的局面）。
>
> 下面第 4 項（meeple 動作置換）也做對了：`SideGroups` + `RotateMeepleSide` 是
> **重算正規代表**，不是單純循環邊編號。手算驗證 type 10：
> `groups = [0,1,1,0]` → `RotateMeepleSide(0,k=1) = min{1,0} = 0`、
> `RotateMeepleSide(1,k=1) = min{2,3} = 2`，與 §3.10 推導的 `rot0 {0,1} → rot1 {0,2}` 吻合。
> `RotateObservation` 也用同一個函式轉 `kLegalMeeplePlane`，一致。

整盤旋轉 90°/180°/270° 是**嚴格合法**的對稱，**免費 4 倍資料**。要同步旋轉的東西有四樣：

1. 盤面張量的 (y, x)
2. 每張磚的 `rotation`，以及所有 **per-side 平面**（4 邊地形、元件平面 +0..+4）的邊順序
3. policy target 的落子部分 `(x, y, rot)`
4. **policy target 的 meeple 部分** —— 4 個邊動作要循環置換（N→E→S→W），
   `skip`（index 900）與 `monastery`（index 905）不動

> **第 4 項最容易漏。** 漏了不會報錯、不會 assert，只會讓 25% 的 meeple policy target
> 系統性錯位。寫一個往返測試：旋轉 4 次應該回到原狀態，且 policy target 逐項相等。

鏡射看起來也合法（type 17 CGRR 與 type 18 CRRG 互為鏡像且各 3 張），
但請先寫測試驗證每個 type 的鏡像都在牌組裡且張數相同再啟用。

> 旋轉增強還有一個好處：它會把 225 個位置權重的訓練訊號**平均化**，
> 直接緩解 §2.4-c 那個 185× 的占用率失衡。
> **但這個好處在做完 §7 #1（global pooling）與 #4（conv policy head）之後會消失**
> —— 那時已經沒有位置相依權重了。剩下的是純粹「教 trunk 旋轉對稱」，真實但比較小。
> 所以現在量到的 augmentation 效果會比之後大。

### 6.5b `replay_buffer_reuse` 要不要跟著 ×4？不要。

`learn_rate = buffer_size / reuse` 是每個 learner step 消耗的**新**狀態數；
每步做 `buffer.Size() / train_batch_size = 262144/2048 = 128` 次 gradient update。
所以每個狀態在被擠出 buffer 前期望被抽到約 `reuse` 次。

1. **augmentation 給的是 ×4 的輸入多樣性，不是 ×4 的資訊。** 同一局面的 4 個旋轉共用
   完全相同的 `z` 和完全相同（只是置換過）的 policy target —— 完全相關的樣本，
   拿不到 4 個獨立樣本該有的 √4 變異數縮減。buffer 的資訊量沒變。
2. **`reuse` 的真正約束是 staleness，不是記憶化。** 旋轉不會讓資料變新。
3. **目前也不是 overfit**（`policy_kl = 0.436`、`pred_entropy > target_entropy`）。
   reuse 太高的病徵是相反的：KL→0、pred_entropy 貼上 target，但 eval 停滯。

唯一支持**小幅**提高的理由：採樣時隨機挑 1/4 方位，抽 `reuse` 次的期望相異方位數是
`4·(1 − (3/4)^reuse)`，而且**在 4 個方位就飽和**：

| reuse | 4 | **6** | 8 | 12 | 16 |
|---|---|---|---|---|---|
| 期望相異方位 | 2.73 | **3.29** | 3.60 | 3.87 | 3.96 |

要動的話 **4 → 6**，不是 4 → 16。超過 8 之後就是純重複，staleness 的代價重新接管。

### 6.6 優化器與 lr（v6 重寫）

**先更正**：optimizer 是 **`torch::optim::Adam`**（`vpnet.cc:178`），不是 SGD。
Adam + batch 2048 + `lr = 1e-4` 是個**正常值**，不是明顯壞掉的值。
先前這裡寫「lr 太低是瓶頸」是在沒看 optimizer 的情況下寫的，講太滿。

**而且 gradient update 的數量我算錯過**：

```
193 learner steps × 128 updates/step = 24,704 次 mini-batch（batch 2048）
AGZ:                                   700,000 次 mini-batch（batch 2048）
```

差 **28 倍，不是 3600 倍**。「193 steps vs AGZ 700k steps」把 learner step 和
gradient update 混為一談了。

**「underfit」這個判斷本身也要打折。** `pred_entropy (1.504) > target_entropy (1.063)`
不一定是 underfit：policy target 是**隨機變數**（每次搜尋重抽 Dirichlet 噪音 ε=0.25），
而 Carcassonne 局面幾乎不重複，網路學的是位置空間上的平滑函數 ——
**一群尖銳但峰值不同的分佈，期望必然比任一個都平。** 這部分不可消除。

**所以先量，不要先調**（見 §6.7 的 held-out loss）。

**若確認要調 lr**，用 `replay_buffer.data`（`alpha_zero.cc:464` 有存檔）做**離線** lr range test：
載入 `checkpoint-N.pt` + buffer，對 `lr ∈ {1e-4, 3e-4, 1e-3, 3e-3}` 各跑 500 次 update 畫曲線。
一小時內回答，不用動 self-play。

改的話（Adam，batch 2048）：第一步試 **3e-4**（不是 1e-3，BN+resnet 容易震盪）；
**warmup 比 peak 重要**（500–2000 次 update 線性 warmup）；cosine 對著**總 update 數**而非 learner step。
libtorch 沒內建 scheduler，每次 update 前直接改：

```cpp
static_cast<torch::optim::AdamOptions&>(
    model_optimizer_.param_groups()[0].options()).lr(new_lr);
```

**Gotcha**：`LoadCheckpoint` 會一併載入 Adam 的 optimizer state；改 lr 通常無妨，
但**改架構時絕對不能載**（形狀對不上）。另見 §2.4-c 關於改用 AdamW 的段落。

容量方面 `nn_width = 128 / depth = 10`，但等 feature 與 head 修好再放大。

### 6.7 觀測指標

- **加一個 held-out loss（優先度最高的一項儀表）**：對剛從 queue 拿出來、
  **還沒進 buffer** 的 trajectory 跑一次 forward-only 的 `model->losses()`。
  那是天然的 held-out set —— 當前網路產生、從沒訓練過。幾十行，一次回答整類調參問題：

  | `fresh_kl` vs `train_kl` | 意思 | 該做什麼 |
  |---|---|---|
  | 兩者都 ≈ 0.44 | 到了容量或噪音底線 | 調 lr / reuse 都沒用；改架構（§7 #1、#4） |
  | `fresh ≫ train` | overfit | 降 reuse、加正則 |
  | 兩者都高，且單步 128 次 update 內 loss 仍穩定下降 | 真 underfit | 這時候才調 lr |

- 把 **raw NN value 的 sign accuracy** 與 MCTS root value 分開記錄（**已實作**：
  `raw_value_accuracy` / `raw_value_prediction`）。
- 每次 checkpoint 跑一次 `tools/check_value_head.py`。改成 pooling 之前盯 `β/|γ|`、`cos(eff, 1)`、
  `corr(占用率, W1 per-pos std)`；改完之後這幾個指標消失（權重被硬編碼），
  改盯 `value_conv` 32 個通道的 BN γ 分佈（若大部分塌到 0 表示通道用不完，可以減）。

---

## 7. 落地順序

| # | 改動 | 工作量 | 動到哪 | 解決 | 效果 |
|---|---|---|---|---|---|
| **0** | **held-out loss**（對還沒進 buffer 的 trajectory 算一次 forward-only loss） | ~40 行 | `alpha_zero.cc` | §6.7 | **先做這個** —— 在它之前調 lr / reuse 都是猜 |
| ~~1~~ | ~~value head：1→32 通道 + global pooling~~ | — | — | — | **已完成並驗證（§1.5）**：eval −0.93 → **+0.14**，stage 4–5 準確率 +13 分 |
| **2** | `static_diff` / `pending` 進觀測（多尺度 clip，**不要 bucket one-hot**），`tanh(d/30)` → `clip` | ~60 行 | `game.hpp`, `carcassonne.h/cc` | §2.4-a 部分 | 大 |
| **3** | `opens` / `getScore()` / `2×getScore()` 元件廣播平面 | ~60 行 | `carcassonne.h/cc` | §3.3 | 大 |
| **4** | **policy head 改 conv + gpool 偏置；meeple head 改空間讀取**（§5.5，程式碼 §9.4） | ~40 行 | 只有 `model.h`/`model.cc` | §2.4-e, §3.9, §3.10 | **大**（參數 408,606 → 4,586；修掉實測的權重歸零） |
| **5** | `BOARD_SIZE 15 → 21`（**必須排在 #4 之後**，見 §5.5.7） | 一個常數 + action space | `game.hpp` | §3.6 | 中高 |
| ~~6~~ | ~~旋轉增強 ×4~~ | — | — | — | **已完成並驗證**（§6.5） |
| **7** | `temperature_drop` 單位、`policy_alpha=10/n`；sims 800→300 **需先做對打實驗**（§6.3） | 幾行 | `alpha_zero.cc` | §6.1–6.3 | 中 |
| **8** | Adam → AdamW（移除 `losses()` 的顯式 L2） | ~10 行 | `vpnet.cc`, `model.cc` | §2.4-c 的耦合衰減 | 中（#1、#4 之後傷害已減輕） |
| **9** | 全域向量獨立輸入，移除 50 個廣播 plane | 中等 | model 輸入介面 | §3.5 | 中 |
| **10** | **score-margin 分佈頭（WDL 由它導出）** | 大 | `losses()`, `vpnet.cc` | §2.4-a, §2.4-d | 大 |
| **11** | afterstate value（chance node 直接評估） | 大 | `mcts.cc`, `carcassonne.cc` | §6.3 | 大 |

> **不建議單獨做 WDL。** 它的效益只有 §2.4-d 的梯度形狀 + 校準可觀測性（中等），
> 而真正解監督問題的是 score head。既然兩者動到同樣的檔案，就一起做（#9）。

---

## 8. 驗證測試

1. **不用訓練的 sanity check**：取 10 萬個 self-play 狀態，分別用
   (a) 現在的 `kScoreDiffPlane`、(b) `static_diff`，做 logistic regression 預測勝者，比較各 stage 的 AUC。
   隨機對局是 0.717 vs 0.992（最後一手），請在你的 self-play 分佈上再確認一次。
2. **終局倒數一手單元測試**：隨機生成 1000 個「再一個決策就結束」的狀態，
   檢查 `sign(net_value) == sign(true_result)`。改動 #1+#2 之後應 > 0.97。
3. **權重健康檢查**（`tools/check_value_head.py`）：改動 #1 之前 `β/|γ|` 應 ≫ 1（實測 0.07）、
   `cos(eff, 1)` 應接近 1（實測 0.048）。
4. **有號探針**（`tools/erf.py`）：`plane15 − plane20` 方向的端到端敏感度應處處為正。
   實測只有 89/225 為正 —— 這個數字改完 #1 之後應該大幅上升。
5. ~~**回歸指標**：AZ vs 800-sim rollout MCTS 的勝率。目前 **−0.656**。改完 #1+#2 應該要翻正。~~
   → **通過，而且只做 #1 就翻正了**（0916 step 62 = **+0.14**，§1.5）。下一個門檻：做完 #2+#3 之後，
   `raw_value_accuracy` 的 stage 6 應該從 **0.918 → >0.98**，stage 4 從 **0.706 → >0.85**。
6. **policy head 索引對齊**（改動 #4，§5.5.3）：建一個值為 `(y*W+x)*4+rot` 的 `[1,4,H,W]` 張量，
   跑 `permute({0,2,3,1}).contiguous().flatten(1)`，斷言結果等於 `arange(900)`。
   再對每個動作建單一合法 mask，檢查 `argmax(logits)` 等於該動作。**這一步錯了不會 crash，只會學不起來。**
7. **policy 權重衰減回歸**（`tools/readckpt.py`）：改動 #4 之前，`policy_linear.weight` 的
   per-(x,y) 列範數在 r=7 是 0.087、(7,7) 是 0.000（初始 0.577）。改完之後這個指標不存在了 ——
   改為檢查 `tile_conv_.weight` 的 4 個輸出通道範數彼此相近（rot 對稱，旋轉增強下應該幾乎相等）。
8. **`opens` 溢位檢查**：`Feature::opens` 是 `uint8_t`，`FeatureModule::placeTileOnBoard`
   裡無條件 `opens -= 2`。分析上不會為負，加一個 `SPIEL_CHECK_GE(opens, 2)` 幾乎零成本。

---

## 9. 程式碼片段

### 9.1 value head（改動 #1）

`model.cc` config（約 line 299）：

```cpp
constexpr int kValueFilters = 32;
ResOutputBlockConfig output_config = {
    /*input_channels=*/config.nn_width,
    /*value_filters=*/kValueFilters,                  // 1 -> 32
    /*policy_filters=*/2,
    /*kernel_size=*/1,
    /*padding=*/0,
    /*value_linear_in_features=*/2 * kValueFilters,   // avg ⊕ max，不再是 1*w*h
    /*value_linear_out_features=*/256,                // 原本是 config.nn_width
    /*policy_linear_in_features=*/2 * width * height,
    /*policy_linear_out_features=*/config.number_of_actions,
    /*value_observation_size=*/2 * kValueFilters,     // 不再用來 view()
    /*policy_observation_size=*/2 * width * height};
```

`ResOutputBlockImpl::forward`：

```cpp
torch::Tensor v = torch::relu(value_batch_norm_(value_conv_(x)));  // [B, 32, H, W]
torch::Tensor flat = v.flatten(2);                                 // [B, 32, H*W]
torch::Tensor avg  = flat.mean(2);                                 // [B, 32]
torch::Tensor mx   = std::get<0>(flat.max(2));                     // [B, 32]
torch::Tensor value_output = torch::cat({avg, mx}, 1);             // [B, 64]
value_output = torch::relu(value_linear1_(value_output));
value_output = torch::tanh(value_linear2_(value_output));
```

（刪掉原本的 `view({-1, value_observation_size_})`。）

### 9.2 `static_diff`（改動 #2）

`game.hpp` — `Carcassonne` 加 const 方法：

```cpp
// 「若現在立刻結束」的終局補分（不含已入袋分數）
void getPendingScore(int *pending) const {
    pending[0] = pending[1] = 0;
    if (current_phase == PHASE_TERMINAL) return;   // ← 終局已經加過一次，再算會重複計分
    Carcassonne copy = *this;                      // ~14KB trivial copy，約 3-5us
    int before[2] = {player_scores[0], player_scores[1]};
    copy.resolveEndGameScore();                    // 注意：非冪等，只能對複本呼叫
    pending[0] = copy.player_scores[0] - before[0];
    pending[1] = copy.player_scores[1] - before[1];
}
```

> 正式版應改成不複製的 const 累加器（`FeatureModule` / `MonasteryModule` 各加一個
> `accumulatePending(int*) const`，用 `find(i) == i` 掃 root），約 0.5–1 µs。
> 用上面的複製版當 oracle 寫等價測試 —— `tools/common.hpp` 的 `diag::PendingDiff` 就是它。

`carcassonne.cc` — `ObservationTensor` 尾段：

```cpp
int pending[2];
game_state_.getPendingScore(pending);
const int opp = 1 - player;
const float banked = float(game_state_.player_scores[player] -
                           game_state_.player_scores[opp]);
const float statd  = banked + float(pending[player] - pending[opp]);

auto clip = [](float v) { return std::max(-1.0f, std::min(1.0f, v)); };

BroadcastPlane(values, kScoreDiffPlane,      clip(banked / 20.0f));  // tanh(d/30) -> clip
BroadcastPlane(values, kStaticDiffPlane,     clip(statd  / 30.0f));  // 大局量級
BroadcastPlane(values, kStaticDiffMidPlane,  clip(statd  / 10.0f));  // 中段
BroadcastPlane(values, kStaticDiffFinePlane, clip(statd  /  3.0f));  // 決勝區間
BroadcastPlane(values, kPendingMinePlane,    pending[player] / 20.0f);
BroadcastPlane(values, kPendingOppPlane,     pending[opp]    / 20.0f);
```

plane 配置（`kGlobalFeaturePlanes` 6 → 11，`kObservationPlanes` 80 → 85）：

```cpp
inline constexpr int kStaticDiffPlane     = 80;
inline constexpr int kStaticDiffMidPlane  = 81;
inline constexpr int kStaticDiffFinePlane = 82;
inline constexpr int kPendingMinePlane    = 83;
inline constexpr int kPendingOppPlane     = 84;
static_assert(kObservationPlanes == kPendingOppPlane + 1);
```

**三個必寫的測試**：
(1) 不複製的 const 版對複製版等價；
(2) terminal 狀態 `getPendingScore` 回 `{0,0}`；
(3) 同一盤面 `ObservationTensor(0)` 與 `ObservationTensor(1)` 的 `kStaticDiff*` 平面剛好相反號。

**連帶**：舊 checkpoint 不能載（輸入通道變了）、inference cache 全失效、
`integration_tests/playthroughs/carcassonne.txt` 要重新產生。

### 9.3 元件廣播平面（改動 #3）

放在 `ObservationTensor` 既有的 `for (y) for (x)` 迴圈內：

```cpp
const int me = player, opp = 1 - player;
for (int s = 0; s < 4; ++s) {
    if (tile.edge[s] == GRASS) continue;
    const Feature &f = game_state_.featureAt(placement.id, s);   // 新增一個 const accessor
    const float sc = float(f.getScore());
    const int mm = f.meeple_count[me], mo = f.meeple_count[opp];
    const float sgn = (mm > mo) ? 1.0f : ((mo > mm) ? -1.0f : 0.0f);

    SetPlaneValue(values, kOpensPlane      + s, x, y, std::min<int>(f.opens, 6) / 6.0f);
    SetPlaneValue(values, kScoreNowPlane   + s, x, y, std::min(sc / 12.0f, 1.0f));
    SetPlaneValue(values, kScoreClosedPlane+ s, x, y,
                  f.type == CITY ? std::min(2.0f * sc / 24.0f, 1.0f) : std::min(sc / 12.0f, 1.0f));
    SetPlaneValue(values, kMeepleDiffPlane + s, x, y, std::clamp((mm - mo) / 3.0f, -1.0f, 1.0f));
    SetPlaneValue(values, kSignedPendPlane + s, x, y, std::clamp(sgn * sc / 12.0f, -1.0f, 1.0f));
}
```

記得同步更新 `kObservationPlanes`、各 offset 常數與 `static_assert`。

---

### 9.4 policy head 改 conv（改動 #4）

**`model.h`** —— 換掉 `ResOutputBlockImpl` 的 policy 成員：

```cpp
  // 移除：torch::nn::Linear policy_linear_;  int policy_observation_size_;
  torch::nn::Conv2d   policy_conv_;        // W  -> PF (=32)
  torch::nn::Conv2d   gpool_conv_;         // W  -> PF
  torch::nn::Linear   gpool_fc_;           // 2*PF -> PF
  torch::nn::BatchNorm2d policy_batch_norm_;  // PF
  torch::nn::Conv2d   tile_conv_;          // PF -> 4
  torch::nn::Conv2d   meeple_conv_;        // PF -> kMeepleActionCount (=6)
  int num_extra_actions_;                  // = num_actions - 4*H*W
```

`ResOutputBlockConfig` 把 `policy_filters` 沿用成 PF（改成 32），
`policy_observation_size` 不再需要，改存 `num_extra_actions`。

**`model.cc`** —— `ResOutputBlockImpl::forward` 的 policy 段：

```cpp
std::vector<torch::Tensor> ResOutputBlockImpl::forward(torch::Tensor x,
                                                       torch::Tensor mask,
                                                       torch::Tensor last_plane) {
  // ... value 段不動 ...

  // --- policy ---
  torch::Tensor p = policy_conv_(x);                       // [B,PF,H,W]

  // 全域池化偏置（§5.5.5）
  torch::Tensor g = gpool_conv_(x).flatten(2);             // [B,PF,HW]
  g = torch::cat({g.mean(2), g.amax(2)}, 1);               // [B,2*PF]
  g = gpool_fc_(g).unsqueeze(-1).unsqueeze(-1);            // [B,PF,1,1]

  p = torch::relu(policy_batch_norm_(p + g));              // [B,PF,H,W]

  // 落子：[B,4,H,W] -> [B,H,W,4] -> [B,H*W*4]，索引 = (y*W+x)*4+rot（§5.5.3）
  torch::Tensor tile_logits = tile_conv_(p)
                                  .permute({0, 2, 3, 1})
                                  .contiguous()
                                  .flatten(1);

  // meeple：用 last-placed one-hot 從那一格讀出 6 個 logit（§5.4）
  torch::Tensor meeple_logits =
      (meeple_conv_(p) * last_plane).sum(/*dim=*/{2, 3});  // [B,6]

  torch::Tensor policy_logits = torch::cat({tile_logits, meeple_logits}, 1);
  policy_logits = torch::where(mask, policy_logits,
                               -(1 << 16) * torch::ones_like(policy_logits));

  return {value_output, policy_logits};
}
```

**`ModelImpl::forward_`** —— 把 last-placed 平面從原始觀測切出來傳進去。
輸入張量進來時是攤平的 `{B, flat_input_size}`，reshape 發生在 `ResInputBlock` 裡，
所以在迴圈前先自己 view 一次：

```cpp
  // 需要在 ModelImpl 存下 channels_/height_/width_ 與 last_placed_plane_idx_
  torch::Tensor obs = x.view({-1, channels_, height_, width_});
  torch::Tensor last_plane = obs.narrow(1, last_placed_plane_idx_, 1);  // [B,1,H,W]
```

`last_placed_plane_idx_` 從 `ModelConfig` 傳進來（Carcassonne 是 `kLastPlacedPlane = 40`），
不要在 `model.cc` 裡 include 遊戲的標頭 —— 那會把通用的 alpha_zero_torch 綁死在這個遊戲上。

**不用改的**：`vpnet.cc`（`Inference` 讀的還是 `[B,906]` 的 logits）、
`losses()`、`mcts.cc`、`carcassonne.cc`。

**必做的斷言**（`ResOutputBlockImpl` 建構時）：

```cpp
SPIEL_CHECK_EQ(num_actions, 4 * height * width + num_extra_actions_);
```

---

## 附錄：診斷工具

全部的量測程式都已收進 **`AlphaCarcassonne/tools/`**，附 `Makefile` 與 `README.md`。
不需要 OpenSpiel 本體、CMake 或 libtorch：C++ 只依賴 `open_spiel/games/carcassonne/game/`
底下的純引擎，Python 只需要 numpy。

```bash
cd tools
make && make run                                   # 6 支 C++ 診斷 -> results/
cd results
python3 ../readckpt.py /path/to/checkpoint-193.pt ck.npz
python3 ../check_value_head.py ck.npz              # value head 三項健康檢查
python3 ../erf.py            ck.npz                # 有效感受野 / 端到端 / 有號探針
```

| 程式 | 量什麼 | 本文小節 |
|---|---|---|
| `diag_value_gap` | 每個決策點的 `banked` vs `static` 分差，預測最終勝者的準確率 | §2.1 |
| `diag_local_decomp` | 「完美 per-cell CNN + 求和」在現有 80 plane 下能否重建 pending | §2.2 |
| `diag_board` / `diag_board25` | bounding box、合法落點數、跨度分佈、終局待結分組成 | §3.6 |
| `diag_opens` | 有盾磚型的城市元件數；`opens` vs 最終封口率 | §3.2, §3.3 |
| `diag_occupancy` | 每格「有磚」的機率熱圖（輸出 `occupancy.csv`） | §2.4-c |
| `readckpt.py` | 純 numpy 讀 libtorch checkpoint → `.npz` | §2.4-c/e |
| `check_value_head.py` | `β/\|γ\|`、`cos(eff, 1)`、per-position std vs 占用率 | §2.4-c |
| `erf.py` | trunk 有效感受野（FFT 卷 17 個核）、端到端敏感度、有號探針 | §2.4-b |

每支程式的原始碼開頭都註明了本文引用的參考結果，`tools/README.md` 有完整說明
與三項 caveat（`#define private public`、線性化的 Jacobian、隨機對局的分佈偏差）。

**改完 §7 第 1 項之後**：`check_value_head.py` 的 (2)(3) 與 `erf.py` 的 (2)(3)
會自動偵測新架構並跳過（權重被硬編碼，這些指標失去意義）。
改盯 `value_conv` 各通道的 BN γ 分佈，以及 raw NN value 分 stage 的 sign accuracy。