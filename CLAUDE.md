# AlphaCarcassonne 診斷 + Feature / 訓練重新設計

> **依據**
> - 原始碼：`open_spiel/games/carcassonne/{carcassonne.h,carcassonne.cc,game/*}`、
>   `open_spiel/algorithms/alpha_zero_torch/{model.cc,model.h,alpha_zero.cc,vpevaluator.cc,vpnet.cc}`、
>   `open_spiel/algorithms/mcts.cc`
> - 訓練紀錄：`0904/learner (7).jsonl`（step 193）、`0904/config (3).json`
> - **權重實測**：`0904/checkpoint-193.pt`（用附錄 B 的純 numpy 工具讀出）
> - **模擬實測**：用你的 `game/` 原始碼直接編譯的 6 支診斷程式（附錄 A）
>
> **設定**：2 人、基本版、**關閉農夫**。
>
> **修訂紀錄**
> | 版本 | 內容 |
> |---|---|
> | v1 | 主張「CNN 在結構上算不出終局待結分」。**錯誤**，v2 已整節重寫 |
> | v2 | 修正 §2；刪除「元件大小 / 盾牌數 / 封口旗標需要廣播」三個錯誤建議 |
> | v3 | §2.4-c 從論證升級為權重實測；加入參數預算與占用率熱圖 |
> | **v4（本版）** | 加入 trunk 有效感受野的實測（§2.4-b）；新增 §2.4-d 損失函數；修正 §3.8 對和局的誇大；把 WDL 從落地順序第 1 步拆出（§7）；整併所有實測到附錄 A |

---

## 0. TL;DR

| # | 問題 | 證據 | 嚴重度 |
|---|---|---|---|
| 1 | **value 的「和」只存在於最後一步**，trunk 傳不動、policy head 看不到。實測 trunk 有效感受野在 r=7 已衰減到 **0.29**、r=10 剩 **0.047** | §2.4-b | 最高 |
| 2 | **value head 的形狀跟「有號空間求和」不合**：1 通道 + ReLU 砍掉 **47%** 的格子；有效讀出與均勻求和的 cosine 只有 **0.048**；93 個週邊格子的權重**還停在初始值** | §2.4-c | 最高 |
| 3 | **沒有對「數值」的直接監督**：唯一訊號是對 ±1 做 MSE，中間隔著硬閾值 | §2.4-a | 高 |
| 4 | **`opens` 完全不在觀測裡** —— 唯一無法由局部推導的全域量，而封口 ×2 是最大的單一分數槓桿 | §3.3 | 高 |
| 5 | **15×15 棋盤 31% 的對局會被截斷**，還製造永遠填不掉的假開口 | §3.6 | 中高 |
| 6 | **69.5% 的參數卡在 policy head 的一層全連接**，value head 只有 1.2% | §2.4-e | 中高 |
| 7 | **tanh 飽和在錯得最離譜時把梯度掐掉**（stage 3 的 max 處衰減 1.1 萬倍） | §2.4-d | 中 |
| 8 | 62% 的輸入張量是空間常數廣播平面 | §3.5 | 中 |
| 9 | `temperature_drop` 單位錯誤、`policy_alpha=1.0` 過大、800 sims 幾乎沒有深度 | §6 | 中 |

**最重要的單一指標**：`eval.results = [-0.65625]` —— 同樣 800 sims 下，**AZ 正在大輸給隨機 rollout MCTS**。
rollout 之所以贏，正是因為它走到終局、免費拿到了網路算不出來的那半邊分數。

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
改成 conv policy head（§5）可以把這 40 萬參數還給 trunk。

另外：62% 的輸入張量是常數平面（§3.5）；193 個 training step（AGZ 是 700k）。

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
應改成 `clip(diff/20)` **加一組 bucket one-hot**（§4.3）。

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
| **31** | **banked 分差 bucket one-hot（−15..+15，兩端飽和）** |
| 2 | 我方 / 對手 pending 分數 `/20` |
| 1 | **`static_diff` = banked + pending，`clip(d/20)`** |
| **31** | **`static_diff` bucket one-hot（−15..+15）** |
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

**bucket one-hot 很重要**：勝負由 ±1 分決定，壓進一個 scalar 等於要網路在 `tanh` 的近線性區
用一個維度做出銳利閾值；one-hot 讓它線性可分。

**注入方式**（KataGo 做法）：`Linear(160 → C)` 後加到 input conv 輸出當 per-channel bias
（每 2 個 block 再做一次），**同時** concat 到 value / policy head 的 global pooling 之後。

---

## 5. 網路架構

```
trunk:   input conv(48→W) + global-bias  →  D × residual block (W ch, 3×3, BN, ReLU)
                                             ↑ 每 2 block 插一次 global pooling bias

policy:  1×1 conv(W→4)                    → 4×21×21 = 1764 落子 logits   (取代 40 萬參數的 dense)
         pooled(W) ⊕ global → MLP → 6     → meeple logits
         concat → mask → softmax

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

### 6.5 沒有做資料增強

整盤旋轉 90°/180°/270°（同步旋轉每張磚的 rotation、side plane、policy target 的 (x,y,rot)）
是**嚴格合法**的對稱，**免費 4 倍資料**。
鏡射看起來也合法（type 17 CGRR 與 type 18 CRRG 互為鏡像且各 3 張），
但請先寫測試驗證每個 type 的鏡像都在牌組裡且張數相同再啟用。

> 旋轉增強還有一個好處：它會把 225 個位置權重的訓練訊號**平均化**，
> 直接緩解 §2.4-c 那個 185× 的占用率失衡。即使暫時不換 global pooling，這一項也有效。

### 6.6 優化超參

`lr = 1e-4` 固定、193 steps、`total_states = 16M`。以 AZ 標準這仍是**極早期**（AGZ 是 700k steps）。
建議 warmup + cosine，peak `lr = 2e-4 ~ 1e-3`（batch 2048），
`nn_width = 128 / depth = 10`（等 feature 與 head 修好再放大）。

### 6.7 觀測指標

- 把 **raw NN value 的 sign accuracy** 與 MCTS root value 分開記錄。
- 每次 checkpoint 跑一次附錄 B 的權重檢查。改成 pooling 之前盯 `β/|γ|`、`cos(eff, 1)`、
  `corr(占用率, W1 per-pos std)`；改完之後這幾個指標消失（權重被硬編碼），
  改盯 `value_conv` 32 個通道的 BN γ 分佈（若大部分塌到 0 表示通道用不完，可以減）。

---

## 7. 落地順序

| # | 改動 | 工作量 | 動到哪 | 解決 | 效果 |
|---|---|---|---|---|---|
| **1** | value head：1→32 通道 + global pooling（輸出仍是 tanh scalar） | ~10 行 | 只有 `model.h`/`model.cc` | §2.4-b/c | **最大** |
| **2** | `static_diff` / `pending` 進觀測（含 bucket one-hot），`tanh(d/30)` → `clip(d/20)` | ~60 行 | `game.hpp`, `carcassonne.h/cc` | §2.4-a 部分 | 大 |
| **3** | `opens` / `getScore()` / `2×getScore()` 元件廣播平面 | ~60 行 | `carcassonne.h/cc` | §3.3 | 大 |
| **4** | policy head 改 conv（釋放 40 萬參數） | ~30 行 | `model.cc` | §2.4-e, §3.9 | 中高 |
| **5** | `BOARD_SIZE 15 → 21` | 一個常數 + action space | `game.hpp` | §3.6 | 中高（做完 #1 後幾乎零成本） |
| **6** | 旋轉增強 ×4 | ~100 行 | `alpha_zero.cc` | 資料量 + 位置權重均衡 | 中高 |
| **7** | `temperature_drop` 單位、`policy_alpha=10/n`、sims 800→300 | 幾行 | `alpha_zero.cc` | §6.1–6.3 | 中 |
| **8** | 全域向量獨立輸入，移除 50 個廣播 plane | 中等 | model 輸入介面 | §3.5 | 中 |
| **9** | **score-margin 分佈頭（WDL 由它導出）** | 大 | `losses()`, `vpnet.cc` | §2.4-a, §2.4-d | 大 |
| **10** | afterstate value（chance node 直接評估） | 大 | `mcts.cc`, `carcassonne.cc` | §6.3 | 大 |

> **不建議單獨做 WDL。** 它的效益只有 §2.4-d 的梯度形狀 + 校準可觀測性（中等），
> 而真正解監督問題的是 score head。既然兩者動到同樣的檔案，就一起做（#9）。

---

## 8. 驗證測試

1. **不用訓練的 sanity check**：取 10 萬個 self-play 狀態，分別用
   (a) 現在的 `kScoreDiffPlane`、(b) `static_diff`，做 logistic regression 預測勝者，比較各 stage 的 AUC。
   隨機對局是 0.717 vs 0.992（最後一手），請在你的 self-play 分佈上再確認一次。
2. **終局倒數一手單元測試**：隨機生成 1000 個「再一個決策就結束」的狀態，
   檢查 `sign(net_value) == sign(true_result)`。改動 #1+#2 之後應 > 0.97。
3. **權重健康檢查**（附錄 B）：改動 #1 之前 `β/|γ|` 應 ≫ 1（實測 0.07）、
   `cos(eff, 1)` 應接近 1（實測 0.048）。
4. **有號探針**（附錄 A / `erf.py`）：`plane15 − plane20` 方向的端到端敏感度應處處為正。
   實測只有 89/225 為正 —— 這個數字改完 #1 之後應該大幅上升。
5. **回歸指標**：AZ vs 800-sim rollout MCTS 的勝率。目前 **−0.656**。改完 #1+#2 應該要翻正。
6. **`opens` 溢位檢查**：`Feature::opens` 是 `uint8_t`，`FeatureModule::placeTileOnBoard`
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
    Carcassonne copy = *this;                 // ~7KB，約 20-30us，佔推論時間 <1%
    int before[2] = {player_scores[0], player_scores[1]};
    copy.resolveEndGameScore();
    pending[0] = copy.player_scores[0] - before[0];
    pending[1] = copy.player_scores[1] - before[1];
}
```

`carcassonne.cc` — `ObservationTensor` 尾段：

```cpp
int pending[2];
game_state_.getPendingScore(pending);
const int opp = 1 - player;
const float banked = float(game_state_.player_scores[player] -
                           game_state_.player_scores[opp]);
const float statd  = banked + float(pending[player] - pending[opp]);

BroadcastPlane(values, kScoreDiffPlane,   std::clamp(banked / 20.0f, -1.0f, 1.0f));
BroadcastPlane(values, kStaticDiffPlane,  std::clamp(statd  / 20.0f, -1.0f, 1.0f));
BroadcastPlane(values, kPendingMinePlane, pending[player] / 20.0f);
BroadcastPlane(values, kPendingOppPlane,  pending[opp]    / 20.0f);

const int b = std::clamp(int(std::lround(statd)), -15, 15) + 15;
BroadcastPlane(values, kStaticDiffBucketPlane + b, 1.0f);
```

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

## 附錄 A：診斷程式清單

全部用 `#define private public` 直接 include 你的 `game/*.hpp`，
以 `g++ -O2 -std=c++17` 連同 `game/*.cpp` 編譯。

| 程式 | 量什麼 | 結果在 |
|---|---|---|
| `diag.cpp` | 每個決策點的 `banked` / `static` 分差 vs 最終勝者，分 10 個 bucket | §2.1 |
| `diag2.cpp` | 棋盤 bounding box、每手合法落點數、終局 pending 的城/路/修道院組成 | §2.1, §3.6 |
| `diag3.cpp` | 對局最大跨度分佈（用 `BOARD_SIZE=25` 的副本編譯） | §3.6 |
| `diag4.cpp` | 「完美 per-cell CNN + 求和」在現有 80 plane 下能否重建 pending | §2.2 |
| `diag5.cpp` | 有盾磚型的城市元件數；`opens` vs 最終封口率 | §3.2, §3.3 |
| `diag6.cpp` | 每格「有磚」的機率熱圖 | §2.4-c |
| `erf.py` | trunk 有效感受野（FFT 卷積 17 個核）、端到端敏感度、有號探針 | §2.4-b |
| `readckpt.py` | 純 numpy 讀 libtorch checkpoint（附錄 B） | §2.4-c/e |

`diag4.cpp` 的核心（局部可分解性檢驗）：

```cpp
// 只用：邊地形 + shield + 磚內 link 去重 + 已廣播的元件 meeple 數
static int localEstimateDiff(const Carcassonne &g, int player){
  int opp = 1-player, total = 0;
  for (int y=0; y<BOARD_SIZE; ++y) for (int x=0; x<BOARD_SIZE; ++x) {
    Placement p = g.getPlacement(x,y);
    if (!p.id) continue;
    const Tile &t = full_deck[p.id][p.rotation];
    int seenLink[4]; int nSeen = 0;
    for (int i=0; i<4; ++i) {
      if (t.edge[i] == GRASS) continue;
      bool dup = false;
      for (int k=0; k<nSeen; ++k) if (seenLink[k] == t.link[i]) dup = true;
      if (dup) continue;                       // 只靠磚內 link 去重（網路能做到的極限）
      seenLink[nSeen++] = t.link[i];
      const Feature &f = g.features.featureMap.getSetData(g.features.edgeIndex(p.id, i));
      int mm = f.meeple_count[player], mo = f.meeple_count[opp];
      int sgn = (mm > mo) ? 1 : ((mo > mm) ? -1 : 0);
      if (!sgn) continue;
      total += sgn * (1 + ((t.edge[i] == CITY && t.shield) ? 1 : 0));
    }
  }
  for (int i=0; i<g.monasteries.active_monasteries.size(); ++i) {
    const MonasteryTracker &m = g.monasteries.active_monasteries[i];
    total += (m.owner == player ? 1 : -1) * m.tile_count;
  }
  return total;
}
```

`erf.py` 的核心（有效感受野）：

```python
def fold_bn(w, prefix):                       # BN 折進 conv
    g, var = d[prefix+'.weight'], d[prefix+'.running_var']
    return w * (g/np.sqrt(var+1e-3))[:,None,None,None]

def emb(k):                                   # 空間核 -> FFT，中心置於 index 0
    s = k.shape[-1]; c = s//2
    o = np.zeros(k.shape[:2]+(N,N)); o[..., :s, :s] = k
    return np.fft.fft2(np.roll(o, (-c,-c), axis=(-2,-1)))

compose = lambda FA, FB: np.einsum('acuv,cbuv->abuv', FA, FB)   # 先 B 後 A

F = emb(fold_bn(d['layers.0.input_conv.weight'], 'layers.0.input_batch_norm'))
I32 = np.zeros((32,32,1,1)); I32[np.arange(32), np.arange(32), 0, 0] = 1
for i in range(8):                            # residual: I + W2'∘W1'
    L = f'layers.{i+1}.res_{i}_'
    W1 = fold_bn(d[L+'conv_1.weight'], L+'batch_norm_1')
    W2 = fold_bn(d[L+'conv_2.weight'], L+'batch_norm_2')
    F = compose(emb(I32) + compose(emb(W2), emb(W1)), F)
F = compose(emb(fold_bn(d['layers.9.value_conv.weight'], 'layers.9.value_batch_norm')), F)
K = np.roll(np.real(np.fft.ifft2(F)), (N//2, N//2), axis=(-2,-1))
M = np.sqrt((K[0]**2).sum(0))                 # [N,N] 空間敏感度 -> 取環平均即得上表
```

---

## 附錄 B：checkpoint 權重檢查工具（純 numpy，不需要 torch）

`torch::save` 產生的是 TorchScript zip archive，Python 端不裝 torch 也能讀。
建議放成 `tools/readckpt.py`。

```python
import pickle, zipfile, numpy as np, sys, io, collections

path = sys.argv[1]
z = zipfile.ZipFile(path)
root = z.namelist()[0].split('/')[0]
DT = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'LongStorage': np.int64,
      'IntStorage': np.int32, 'HalfStorage': np.float16, 'BoolStorage': np.bool_}

def maketensor(storage, offset, size, stride, *rest):
    _, stype, key, _, numel = storage
    nm = stype if isinstance(stype, str) else getattr(stype, '__name__', str(stype))
    arr = np.frombuffer(z.read(f'{root}/data/{key}'), dtype=DT.get(nm.split('.')[-1], np.float32))
    n = int(np.prod(size)) if len(size) else 1
    return arr[offset:offset + n].reshape(tuple(size))

class Base:
    def __setstate__(self, st): self.__st__ = st
    def __init__(self, *a, **k):
        if a: self.__st__ = a

cache = {}
def mkclass(full):
    if full not in cache:
        cache[full] = type('C_' + full.replace('.', '_'), (Base,), {'__full__': full})
    return cache[full]

class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod == 'collections' and name == 'OrderedDict': return collections.OrderedDict
        if '_rebuild_tensor' in name: return maketensor
        if name.startswith('build_') or name == 'restore_type_tag':
            return lambda *a, **k: (a[0] if a else None)
        return mkclass(mod + '.' + name)
    def persistent_load(self, pid): return pid

obj = U(io.BytesIO(z.read(f'{root}/data.pkl'))).load()

found, seen = {}, set()
def walk(o, prefix='', d=0):
    if d > 14: return
    if isinstance(o, np.ndarray): found[prefix] = o; return
    if id(o) in seen: return
    if isinstance(o, (dict, list, tuple, Base)): seen.add(id(o))
    if isinstance(o, dict):
        for k, v in o.items(): walk(v, f'{prefix}.{k}' if prefix else str(k), d + 1)
    elif isinstance(o, (list, tuple)):
        for j, v in enumerate(o): walk(v, f'{prefix}[{j}]', d + 1)
    elif isinstance(o, Base):
        if getattr(o, '__st__', None) is not None: walk(o.__st__, prefix, d + 1)
        for k, v in vars(o).items():
            if not k.startswith('__'): walk(v, f'{prefix}.{k}' if prefix else k, d + 1)
walk(obj)
np.savez(sys.argv[2], **found)
for k, v in found.items(): print(f'{k:70s} {str(v.shape):18s} {v.dtype}')
```

健康檢查（改動 #1 之前適用）：

```python
d = np.load('ck.npz')
g, b = d['layers.9.value_batch_norm.weight'][0], d['layers.9.value_batch_norm.bias'][0]
print('beta/|gamma| =', b / abs(g))                       # 要 >> 1；checkpoint-193 實測 0.07
eff = d['layers.9.value_linear_2.weight'][0] @ d['layers.9.value_linear_1.weight']
print('cos(eff, 1)  =', eff.sum() / (np.linalg.norm(eff) * np.sqrt(eff.size)))   # 實測 0.048
print('per-pos std  =', d['layers.9.value_linear_1.weight'].std(0))              # 對照占用率熱圖
```