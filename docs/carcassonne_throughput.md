# 自我對弈吞吐與資料新鮮度（2026-09-20 / 09-21）

接續「推論瓶頸調查」（2026-09-16，GPU 17%、160 核只用到 8.4 核）。這份記錄
這兩天做的兩項改動、實測結果，以及瓶頸移到哪裡之後量到的新事實。

量測機器：RTX 5090 + AMD EPYC 9V74（160 執行緒）+ 188 GB RAM。

---

## 1. 做了什麼

### 1.1 多份 model 副本（commit `usage`）

原本的分析說推論被 `vpnet.cc` 的 `InferenceMutexForDevice` 串成一條線，**那不是
主因**。真正綁住全部推論的是 `DeviceManager::Get()`：它鎖住該 device 的
`model_mu`，而 `device_manager_.Get(n)->Inference(inputs)` 這個寫法會把整個
`Inference()`（含打包資料）留在鎖裡。`--devices=cuda:0` 只有一份 model，所以
16 個推論執行緒一次只跑得動一個。

`vpnet.cc` 那把鎖是**第二把**、按 device 名稱鎖的鎖，來自 `a0eea036`
（2026-05-26，配對評估工具一次開兩個 DeviceManager），不是 `fix seg fault`
（`b37e5e5b` 沒動過 `vpnet.cc`，修的是 `PlayGame` 用 `bots[-4]`）。
`docs/carcassonne_segfault.md` 也寫明 model 是由 `DeviceLoan` 的 mutex 保護的。

改動：刪掉那把按名稱鎖的 mutex。每份 model 仍由自己的 `model_mu` 保護，因此
`--devices=cuda:0,cuda:0,...` 就能在同一張 GPU 上跑多份副本並行推論
（`AddDevice` 逐份建立、`Get()` 挑 outstanding 最少的、每步存 checkpoint 後逐份
同步）。

### 1.2 Staging：把 CPU 工作移出鎖外（commit `usage 2`）

- 新增 `VPNetModel::InferenceStaging`：呼叫端持有、可重複使用的緩衝區，CUDA 時
  用 pinned memory。`Pack()` 擺放批次、`Unpack()` 讀回結果，兩者都不需要 model。
- 新增 `VPNetModel::RunInference(staging)`：只做 H2D → forward → D2H，拿掉多餘的
  `.clone()`，H2D 改成 pinned + `non_blocking`。
- `VPNetEvaluator::Runner()` 改成「鎖外打包 → 持鎖只跑 `RunInference` → 鎖外讀回」。
- 佇列不再複製觀測：`ThreadedQueue::Push` 改成傳值、`Pop` 用 `std::move`，
  Runner 用 `std::move(item->inputs)`。
- `model_->eval()` 從每次推論裡移出（建構子、載入 checkpoint 後、`Learn()` 結尾各
  一次）。那個寫入是唯一讓「多執行緒共用同一份 model」不安全的地方。

**為什麼佇列那兩次複製是關鍵**：`Runner()` 收批次時持有 `inference_queue_m_`，
而且設計上一次只有一個執行緒在收（「maximize batch size」）。每筆請求在那段裡被
複製兩次（86 KB × 2），一個 128 的批次就是 22 MB。本機實測單核複製頻寬
**10.9 GB/s**，等於**每批 2.06 ms 的序列化 memcpy**，換算成硬上限：

```
10.9 GB/s ÷ 172 KB per request ≈ 63,000 requests/s
```

跟批次大小、副本數、GPU 速度都無關。改完之後這一段只剩 deque 的指標搬移。

---

## 2. 實測效果

`0919`（1 份 model）對 `0921`（4 份副本 + staging），其餘設定完全相同：

| | 0919 | 0921 |
|---|---|---|
| requests/s | 28.5k | **72–80k** |
| states/s | 110 | **337** |
| 每步 `collect` | 423–1205 s | **1 s** |
| GPU | 17–25%、116 W | **54%、299 W** |
| 跑到 852k states | 502 分 | **148 分**（3.4×） |

驗證（本機 RTX 4060）：同批次大小下 staged 與舊路徑的輸出**逐位元相同**
（CPU 與 CUDA 都是 0 差異）；不同批次大小的差異是 float 噪音（CPU 1.5e-08、
CUDA 3.4e-06，後者來自 TF32 與不同 conv 演算法）。4 份副本 × 50 批的並行推論
與單執行緒結果相同。

---

## 3. 瓶頸移到哪裡

### 3.1 learner 端：每步 140 秒在存 replay buffer

目前一步是 `collect 1 s` + `buffer_save 140 s` + `learn 62 s` ≈ 202 s。

`checkpoint_freq` **管不到** buffer save，兩者無關：

- `alpha_zero.cc:467`：`replay_buffer.SaveBuffer(...)` 每步無條件執行。
- `alpha_zero.cc:514`：`checkpoint_freq` 只決定要不要多存一份有編號的
  `checkpoint-N.pt`；`checkpoint--1.pt` 每步都要存，因為其他副本靠它同步。

程式裡沒有 `buffer_save_freq` 這個選項。以 21×21 觀測計，`replay_buffer.data`
每步重寫約 29 GB（libnop 對 `vector<float>` 逐元素寫入，每個 float 多 1 byte
型別前綴）。

### 3.2 自我對弈端：一波 2048 局，餵飽 4.4 個 step

只有 actor 0–19 會寫 log（`alpha_zero.cc:203`，`if (num < 20)  // Limit the
number of open files`），這 20 個是 2048 個的抽樣。02:30 重啟後，每個抽樣 actor
到 03:30 只下完 **1 局** —— 一局約 **55–60 分鐘**，而且 2048 局幾乎同時完成：

| step | collect | queue（局） |
|---|---|---|
| 26 | **3380 s**（56 分，等第一波） | 0 |
| 27 | 1 s | 958 |
| 28 | 1 s | 1029 |
| 29 | 1 s | 656 |

learner 每步吃 462 局，所以一波 2048 局 ≈ **4.4 個 step**，而這 2048 局幾乎都是
同一份權重產生的。

**資料被用到時的平均年齡**：約 30 分（局中位置）+ 約 7 分（排隊）+ 約 27 分
（buffer 內 reuse 4）≈ **1 小時，約 4–5 個 step 的權重差**。

---

## 4. 兩個實驗結果

### 4.1 `inference_batch_size` 128 → 512：變慢 15%

| | step 24–25（128、4 副本） | step 27–29（512、6 副本） |
|---|---|---|
| requests/s | 75–77k | 64–68k |
| states/s | 318–320 | 315–319 |
| batch 平均 | 128.0（std 1） | 283（std 224） |
| 滿批比例 | 99.8% | 43%（30% 的批次 < 64） |

原因：**in-flight 請求數被 actor 數綁死**。每個 actor 的 MCTS 是循序的，一次只有
一個請求在外，所以 in-flight ≤ actors + evaluators = 2052。一個收集執行緒要湊
512 筆，16 個執行緒就要 8192 筆 —— 湊不到就等滿 `batch_wait_ms=1` 然後發出殘批。

而在封閉迴圈裡 `throughput = in-flight / latency`：

- batch 128：2052 / 77k ≈ **27 ms**
- batch 512：2052 / 64k ≈ **32 ms**

做完 staging 之後，每批的固定成本只剩一次上鎖與幾次 kernel launch，真正花錢的
（打包、PCIe 位元組、讀回）都是**每筆**成本，所以放大批次買不到任何攤提，只買到
延遲。**結論：改回 128。**
（這次比較同時把副本 4 → 6，不是乾淨的單變數實驗，但方向明確。）

### 4.2 `actors` 2048 → 4096：只會讓資料更舊

伺服端接近飽和時吞吐固定，Little's law 給出 `latency = in-flight / throughput`，
所以 actor 加倍 → 每個請求的延遲加倍 → 一局要 2 小時。而

```
每小時局數 = actors / 一局時間 = 2N / (2 × 60 分) = N / 60 分   ← 不變
```

資料量一樣，但每一筆都是「舊一倍的權重」產生的，而且一波變成 4096 局 ≈ 8.8 個
step 用同一份權重。佇列裡本來就積著 600–1000 局（`collect` 只有 1 s），代表自我
對弈的產出已經追上 learner 的消耗，**多出來的產能只會變成更長的排隊**。

反過來才是對的：actor 減半 → 延遲減半 → 一局 30 分鐘 → 每小時局數不變、新鮮度
加倍。下限是要餵得飽伺服端（6 副本 × 128 ≈ 768 筆同時在途），所以約 1024 個
actor 是底線。

---

## 5. 訓練指標看起來不動、但 eval 上升

`0921` 前 13 步：value loss 0.67 → 0.75、policy loss 1.81 → 1.82、
KL 0.075 → 0.32，而 eval 從 −1.0 升到 −0.21。原因有四個，都有數據：

1. **量的東西不一樣**。eval 是 AZ（網路 + 800 sims 搜尋）對**固定**對手
   （`RandomRolloutEvaluator` 的 MCTS，永遠不變）；loss 與 accuracy 則是拿
   網路自己搜尋出來的 target 在自己產生的局面上算 —— 分母跟著一起變強，
   絕對強度的進步在這些指標裡會被抵銷。
2. **自我對弈的局變難了**。每步約 460 局裡的和局數 13 → 19 → 32，同期
   stage 5 的 value accuracy 0.86 → 0.77、value loss 0.67 → 0.75。是局勢變接近，
   不是 value head 變差。
3. **policy 落後自己的搜尋**：target entropy 1.74 → 1.49（搜尋變銳利），
   pred entropy 一直在 1.81 不動，所以 KL 擴大。eval 量的是「網路 + 搜尋」，
   搜尋的進步先反映在那裡。
4. **還很早，而且上一輪長得一模一樣**。`0919` 第 13 步是
   value 0.640 / policy 1.829 / KL 0.277 / eval −0.04；它的 policy loss 一直到
   第 17 步左右才開始掉，最後到 1.33、eval +0.70。

另外 eval 本身很吵：n = 12–28 局、50 局滑動視窗，相鄰步共用大部分對局
（所以會出現 −0.312 連 4 步、−0.208 連 3 步這種重複值），n=28 時標準誤約 0.19。

想在這個「平原期」看出進展，要的是 CLAUDE.md §6.7 的 held-out loss：對剛出佇列、
還沒進 buffer 的 trajectory 跑一次 forward-only 的 `losses()`，跟訓練 loss 比較，
才分得出「到了容量/噪音底線」還是「在過擬合 buffer」。

---

## 6. 建議順序

1. `inference_batch_size` 改回 **128**（實測 512 慢 15%）。
2. **不要**加 actor；要動就往下調到約 1024，換取資料新鮮度。
3. `max_simulations` 800 → **300**：一局約 20 分鐘，資料新鮮度約 3 倍、每小時局數
   約 2.7 倍。唯一同時改善兩個軸的槓桿，但會改變每局棋力，要先做對打實驗。
4. 加 `buffer_save_freq`（現在沒有這個選項），把每步 140 s 的 buffer 寫入降下來；
   或把 libnop 的逐元素序列化換成整塊寫入（約可省 40% 時間與檔案大小）。
5. （可選，小改動）讓 actor 錯開起跑，打散「一波 2048 局」的叢聚。
6. 加 held-out loss，讓平原期也有可讀的指標。

---

## 7. 還沒驗證的

- 拿掉那把 device 名稱鎖之後，配對評估工具（`alpha_zero_torch_game_example`，
  兩個 DeviceManager 指到同一張 GPU）還沒做過長時間 soak。
- 6 份副本相對 4 份的單獨效果沒有分離出來（和 batch 512 同時改）。
- per-thread CUDA stream 沒有做：`alpha_zero_torch` 這個 object library 的 include
  裡沒有 CUDA 標頭路徑，要動 CMake。目前 GPU 54%，還沒到需要它的時候。
