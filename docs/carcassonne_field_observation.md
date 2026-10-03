# 農田觀測：連接表與元件身分（2026-10-03，待消融）

農夫規則加進引擎之後（`farm` 分支），討論過觀測裡要不要再補農田的「連接表」，
以及空格四周朝向它的元件「是不是同一個」。結論是**這次都不加**，先把分析、
候選做法與成本記下來，將來用消融實驗決定。

本文的平面數與記憶體都以 `BOARD_SIZE = 21`、replay buffer 262,144 筆、
觀測以 float 存（`TrainInputs::observations`）計算。

---

## 1. 現況：農田在觀測裡有什麼

**空間平面**（`carcassonne.h`，接在修道院平面之後）：

| 平面 | 內容 | 正規化 |
|---|---|---|
| `kFieldMyFarmersPlane` + 半邊 0..7 | 該半邊所屬農田（元件）的我方農夫數 | `/7` |
| `kFieldOpponentFarmersPlane` + 半邊 | 同上，對手 | `/7` |
| `kFieldScorePlane` + 半邊 | `3 × 已完成相鄰城數` | `/30` clip |
| `kFieldSizePlane` + 半邊 | 田的大小（涵蓋的磚數） | `/30` clip |
| `kFieldOpenCitiesPlane` + 半邊 | 相鄰的未完成城數 | `/10` clip |
| `kInnerFieldMyFarmersPlane` … `OpenCities` | 同上五項，對象是該格的 inner field | 同上 |

大小與未完成城數在 2026-10-03 加入：這兩個量要沿田與城的元件走才算得出來，卷積算不了。
隨機對局每半邊的分佈：大小 p99 28 磚（最後一手 32、max 41），相鄰未完成城 p99 9–10 座（max 16）。

半邊編號：`e = 2*side + h`，順時針 `0 N西 1 N東 2 E北 3 E南 4 S東 5 S西 6 W南 7 W北`。
城市邊的半邊一律為 0。

**global vector**：
- 我方 / 對手的農田 pending，`clip(/40)`；它們也包含在總 pending 裡。
- 合法 meeple 遮罩 15 維，其中 `5..12` 是農夫（以該田在本磚的最小半邊指名），`13` 是 inner field。

**手上的磚**：
- global 只有 24 維 type one-hot，只在 tile 期有值；
- 加上空間的 4 個合法落點平面（`kLegalPlacementPlane`，rot 0..3）。

**沒有的**：
- 一格磚內「哪些半邊同一塊田、哪塊田貼哪一邊的城」；
- 空格四周朝向它的城、路、田，彼此是不是同一個元件；
- 手上那張磚的結構（地形、連接、農田）。只能靠 one-hot 學出來。

---

## 2. 結論與反例

### 2.1 基本牌組：磚內佈局可以推出來

一格的農田分區與貼城關係，在基本牌組下完全由該格的
「4 邊地形 + 6 個 side-link 平面 + 盾/修道院」決定。這是單格的局部函數，CNN 很容易學。

`carcassonne_test.cc` 的 `FieldLayoutTest` 把這件事釘成測試：
`TileLook`（地形 + side-link）相同的兩種「磚型 × 旋轉」，`CanonicalFieldLayout` 必須相同。
它對引擎實際使用的牌組（`tile.hpp` 的 `base_deck`）逐一檢查。

### 2.2 擴充牌組：有反例

上面的前提**在擴充版不成立**：已找到地形與 side-link 完全相同、農田佈局卻不同的磚。

- 反例磚：（待補：磚名，以及兩種佈局的差異）

後果：
- 換到那個牌組時，`FieldLayoutTest` 的 `TileLook` 檢查會失敗。這是它設計要抓的情況，不會靜默算錯。
- 到那時，§3 的**候選 A 就不再是可選的消融臂，而是必要的修正**。否則網路從觀測裡分不出那些磚的農田佈局。

### 2.3 global vector 放不下盤面磚

global 每個狀態只有一份，只能描述一張磚（手上那張，或剛放下那張）。
盤面上七十多張已放的磚，各自的佈局只能用空間平面表達。

### 2.4 手上的磚不受反例影響

one-hot 對每種磚型仍然唯一，所以佈局永遠可以「查表」推出。
是否要給結構化描述（候選 C）純粹是表示方式的問題，不是資訊量的問題。

---

## 3. 候選臂

### A. 磚內農田連接表（空間，65 平面）

每個**已放的磚**那一格：

| 區塊 | 平面數 | 內容 |
|---|---|---|
| 半邊兩兩同田 | 28 | pair `(a, b)`，`a < b`，兩個半邊都不是城邊而且 `field[a] == field[b]` |
| 半邊 × 城邊 | 32 | `4e + s`：半邊 e 那塊田貼著本磚 side s 的城（`field_city_sides[field[e]]` 的 bit s） |
| 有 inner field | 1 | `tile.innerField() != -1` |
| inner field 貼城邊 | 4 | inner field 的 `field_city_sides` 的 bit s |

- **pair 順序**：`(0,1), (0,2), …, (6,7)`，索引 = `a*(15-a)/2 + (b-a-1)`（n = 8 時的 `a*(2n-a-1)/2 + (b-a-1)`）。
- **旋轉 k 次**：半邊 `e → (e+2k) % 8`、side `s → (s+k) % 4`。
  - pair 改成重新取索引，比照 `kSideLinkPlane` 的 `SidePairIndex`；
  - `kInnerFieldPlane` 只移動格子。
- **完整性**：28 + 32 + 5 能還原 `CanonicalFieldLayout` 的全部內容（分區、田數、各田城邊）。
- **預期效果**：基本牌組下是冗餘資訊，大概量不出差別；擴充牌組下是必要的。

### B. 空格朝向元件是否同一個（空間，34 平面）

每個**空格**（`isFrontier`），用空格自己的邊與半邊編號：

| 區塊 | 平面數 | 內容 |
|---|---|---|
| 城/路 pair | 6 | `kSidePairs` 順序；side a、b 的鄰居朝向邊都是城/路，而且是同一個 feature root |
| 田 pair | 28 | 同 A 的 pair 順序；半邊 a、b 朝向的鄰居半邊是同一個 field root |

- **朝向規則**：空格的半邊 `e = 2s + h` 面對鄰居的半邊 `2*((s+2)%4) + 1 - h`，與 `FieldModule::placeTileOnBoard` 的相接規則相同。城/路要用 feature root，需要在 `Carcassonne` 加 `featureRoot(tile_id, side)`（`fieldRoot` 已經有）。
- **旋轉**：城/路 pair 與 `kSideLinkPlane` 同規則；田 pair 同 A。
- **動機**：CLAUDE.md §2.6「一手合併兩個元件」。每個朝向的邊/半邊都有元件層級的數值（meeple / 農夫數、分數、`opens`），但沒有身分；要分辨是否同一個元件，得沿元件繞過大半個盤面，卷積做不到。
  - 對農田特別關鍵：搶農田多數幾乎都靠把兩塊田接起來，而盤上大多數半邊屬於同一塊巨大農田。
- **跟 A 的關係**：田 pair 和 A 的連接 pair 是同一個索引空間，「放下去會怎麼合併」可以直接對位比較。
- **便宜的替代 B′（3 平面）**：每個空格「朝向的城、路、田各有幾個不同元件」。有損：丟掉「哪兩個是同一個」。

**手算測試案例**（U 形路）：
- 起始磚（type 20 rot 0）在 `(c, c)`，依序放 4 張 GGRR（type 22）：
  - `(c-1, c)` rot 2（N、E 路）
  - `(c+1, c)` rot 1（N、W 路）
  - `(c-1, c-1)` rot 3（E、S 路）
  - `(c+1, c-1)` rot 0（S、W 路）
- 空格 `(c, c-1)` 被這條 U 形路與起始磚城口夾住，預期：
  - 城/路：E-W pair = 1（同一條路）；S 邊是城，跟 E、W 都是 0。
  - 田：
    - `(3,6)` = 1：U 內那塊田，繞過起始磚城口；
    - `(2,7)` = 1：U 外那塊田，從南邊繞一圈連起來；
    - `(2,3)`、`(2,6)`、`(3,7)`、`(6,7)` = 0；
    - 半邊 0、1、4、5 不朝向任何田（N 沒有鄰居、S 是城）。
- 只有起始磚時：
  - 南邊空格 `(c, c+1)` 的田 pair `(0,1)` = 1；
  - 東、西空格朝向的兩個半邊分屬路的南北兩田，為 0；
  - 全盤田 pair 總和 = 1，城/路 pair 全 0。

### C. 手上磚的結構描述（global，85 或 65 維）

用磚自己的 rot 0 框架描述手上那張磚，one-hot 保留：

| 版本 | 內容 | global 維度 |
|---|---|---|
| 完整 | 地形 12 + side-link 6 + 盾/修道院 2 + 農田連接 28 + 貼城 32 + inner 5 = **85** | 81 → 166 |
| 只農田 | 農田連接 28 + 貼城 32 + inner 5 = **65** | 81 → 146 |

- **記憶體不變**：global vector 住在一個 441 格的平面裡，166 也放得下。
- **參數約 +4 萬**：5 個 global bias Linear + value head + policy gpool，全網路約 20 萬。
- **旋轉增強不用改**：手上的磚還沒有方向，rot 0 框架描述跟盤面旋轉無關。
- **預期效果**：基本牌組下量不出差別，因為 24 種磚每局都出現很多次，one-hot 的 24 個向量很快學好。價值在擴充牌組（磚型多、每種樣本少），以及城、路、田表示一致。

### global vector 怎麼進網路

這是 A/B 與 C 定位不同的原因。`model.cc` 把 global 經 `Linear(G → width)` 變成 **per-channel bias**：
加在 input conv 之後和每兩個 residual block，再接進 value head 與 policy 的 gpool 分支。
它對每一格加的是**同一個偏移**：能告訴整盤「手上是什麼磚」，但「放在這格、轉 r 之後會接到什麼」仍要 trunk 在每格自己算。
所以每格各自不同的資訊（A、B）只能放空間平面。

---

## 4. 成本

| 範圍 | 空間平面 | 觀測平面 | 每狀態 | replay buffer |
|---|---|---|---|---|
| 現況（含田的大小、未完成城數） | 94 | 95 | 168 KB | 43.9 GB |
| +A | 159 | 160 | 282 KB | 74.0 GB |
| +B | 128 | 129 | 228 KB | 59.7 GB |
| +A+B | 193 | 194 | 342 KB | 89.7 GB |
| +C | 不變 | 不變 | 不變 | 不變 |

**`max_memory_mb` 不是記憶體預算，跟 replay buffer 無關。**
- 它是每個 `MCTSBot` 每次搜尋的樹節點上限：`mcts.cc:222` 的 `max_nodes = (max_memory_mb << 20) / sizeof(SearchNode)`，`SearchNode` 約 80 B。
- 節點數超過時，`GarbageCollect` 會剪掉訪問數低的節點。
- 0921 設 100000，等於約 13 億個節點；800 sims 的樹遠小於此，所以實際上從不觸發。`docs/carcassonne_segfault.md` 也記過這點。

replay buffer 的實際上限是**機器 RAM**：量測機是 188 GB（`docs/carcassonne_throughput.md`）。這份 RAM 還要分給：
- 2048 個 actor 的狀態與搜尋樹；
- trajectory queue：容量是 `replay_buffer_size / replay_buffer_reuse` 局完整棋局，`learner` 跟不上時才會被填滿；
- 推論 cache。

+A+B 的 89.7 GB 約佔 RAM 一半，要評估是否調小 `replay_buffer_size`。

將來若要省記憶體：A 的平面全由「磚型, 旋轉」決定，可以改成每格的 tile embedding（磚型 + 旋轉 → 學出來的向量），不必存 65 個稀疏平面。這是架構改動。

---

## 5. 消融設計建議

- **一臂只改一件事。** CLAUDE.md §7 的教訓：0921 把 7a、7b 綁在一起跑，結果誰都無法歸因。建議的臂：基準、+A、+B、+C（若要再分，C 拆成完整 / 只農田）。
- **同訓練量（states）比較**，同 config。
- **各臂觀測形狀不同**（C 只有 global 維度不同），checkpoint 不能互相載入，每臂從頭訓練。
- **要看的指標**：
  - eval；
  - 分 stage 的 `raw_value_accuracy`：B 預期影響中盤；
  - `policy_kl`；
  - 若能拆出 meeple 期，看農夫相關決策的 KL。
- **eval 的解析度**：在高勝率時不足（95% 區間約 ±189 Elo），臂間比較要用 `tools/ladder.py`。
- **何時 A 不再是消融**：換到 §2.2 那種牌組時，A 是必要的（`FieldLayoutTest` 會先失敗提醒）。

---

## 6. 實作指引

- **`open_spiel/games/carcassonne/carcassonne.h`**
  - 新平面接在 `kInnerFieldOpenCitiesPlane` 之後，不動前面的索引，`kLastPlacedPlane == 26` 不變；
  - 更新 `kSpatialPlanes` 的 static_assert；
  - C 則是在 global offsets 末尾追加，更新 `kGlobalFeatures` 的 static_assert。
- **`open_spiel/games/carcassonne/carcassonne.cc`**
  - 匿名 namespace：比照 `kSidePairs` / `SidePairIndex`，加 `kHalfEdgePairs`、`HalfEdgePairIndex`；B 另外需要 `kDx` / `kDy`。
  - `ObservationTensor`：A 在每張已放磚的迴圈裡、現有農田平面旁邊填；B 在同一個迴圈裡、對 `isFrontier(x, y)` 的空格填；C 在 global 區段填。
  - `RotatePlane`：A、B 的新規則，見 §3。C 不用改 `RotateObservation`。
- **`open_spiel/games/carcassonne/game/game.hpp`**：B 需要 `featureRoot`。
- **`open_spiel/examples/carcassonne_observation_dump.cc`**：新平面 / global 欄位的名稱。
- **`open_spiel/games/carcassonne/carcassonne_test.cc`**
  - `ObservationTensorSmokeTest`：`shape[0]`，以及 §3 的起始磚預期值。A 的起始磚：田 pair 恰好 7 個為 1，即 (2,7)、(3,4)、(3,5)、(3,6)、(4,5)、(4,6)、(5,6)；城邊只有 (e=2, N)、(e=7, N)。
  - B：§3 的 U 形路測試。
  - `RelativePerspectiveTest`：新平面兩個視角相同，都跟視角無關。
  - `FieldLayoutTest`：做了 A 之後，`TileLook` 改成包含 A 的平面，驗「平面能完整還原佈局」。
  - `RotationEquivarianceTest` 不用改：它逐值比對旋轉 twin 的整個觀測，會直接驗到新的旋轉規則。
