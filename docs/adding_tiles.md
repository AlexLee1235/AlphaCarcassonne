# 擴充牌：怎麼填牌表

牌表在 `open_spiel/games/carcassonne/game/tile.hpp` 的 `all_tiles[]`。一列是一種牌，也就是一張圖。
基本版的 24 種（type 1–24）已經填好，四個擴充各留了一段空位：

| 擴充 | type（= `tiles/<type>.png`） | 標記 | 遊戲參數 | 整盒張數 |
|---|---|---|---|---|
| 河流 | 25–34 | `EXP_RIVER` | `river` | 12 |
| 旅館與大教堂 | 35–51 | `EXP_INNS_CATHEDRALS` | `inns_cathedrals` | 18 |
| 商人與建築師 | 52–75 | `EXP_TRADERS_BUILDERS` | `traders_builders` | 24 |
| 公主與龍 | 76–104 | `EXP_PRINCESS_DRAGON` | `princess_dragon` | 30 |

`tiles/` 裡的圖已經照這個順序編好號，填表時照著 type 順序一列一列往下填即可。

目前**只做牌的形狀**，擴充的規則一律不算：旅館、大教堂、貨物、龍、公主、魔法門、火山都還沒有效果。

---

## 1. 流程

1. 打開 `tiles/<type>.png`（1–104 都已經有圖）。圖片原樣就是 rot 0，**不要轉向**。
2. 照著那張圖，在 `all_tiles[]` 對應的擴充區段加一列（格式見下）。type 就是圖的編號，必須連續：
   type 25 是第 25 列，所以要從河流開始，照編號順序填。
3. 跑檢查並畫出總覽圖（見 §6），逐張對照圖片看有沒有填錯。
4. 一個擴充整段填完（張數等於整盒）之後，跑 `carcassonne_test`。

---

## 2. 一列的格式

```cpp
{Tile(北, 東, 南, 西,                      // 四邊的種類
      link北, link東, link南, link西,      // 這張牌上哪些邊連在一起
      {{半邊0, 半邊1, …, 半邊7},           // 每個半邊屬於哪塊田（城的邊填 -1）
       田數,                               // 內田也算在內
       {田0貼著的城邊, 田1…}},             // 每塊田貼著的城（SIDE_N | SIDE_W 這種寫法）
      有沒有盾, 有沒有修道院),             // 可省略，預設都是 false
 張數, type, EXP_擴充},
```

### 方向

**圖片原樣就是 rot 0：上方是北**，四邊的順序是 北 → 東 → 南 → 西（順時針）。
`tiles/<type>.png` 必須就是填表時看的那張圖，不能轉。

### 半邊的編號

每一邊分成兩半，從北邊的西半開始順時針編號：

```
            0 北西    1 北東
          ┌─────────┬─────────┐
  7 西北  │                   │  2 東北
          ├                   ┤
  6 西南  │                   │  3 東南
          └─────────┴─────────┘
            5 南西    4 南東
```

---

## 3. 每一欄怎麼填

### 邊：`GRASS` / `CITY` / `ROAD` / `RIVER`

| 圖上看到的 | 填 |
|---|---|
| 草地 | `GRASS` |
| 城牆、城市 | `CITY` |
| 路（**旅館**：路邊有湖的路也算） | `ROAD` |
| 河 | `RIVER` |
| **大教堂** | `CITY` |
| 貨物（酒、布、麥）、火山、龍、公主、魔法門的圖示 | 不管它，照底下的地形填 |

`RIVER` 只能接 `RIVER`，不能放 meeple，也不計分，但會像路一樣把農田切開。

### link：這張牌上哪些邊是同一個東西

- 連在一起的邊填同一個號碼。例如一座城佔了北、東、西三邊，三邊都填 0。
- 從北邊開始依序編 0、1、2…，**每個草地邊各自佔一個號碼**。
- **link 相同的邊，種類也必須相同**。城和路不能共用號碼，否則引擎會把它們合成同一個區塊。

| 牌 | link |
|---|---|
| 一座城佔北、東、西，南邊草地 | `0, 0, 1, 0` |
| 北邊城，路從東彎到南，西邊草地 | `0, 1, 1, 2` |
| 東西兩邊各一座**分開的**城 | `0, 1, 2, 3` |
| 東西兩邊是**同一座**城（中間連通） | `0, 1, 2, 1` |
| 河從北流到南 | `0, 1, 0, 2` |

### 半邊 → 田

- **城的邊**：兩個半邊都填 `-1`。
- **草地的邊**：兩個半邊是同一塊田。
- **路或河的邊**：
  - 會通到另一邊的路或河，一定把兩側切成不同的田。
  - 在這張牌上就結束的路或河（例如通到湖、源頭、修道院前），兩側可能是同一塊田，要看圖。
- **田的編號**：照半邊 0 → 7 的順序，第一次出現的田編 0，下一塊新的田編 1，依此類推。
- **內田**：不碰任何一邊的田，例如四座城圍著中間一塊草地。它排在最後，每張牌最多一塊。
- **田數**：包含內田，最多 4 塊。

### 田貼著的城：`field_city_sides`

每塊田一項，依田的編號排列，列出它碰到的每一座城的**所有**邊。

- 碰到的城佔北、東、西三邊，就要寫 `SIDE_N | SIDE_E | SIDE_W`，不能只寫一邊。
- 沒碰到城的田寫 `0`，或乾脆省略。

### 盾、修道院

- 盾：`true` 或 `false`。**只能出現在「這張牌只有一座城」的牌上**，因為引擎把盾算給這張牌上的每一座城。
  如果真的有兩座分開的城、只有一座有盾，檢查會擋下來。這種牌要先改引擎，讓盾記錄在個別的城上。
- 修道院：`true` 或 `false`。

### 張數、type、擴充

- 張數：這種牌在這個擴充裡有幾張。
- type：列號 + 1，從 25 起連續編號。基本版 1–24 不能動。
- 擴充：該段的 `EXP_…` 標記。

---

## 4. 範例

```cpp
// 基本版 type 17：北邊城，路從南邊彎到西邊。
// 田 0 是外圍那塊（碰到城），田 1 是彎道內側的西南角（半邊 5、6）。
{Tile(CITY, GRASS, ROAD, ROAD, 0, 1, 2, 2, {{-1, -1, 0, 0, 0, 1, 1, 0}, 2, {SIDE_N}}), 3, 17},

// 河從北流到南：西側是田 0，東側是田 1。
{Tile(RIVER, GRASS, RIVER, GRASS, 0, 1, 0, 2, {{0, 1, 1, 1, 1, 0, 0, 0}, 2, {}}), 張數, type, EXP_RIVER},

// 河從北邊流進來，在牌中間的湖結束：草地繞過湖，整張是同一塊田。
{Tile(RIVER, GRASS, GRASS, GRASS, 0, 1, 2, 3, {{0, 0, 0, 0, 0, 0, 0, 0}, 1, {}}), 張數, type, EXP_RIVER},

// 路從東到西，用橋跨過南北向的河：四個角各是一塊田。
{Tile(RIVER, ROAD, RIVER, ROAD, 0, 1, 0, 1, {{0, 1, 1, 2, 2, 3, 3, 0}, 4, {}}), 張數, type, EXP_RIVER},

// 四邊各一座分開的城，中間一塊草地：這塊草地是內田，碰到全部四座城。
{Tile(CITY, CITY, CITY, CITY, 0, 1, 2, 3,
      {{-1, -1, -1, -1, -1, -1, -1, -1}, 1, {SIDE_N | SIDE_E | SIDE_S | SIDE_W}}), 張數, type, EXP_INNS_CATHEDRALS},

// 大教堂：四邊是同一座城，沒有田。
{Tile(CITY, CITY, CITY, CITY, 0, 0, 0, 0, {{-1, -1, -1, -1, -1, -1, -1, -1}, 0, {}}), 張數, type, EXP_INNS_CATHEDRALS},
```

---

## 5. 常見錯誤（檢查會擋下來）

| 訊息大意 | 原因 |
|---|---|
| `share link … but are CITY and ROAD` | 城和路用了同一個 link 號碼 |
| `grass sides … share a link` | 兩個草地邊用了同一個號碼 |
| `is a city: both half-edges must be -1` | 城的邊填了田的編號 |
| `carries on to another side, so its two half-edges must be different fields` | 會通到另一邊的路或河，兩側卻填成同一塊田 |
| `must list that city's side … too` | 田貼著的城只寫了其中一邊 |
| `field N is on no half-edge` | 田的編號跳號，或內田沒排在最後 |
| `a shield on a tile with 2 cities` | 見 §3 的盾 |

---

## 6. 檢查與核對

**Windows（PowerShell）：**
```powershell
cd C:\achieve\Carcassonne\AlphaCarcassonne\tools
g++ -O2 -std=c++17 -I../open_spiel/games/carcassonne/game dump_tiles.cpp -o build/dump_tiles.exe
.\build\dump_tiles.exe results\tile_table.json
python render_tile_table.py
```

**WSL：**
```bash
cd /mnt/c/achieve/Carcassonne/AlphaCarcassonne/tools && make tiles
```

`dump_tiles` 會印出四件事：

- 每個擴充已填幾種、幾張，對照整盒的張數。
- **錯誤**：違反 §3 規則的地方。有錯誤時結束碼是 1。
- **未完成**：已經開始填、但張數還不等於整盒的擴充。
- **觀測分不出農田的牌對**：見 §7。

`render_tile_table.py` 會為每個擴充畫一張 `tools/results/tile_sheet_<擴充>.png`。每種牌的圖上會標出：

- 四邊的色條：綠是草地、橘是城、白是路、藍是河。色條上的字是種類加 link 號碼，例如 `C0`、`R1`。
- 8 個半邊位置上的圓點：點裡的數字是田的編號；中央的大圓點是內田。
- 圖下方：張數、盾或修道院，以及每塊田貼著哪些城邊（`f0:NW`）。

逐張對照：色條和號碼跟圖上的城、路、河一致，同一塊草地的圓點數字相同，路或河兩側的數字不同。
先看 `tile_sheet_base.png`，那是已知正確的基本版，可以拿來對照標示的意思。

整段填完之後跑完整測試（在 WSL）：
```bash
cd /mnt/c/achieve/Carcassonne/AlphaCarcassonne/build && make carcassonne_test -j8 && ./games/carcassonne_test
```

測試會檢查：

- 整張牌表合乎規則，每個擴充張數正確。
- 每個參數只發該擴充的牌。
- 開全部擴充跑隨機對局，河的邊不會出現在元件平面。
- 開了擴充之後，旋轉對稱仍然成立。

---

## 7. 觀測分不出農田的牌對

觀測裡只有每張牌的地形和 link，沒有它自己的農田切法（`docs/carcassonne_field_observation.md` §2.2）。
如果兩種牌在某個方向下地形和 link 完全一樣，農田卻切得不同，網路就分不出來。

`dump_tiles` 會列出這種牌對，`carcassonne_test` 也會擋下來。處理方式：

- 先確認不是填錯了。
- 確認沒填錯，就把那一對加進 `carcassonne_test.cc` 的 `kAcceptedTileLookConflicts`，並在旁邊寫明原因。
  這等於接受觀測的這個限制；要真正解決，得做 §2.2 的候選 A（每張牌自己的農田平面）。

---

## 8. 開擴充來玩或訓練

```
carcassonne(inns_cathedrals=tiles,traders_builders=tiles,river=tiles,princess_dragon=tiles)
```

- 每個參數是 `off`（預設）或 `tiles`。基本版的牌一定都在。
- **觀測和動作的維度不隨參數改變**：global vector 為牌表裡的每一種牌都留了位置（33 + 2 × 牌種數），沒發的牌種一律是 0。
  所以**牌表每加一種牌，維度就變一次**。等牌表定案再開始訓練；舊的 checkpoint 也不能載入，`carcassonne_bot_cli` 會直接報錯。
- 河流牌直接混在牌堆裡，沒有「先鋪河」的抽牌順序。抽到放不下的牌會被丟掉；河口旁的空格只能接河，多半會一直空著。
- 盤面：牌變多，對局也變大。填完後用下面的指令量全開時的跨度，再決定 `BOARD_SIZE`（目前 21）要不要加大：
  ```bash
  cd tools && make build/diag_board31 && ./build/diag_board31 3000 all
  ```
