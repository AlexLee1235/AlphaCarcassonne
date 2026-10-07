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

目前除了河流、旅館與大教堂、商人與建築師（§8）之外**只做牌的形狀**，擴充的規則一律不算：龍、公主、魔法門、火山都還沒有效果。

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
      {北的標記, 東, 南, 西},              // 盾、公主、貨物、旅館；可省略
      整張牌的標記),                       // 修道院、龍、火山、魔法門；可省略
 張數, type, EXP_擴充},
```

只有後面兩欄可以省略。如果要寫整張牌的標記、又沒有邊的標記，邊的標記那一欄寫 `{}`，例如 `…, {}, TILE_MONASTERY)`。

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
| 盾、公主、貨物（酒、布、麥）、旅館、龍、火山、魔法門的圖示 | 邊照底下的地形填；圖示另外記在標記欄（見下） |

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

### 標記：跟著城或路的

盾、公主、貨物屬於某一座城，旅館屬於某一條路，所以要寫在**那座城或那條路的邊**上。
順序是北、東、南、西，各一格：

| 圖示 | 寫 | 放在 |
|---|---|---|
| 盾 | `MARK_SHIELD` | 城的邊 |
| 公主 | `MARK_PRINCESS` | 城的邊 |
| 貨物：酒、布、麥 | `MARK_WINE` / `MARK_CLOTH` / `MARK_WHEAT` | 城的邊；一座城最多一種 |
| 旅館（路邊的湖） | `MARK_INN` | 路的邊 |

- 一座城佔好幾邊時，**寫在其中任一邊即可**，引擎會把同一座城的邊合起來看。建議寫在最前面那一邊（照北、東、南、西的順序）。
- 同一格有兩個標記時用 `|` 連起來，例如 `MARK_SHIELD | MARK_WINE`。
- 例子：北邊和南邊各一座分開的城，只有北邊那座有盾，寫成 `{MARK_SHIELD, 0, 0, 0}`；南邊的城不算盾。
- 目前**只有盾會計分**。其他標記只是先記下來，畫在總覽圖上，等之後做規則時再用。
- 大教堂不用標記：整個牌組只有 type 36 這一種，之後做規則時直接用 type 判斷。

### 標記：整張牌的

`TILE_MONASTERY`（修道院）、`TILE_DRAGON`（龍）、`TILE_VOLCANO`（火山）、`TILE_PORTAL`（魔法門）。
多個時用 `|` 連起來。目前只有修道院有規則。

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

// 北、南兩座分開的城，只有北邊那座有盾，南邊那座有酒。
{Tile(CITY, GRASS, CITY, GRASS, 0, 1, 2, 3, {{-1, -1, 0, 0, -1, -1, 0, 0}, 1, {SIDE_N | SIDE_S}},
      {MARK_SHIELD, 0, MARK_WINE, 0}), 張數, type, EXP_TRADERS_BUILDERS},

// 修道院加上一條通到南邊的路（基本版 type 2）：整張牌的標記。
{Tile(GRASS, GRASS, ROAD, GRASS, 0, 1, 2, 3, {{0, 0, 0, 0, 0, 0, 0, 0}, 1, {}}, {}, TILE_MONASTERY), 2, 2},
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
| `MARK_INN goes on a road side` | 旅館寫在城的邊上 |
| `shield, princess and goods go on a city side` | 盾、公主或貨物寫在路的邊上 |
| `has a mark; marks go on a side of the city or road` | 標記寫在草地或河的邊上 |
| `one city with more than one kind of goods` | 同一座城寫了兩種貨物 |

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
- 黃字：寫在那一邊的標記，`SH` 盾、`PR` 公主、`WI` / `CL` / `WH` 酒 / 布 / 麥、`INN` 旅館。
- 圖下方：張數、整張牌的標記（monastery、dragon、volcano、portal），以及每塊田貼著哪些城邊（`f0:NW`）。

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
- 確認沒填錯，就把造成差異的那一種牌加進 `carcassonne_test.cc` 的 `kHiddenFieldTypes`，並在旁邊寫明原因。
  只要一對裡有一種在這個清單上，測試就接受。目前清單上有兩種情況：
  - 42、62、65、67、94、95：城牆延伸到牌角，把那個角兩側的草地切開。
  - 76：路在草地中間就結束，沒接到城，所以沒有把草地切開（對手 45、72 的路接到城門）。
  這等於接受觀測的這個限制；要真正解決，得做 §2.2 的候選 A（每張牌自己的農田平面）。

---

## 8. 開擴充來玩或訓練

```
carcassonne(inns_cathedrals=on,traders_builders=on,river=on,princess_dragon=tiles)
```

- 每個參數是 `off`（預設）、`tiles`（只發牌、不算規則）或 `on`（發牌並套用規則，只有規則做好的擴充才有）。基本版的牌一定都在。
  - 河流：`off`／`on`，沒有只發牌的選項。
  - 旅館與大教堂：`off`／`tiles`／`on`。`on` = 大米寶＋旅館＋大教堂；`tiles` 的旅館、大教堂只是一般的路和城。
  - 商人與建築師：`off`／`tiles`／`on`。`on` = 建築師＋小豬＋貨物。
  - 公主與龍：`off`／`tiles`。
- **觀測和動作的維度不隨參數改變**：global vector 為牌表裡的每一種牌都留了位置（86 + 2 × 牌種數），沒發的牌種一律是 0。
  所以**牌表每加一種牌，維度就變一次**。等牌表定案再開始訓練；舊的 checkpoint 也不能載入，`carcassonne_bot_cli` 會直接報錯。
- 河流規則（`river=on`）：河源取代起始牌放在中央，基本版的起始牌拿掉不用（牌組 72 + 12 − 1 = 83 張）；先抽完河流牌，湖一定最後，之後才抽一般牌。
  每張河流牌都要接在河的出口上，而且連續兩個彎不能往同一邊轉（中間隔著直流也算），所以河只會在兩個方向間交替、不會流回自己旁邊。
  河不能流出盤面；放不下的河流牌（只會在盤面邊緣發生）跟一般牌一樣丟掉。規則寫在 `game/game.cpp` 的 `isLegalPlacement` / `advanceRiver`。
- 大米寶（`inns_cathedrals=on`）：每人多 1 個大米寶，可以取代一般 meeple 放在任何能放 meeple 的地方（城、路、修道院、農田）。
  多數決時算 2 個，分數不加倍；完成後一樣收回，當農夫就留到終局。只剩大米寶時也能放。
  動作是另外 14 個（`place_big_meeple(...)`，接在一般 meeple 動作後面）；規則寫在 `game/game.cpp` 的 `getLegalMeepleMoves` / `placeMeeple`。
- 旅館與大教堂（`inns_cathedrals=on`）：路上只要有一間旅館（`MARK_INN`，只算它旁邊那段路），完成時每張磚 2 分；
  城裡只要有一座大教堂（type 36，`CATHEDRAL_TYPE`），完成時每張磚、每個盾 3 分。兩者終局沒完成都是 0 分。
  規則寫在 `game/Feature.cpp` 的 `getScore()`；旅館與大教堂在 `FeatureModule::placeTileOnBoard` 記進元件。
- 建築師（`traders_builders=on`）：每人 1 個，可以取代 meeple 放在剛放的磚上、自己已經有 follower（含大米寶）的城或路；
  上面有對手的 meeple 或建築師也可以，不能放農田或修道院。它不是 follower，不算多數決，城／路完成時跟 follower 一起收回。
  之後自己放的磚延伸到它所在的城／路，這回合結束後**一定**再放一張（雙回合），第二張不會再觸發第三張；
  那張磚順便完成城／路、把建築師收回，第二張照樣有。動作是另外 4 個（`place_builder(edge=N)`）；
  規則寫在 `game/game.cpp` 的 `placeTile`（判斷延伸）、`getLegalMeepleMoves`、`placeMeeple`（換人）。
- 小豬（`traders_builders=on`）：每人 1 隻，可以取代 meeple 放在剛放的磚上、自己已經有農夫的農田（上面有對手的農夫或小豬也可以）；
  inner field 不可能有小豬（它只在放下的當回合出現，還沒有農夫）。小豬不算多數決，放下就留到終局。
  終局時，田的多數（含平手）玩家若有小豬在田上，每座完成的城算 4 分而不是 3 分。動作是另外 8 個（`place_pig(field=N)`）；
  計分寫在 `game/FieldModule.cpp` 的 `accumulateScore`。
- 貨物（`traders_builders=on`）：城完成時，放下完成那張磚的玩家每個貨物符號拿 1 個 token（酒、麥、布；城裡有沒有他的騎士都一樣）。
  終局時每種 token 最多的玩家得 10 分，平手都得，雙方都是 0 個就不給。token 在 `game/game.cpp` 的 `placeTile` 發，終局分在 `accumulateGoodsScore`。
  牌表的貨物數（`goodsInTable`）在 `tile.hpp` 有 static_assert 對 9／6／5，改 T&B 的貨物標記要跟著改。
- 盤面：牌變多，對局也變大。引擎只在 `VIEW_SIZE`（目前 21）格的視窗裡放磚，視窗每手置中在已放的磚上。
  填完後量全開時要多大的視窗，再決定 `VIEW_SIZE` 要不要加大（`BOARD_SIZE` 只要容得下視窗能移到的地方）：
  ```bash
  cd tools && make build/diag_board_free && ./build/diag_board_free 3000 all
  ```
