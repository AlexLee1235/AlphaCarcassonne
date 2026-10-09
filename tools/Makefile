# AlphaCarcassonne 診斷工具
#
#   make            建立所有診斷程式到 build/
#   make run        依序跑一遍並把輸出存到 results/
#   make tiles      檢查牌表，並把它畫在 ../tiles/<type>.png 上（results/tile_sheet_*.png）
#   make clean
#
# 這些工具只依賴 open_spiel/games/carcassonne/game/ 底下的純 C++ 引擎，
# 不需要 OpenSpiel 本體、不需要 CMake、不需要 libtorch。

GAME     := ../open_spiel/games/carcassonne/game
CXXFLAGS := -O2 -std=c++17 -Wall
SRCS     := $(GAME)/Feature.cpp $(GAME)/FeatureModule.cpp $(GAME)/FieldModule.cpp $(GAME)/MonasteryModule.cpp \
            $(GAME)/BoardModule.cpp $(GAME)/FrontierModule.cpp $(GAME)/DeckModule.cpp \
            $(GAME)/game.cpp

# VIEW_SIZE = BOARD_SIZE = 41 的副本：視窗蓋住整個盤面，什麼都不擋，用來量對局本來會長多大
GFREE    := build/game_free
SRCSFREE := $(GFREE)/Feature.cpp $(GFREE)/FeatureModule.cpp $(GFREE)/FieldModule.cpp $(GFREE)/MonasteryModule.cpp \
            $(GFREE)/BoardModule.cpp $(GFREE)/FrontierModule.cpp $(GFREE)/DeckModule.cpp \
            $(GFREE)/game.cpp
# 副本要跟著引擎更新（牌表 tile.hpp 常常在改）
ENGINE   := $(wildcard $(GAME)/*.hpp $(GAME)/*.cpp)

TOOLS    := diag_value_gap diag_local_decomp diag_board diag_opens diag_occupancy diag_branching diag_luck diag_value_gap_strong diag_pending_scale diag_plane_scale diag_replay_scale diag_game_report game_mix
BINS     := $(addprefix build/,$(TOOLS)) build/diag_board_free build/dump_tiles
PYTHON   ?= python3

.PHONY: all run tiles clean
all: $(BINS)

# 只用到牌表，不需要引擎的 .cpp
build/dump_tiles: dump_tiles.cpp $(GAME)/tile.hpp $(GAME)/tile_check.hpp | build
	$(CXX) $(CXXFLAGS) -I$(GAME) $< -o $@

# 牌表有錯時 dump_tiles 回傳 1，但照樣畫圖，方便對著圖找錯
tiles: build/dump_tiles
	mkdir -p results
	-./build/dump_tiles results/tile_table.json
	$(PYTHON) render_tile_table.py results/tile_table.json

build:
	mkdir -p build

$(GFREE)/game.hpp: $(ENGINE) | build
	mkdir -p $(GFREE)
	cp $(GAME)/*.hpp $(GAME)/*.cpp $(GFREE)/
	sed -e 's/constexpr int BOARD_SIZE = [0-9]*;/constexpr int BOARD_SIZE = 41;/' \
	    -e 's/constexpr int VIEW_SIZE = [0-9]*;/constexpr int VIEW_SIZE = 41;/' $(GAME)/game.hpp > $(GFREE)/game.hpp
	@grep -n '_SIZE = ' $(GFREE)/game.hpp

build/diag_board_free: diag_board.cpp common.hpp $(GFREE)/game.hpp
	$(CXX) $(CXXFLAGS) -I$(GFREE) $< $(SRCSFREE) -o $@

build/%: %.cpp common.hpp scale_stats.hpp actor_log.hpp $(ENGINE) | build
	$(CXX) $(CXXFLAGS) -I$(GAME) $< $(SRCS) -o $@

run: all
	mkdir -p results
	cd results && ../build/diag_occupancy      | tee diag_occupancy.txt
	cd results && ../build/diag_value_gap      | tee diag_value_gap.txt
	cd results && ../build/diag_local_decomp   | tee diag_local_decomp.txt
	cd results && ../build/diag_board          | tee diag_board.txt
	cd results && ../build/diag_board_free     | tee diag_board_free.txt
	cd results && ../build/diag_opens          | tee diag_opens.txt
	cd results && ../build/diag_branching      | tee diag_branching.txt
	@echo
	@echo "輸出在 results/ ，occupancy.csv 也在那裡（給 erf.py / check_value_head.py 用）"

clean:
	rm -rf build results
