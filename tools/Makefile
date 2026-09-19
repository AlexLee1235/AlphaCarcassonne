# AlphaCarcassonne 診斷工具
#
#   make            建立所有診斷程式到 build/
#   make run        依序跑一遍並把輸出存到 results/
#   make clean
#
# 這些工具只依賴 open_spiel/games/carcassonne/game/ 底下的純 C++ 引擎，
# 不需要 OpenSpiel 本體、不需要 CMake、不需要 libtorch。

GAME     := ../open_spiel/games/carcassonne/game
CXXFLAGS := -O2 -std=c++17 -Wall
SRCS     := $(GAME)/Feature.cpp $(GAME)/FeatureModule.cpp $(GAME)/MonasteryModule.cpp \
            $(GAME)/BoardModule.cpp $(GAME)/FrontierModule.cpp $(GAME)/DeckModule.cpp \
            $(GAME)/game.cpp

# BOARD_SIZE=25 的副本，用來量「沒有被裁切時」對局有多大
G25      := build/game25
SRCS25   := $(G25)/Feature.cpp $(G25)/FeatureModule.cpp $(G25)/MonasteryModule.cpp \
            $(G25)/BoardModule.cpp $(G25)/FrontierModule.cpp $(G25)/DeckModule.cpp \
            $(G25)/game.cpp

TOOLS    := diag_value_gap diag_local_decomp diag_board diag_opens diag_occupancy diag_branching diag_pending_scale
BINS     := $(addprefix build/,$(TOOLS)) build/diag_board25

.PHONY: all run clean
all: $(BINS)

build:
	mkdir -p build

$(G25)/game.hpp: $(GAME)/game.hpp | build
	mkdir -p $(G25)
	cp $(GAME)/*.hpp $(GAME)/*.cpp $(G25)/
	sed 's/constexpr int BOARD_SIZE = [0-9]*;/constexpr int BOARD_SIZE = 25;/' $(GAME)/game.hpp > $(G25)/game.hpp
	@grep -n 'BOARD_SIZE = ' $(G25)/game.hpp

build/diag_board25: diag_board.cpp common.hpp $(G25)/game.hpp
	$(CXX) $(CXXFLAGS) -I$(G25) $< $(SRCS25) -o $@

build/%: %.cpp common.hpp | build
	$(CXX) $(CXXFLAGS) -I$(GAME) $< $(SRCS) -o $@

run: all
	mkdir -p results
	cd results && ../build/diag_occupancy      | tee diag_occupancy.txt
	cd results && ../build/diag_value_gap      | tee diag_value_gap.txt
	cd results && ../build/diag_local_decomp   | tee diag_local_decomp.txt
	cd results && ../build/diag_board          | tee diag_board.txt
	cd results && ../build/diag_board25        | tee diag_board25.txt
	cd results && ../build/diag_opens          | tee diag_opens.txt
	cd results && ../build/diag_branching      | tee diag_branching.txt
	@echo
	@echo "輸出在 results/ ，occupancy.csv 也在那裡（給 erf.py / check_value_head.py 用）"

clean:
	rm -rf build results
