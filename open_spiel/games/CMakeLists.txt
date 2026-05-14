set(GAME_SOURCES
  carcassonne/carcassonne.cc
  carcassonne/carcassonne.h
  carcassonne/game/BoardModule.cpp
  carcassonne/game/DeckModule.cpp
  carcassonne/game/Feature.cpp
  carcassonne/game/FeatureModule.cpp
  carcassonne/game/FrontierModule.cpp
  carcassonne/game/MonasteryModule.cpp
  carcassonne/game/game.cpp
  carcassonne/game/game.hpp
  carcassonne/game/tile.hpp
  connect_four/connect_four.cc
  connect_four/connect_four.h
  othello/othello.cc
  othello/othello.h
  tic_tac_toe/tic_tac_toe.cc
  tic_tac_toe/tic_tac_toe.h
)

add_library(games OBJECT ${GAME_SOURCES})

target_include_directories(games PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(games PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../pybind11_abseil)
target_include_directories(games PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../pybind11_json)
target_include_directories(games PUBLIC ${Python3_INCLUDE_DIRS})

add_executable(carcassonne_test carcassonne/carcassonne_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(carcassonne_test carcassonne_test)

add_executable(connect_four_test connect_four/connect_four_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(connect_four_test connect_four_test)

add_executable(othello_test othello/othello_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(othello_test othello_test)

add_executable(tic_tac_toe_test tic_tac_toe/tic_tac_toe_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(tic_tac_toe_test tic_tac_toe_test)
