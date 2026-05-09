add_executable(benchmark_game benchmark_game.cc ${OPEN_SPIEL_OBJECTS})
add_test(benchmark_game_test benchmark_game --game=tic_tac_toe --sims=100 --attempts=2)

add_executable(example example.cc ${OPEN_SPIEL_OBJECTS})
add_test(example_test example --game=tic_tac_toe --seed=0)

if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  add_executable(alpha_zero_torch_example
    alpha_zero_torch_example.cc
    ${OPEN_SPIEL_OBJECTS}
    $<TARGET_OBJECTS:alpha_zero_torch>)
  target_link_libraries(alpha_zero_torch_example ${TORCH_LIBRARIES})

  add_executable(alpha_zero_torch_game_example
    alpha_zero_torch_game_example.cc
    ${OPEN_SPIEL_OBJECTS}
    $<TARGET_OBJECTS:alpha_zero_torch>)
  target_link_libraries(alpha_zero_torch_game_example ${TORCH_LIBRARIES})
endif()
