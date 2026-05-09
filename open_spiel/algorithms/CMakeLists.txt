add_library(algorithms OBJECT
  best_response.cc
  best_response.h
  cfr.cc
  cfr.h
  cfr_br.cc
  cfr_br.h
  corr_dev_builder.cc
  corr_dev_builder.h
  deterministic_policy.cc
  deterministic_policy.h
  evaluate_bots.cc
  evaluate_bots.h
  expected_returns.cc
  expected_returns.h
  external_sampling_mccfr.cc
  external_sampling_mccfr.h
  fsicfr.cc
  fsicfr.h
  get_all_histories.cc
  get_all_histories.h
  get_all_infostates.cc
  get_all_infostates.h
  get_all_states.cc
  get_all_states.h
  get_legal_actions_map.cc
  get_legal_actions_map.h
  history_tree.cc
  history_tree.h
  infostate_tree.cc
  infostate_tree.h
  is_mcts.cc
  is_mcts.h
  matrix_game_utils.cc
  matrix_game_utils.h
  mcts.cc
  mcts.h
  minimax.cc
  minimax.h
  nfg_writer.cc
  nfg_writer.h
  observation_history.cc
  observation_history.h
  oos.cc
  oos.h
  outcome_sampling_mccfr.cc
  outcome_sampling_mccfr.h
  policy_iteration.cc
  policy_iteration.h
  state_distribution.cc
  state_distribution.h
  tabular_best_response_mdp.cc
  tabular_best_response_mdp.h
  tabular_exploitability.cc
  tabular_exploitability.h
  tabular_q_learning.cc
  tabular_q_learning.h
  tabular_sarsa.cc
  tabular_sarsa.h
  tensor_game_utils.cc
  tensor_game_utils.h
  trajectories.cc
  trajectories.h
  value_iteration.cc
  value_iteration.h
)

target_include_directories(algorithms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  if(NOT OPEN_SPIEL_BUILD_WITH_LIBNOP)
    message(FATAL_ERROR
      "alpha_zero_torch requires libnop (OPEN_SPIEL_BUILD_WITH_LIBNOP)")
  endif()

  add_library(alpha_zero_torch OBJECT
    alpha_zero_torch/alpha_zero.h
    alpha_zero_torch/alpha_zero.cc
    alpha_zero_torch/device_manager.h
    alpha_zero_torch/model.h
    alpha_zero_torch/model.cc
    alpha_zero_torch/vpevaluator.h
    alpha_zero_torch/vpevaluator.cc
    alpha_zero_torch/vpnet.h
    alpha_zero_torch/vpnet.cc
  )
  target_include_directories(alpha_zero_torch PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/alpha_zero_torch)
endif()

if (${OPEN_SPIEL_BUILD_WITH_ORTOOLS})
  add_subdirectory(ortools)
endif()
