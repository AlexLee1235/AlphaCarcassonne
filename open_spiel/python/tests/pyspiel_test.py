# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""General tests for the trimmed pyspiel bindings."""

from absl.testing import absltest

import pyspiel


EXPECTED_GAMES = ["carcassonne", "connect_four", "othello", "tic_tac_toe"]


class PyspielTest(absltest.TestCase):

  def test_registered_names_are_the_supported_games(self):
    self.assertEqual(pyspiel.registered_names(), EXPECTED_GAMES)

  def test_registered_concrete_names_are_the_supported_games(self):
    self.assertEqual(pyspiel.registered_concrete_names(), EXPECTED_GAMES)

  def test_can_load_supported_games(self):
    for game_name in EXPECTED_GAMES:
      game = pyspiel.load_game(game_name)
      state = game.new_initial_state()
      self.assertEqual(game.get_type().short_name, game_name)
      self.assertFalse(state.is_terminal())

  def test_removed_games_are_not_loadable(self):
    for game_name in ["kuhn_poker", "python_tic_tac_toe"]:
      with self.assertRaisesRegex(RuntimeError, f"Unknown game '{game_name}'"):
        pyspiel.load_game(game_name)

  def test_tic_tac_toe_random_sim(self):
    game = pyspiel.load_game("tic_tac_toe")
    pyspiel.random_sim_test(game, num_sims=5, serialize=True, verbose=False)

  def test_othello_random_sim(self):
    game = pyspiel.load_game("othello")
    pyspiel.random_sim_test(game, num_sims=3, serialize=True, verbose=False)

  def test_carcassonne_random_sim(self):
    game = pyspiel.load_game("carcassonne")
    pyspiel.random_sim_test(game, num_sims=1, serialize=False, verbose=False)


if __name__ == "__main__":
  absltest.main()
