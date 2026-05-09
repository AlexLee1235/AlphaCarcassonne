#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${ROOT_DIR}/build/examples/alpha_zero_torch_game_example"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/0509}"
CHECKPOINT="${CHECKPOINT:--1}"
NUM_GAMES="${NUM_GAMES:-20}"
SIMS="${SIMS:-1 10 100}"
SEED="${SEED:-0}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/eval_results/carcassonne_az_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -x "${BIN}" ]]; then
  echo "Missing executable: ${BIN}" >&2
  echo "Build alpha_zero_torch_game_example first." >&2
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/vpnet.pb" ]]; then
  echo "Missing model graph: ${MODEL_DIR}/vpnet.pb" >&2
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/checkpoint-${CHECKPOINT}.pt" ]]; then
  echo "Missing checkpoint: ${MODEL_DIR}/checkpoint-${CHECKPOINT}.pt" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
export LD_LIBRARY_PATH="${ROOT_DIR}/open_spiel/libtorch/libtorch/lib:${LD_LIBRARY_PATH:-}"

SUMMARY="${OUT_DIR}/summary.txt"
: > "${SUMMARY}"

run_match() {
  local label="$1"
  local player1="$2"
  local player2="$3"
  local sims="$4"
  local log="${OUT_DIR}/${label}.log"

  echo "== ${label} ==" | tee -a "${SUMMARY}"
  echo "players=${player1},${player2} sims=${sims} games=${NUM_GAMES} seed=${SEED}" | tee -a "${SUMMARY}"

  "${BIN}" \
    --game=carcassonne \
    --player1="${player1}" \
    --player2="${player2}" \
    --az_path="${MODEL_DIR}" \
    --az_checkpoint="${CHECKPOINT}" \
    --max_simulations="${sims}" \
    --num_games="${NUM_GAMES}" \
    --seed="${SEED}" \
    --quiet=true \
    > "${log}" 2>&1

  grep -E "^(Number of games played|Players:|Overall wins:|Overall returns:)" "${log}" | tee -a "${SUMMARY}" || true
  echo "log=${log}" | tee -a "${SUMMARY}"
  echo | tee -a "${SUMMARY}"
}

echo "root=${ROOT_DIR}" | tee -a "${SUMMARY}"
echo "model=${MODEL_DIR}" | tee -a "${SUMMARY}"
echo "checkpoint=${CHECKPOINT}" | tee -a "${SUMMARY}"
echo "num_games=${NUM_GAMES}" | tee -a "${SUMMARY}"
echo "sims=${SIMS}" | tee -a "${SUMMARY}"
echo "out=${OUT_DIR}" | tee -a "${SUMMARY}"
echo | tee -a "${SUMMARY}"

for sims in ${SIMS}; do
  run_match "az_first_random_sims${sims}" "az" "random" "${sims}"
  run_match "az_second_random_sims${sims}" "random" "az" "${sims}"
done

for sims in ${SIMS}; do
  run_match "mcts_first_random_sims${sims}" "mcts" "random" "${sims}"
  run_match "mcts_second_random_sims${sims}" "random" "mcts" "${sims}"
done

echo "Summary written to ${SUMMARY}"
