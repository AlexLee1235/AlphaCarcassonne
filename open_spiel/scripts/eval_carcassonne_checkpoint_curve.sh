#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${ROOT_DIR}/build/examples/alpha_zero_torch_game_example"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/0509}"
NUM_GAMES="${NUM_GAMES:-20}"
SIMS="${SIMS:-100}"
SEED="${SEED:-0}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/eval_results/carcassonne_checkpoint_curve_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -x "${BIN}" ]]; then
  echo "Missing executable: ${BIN}" >&2
  echo "Build alpha_zero_torch_game_example first." >&2
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/vpnet.pb" ]]; then
  echo "Missing model graph: ${MODEL_DIR}/vpnet.pb" >&2
  exit 1
fi

if [[ -n "${CHECKPOINTS:-}" ]]; then
  checkpoints="${CHECKPOINTS}"
else
  checkpoints="$(
    find "${MODEL_DIR}" -maxdepth 1 -type f -name 'checkpoint-*.pt' \
      ! -name '*-optimizer.pt' \
      -printf '%f\n' \
      | sed -E 's/^checkpoint-(-?[0-9]+)\.pt$/\1/' \
      | grep -E '^-?[0-9]+$' \
      | sort -n \
      | tr '\n' ' '
  )"
fi

if [[ -z "${checkpoints// }" ]]; then
  echo "No checkpoints found in ${MODEL_DIR}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
export LD_LIBRARY_PATH="${ROOT_DIR}/open_spiel/libtorch/libtorch/lib:${LD_LIBRARY_PATH:-}"

SUMMARY="${OUT_DIR}/summary.tsv"
DETAIL="${OUT_DIR}/details.txt"
: > "${DETAIL}"
printf "checkpoint\tsims\tgames\taz_first_wins\trandom_second_wins\taz_second_wins\trandom_first_wins\n" > "${SUMMARY}"

extract_wins() {
  local log="$1"
  grep -E '^Overall wins:' "${log}" \
    | tail -1 \
    | sed -E 's/^Overall wins:[[:space:]]*([0-9]+),[[:space:]]*([0-9]+).*/\1 \2/'
}

run_one() {
  local checkpoint="$1"
  local player1="$2"
  local player2="$3"
  local label="$4"
  local log="${OUT_DIR}/${label}_checkpoint${checkpoint}.log"

  "${BIN}" \
    --game=carcassonne \
    --player1="${player1}" \
    --player2="${player2}" \
    --az_path="${MODEL_DIR}" \
    --az_checkpoint="${checkpoint}" \
    --max_simulations="${SIMS}" \
    --num_games="${NUM_GAMES}" \
    --seed="${SEED}" \
    --quiet=true \
    > "${log}" 2>&1

  {
    echo "== ${label} checkpoint=${checkpoint} =="
    grep -E "^(Number of games played|Players:|Overall wins:|Overall returns:)" "${log}" || true
    echo "log=${log}"
    echo
  } >> "${DETAIL}"

  extract_wins "${log}"
}

echo "root=${ROOT_DIR}" | tee -a "${DETAIL}"
echo "model=${MODEL_DIR}" | tee -a "${DETAIL}"
echo "checkpoints=${checkpoints}" | tee -a "${DETAIL}"
echo "sims=${SIMS}" | tee -a "${DETAIL}"
echo "num_games=${NUM_GAMES}" | tee -a "${DETAIL}"
echo "seed=${SEED}" | tee -a "${DETAIL}"
echo "out=${OUT_DIR}" | tee -a "${DETAIL}"
echo | tee -a "${DETAIL}"

for checkpoint in ${checkpoints}; do
  if [[ ! -f "${MODEL_DIR}/checkpoint-${checkpoint}.pt" ]]; then
    echo "Skipping missing checkpoint-${checkpoint}.pt" | tee -a "${DETAIL}"
    continue
  fi

  echo "Running checkpoint ${checkpoint}..."
  first_wins="$(run_one "${checkpoint}" "az" "random" "az_first")"
  second_wins="$(run_one "${checkpoint}" "random" "az" "az_second")"

  az_first_wins="$(awk '{print $1}' <<< "${first_wins}")"
  random_second_wins="$(awk '{print $2}' <<< "${first_wins}")"
  random_first_wins="$(awk '{print $1}' <<< "${second_wins}")"
  az_second_wins="$(awk '{print $2}' <<< "${second_wins}")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${checkpoint}" "${SIMS}" "${NUM_GAMES}" \
    "${az_first_wins}" "${random_second_wins}" \
    "${az_second_wins}" "${random_first_wins}" \
    | tee -a "${SUMMARY}"
done

echo "Summary written to ${SUMMARY}"
echo "Details written to ${DETAIL}"
