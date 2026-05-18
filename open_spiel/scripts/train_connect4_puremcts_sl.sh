#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  open_spiel/scripts/train_connect4_puremcts_sl.sh [--no-build]

Runs the Connect 4 pure-MCTS supervised-learning loop:
  generate pure-MCTS self-play data
  train a VPNet student
  evaluate AZ-MCTS against pure MCTS from both seats
  stop when the combined non-draw AZ win rate reaches PASS_RATE

Important environment overrides:
  RUN                 Output directory. Default: checkpoints/connect4_puremcts_sl_<timestamp>
  GAME                Default: connect_four(egocentric_obs_tensor=true)
  ROUNDS              Default: 3
  SAMPLES             Default: 262144
  HOLDOUT             Default: 65536
  MCTS_SIMS           Default: 1600
  ROLLOUTS            Default: 64
  GEN_WORKERS         Default: 32
  NN_WIDTH            Default: 128
  NN_DEPTH            Default: 16
  TRAIN_STEPS         Default: 20000
  BATCH_SIZE          Default: 1024
  DEVICE              Default: /cuda:0
  EVAL_GAMES          Default: 400
  EVAL_WORKERS        Default: 32
  PASS_RATE           Default: 0.55
  BASE_SEED           Default: 1
  SOLVE               Default: true
  BUILD_DIR           Default: <repo>/build
  BUILD_JOBS          Default: nproc

Example smoke test:
  ROUNDS=1 SAMPLES=128 HOLDOUT=32 MCTS_SIMS=8 ROLLOUTS=1 \
  GEN_WORKERS=2 NN_WIDTH=16 NN_DEPTH=1 TRAIN_STEPS=4 BATCH_SIZE=8 \
  DEVICE=/cpu:0 EVAL_GAMES=4 EVAL_WORKERS=1 \
  open_spiel/scripts/train_connect4_puremcts_sl.sh --no-build
EOF
}

build=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build)
      build=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

default_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  else
    echo 4
  fi
}

quote_command() {
  printf '%q ' "$@"
}

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
    echo "${name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
PRETRAIN="${PRETRAIN:-${BUILD_DIR}/examples/alpha_zero_torch_dataset_pretrain}"
PLAY="${PLAY:-${BUILD_DIR}/examples/alpha_zero_torch_game_example}"

BUILD_JOBS="${BUILD_JOBS:-$(default_jobs)}"
GAME="${GAME:-connect_four(egocentric_obs_tensor=true)}"
RUN="${RUN:-${ROOT_DIR}/checkpoints/connect4_puremcts_sl_$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-${RUN}/datasets}"
MODEL_DIR="${MODEL_DIR:-${RUN}/models}"
EVAL_DIR="${EVAL_DIR:-${RUN}/eval}"
LOG_DIR="${LOG_DIR:-${RUN}/logs}"

ROUNDS="${ROUNDS:-3}"
SAMPLES="${SAMPLES:-262144}"
HOLDOUT="${HOLDOUT:-65536}"
MCTS_SIMS="${MCTS_SIMS:-1600}"
ROLLOUTS="${ROLLOUTS:-64}"
GEN_WORKERS="${GEN_WORKERS:-32}"

NN_WIDTH="${NN_WIDTH:-128}"
NN_DEPTH="${NN_DEPTH:-16}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
DEVICE="${DEVICE:-/cuda:0}"

EVAL_GAMES="${EVAL_GAMES:-400}"
EVAL_WORKERS="${EVAL_WORKERS:-32}"
PASS_RATE="${PASS_RATE:-0.55}"
BASE_SEED="${BASE_SEED:-1}"
SOLVE="${SOLVE:-true}"

require_positive_int ROUNDS "${ROUNDS}"
require_positive_int SAMPLES "${SAMPLES}"
require_positive_int HOLDOUT "${HOLDOUT}"
require_positive_int MCTS_SIMS "${MCTS_SIMS}"
require_positive_int ROLLOUTS "${ROLLOUTS}"
require_positive_int GEN_WORKERS "${GEN_WORKERS}"
require_positive_int NN_WIDTH "${NN_WIDTH}"
require_positive_int NN_DEPTH "${NN_DEPTH}"
require_positive_int TRAIN_STEPS "${TRAIN_STEPS}"
require_positive_int BATCH_SIZE "${BATCH_SIZE}"
require_positive_int EVAL_GAMES "${EVAL_GAMES}"
require_positive_int EVAL_WORKERS "${EVAL_WORKERS}"
require_positive_int BASE_SEED "${BASE_SEED}"

mkdir -p "${DATA_DIR}" "${MODEL_DIR}" "${EVAL_DIR}" "${LOG_DIR}"

SUMMARY="${RUN}/summary.txt"
SUMMARY_TSV="${RUN}/summary.tsv"
CONFIG="${RUN}/config.env"
: > "${SUMMARY}"
printf "round\tmodel\taz_first_wins\tmcts_second_wins\taz_second_wins\tmcts_first_wins\taz_wins\tmcts_wins\tdecisive\taz_non_draw_win_rate\tpassed\n" > "${SUMMARY_TSV}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${SUMMARY}"
}

run_to_log() {
  local log_file="$1"
  shift

  log "Running -> ${log_file}"
  {
    printf 'command: '
    quote_command "$@"
    printf '\n'
    "$@"
  } > "${log_file}" 2>&1 || {
    local status=$?
    echo "Command failed with status ${status}: ${log_file}" >&2
    echo "Last 80 log lines:" >&2
    tail -80 "${log_file}" >&2 || true
    exit "${status}"
  }
}

extract_wins() {
  local log_file="$1"
  local line
  line="$(grep -E '^Overall wins:' "${log_file}" | tail -1 || true)"
  if [[ ! "${line}" =~ ^Overall\ wins:[[:space:]]*([0-9]+),[[:space:]]*([0-9]+) ]]; then
    echo "Could not parse Overall wins from ${log_file}" >&2
    tail -80 "${log_file}" >&2 || true
    exit 1
  fi
  printf '%s %s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
}

write_config() {
  {
    printf 'ROOT_DIR=%q\n' "${ROOT_DIR}"
    printf 'BUILD_DIR=%q\n' "${BUILD_DIR}"
    printf 'PRETRAIN=%q\n' "${PRETRAIN}"
    printf 'PLAY=%q\n' "${PLAY}"
    printf 'RUN=%q\n' "${RUN}"
    printf 'GAME=%q\n' "${GAME}"
    printf 'ROUNDS=%q\n' "${ROUNDS}"
    printf 'SAMPLES=%q\n' "${SAMPLES}"
    printf 'HOLDOUT=%q\n' "${HOLDOUT}"
    printf 'MCTS_SIMS=%q\n' "${MCTS_SIMS}"
    printf 'ROLLOUTS=%q\n' "${ROLLOUTS}"
    printf 'GEN_WORKERS=%q\n' "${GEN_WORKERS}"
    printf 'NN_WIDTH=%q\n' "${NN_WIDTH}"
    printf 'NN_DEPTH=%q\n' "${NN_DEPTH}"
    printf 'TRAIN_STEPS=%q\n' "${TRAIN_STEPS}"
    printf 'BATCH_SIZE=%q\n' "${BATCH_SIZE}"
    printf 'DEVICE=%q\n' "${DEVICE}"
    printf 'EVAL_GAMES=%q\n' "${EVAL_GAMES}"
    printf 'EVAL_WORKERS=%q\n' "${EVAL_WORKERS}"
    printf 'PASS_RATE=%q\n' "${PASS_RATE}"
    printf 'BASE_SEED=%q\n' "${BASE_SEED}"
    printf 'SOLVE=%q\n' "${SOLVE}"
  } > "${CONFIG}"
}

if [[ "${build}" == true ]]; then
  log "Building alpha_zero_torch_dataset_pretrain and alpha_zero_torch_game_example"
  cmake --build "${BUILD_DIR}" \
    --target alpha_zero_torch_dataset_pretrain alpha_zero_torch_game_example \
    -j"${BUILD_JOBS}"
fi

if [[ ! -x "${PRETRAIN}" ]]; then
  echo "Missing executable: ${PRETRAIN}" >&2
  exit 1
fi
if [[ ! -x "${PLAY}" ]]; then
  echo "Missing executable: ${PLAY}" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${ROOT_DIR}/open_spiel/libtorch/libtorch/lib:${LD_LIBRARY_PATH:-}"
write_config

log "Run directory: ${RUN}"
log "Config: ${CONFIG}"
log "Summary TSV: ${SUMMARY_TSV}"
log "Game: ${GAME}"
log "Pass condition: AZ non-draw win rate >= ${PASS_RATE}"

prev_model=""

for round in $(seq 1 "${ROUNDS}"); do
  train_data="${DATA_DIR}/train_r${round}.nop"
  holdout_data="${DATA_DIR}/holdout_r${round}.nop"
  model="${MODEL_DIR}/round${round}"
  seed=$((BASE_SEED + round * 10000))

  generate_log="${LOG_DIR}/round${round}_generate.log"
  train_log="${LOG_DIR}/round${round}_train.log"
  first_log="${EVAL_DIR}/round${round}_az_first.log"
  second_log="${EVAL_DIR}/round${round}_az_second.log"

  log "Round ${round}: generating pure-MCTS dataset"
  run_to_log "${generate_log}" \
    "${PRETRAIN}" \
    --mode=generate \
    --game="${GAME}" \
    --dataset="${train_data}" \
    --holdout_dataset="${holdout_data}" \
    --samples="${SAMPLES}" \
    --holdout_samples="${HOLDOUT}" \
    --teacher=pure_mcts \
    --max_simulations="${MCTS_SIMS}" \
    --rollout_count="${ROLLOUTS}" \
    --mcts_policy_temperature=1.0 \
    --value_target=terminal \
    --value_is_current_player=true \
    --num_workers="${GEN_WORKERS}" \
    --seed="${seed}"

  log "Round ${round}: training student"
  if [[ -z "${prev_model}" ]]; then
    run_to_log "${train_log}" \
      "${PRETRAIN}" \
      --mode=train \
      --game="${GAME}" \
      --dataset="${train_data}" \
      --holdout_dataset="${holdout_data}" \
      --student_path="${model}" \
      --init_from_checkpoint=false \
      --nn_model=resnet \
      --nn_width="${NN_WIDTH}" \
      --nn_depth="${NN_DEPTH}" \
      --train_steps="${TRAIN_STEPS}" \
      --batch_size="${BATCH_SIZE}" \
      --learning_rate=0.0003 \
      --weight_decay=0.0001 \
      --device="${DEVICE}" \
      --value_is_current_player=true \
      --report_every=500 \
      --save_final_checkpoint=true \
      --save_best_holdout_checkpoint=true
  else
    run_to_log "${train_log}" \
      "${PRETRAIN}" \
      --mode=train \
      --game="${GAME}" \
      --dataset="${train_data}" \
      --holdout_dataset="${holdout_data}" \
      --student_path="${model}" \
      --init_from_checkpoint=true \
      --az_path="${prev_model}" \
      --az_checkpoint=-3 \
      --train_steps="${TRAIN_STEPS}" \
      --batch_size="${BATCH_SIZE}" \
      --learning_rate=0.0001 \
      --weight_decay=0.0001 \
      --device="${DEVICE}" \
      --value_is_current_player=true \
      --report_every=500 \
      --save_final_checkpoint=true \
      --save_best_holdout_checkpoint=true
  fi

  log "Round ${round}: evaluating AZ as player 1"
  run_to_log "${first_log}" \
    "${PLAY}" \
    --game="${GAME}" \
    --player1=az \
    --player2=mcts \
    --az_path="${model}" \
    --az_checkpoint=-3 \
    --az_device="${DEVICE}" \
    --az_value_is_current_player=true \
    --az_batch_size=64 \
    --az_threads=16 \
    --az_cache_size=65536 \
    --az_cache_shards=16 \
    --max_simulations="${MCTS_SIMS}" \
    --rollout_count="${ROLLOUTS}" \
    --num_games="${EVAL_GAMES}" \
    --num_workers="${EVAL_WORKERS}" \
    --solve="${SOLVE}" \
    --quiet=true \
    --seed=$((seed + 101))

  log "Round ${round}: evaluating AZ as player 2"
  run_to_log "${second_log}" \
    "${PLAY}" \
    --game="${GAME}" \
    --player1=mcts \
    --player2=az \
    --az_path="${model}" \
    --az_checkpoint=-3 \
    --az_device="${DEVICE}" \
    --az_value_is_current_player=true \
    --az_batch_size=64 \
    --az_threads=16 \
    --az_cache_size=65536 \
    --az_cache_shards=16 \
    --max_simulations="${MCTS_SIMS}" \
    --rollout_count="${ROLLOUTS}" \
    --num_games="${EVAL_GAMES}" \
    --num_workers="${EVAL_WORKERS}" \
    --solve="${SOLVE}" \
    --quiet=true \
    --seed=$((seed + 202))

  first_wins="$(extract_wins "${first_log}")"
  second_wins="$(extract_wins "${second_log}")"

  az_first="$(awk '{print $1}' <<< "${first_wins}")"
  mcts_second="$(awk '{print $2}' <<< "${first_wins}")"
  mcts_first="$(awk '{print $1}' <<< "${second_wins}")"
  az_second="$(awk '{print $2}' <<< "${second_wins}")"

  az_wins=$((az_first + az_second))
  mcts_wins=$((mcts_first + mcts_second))
  decisive=$((az_wins + mcts_wins))

  rate="$(awk -v a="${az_wins}" -v d="${decisive}" 'BEGIN { if (d == 0) print 0; else printf "%.6f", a / d }')"
  passed=false

  if awk -v r="${rate}" -v p="${PASS_RATE}" 'BEGIN { exit !(r >= p) }'; then
    passed=true
  fi

  {
    echo "round=${round}"
    echo "model=${model}"
    echo "az_first_wins=${az_first}"
    echo "mcts_second_wins=${mcts_second}"
    echo "az_second_wins=${az_second}"
    echo "mcts_first_wins=${mcts_first}"
    echo "az_wins=${az_wins}"
    echo "mcts_wins=${mcts_wins}"
    echo "decisive=${decisive}"
    echo "az_non_draw_win_rate=${rate}"
    echo "passed=${passed}"
    echo "generate_log=${generate_log}"
    echo "train_log=${train_log}"
    echo "first_log=${first_log}"
    echo "second_log=${second_log}"
    echo
  } | tee -a "${SUMMARY}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${round}" "${model}" "${az_first}" "${mcts_second}" "${az_second}" \
    "${mcts_first}" "${az_wins}" "${mcts_wins}" "${decisive}" "${rate}" \
    "${passed}" >> "${SUMMARY_TSV}"

  if [[ "${passed}" == true ]]; then
    log "PASS round=${round} model=${model} rate=${rate}"
    exit 0
  fi

  prev_model="${model}"
done

log "NOT_PASS best logs in ${RUN}"
exit 1
