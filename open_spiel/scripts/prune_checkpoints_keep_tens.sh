#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  open_spiel/scripts/prune_checkpoints_keep_tens.sh [--delete] [--keep-every N] CHECKPOINT_DIR

Keeps:
  checkpoint-0.pt
  checkpoint-10.pt
  checkpoint-20.pt
  checkpoint-<multiple-of-N>.pt
  matching checkpoint-<step>-optimizer.pt files
  checkpoint--1.pt and checkpoint--1-optimizer.pt

Deletes, only with --delete:
  checkpoint-16.pt
  checkpoint-16-optimizer.pt
  any positive checkpoint step that is not a multiple of N

Default mode is dry-run.
EOF
}

delete=false
keep_every=10
checkpoint_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --delete)
      delete=true
      shift
      ;;
    --dry-run)
      delete=false
      shift
      ;;
    --keep-every)
      if [[ $# -lt 2 ]]; then
        echo "Missing value after --keep-every" >&2
        usage >&2
        exit 2
      fi
      keep_every="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "${checkpoint_dir}" ]]; then
        echo "Unexpected extra argument: $1" >&2
        usage >&2
        exit 2
      fi
      checkpoint_dir="$1"
      shift
      ;;
  esac
done

if [[ -z "${checkpoint_dir}" ]]; then
  echo "Missing CHECKPOINT_DIR" >&2
  usage >&2
  exit 2
fi

if [[ ! -d "${checkpoint_dir}" ]]; then
  echo "Not a directory: ${checkpoint_dir}" >&2
  exit 1
fi

if ! [[ "${keep_every}" =~ ^[0-9]+$ ]] || (( keep_every <= 0 )); then
  echo "--keep-every must be a positive integer, got: ${keep_every}" >&2
  exit 2
fi

declare -a keep_files=()
declare -a delete_files=()

while IFS= read -r -d '' file; do
  base="${file##*/}"
  if [[ "${base}" =~ ^checkpoint-(-?[0-9]+)(-optimizer)?\.pt$ ]]; then
    step="${BASH_REMATCH[1]}"
    if (( step < 0 || step % keep_every == 0 )); then
      keep_files+=("${file}")
    else
      delete_files+=("${file}")
    fi
  fi
done < <(find "${checkpoint_dir}" -maxdepth 1 -type f -name 'checkpoint-*.pt' -print0 | sort -z)

echo "Directory: ${checkpoint_dir}"
echo "Keep every: ${keep_every}"
if [[ "${delete}" == true ]]; then
  echo "Mode: delete"
else
  echo "Mode: dry-run"
fi
echo

echo "Keeping ${#keep_files[@]} file(s)."
if ((${#keep_files[@]} > 0)); then
  printf '  %s\n' "${keep_files[@]}"
fi
echo

if ((${#delete_files[@]} == 0)); then
  echo "Nothing to delete."
  exit 0
fi

if [[ "${delete}" == true ]]; then
  echo "Deleting ${#delete_files[@]} file(s):"
  printf '  %s\n' "${delete_files[@]}"
  rm -- "${delete_files[@]}"
else
  echo "Would delete ${#delete_files[@]} file(s):"
  printf '  %s\n' "${delete_files[@]}"
  echo
  echo "Re-run with --delete to actually remove them."
fi
