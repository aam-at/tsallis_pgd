#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"

dataset="${1:-}"

if [[ -z "${dataset}" ]]; then
  echo "Usage: $0 <cityscapes|pascal-voc|pascal-voc-aug|ade20k|all> [root]"
  exit 1
fi

shift

root="${DATA_ROOT:-${project_root}/data}"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  root="${1}"
  shift
fi

if [[ "${dataset}" == "cityscapes" ]]; then
  bash "${project_root}/scripts/download_cityscapes.sh" "${root}" "$@"
  exit 0
fi

python "${project_root}/scripts/prepare_data.py" --dataset "${dataset}" --root "${root}" "$@"
