#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"

root="${DATA_ROOT:-${project_root}/data}"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  root="${1}"
  shift
fi

cityscapes_username="${CITYSCAPES_USERNAME:-}"
if [[ -z "${cityscapes_username}" ]]; then
  read -r -p "Cityscapes username or email address: " cityscapes_username
fi

read -r -s -p "Cityscapes password: " cityscapes_password
echo

credentials_file="$(
  python - <<'PY'
import appdirs
import os

print(
    os.path.join(
        appdirs.user_data_dir("cityscapesscripts", "cityscapes"),
        "credentials.json",
    )
)
PY
)"

cleanup() {
  rm -f "${credentials_file}"
}

trap cleanup EXIT
mkdir -p "$(dirname "${credentials_file}")"

python - <<'PY' "${credentials_file}" "${cityscapes_username}" "${cityscapes_password}"
import json
import os
import stat
import sys

credentials_file, username, password = sys.argv[1:4]
with open(credentials_file, "w", encoding="utf-8") as f:
    json.dump({"username": username, "password": password}, f)
os.chmod(credentials_file, stat.S_IREAD | stat.S_IWRITE)
PY

python "${project_root}/scripts/prepare_data.py" --dataset cityscapes --root "${root}" "$@"
