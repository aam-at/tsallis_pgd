#!/bin/bash
#
# Shared utilities for attack test scripts.
# Source this file, then call init_attack_script, build_command, and dispatch_commands.
#

source ./parse_utils.sh
source ./run_utils.sh

# ── Default job IDs ──────────────────────────────────────────────────────────
DEFAULT_JOB_IDS=(0 1 2)

# ── Default attacks ───────────────────────────────────────────────────────────
DEFAULT_ATTACKS=(
  auto_multi_pgd_ce                                       # pgd
  auto_multi_pgd_ce_cossim                                # cospgd
  auto_multi_pgd_js                                       # js pgd
  "auto_multi_pgd_ce_annealed loss@attack.loss=ce_loss"   # segpgd (ce)
  "auto_multi_pgd_ce_annealed loss@attack.loss=dice_loss" # segpgd (dice)
  auto_multi_pgd_masked_ce                                # masked pgd
  "auto_multi_pgd_tsallis_ce_adaptive loss@attack.loss=adaptive_tsallis_ce_loss_q_linear attack.loss.params.q_start=-2.0 attack.loss.params.q_end=1.0"
)

# ── init_attack_script ────────────────────────────────────────────────────────
# Sets up defaults and parses CLI args.  Call as: init_attack_script "$@"
init_attack_script() {
  dry_run=false
  gpus_str=0
  gpus_list=(0)
  parse_args "$@"
  commands_to_run=()
  attacks=("${DEFAULT_ATTACKS[@]}")
  job_ids=("${DEFAULT_JOB_IDS[@]}")
}

# ── build_command ─────────────────────────────────────────────────────────────
# Builds a single Hydra command line and prints it to stdout.
#
# Usage:
#   build_command <config_name> <model> <attack> <task_name> \
#                 <epsilon> <iterations> <random_start> [extra_cfg ...]
#
# The extra_cfg arguments are appended verbatim (e.g. "data.normalize=true").
build_command() {
  local config_name=$1
  local model_var=$2
  local attack_var=$3
  local task_name=$4
  local epsilon_var=$5
  local iterations=$6
  local random_var=$7
  shift 7

  local cfgs=()
  cfgs+=("python test.py -cn $config_name")
  cfgs+=("model=$model_var")
  cfgs+=("attack=$attack_var")
  cfgs+=("task_name=${task_name}")
  cfgs+=("attack.base_epsilon=${epsilon_var}")
  cfgs+=("attack.base_iterations=${iterations}")
  cfgs+=("attack.random_start=${random_var}")
  cfgs+=("return_at_iterations='[]'")
  cfgs+=("save_adv_predictions=true")

  # Append any extra overrides
  for extra in "$@"; do
    cfgs+=("$extra")
  done

  printf '%s ' "${cfgs[@]}"
}

# ── make_task_name ────────────────────────────────────────────────────────────
# Builds a task name from model + epsilon + random flag.
#
# Usage: make_task_name <model> <epsilon> <random_start>
# Prints: <model>_eps_<epsilon>_rand  or  <model>_eps_<epsilon>_norand
make_task_name() {
  local model=$1
  local epsilon=$2
  local random_var="${3,,}"
  local task_name="${model}_eps_${epsilon}"
  if [[ $random_var == true ]]; then
    task_name+="_rand"
  else
    task_name+="_norand"
  fi
  echo "$task_name"
}

# ── dispatch_commands ─────────────────────────────────────────────────────────
# Either prints (dry-run) or runs commands from the commands_to_run array.
#
# Usage: dispatch_commands <result_dir>
dispatch_commands() {
  local result_dir=$1
  if [ "$dry_run" = true ]; then
    printf '%s\n' "${commands_to_run[@]}"
  else
    run_in_parallel "$result_dir" "$gpus_str" "${commands_to_run[@]}"
  fi
}

export -f build_command
export -f make_task_name
