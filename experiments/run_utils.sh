#!/bin/bash

# Run commands with GPU
run_with_gpu() {
  local gpus_str_ref="$1"
  local job_id_ref="$2"
  local cmd_ref="$3"
  IFS=',' read -r -a gpus_list_ref <<<"$gpus_str_ref"
  local current_gpu_ref="${gpus_list_ref[job_id_ref - 1]}"
  export CUDA_VISIBLE_DEVICES=$current_gpu_ref
  bash -c "$cmd_ref"
}

# Run commands in parallel with GPU
run_in_parallel() {
  local log_dir=$1
  local gpus_str_ref=$2
  shift 2
  local commands_list_ref=("$@")

  mapfile -t commands_list_ref < <(printf "%s\n" "${commands_list_ref[@]}" | shuf)

  IFS=',' read -r -a gpus_list_ref <<<"$gpus_str_ref"
  local gpus_num_ref=${#gpus_list_ref[@]}
  parallel --results $log_dir -j "$gpus_num_ref" run_with_gpu $gpus_str_ref {%} {} ::: "${commands_list_ref[@]}"
}

export -f run_with_gpu
export -f run_in_parallel
