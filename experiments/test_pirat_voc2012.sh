#!/bin/bash

source ./attack_utils.sh
init_attack_script "$@"

config_name="test_voc2012"
models=(
  convnext_t_cvst_robust_voc2012
)
epsilons=(4 8 12)

for model in "${models[@]}"; do
  for attack in "${attacks[@]}"; do
    for epsilon in "${epsilons[@]}"; do
      for job_id in "${job_ids[@]}"; do
        task_name=$(make_task_name "$model" "$epsilon" true)
        task_name="${task_name/_rand/_center_crop_rand}"
        commands_to_run+=("$(build_command "$config_name" "$model" "$attack" "$task_name" "$epsilon" 300 true \
          "job_id=$job_id")")
      done
    done
  done
done

dispatch_commands "pirat_voc2012_results"
