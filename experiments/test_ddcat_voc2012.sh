#!/bin/bash

source ./attack_utils.sh
init_attack_script "$@"

config_name="test_voc2012"
models=(pspnet_voc2012_ddcat)
epsilons=(0.25 0.5 1.0)

for model in "${models[@]}"; do
  for attack in "${attacks[@]}"; do
    for epsilon in "${epsilons[@]}"; do
      for job_id in "${job_ids[@]}"; do
        task_name=$(make_task_name "$model" "$epsilon" true)
        task_name="${task_name/_rand/_center_crop_rand}"
        commands_to_run+=("$(build_command "$config_name" "$model" "$attack" "$task_name" "$epsilon" 300 true \
          "job_id=$job_id" "data.normalize=false" "data.info.ignore_index=255" "data.dataset.croce_labels=false")")
      done
    done
  done
done

dispatch_commands "ddcat_voc2012_results"
