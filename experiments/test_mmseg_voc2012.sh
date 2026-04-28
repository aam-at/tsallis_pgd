#!/bin/bash

source ./attack_utils.sh
init_attack_script "$@"

config_name="test_voc2012"
models=(
  pspnet_resnet50_voc2012
)
epsilons=(0.25 0.5 1.0)

for model in "${models[@]}"; do
  for attack in "${attacks[@]}"; do
    for epsilon in "${epsilons[@]}"; do
      for job_id in "${job_ids[@]}"; do
        task_name=$(make_task_name "$model" "$epsilon" true)
        task_name="${task_name/_rand/_center_crop_rand}"
        commands_to_run+=("$(build_command "$config_name" "$model" "$attack" "$task_name" "$epsilon" 300 true \
          "job_id=$job_id" "data.normalize=true" "data.crop_size=512" "data.info.ignore_index=255")")
      done
    done
  done
done

dispatch_commands "mmseg_voc2012_results"
