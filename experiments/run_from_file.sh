#!/bin/bash

source ./parse_utils.sh
source ./run_utils.sh

cmd_file=""
dry_run=false
gpus_str=0
gpus_list=(0)

# Temporarily capture all arguments
filtered_args=()

# Pre-parse to extract --cmd_file and rebuild args for parse_args
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --cmd_file)
            shift
            cmd_file="$1"
            ;;
        *)
            filtered_args+=("$1")
            ;;
    esac
    shift
done

# Run original parser
parse_args "${filtered_args[@]}"

if [[ -n "$cmd_file" ]]; then
    if [[ ! -f "$cmd_file" ]]; then
        echo "Command file '$cmd_file' not found."
        exit 1
    fi
    mapfile -t commands_to_run < "$cmd_file"
else
    echo "No --cmd_file provided. Exiting or handle differently."
    exit 1
fi


if [[ "$dry_run" == true ]]; then
    printf '%s\n' "${commands_to_run[@]}"
else
    run_in_parallel "parallel_job_from_file" "$gpus_str" "${commands_to_run[@]}"
fi
