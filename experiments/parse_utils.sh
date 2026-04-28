#!/bin/bash

parse_args() {
  while [[ "$#" -gt 0 ]]; do
    case $1 in
    --help)
      print_help
      exit 0
      ;;
    --dry-run)
      dry_run=true
      ;;
    --gpus)
      shift
      gpus_str="$1"
      IFS=',' read -r -a gpus_list <<<"$1"
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
    esac
    shift
  done
}

print_help() {
  echo "Usage: $0 [OPTION]..."
  echo "Run the script with the following options:"
  echo
  echo "Options:"
  echo "  --help                Display this help message and exit"
  echo "  --dry-run             Run in dry-run mode without executing commands"
  echo "  --gpus GPUS           Specify GPUs to use (comma-separated list of GPU IDs)"
  echo
  echo "Example:"
  echo "  $0 --gpus 0,1,2,3 --dry-run"
}

export -f parse_args
