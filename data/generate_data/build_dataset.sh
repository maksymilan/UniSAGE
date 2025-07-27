#!/bin/bash

# ==============================================================================
#                Dataset Build Script for Relational Graphs
#
# Description:
#   This script automates the process of building the nested JSONL graph
#   representations for various relational datasets using 'raw_to_json.py'.
#
#   It processes raw Parquet files and outputs structured JSONL files,
#   which serve as the input for subsequent model training stages.
#
# Usage:
#   1. To build ALL datasets sequentially:
#      ./build_datasets.sh
#
#   2. To build a SINGLE specific dataset:
#      ./build_datasets.sh [dataset_name]
#      Example: ./build_datasets.sh rel-amazon
#
# Supported dataset_names:
#   rel-amazon, rel-avito, rel-event, rel-f1, rel-hm, rel-stack, rel-trial
#
# Prerequisites:
#   - The main Python script 'raw_to_json.py' must be in the same directory.
#   - A configured Python 3 environment with all dependencies from
#     requirements.txt installed.
#   - Raw dataset files must be located in their default path, which is
#     typically '~/.cache/relbench/[dataset_name]/db'.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status, ensuring
# that the script stops on any build error.
set -e

# --- Build Functions for Each Dataset ---

build_amazon() {
    echo "Building dataset: rel-amazon"
    python raw_to_json.py rel-amazon customer
    python raw_to_json.py rel-amazon product
    echo "Build complete for rel-amazon"
}

build_avito() {
    echo "Building dataset: rel-avito"
    python raw_to_json.py rel-avito ad
    python raw_to_json.py rel-avito user
    echo "Build complete for rel-avito"
}

build_event() {
    echo "Building dataset: rel-event"
    python raw_to_json.py rel-event user
    echo "Build complete for rel-event"
}

build_f1() {
    echo "Building dataset: rel-f1"
    python raw_to_json.py rel-f1 driver
    echo "Build complete for rel-f1"
}

build_hm() {
    echo "Building dataset: rel-hm"
    python raw_to_json.py rel-hm customer
    python raw_to_json.py rel-hm article
    echo "Build complete for rel-hm"
}

build_stack() {
    echo "Building dataset: rel-stack"
    python raw_to_json.py rel-stack post
    python raw_to_json.py rel-stack user
    echo "Build complete for rel-stack"
}

build_trial() {
    echo "Building dataset: rel-trial"
    python raw_to_json.py rel-trial study
    echo "Build complete for rel-trial"
}

build_all() {
    echo "Starting build process for all datasets."
    echo "========================================="
    build_amazon
    echo "-----------------------------------------"
    build_avito
    echo "-----------------------------------------"
    build_event
    echo "-----------------------------------------"
    build_f1
    echo "-----------------------------------------"
    build_hm
    echo "-----------------------------------------"
    build_stack
    echo "-----------------------------------------"
    build_trial
    echo "========================================="
    echo "All dataset builds have been completed successfully."
}

# --- Main Execution Logic ---

# Check if a specific dataset name was passed as an argument
if [ -z "$1" ]; then
    # No argument provided, so build all datasets.
    build_all
else
    # An argument was provided. Build the specified dataset.
    case "$1" in
        rel-amazon) build_amazon ;;
        rel-avito)  build_avito ;;
        rel-event)  build_event ;;
        rel-f1)     build_f1 ;;
        rel-hm)     build_hm ;;
        rel-stack)  build_stack ;;
        rel-trial)  build_trial ;;
        *)
            echo "Error: Invalid dataset name '$1'."
            echo "Please use one of the supported names or no argument to build all."
            echo
            echo "Usage: $0 [dataset_name]"
            echo "Supported names: rel-amazon, rel-avito, rel-event, rel-f1, rel-hm, rel-stack, rel-trial"
            exit 1
            ;;
    esac
    echo "Specified dataset build finished successfully."
fi

exit 0