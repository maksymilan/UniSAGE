# ==============================================================================
#            Automated Training Script for Regression Tasks
#
# Description:
#   This script automates the training for all predefined regression tasks.
#   It now binds the training process to specific CPU cores to manage resources.
#
# Usage:
#   ./run_training_regression.sh
# ==============================================================================

# --- Configuration ---

# Path to your Python script for regression training.
TRAIN_SCRIPT="train_regression.py"

# Root directory for all preprocessed and sampled data.
DATA_ROOT="${HOME}/relbench-data-pt"
TASK_DATA_ROOT="${HOME}/relbench-data-test/sampling"

CPU_CORES_TO_USE="10-19"

# List of all regression tasks to be trained.
TASKS=(
    "rel-amazon/tasks/user-ltv"
    "rel-amazon/tasks/item-ltv"
    "rel-avito/tasks/ad-ctr"
    "rel-event/tasks/user-attendance"
    "rel-f1/tasks/driver-position"
    "rel-hm/tasks/item-sales"
    "rel-stack/tasks/post-votes"
    "rel-trial/tasks/study-adverse"
)

# --- Task Metadata Configuration ---

# DATASET_ID_KEY: The entity ID key used in the preprocessed .pt graph files.
declare -A DATASET_ID_KEYS=(
    ["rel-amazon/tasks/user-ltv"]="customer_id"
    ["rel-amazon/tasks/item-ltv"]="product_id"
    ["rel-avito/tasks/ad-ctr"]="AdID"
    ["rel-event/tasks/user-attendance"]="user_id"
    ["rel-f1/tasks/driver-position"]="driverId"
    ["rel-hm/tasks/item-sales"]="article_id"
    ["rel-stack/tasks/post-votes"]="PostId"
    ["rel-trial/tasks/study-adverse"]="nct_id"
)

# TASK_ID_KEY: The entity ID key used in the task's .parquet files.
declare -A TASK_ID_KEYS=(
    ["rel-amazon/tasks/user-ltv"]="customer_id"
    ["rel-amazon/tasks/item-ltv"]="product_id"
    ["rel-avito/tasks/ad-ctr"]="AdID"
    ["rel-event/tasks/user-attendance"]="user"
    ["rel-f1/tasks/driver-position"]="driverId"
    ["rel-hm/tasks/item-sales"]="article_id"
    ["rel-stack/tasks/post-votes"]="PostId"
    ["rel-trial/tasks/study-adverse"]="nct_id"
)

# LABEL_KEY: The column name of the target label in the task's .parquet files.
declare -A LABEL_KEYS=(
    ["rel-amazon/tasks/user-ltv"]="ltv"
    ["rel-amazon/tasks/item-ltv"]="ltv"
    ["rel-avito/tasks/ad-ctr"]="num_click"
    ["rel-event/tasks/user-attendance"]="target"
    ["rel-f1/tasks/driver-position"]="position"
    ["rel-hm/tasks/item-sales"]="sales"
    ["rel-stack/tasks/post-votes"]="popularity"
    ["rel-trial/tasks/study-adverse"]="num_of_adverse_events"
)

# TIMESTAMP_KEY: The column name of the timestamp in the task's .parquet files.
declare -A TIMESTAMP_KEYS=(
    ["rel-amazon/tasks/user-ltv"]="timestamp"
    ["rel-amazon/tasks/item-ltv"]="timestamp"
    ["rel-avito/tasks/ad-ctr"]="timestamp"
    ["rel-event/tasks/user-attendance"]="timestamp"
    ["rel-f1/tasks/driver-position"]="date"
    ["rel-hm/tasks/item-sales"]="timestamp"
    ["rel-stack/tasks/post-votes"]="timestamp"
    ["rel-trial/tasks/study-adverse"]="timestamp"
)


# --- Main Execution Logic ---

echo "=== Starting Automated Training for Regression Tasks ==="
echo "=== All runs will be bound to CPU cores: [${CPU_CORES_TO_USE}] ==="

for task in "${TASKS[@]}"; do
    echo
    echo "======================================================================"
    echo "Preparing to train task: $task"
    echo "======================================================================"

    dataset_id_key=${DATASET_ID_KEYS[$task]}
    task_id_key=${TASK_ID_KEYS[$task]}
    label_key=${LABEL_KEYS[$task]}
    timestamp_key=${TIMESTAMP_KEYS[$task]}

    if [ -z "$dataset_id_key" ] || [ -z "$task_id_key" ] || [ -z "$label_key" ] || [ -z "$timestamp_key" ]; then
        echo "Error: Metadata for task '$task' is incomplete. Skipping."
        continue
    fi

    processed_path="${DATA_ROOT}/${task}/graph_dataset.pt"
    task_path="${TASK_DATA_ROOT}/${task}/"
    log_file_dir="${DATA_ROOT}/${task}"
    log_file="${log_file_dir}/train_regression.log"

    mkdir -p "$log_file_dir"

    echo "  - Processed data: ${processed_path}"
    echo "  - Task files dir: ${task_path}"
    echo "  - Log file:       ${log_file}"
    echo "  - Dataset ID key: ${dataset_id_key}"
    echo "  - Task ID key:    ${task_id_key}"
    echo "  - Label key:      ${label_key}"
    echo "  - Timestamp key:  ${timestamp_key}"

    if [ ! -f "$processed_path" ]; then
        echo "Error: Preprocessed file not found: '${processed_path}'. Please run preprocessing first. Skipping."
        continue
    fi

    COMMAND="taskset -c ${CPU_CORES_TO_USE} python3 ${TRAIN_SCRIPT} \
        --processed_path \"${processed_path}\" \
        --task_path \"${task_path}\" \
        --dataset_id_key \"${dataset_id_key}\" \
        --task_id_key \"${task_id_key}\" \
        --label_key \"${label_key}\" \
        --timestamp_key \"${timestamp_key}\" \
        --batch_size 1 \
        --gpu_ids 0 \
        --seed 42"

    echo
    echo "Executing command:"
    echo "$COMMAND"
    echo

    eval "$COMMAND" > "$log_file" 2>&1

    echo "Training for task '$task' complete. Log saved to ${log_file}"
done

echo
echo "======================================================================"
echo "All regression training tasks have been executed."
echo "======================================================================"